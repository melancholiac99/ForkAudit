"""Small compatibility hook for old verl-agent, following modern verl bypass PPO.

No actor loss is replaced. Native mode logs mismatch only. Bypass mode replaces
the PPO anchor with actual sampler probabilities once, without another IS weight.
"""
import inspect
import json
import math
import os
import textwrap
from pathlib import Path

import torch


def ess_fraction(log_weights):
    if log_weights.numel() == 0:
        return None
    x = log_weights.double()
    return float((2 * torch.logsumexp(x, 0) - torch.logsumexp(2*x, 0) - math.log(len(x))).exp())


def metrics(training, rollout, mask):
    assert training.shape == rollout.shape == mask.shape
    mask = mask.bool()
    delta = (training.detach().double() - rollout.detach().double()).cpu()
    selected = delta[mask.cpu()]
    assert selected.numel() > 0 and torch.isfinite(selected).all(), 'Nonfinite actual sampler/trainer probabilities'
    lengths = mask.sum(-1).cpu()
    sums = torch.where(mask.cpu(), delta, 0).sum(-1)
    sums = sums[lengths > 0]
    ratio = selected.exp()
    k3 = torch.expm1(-selected) + selected
    assert torch.isfinite(ratio).all() and torch.isfinite(k3).all(), 'Overflowing ratio; stop before updating'
    result = dict(token_count=selected.numel(), response_count=len(sums),
        sampled_behavior_k3=float(k3.mean()), clip_fraction=float(((ratio < .8) | (ratio > 1.2)).double().mean()),
        token_ESS_fraction=ess_fraction(selected), response_ESS_fraction=ess_fraction(sums),
        token_ratio_min=float(ratio.min()), token_ratio_max=float(ratio.max()),
        response_log_ratio_min=float(sums.min()), response_log_ratio_max=float(sums.max()),
        long_response_count=int((lengths >= 256).sum()))
    for name, q in [('p01', .01), ('p50', .5), ('p99', .99), ('p999', .999)]:
        result['token_ratio_' + name] = float(torch.quantile(ratio, q))
    long_mask = mask.cpu() & (lengths >= 256)[:, None]
    if long_mask.any():
        d = delta[long_mask]
        r = d.exp()
        result.update(long_sampled_k3=float((torch.expm1(-d) + d).mean()),
            long_clip_fraction=float(((r < .8) | (r > 1.2)).double().mean()),
            long_token_ESS_fraction=ess_fraction(d))
    return result


def apply(data, mode, step=None):
    assert mode in ['native', 'bypass_ppo_clip']
    actor = data.batch['old_log_probs']
    rollout = data.batch['rollout_log_probs']
    mask = data.batch['response_mask']
    result = metrics(actor, rollout, mask)
    output = os.environ.get('FORKAUDIT_VERL_OUTPUT')
    if output and step is not None:
        import numpy as np
        folder = Path(output)
        np.savez_compressed(folder / f'probabilities_step_{step}.npz',
            actor=actor.detach().float().cpu().numpy(), rollout=rollout.detach().float().cpu().numpy(),
            mask=mask.detach().cpu().numpy())
        with (folder / 'mismatch.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step, mode=mode, **result)) + '\n')
    assert result['sampled_behavior_k3'] <= .02, 'Batch behavior k3 exceeds frozen 0.02 gate'
    assert result['clip_fraction'] <= .30, 'Batch clip fraction exceeds frozen 0.30 gate'
    if mode == 'bypass_ppo_clip':
        data.batch['old_log_probs'] = rollout.detach().clone()
    return result


def install():
    from verl.trainer.ppo import ray_trainer
    if getattr(ray_trainer.RayPPOTrainer.fit, '_forkaudit_installed', False):
        return
    mode = os.environ.get('FORKAUDIT_VERL_CORRECTION', 'native')
    assert mode in ['native', 'bypass_ppo_clip']
    original = ray_trainer.RayPPOTrainer.fit
    source = textwrap.dedent(inspect.getsource(original))
    anchor = '                if self.use_reference_policy:\n'
    # The branch occurs in the main training loop once; fail on upstream drift.
    assert source.count(anchor) == 1, 'Pinned trainer layout changed; inspect before applying compatibility hook'
    addition = '''            import correction as forkaudit_correction
            forkaudit_stats = forkaudit_correction.apply(batch, __import__('os').environ.get('FORKAUDIT_VERL_CORRECTION', 'native'), self.global_steps)
            metrics.update({'forkaudit/mismatch/' + k: v for k, v in forkaudit_stats.items() if v is not None})

'''
    addition = textwrap.indent(textwrap.dedent(addition), ' ' * 16)
    revised = source.replace(anchor, addition + anchor)
    scope = dict(ray_trainer.__dict__)
    exec(compile(revised, '<forkaudit_old_verl_bypass_compat>', 'exec'), scope)
    ray_trainer.RayPPOTrainer.fit = scope['fit']
    ray_trainer.RayPPOTrainer.fit._forkaudit_installed = True
    original_adjust = ray_trainer.adjust_batch

    def audited_adjust(config, batch):
        before = len(batch)
        adjusted = original_adjust(config, batch)
        output = os.environ.get('FORKAUDIT_VERL_OUTPUT')
        if output:
            record = dict(raw_response_rows=before, adjusted_response_rows=len(adjusted),
                          native_repeated_padding_rows=len(adjusted)-before)
            with (Path(output) / 'native_padding.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
        return adjusted

    ray_trainer.adjust_batch = audited_adjust
    scope['adjust_batch'] = audited_adjust
