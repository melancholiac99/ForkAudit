"""Unclipped forward KL on individual supervised rows, with detached conditional target."""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


def target(lp, lq, ids, conditional):
    with torch.no_grad():
        if not len(ids):
            z = torch.zeros_like(lp[:, 0])
            return lq.detach(), z, z, z.bool()
        ps = lp[:, ids].exp().sum(-1)
        qt = lq[:, ids].exp().sum(-1)
        if not conditional:
            return lq.detach(), ps, qt, torch.zeros_like(ps, dtype=torch.bool)
        special = torch.zeros(lp.shape[-1], device=lp.device, dtype=torch.bool)
        special[ids] = True
        a = lp.masked_fill(special, -torch.inf).logsumexp(-1)
        b = lq.masked_fill(special, -torch.inf).logsumexp(-1)
        guard = (a.exp() < 1e-6) | (b.exp() < 1e-6)
        result = lq + (a - b)[:, None]
        result[:, ids] = lp[:, ids]
        return torch.where(guard[:, None], lp, result), ps, qt, guard


def row_loss(s, t, ids, conditional, temperature):
    lp = (s / temperature).log_softmax(-1)
    lq = (t.detach() / temperature).log_softmax(-1)
    goal, ps, qt, guard = target(lp.detach(), lq, ids, conditional)
    terms = F.kl_div(lp, goal, reduction='none', log_target=True)
    with torch.no_grad():
        original = F.kl_div(lp.detach(), lq, reduction='sum', log_target=True)
        stats = torch.stack([terms.detach().sum(), original, ps.sum(), qt.sum(), guard.sum()])
    return terms, stats


def trainer_loss(trainer, s, t, labels):
    assert trainer.fixed_teacher and trainer.beta == 0 and trainer.jsd_token_clip is None
    assert not trainer.top_k_loss and float(trainer.temperature) == 1.1
    if not hasattr(trainer, '_control_ids'):
        eos = trainer.generation_config.eos_token_id
        ids = sorted(set(eos if isinstance(eos, list) else [eos]) | set(trainer.processing_class.added_tokens_decoder))
        trainer._control_ids = ids
        print('cond_kl_control_tokens', json.dumps({i:trainer.processing_class.convert_ids_to_tokens(i) for i in ids}), flush=True)
    conditional = os.environ['OPSD_ARM'] == 'R-cond'
    valid = (labels != -100).nonzero()
    loss = 0.
    stats = torch.zeros(5, device=s.device, dtype=torch.float64)
    # Recompute short vocab blocks in backward to bound memory on the original microbatch.
    from torch.utils.checkpoint import checkpoint
    for ix in valid.split(64):
        def block(logits, teachers):
            terms, values = row_loss(logits.float(), teachers.float(), trainer._control_ids, conditional, trainer.temperature)
            return terms.sum(), values
        value, sums = checkpoint(block, s[ix[:,0],ix[:,1]], t[ix[:,0],ix[:,1]], use_reentrant=False)
        loss = loss + value
        stats += sums.detach()
    row = dict(step=trainer.state.global_step+1, count=len(valid), sums=stats.tolist())
    path = Path(trainer.args.output_dir) / ('kl-rank-'+os.environ['RANK']+'.jsonl')
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open('a') as stream:
        print(json.dumps(row), file=stream)
    return loss/len(valid)
