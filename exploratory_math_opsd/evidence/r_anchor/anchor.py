"""Frozen student-input distribution and monitoring. R loss is computed upstream."""
import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback


def log_probs(logits, temperature):
    return (logits.float() / temperature).log_softmax(-1)


def anchor_kl(actor, base, temperature):
    student = log_probs(actor, temperature)
    frozen = log_probs(base.detach(), temperature)
    return (frozen.exp() * (frozen - student)).sum(-1)


def gradient_check():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        actor = torch.randn(5, 17, requires_grad=True)
        base = torch.randn_like(actor)
        (0.1 * anchor_kl(actor, base, 1).mean()).backward()
        expected = .1 * (actor.softmax(-1) - base.softmax(-1)) / len(actor)
        error = (actor.grad - expected).abs().max().item()
        assert error < 1e-7, error
        return error


def augment(trainer, model, inputs, actor, teacher, labels, official):
    assert trainer.temperature == 1.1 and trainer.jsd_token_clip == .05 and trainer.beta == 0
    with torch.no_grad(), trainer.accelerator.unwrap_model(model).disable_adapter():
        base = model(input_ids=inputs['student_input_ids'], attention_mask=inputs['student_attention_mask']).logits
        base = base[:, inputs['student_prompt_length'] - 1:-1]
    assert base.shape == actor.shape == teacher.shape
    # Chunk the vocabulary-wide work over rows; checkpoint activations to bound memory.
    from torch.utils.checkpoint import checkpoint
    total = actor.new_zeros((), dtype=torch.float32)
    count = clipped = 0
    for b in range(actor.shape[0]):
        for ids in torch.where(labels[b] != -100)[0].split(64):
            if not len(ids):
                continue
            a, z = actor[b, ids], base[b, ids]
            total = total + checkpoint(anchor_kl, a, z, 1.1, use_reentrant=False).sum()
            with torch.no_grad():
                # Match the official loss's dtypes and comparisons.
                lp = (a / 1.1).log_softmax(-1)
                lq = (teacher[b, ids] / 1.1).log_softmax(-1)
                k = torch.nn.functional.kl_div(lp, lq, reduction='none', log_target=True)
                clipped += int((k > .05).sum())
            count += len(ids)
    mean = total / count
    value = mean.detach().item()
    if trainer.state.global_step == 0:
        if not abs(value) <= .001:
            raise RuntimeError(f'Initial anchor KL exceeds BF16 reference: {value}')
        with (Path(trainer.args.output_dir)/f'initial-rank{trainer.accelerator.process_index}.jsonl').open('a') as f:
            print(json.dumps(dict(kl=value, positions=count)), file=f)
    trainer.anchor_counts[2] += value * count
    trainer.anchor_counts[3] += official.detach().item() * count
    trainer.anchor_counts[4] += count
    trainer.anchor_counts[5] += clipped
    trainer.anchor_counts[6] += count * actor.shape[-1]
    return official + .1 * mean


class Monitor(TrainerCallback):
    def __init__(self, trainer, model_path):
        self.trainer = trainer
        trainer.anchor_counts = [0.] * 7
        trainer.anchor_eos = set(json.loads((Path(model_path)/'generation_config.json').read_text())['eos_token_id'])

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 25:
            return
        counts = torch.tensor(self.trainer.anchor_counts, dtype=torch.float64, device=self.trainer.accelerator.device)
        dist.all_reduce(counts)
        ended, n, kl, r, positions, clipped, elements = counts.tolist()
        assert n == 800
        stop = ended / n < .15
        if state.is_world_process_zero:
            row = dict(step=state.global_step, natural_end_rate=ended/n, responses=int(n),
                       positions=int(positions), anchor_kl=kl/positions, official_R=r/positions,
                       clip_fraction=clipped/elements, stopped=stop,
                       scheduler=args.lr_scheduler_type.value, warmup_steps=args.warmup_steps,
                       lr=self.trainer.lr_scheduler.get_last_lr())
            with (Path(args.output_dir)/'anchor_windows.jsonl').open('a') as f:
                print(json.dumps(row), file=f)
        self.trainer.anchor_counts = [0.] * 7
        if stop:
            control.should_save = control.should_training_stop = True
        return control
