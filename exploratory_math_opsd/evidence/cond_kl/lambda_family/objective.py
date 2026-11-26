"""Conditional KL and the independently adjustable gradient of retained mass."""
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def block(student, teacher, coefficient):
    lp = (student / 1.1).log_softmax(-1)
    with torch.no_grad():
        lq = (teacher / 1.1).log_softmax(-1)
        clipped = F.kl_div(lp.detach(), lq, reduction="none", log_target=True) > .05
        q = lq.exp()
        retained = q.masked_fill(clipped, 0)
        qu = retained.sum(-1)
        log_qu = lq.masked_fill(clipped, -torch.inf).logsumexp(-1)
        pa = lp.detach().exp().masked_fill(~clipped, 0).sum()
        qa = q.masked_fill(~clipped, 0).sum()
    log_pu = lp.masked_fill(clipped, -torch.inf).logsumexp(-1)
    skip = log_pu.detach() < -13.815510557964274  # log(1e-6)
    weighted_kl = (retained * (lq - lp)).sum(-1) - qu * log_qu
    loss = weighted_kl + (1 - coefficient) * qu * log_pu
    stats = torch.stack([clipped.sum(), qa, pa, skip.sum()])
    return loss.masked_fill(skip, 0).sum(), stats.detach()


def training_block(student, teacher, coefficient):
    return block(student.float(), teacher.float(), coefficient)


def trainer_loss(trainer, s, t, labels):
    assert trainer.fixed_teacher and trainer.beta == 0 and not trainer.top_k_loss
    assert float(trainer.temperature) == 1.1
    coefficient = float(os.environ['OPSD_LAMBDA'])
    valid = (labels != -100).nonzero()
    total = s.new_zeros((), dtype=torch.float32)
    stats = torch.zeros(4, device=s.device, dtype=torch.float64)
    for ix in valid.split(64):
        value, chunk_stats = checkpoint(training_block, s[ix[:,0],ix[:,1]], t[ix[:,0],ix[:,1]],
                                         coefficient, use_reentrant=False)
        total = total + value
        stats += chunk_stats
    # Each rank follows the official local supervised-token reduction.
    step = trainer.state.global_step + 1
    path = Path(trainer.args.output_dir) / ('lambda-rank-' + os.environ['RANK'] + '.jsonl')
    with path.open('a') as f:
        print(json.dumps(dict(step=step, count=len(valid), sums=stats.tolist())), file=f)
    return total / (len(valid) - stats[-1].to(total.dtype)).clamp_min(1)
