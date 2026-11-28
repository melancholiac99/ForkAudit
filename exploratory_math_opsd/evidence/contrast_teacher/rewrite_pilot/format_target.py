"""Conditional exponential tilt preserving every format token's student probability."""
import json
from functools import lru_cache
from pathlib import Path
import torch
import torch.nn.functional as F

@lru_cache(maxsize=None)
def format_mask(size,device):
    ids=json.loads(Path(__file__).with_name('format_tokens.json').read_text())['ids']
    mask=torch.zeros(size,dtype=torch.bool,device=device)
    mask[ids]=True
    return mask

def eligible_positions(labels):
    valid=labels.ne(-100)
    if valid.shape[-1]:valid[:,0]=False
    return valid

def targets(student,positive,negative,mask,temperature):
    dtype=torch.float64 if student.dtype==torch.float64 else torch.float32
    lp=F.log_softmax(student.to(dtype)/temperature,-1)
    with torch.no_grad():
        raw=.5*(F.log_softmax(positive.to(dtype)/temperature,-1)-F.log_softmax(negative.to(dtype)/temperature,-1))
        target=lp.detach().clone()
        active=~mask
        if active.any():
            base=lp.detach()[...,active]
            tilted=base+raw[...,active].clamp(-3,3)
            target[...,active]=tilted+(base.logsumexp(-1,keepdim=True)-tilted.logsumexp(-1,keepdim=True))
    return lp,target,raw

def contrast_kl(student,positive,negative,has_negative,temperature=1.1):
    mask=format_mask(student.shape[-1],student.device)
    lp,target,raw=targets(student,positive,negative,mask,temperature)
    target=torch.where(has_negative[...,None],target,lp.detach())
    clipped_mass=(target.exp()*((raw.abs()>3)&~mask)).sum(-1)*has_negative
    return F.kl_div(lp,target,log_target=True,reduction='none').sum(-1)*has_negative,target,clipped_mass
