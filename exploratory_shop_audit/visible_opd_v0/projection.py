"""Student-only rank-eight corrections, task-level calibration and shared KL."""
import torch
from torch import nn
from torch.nn import functional as F

def center(x, probabilities):
    return x-(x*probabilities).sum(-1,keepdim=True)

class VisibleProjection(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        self.state=nn.Linear(hidden_size,8,bias=False)
        self.token=nn.Linear(hidden_size,8,bias=False)

    def forward(self,states,head_rows,qs):
        state=self.state(F.normalize(states.detach().float()))
        tokens=self.token(F.normalize(head_rows.detach().float(),dim=-1))
        return center((state[:,None,:]*tokens).sum(-1),qs.detach())

def candidate_target(snapshot,correction,coefficient,mask):
    with torch.no_grad():
        logits=snapshot.float()+coefficient*correction.float()
        if not mask.any(-1).all() or not logits[mask].isfinite().all():
            raise ValueError('unaligned or nonfinite candidate logits')
        return logits.masked_fill(~mask,-torch.inf).softmax(-1)

def candidate_loss(actor_logits,target,candidate_mask,trajectory_id,task_id):
    """Inputs contain exactly the common valid response positions."""
    if len(target)==0 or not candidate_mask.any(-1).all():raise ValueError('empty support')
    q=target.detach()
    if (q[~candidate_mask]!=0).any() or not torch.allclose(q.sum(-1),torch.ones_like(q[:,0]),atol=1e-5):
        raise ValueError('target/support mismatch')
    actor=actor_logits.float().masked_fill(~candidate_mask,-torch.inf).log_softmax(-1).masked_fill(~candidate_mask,0)
    token_kl=(torch.special.xlogy(q,q)-q*actor).sum(-1)
    means=[]
    for task in task_id.unique():
        same=task_id==task
        means.append(torch.stack([token_kl[same & (trajectory_id==t)].mean() for t in trajectory_id[same].unique()]).mean())
    return torch.stack(means).mean()

def calibrated_slope(pred,real,weights):
    w=weights/weights.sum()
    x=pred-(pred*w).sum();y=real-(real*w).sum()
    var=(w*x*x).sum()
    if not torch.isfinite(var) or var<=1e-10:return None,0.
    raw=(w*x*y).sum()/var
    if not raw.isfinite():return None,0.
    return float(raw),float(raw.clamp(0,1))
