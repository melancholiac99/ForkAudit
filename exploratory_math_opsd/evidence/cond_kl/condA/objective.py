"""Dynamic detached teacher. Loss and mask use the trainer's temperature-scaled logits."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def target(lp,lq):
    with torch.no_grad():
        a=F.kl_div(lp,lq,reduction='none',log_target=True)>.05
        log_pu=lp.masked_fill(a,-torch.inf).logsumexp(-1,keepdim=True)
        log_qu=lq.masked_fill(a,-torch.inf).logsumexp(-1,keepdim=True)
        guard=log_pu.exp()<1e-6
        goal=torch.where(a,lp,lq+log_pu-log_qu)
        return torch.where(guard,lp,goal),log_qu.exp(),a

def block(s,t,temperature):
    lp=(s/temperature).log_softmax(-1)
    lq=(t.detach()/temperature).log_softmax(-1)
    goal,qu,_=target(lp.detach(),lq)
    return (qu*F.kl_div(lp,goal,reduction='none',log_target=True)).sum()

def trainer_loss(trainer,s,t,labels):
    assert trainer.fixed_teacher and trainer.beta==0 and not trainer.top_k_loss
    assert trainer.jsd_token_clip is None and float(trainer.temperature)==1.1
    valid=(labels!=-100).nonzero()
    loss=s.new_zeros((),dtype=torch.float32)
    for ix in valid.split(64):
        loss=loss+checkpoint(block,s[ix[:,0],ix[:,1]].float(),t[ix[:,0],ix[:,1]].float(),1.1,use_reentrant=False)
    return loss/len(valid)
