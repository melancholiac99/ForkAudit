"""Read-only position weights and the Rnt teacher for the registered screens."""
import torch
from torch.nn import functional as F

@torch.no_grad()
def unpredictability(base,inputs,sp,teacher,labels,state):
    hidden=base.model(input_ids=inputs['student_input_ids'],attention_mask=inputs['student_attention_mask'],
                      use_cache=False).last_hidden_state
    head=base.get_output_embeddings()
    if 'vocab' not in state:state['vocab']=F.linear(head.weight.float(),state['B'])
    mask=labels.ne(-100)
    u=torch.zeros_like(labels,dtype=torch.float32)
    for b in range(len(hidden)):
        hs=hidden[b,sp-1:-1];valid=mask[b].nonzero().flatten()
        for ids in valid.split(32):
            if not ids.numel():continue
            h=hs[ids];s=head(h).float();t=teacher[b,ids].float()
            union=torch.cat((s.topk(32).indices,t.topk(32).indices),-1).sort(-1).values
            delta=state['scale']*torch.einsum('pr,pcr->pc',F.linear(h.float(),state['A']),state['vocab'][union])
            # A repeated ID has the same value in both columns. Assign only distinct entries.
            keep=torch.ones_like(union,dtype=torch.bool);keep[:,1:]=union[:,1:]!=union[:,:-1]
            i,j=keep.nonzero(as_tuple=True)
            s[i,union[i,j]]+=delta[i,j]
            lr=F.log_softmax(t/1.1,-1); lp=F.log_softmax(s/1.1,-1)
            u[b,ids]=(lr.exp()*(lr-lp)).sum(-1).clamp_min(0)
    vals=u[mask]
    if vals.numel():
        if vals.max()>0:
            raw=(vals/vals.mean()).clamp_max(5.)
            u[mask]=raw/raw.mean()
        else:u[mask]=1.
    if not torch.isfinite(u).all():raise ValueError('nonfinite Rw weights')
    return u

def remove_think(t,token):
    t[...,token]=-torch.inf
    return t

def weighted_sum(elements,w):
    # Retain the official sum for the uniform component. Applying the remaining
    # weight after the clipped vocabulary reduction avoids a second vocabulary tensor.
    per_position=elements.sum(dim=-1,dtype=torch.float32)
    return elements.sum()+(per_position*(w.float()-1.)).sum()
