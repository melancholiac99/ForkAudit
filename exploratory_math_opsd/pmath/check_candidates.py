"""Exercise the installed official loss on CPU before training either candidate."""
import sys
from common import ROOT
sys.path.insert(0,str(ROOT/'source'))
import torch
from opsd_trainer import OPSDTrainer
from rw_objective import remove_think
from transformers import AutoTokenizer
from common import MODEL
def check():
    gen=torch.Generator().manual_seed(42)
    token=AutoTokenizer.from_pretrained(MODEL).convert_tokens_to_ids('<think>')
    loss=OPSDTrainer.generalized_jsd_loss
    s=torch.randn((2,3,31),generator=gen).bfloat16().requires_grad_()
    t=torch.randn((2,3,31),generator=gen).bfloat16()
    y=torch.tensor([[1,2,-100],[4,5,6]])
    opts=dict(beta=0.,temperature=1.1,token_clip=.05)
    for i,j in (y!=-100).nonzero().tolist():
        sl=s[i:i+1,j:j+1];tl=t[i:i+1,j:j+1];l=y[i:i+1,j:j+1]
        assert torch.equal(loss(sl,tl,labels=l,**opts),loss(sl,tl,labels=l,position_weights=torch.ones_like(l),**opts))
    assert torch.equal(loss(s,t,labels=y,**opts),loss(s,t,labels=y,position_weights=torch.ones_like(y),**opts))
    teacher=torch.randn((1,2,token+2),generator=gen)
    changed=remove_think(teacher.clone(),token)
    assert torch.isneginf(changed[...,token]).all() and (teacher!=changed).sum()==2
    p=(changed/1.1).softmax(-1)
    assert (p[...,token]==0).all()
    student=torch.zeros_like(changed,requires_grad=True)
    v=loss(student,changed,labels=y[:1,:2],zero_target_token=token,**opts)
    v.backward()
    assert torch.isfinite(v) and torch.isfinite(student.grad).all()
    return dict(unit_weight_exact=True,only_think_logit_changed=True,zero_teacher_probability=True,loss_and_gradient_finite=True)
