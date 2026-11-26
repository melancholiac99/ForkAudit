import json
from pathlib import Path
import torch
import torch.nn.functional as F
from objective import block,target
def check():
    torch.manual_seed(42);T=1.1
    errors=[]; scaled=[];rows=0;clipped=0;normalization=0.
    for v,scale in [(8,1),(37,3),(512,6),(64,0)]:
        s=(torch.randn(97,v,dtype=torch.float64)*scale).requires_grad_()
        t=torch.randn_like(s)*scale
        lp=(s/T).log_softmax(-1);lq=(t/T).log_softmax(-1);p=lp.exp();q=lq.exp()
        k=F.kl_div(lp,lq,reduction='none',log_target=True)
        g,=torch.autograd.grad(k.clamp(max=.05).sum()/len(s),s)
        h,=torch.autograd.grad(block(s,t,T)/len(s),s)
        goal,qu,a=target(lp.detach(),lq)
        assert torch.equal(a,k>.05) and not goal.requires_grad and not qu.requires_grad
        pred=torch.where(a,qu*p,-(p*a).sum(-1,keepdim=True)*q)
        errors.append(float((g-h-pred/(len(s)*T)).abs().max().detach()))
        scaled.append(float(((g-h)*len(s)*T-pred).abs().max().detach()))
        normalization=max(normalization,float((goal.exp().sum(-1)-1).abs().max()))
        rows+=len(s);clipped+=int(a.sum())
    assert max(scaled)<1e-12 and normalization<1e-12 and clipped>0
    return dict(dtype='float64',seed=42,temperature=T,positions=rows,elements_in_A=clipped,
                max_abs_error_input_logits_batchmean=max(errors),max_abs_error_scaled_logits_unreduced=max(scaled),
                max_abs_normalization_error=normalization)
if __name__=='__main__':
    data=check();print(data)
    Path(__file__).with_name('gradient_check.json').write_text(json.dumps(data,indent=2)+'\n')
