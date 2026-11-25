"""Measure actual outside probabilities on CPU with the archived candidate IDs."""
import os,sys
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from common import MODEL,WORK,save
from fit import split
from token_kl import TEMPERATURES

@torch.inference_mode()
def run(lane):
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    out=WORK/'rres-outside-probability';out.mkdir(exist_ok=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa').eval()
    for m in model.modules():
        if isinstance(m,torch.nn.Linear):
            m.weight=torch.nn.Parameter(m.weight.float(),requires_grad=False)
            bias=m.bias.float() if m.bias is not None else None
            m.forward=lambda x,m=m,b=bias:F.linear(x.float(),m.weight,b).bfloat16()
    head=model.lm_head.weight
    scale=torch.load(WORK/'projector.pt',weights_only=True)['scale'];folds=split()
    params=[]
    for k in range(4):
        z=torch.load(WORK/f'fit-{k}.pt',weights_only=True)
        params.append((z['A'],F.linear(head,z['B'])))
    rows=torch.load(WORK/f'rollouts-{lane}.pt',weights_only=True)
    for j in range(0,len(rows),5):
        dest=out/f'w{lane}-{j}.pt'
        if dest.exists():continue
        values=[];err=0.
        for idx,_,t,y in rows[j:j+5]:
            x=torch.load(WORK/f'features-{idx}.pt',weights_only=True)
            ht=model.model(input_ids=torch.tensor([t+y]),use_cache=False).last_hidden_state[0,len(t)-1:-1]
            A,v=params[folds[idx]]
            for p in range(0,len(y),64):
                t0=model.lm_head(ht[p:p+64]).float()
                ids=x['ids'][p:p+64].long(); valid=x['valid'][p:p+64]
                h=x['h'][p:p+64].float()
                fix=scale*torch.einsum('pr,pcr->pc',F.linear(h,A),v[ids])
                inside=t0.gather(1,ids)
                if p==0:
                    si=(head[ids[0]]@h[0]).bfloat16().float()
                    d=inside[0]-si
                    d=d-(d*si.masked_fill(~valid[0],-torch.inf).softmax(0)).sum()
                    err=max(err,(x['d'][0]-d).abs()[valid[0]].max().item())
                t0.scatter_(1,ids,-torch.inf)
                probs=[]
                for temp in TEMPERATURES:
                    o=(t0/temp).logsumexp(-1).double()
                    def outside(i):
                        l=(i/temp).masked_fill(~valid,-torch.inf).logsumexp(-1).double()
                        return torch.sigmoid(o-l)
                    probs.append(torch.stack([outside(inside),outside(inside-fix)],1))
                values.append(torch.stack(probs,1))
        save(dest,dict(probabilities=torch.cat(values),questions=len(rows[j:j+5]),max_label_error=err,device='cpu'))
if __name__=='__main__':run(int(sys.argv[1]))
