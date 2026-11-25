"""CPU reconstruction using the archived per-position union and held-out predictor."""
import sys,time,os
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from common import MODEL,WORK,save
from fit import split
from token_kl import TEMPERATURES

OUT=WORK/'rres-cpu'

@torch.inference_mode()
def process(worker,workers):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    OUT.mkdir(exist_ok=True)
    folds=split()
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
                                            attn_implementation='sdpa').eval()
    # These hosts lack native BF16 matmul. GEMM accumulates in FP32, then
    # rounds the output of each linear layer back to the model's BF16 dtype.
    for module in model.modules():
        if isinstance(module,torch.nn.Linear):
            module.weight=torch.nn.Parameter(module.weight.float(),requires_grad=False)
            bias=module.bias.float() if module.bias is not None else None
            module.forward=lambda x,m=module,b=bias:F.linear(x.float(),m.weight,b).bfloat16()
    head=model.lm_head.weight
    scale=torch.load(WORK/'projector.pt',weights_only=True)['scale']
    params=[]
    for f in range(4):
        state=torch.load(WORK/f'fit-{f}.pt',weights_only=True)
        params.append((state['A'],F.linear(head,state['B'])))
    rows=[r for lane in range(4) for r in torch.load(WORK/f'rollouts-{lane}.pt',weights_only=True)]
    rows=rows[worker::workers]
    for start in range(0,len(rows),5):
        dst=OUT/f'w{worker}-{start}.pt'
        if dst.exists():continue
        mass=torch.zeros(2,2,len(head),dtype=torch.float64)
        outside=torch.zeros(2,2,dtype=torch.float64)
        n=0;errsum=0.;maxerr=0.;nc=0
        for idx,_,t,y in rows[start:start+5]:
            x=torch.load(WORK/f'features-{idx}.pt',weights_only=True)
            ht=model.model(input_ids=torch.tensor([t+y]),use_cache=False).last_hidden_state[0,len(t)-1:-1]
            A,v=params[folds[idx]]
            n+=len(y)
            for j in range(0,len(y),64):
                hs=x['h'][j:j+64].float()
                s=F.linear(hs,head).bfloat16().float()
                teacher=F.linear(ht[j:j+64].float(),head).bfloat16().float()
                ids=x['ids'][j:j+64].long()
                mask=x['valid'][j:j+64]
                delta=scale*torch.einsum('pr,pcr->pc',F.linear(hs,A),v[ids])
                adj=(teacher-s).gather(1,ids)
                adj-=((s.gather(1,ids).masked_fill(~mask,-torch.inf).softmax(1))*adj).sum(1,keepdim=True)
                err=(adj-x['d'][j:j+64])[mask].abs()
                errsum+=err.sum().item();maxerr=max(maxerr,err.max().item());nc+=err.numel()
                tgt=teacher.scatter_add(1,ids,-delta.masked_fill(~mask,0))
                for k,temp in enumerate(TEMPERATURES):
                    lp=(s/temp).log_softmax(-1)
                    for a,logits in enumerate((teacher,tgt)):
                        lq=(logits/temp).log_softmax(-1)
                        pos=(lq.exp()*(lq-lp)).clamp_min_(0)
                        mass[k,a]+=pos.sum(0)
                        pos.scatter_(1,ids,0.)
                        outside[k,a]+=pos.sum(dtype=torch.float64)
        save(dst,dict(mass=mass,outside=outside,positions=n,questions=len(rows[start:start+5]),
                      label_absolute_sum=errsum,label_count=nc,label_max=maxerr))
        print(worker,start,n,time.time(),flush=True)
if __name__=='__main__':process(int(sys.argv[1]),int(sys.argv[2]))
