"""GPU reconstruction using the archived per-position union and held-out predictor."""
import sys,time,os
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from common import MODEL,WORK,save
from fit import split
from token_kl import TEMPERATURES

OUT=WORK/'rres-gpu'

@torch.inference_mode()
def process(worker,workers,probability_only=False):
    assert torch.cuda.device_count()==1
    output=WORK/"rres-outside-probability" if probability_only else OUT
    output.mkdir(exist_ok=True)
    folds=split()
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,
                                            attn_implementation='flash_attention_2').cuda().eval()
    head=model.lm_head.weight
    scale=torch.load(WORK/'projector.pt',weights_only=True)['scale']
    params=[]
    for f in range(4):
        state=torch.load(WORK/f'fit-{f}.pt',weights_only=True,map_location='cuda')
        params.append((state['A'],F.linear(head.float(),state['B'])))
    rows=[r for lane in range(4) for r in torch.load(WORK/f'rollouts-{lane}.pt',weights_only=True)]
    rows=rows[worker::workers]
    for start in range(0,len(rows),5):
        dst=output/f'w{worker}-{start}.pt'
        if dst.exists():continue
        mass=torch.zeros(2,2,len(head),dtype=torch.float64,device="cuda")
        outside=torch.zeros(2,2,dtype=torch.float64,device="cuda")
        per_position=[]
        n=0;errsum=0.;maxerr=0.;nc=0
        for idx,_,t,y in rows[start:start+5]:
            x=torch.load(WORK/f'features-{idx}.pt',weights_only=True,map_location='cuda')
            ht=model.model(input_ids=torch.tensor([t+y],device="cuda"),use_cache=False).last_hidden_state[0,len(t)-1:-1]
            A,v=params[folds[idx]]
            n+=len(y)
            for j in range(0,len(y),64):
                hs=x['h'][j:j+64].float()
                s=F.linear(hs.bfloat16(),head).float()
                teacher=F.linear(ht[j:j+64],head).float()
                ids=x['ids'][j:j+64].long()
                mask=x['valid'][j:j+64]
                delta=scale*torch.einsum('pr,pcr->pc',F.linear(hs,A),v[ids])
                adj=(teacher-s).gather(1,ids)
                adj-=((s.gather(1,ids).masked_fill(~mask,-torch.inf).softmax(1))*adj).sum(1,keepdim=True)
                err=(adj-x['d'][j:j+64])[mask].abs()
                errsum+=err.sum().item();maxerr=max(maxerr,err.max().item());nc+=err.numel()
                tgt=teacher.scatter_add(1,ids,-delta.masked_fill(~mask,0))
                if probability_only:
                    chunks=[]
                    for temp in TEMPERATURES:
                        targets=[]
                        for logits in (teacher,tgt):
                            probs=(logits/temp).softmax(-1)
                            probs.scatter_(1,ids,0.)
                            targets.append(probs.sum(-1,dtype=torch.float64))
                        chunks.append(torch.stack(targets,dim=1))
                    per_position.append(torch.stack(chunks,dim=1).cpu())
                    continue
                for k,temp in enumerate(TEMPERATURES):
                    lp=(s/temp).log_softmax(-1)
                    for a,logits in enumerate((teacher,tgt)):
                        lq=(logits/temp).log_softmax(-1)
                        pos=(lq.exp()*(lq-lp)).clamp_min_(0)
                        mass[k,a]+=pos.sum(0)
                        pos.scatter_(1,ids,0.)
                        outside[k,a]+=pos.sum(dtype=torch.float64)
        if probability_only:
            values=torch.cat(per_position)
            assert values.shape==(n,2,2) and torch.isfinite(values).all()
            save(dst,dict(probabilities=values,questions=len(rows[start:start+5]),max_label_error=maxerr))
            continue
        save(dst,dict(mass=mass.cpu(),outside=outside.cpu(),positions=n,questions=len(rows[start:start+5]),
                      label_absolute_sum=errsum,label_count=nc,label_max=maxerr))
        print(worker,start,n,time.time(),flush=True)
if __name__=='__main__':process(int(sys.argv[1]),int(sys.argv[2]),"--probability-only" in sys.argv[3:])
