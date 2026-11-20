"""Teacher force fixed student tapes; retain conditional distributions privately."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from exploratory_shop_audit.grpo_forced_audit.common import MODEL, append, save, digest_file
from .packets import MATCH

def support(s, actual):
    top = s.topk(8).indices
    extra = ~top.eq(actual[:,None]).any(-1)
    ids = torch.cat([top,actual[:,None]],-1)
    mask = torch.ones_like(ids,dtype=torch.bool)
    mask[:,-1]=extra
    return ids,mask

def measure(logp, ids, mask):
    subset=logp.gather(-1,ids)
    mass=subset.masked_fill(~mask,-torch.inf).logsumexp(-1)
    q=(subset-mass[:,None]).exp().masked_fill(~mask,0)
    return subset,q,mass.exp()

def centered(delta,q):
    return delta-(q*delta).sum(-1,keepdim=True)

def kl(q, logp, ref):
    ll=logp-ref
    return (q*ll).sum(-1)

@torch.inference_mode()
def score_row(model,row):
    actual=torch.tensor(row['response'],device='cuda')
    def logits(prompt):
        x=torch.tensor([prompt+row['response']],device='cuda')
        # Only materialize the LM-head output for positions predicting response tokens.
        h=model.model(input_ids=x,use_cache=False).last_hidden_state[:,len(prompt)-1:-1,:]
        s=model.lm_head(h)[0].float().log_softmax(-1)
        if not s.isfinite().all():raise ValueError('SCORER_FAILURE: nonfinite')
        return s
    views=row['views']
    s=logits(views['none']['ids']); ids,mask=support(s,actual)
    base,qs,smass=measure(s,ids,mask)
    greedy_match=s.argmax(-1).eq(actual).tolist()
    del s
    index=(ids==actual[:,None]).long().argmax(-1)[:,None]
    zero=logits(views['none']['ids'])
    empty,_,_=measure(zero,ids,mask)
    error=(empty-base).abs().max().item()
    if error>1e-5:raise ValueError('empty-z scorer mismatch')
    del zero
    dist={}; out={}
    for mode,view in views.items():
        if view['reason']:continue
        p,q,mass=(base,qs,smass) if mode=='none' else measure(logits(view['ids']),ids,mask)
        d=centered(p-base,qs)
        pred=(base+d).masked_fill(~mask,-torch.inf).softmax(-1)
        if not torch.allclose(pred,q,atol=1e-5):raise ValueError('candidate alignment failed')
        logq=p-mass.log()[:,None]
        dist[mode]=(d,q,logq)
        out[mode]=dict(mass=mass.tolist(),delta_y=(p-base).gather(1,index).flatten().tolist(),d_y=d.gather(1,index).flatten().tolist(),d=d.masked_fill(~mask,0).tolist(),
            kl=kl(q,logq,base-smass.log()[:,None]).tolist())
    for m,placeholder in MATCH.items():
        if m not in dist or placeholder not in dist:continue
        d,q,logq=dist[m]; dp,_,logqp=dist[placeholder]
        out[m].update(content_d_y=(d-dp).gather(1,index).flatten().tolist(),content_kl=kl(q,logq,logqp).tolist())
    return {k:row[k] for k in ('goal','rep','step','page','classes')} | dict(actual=row['response'],ids=ids.tolist(),candidate_mask=mask.tolist(),qs=qs.tolist(),scores=out,na={m:v['reason'] for m,v in views.items() if v['reason']},greedy_match=greedy_match,empty_max=error)

def worker(run,rank,size):
    from transformers import AutoModelForCausalLM
    out=run/f'scores_{rank}.jsonl'
    if out.exists():return
    partial=run/f'scores_{rank}_{time.time_ns()}.partial'
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16).eval().to('cuda')
    count=0
    for tape in sorted(run.glob('smoke_batch_*/score_inputs.jsonl')):
        with tape.open() as f:
            for line in f:
                if count%size==rank:
                    row=json.loads(line)
                    append(partial,score_row(model,row))
                count+=1
    if count==0:raise ValueError('empty student tape')
    partial.rename(out)

def score(run):
    if len(list(run.glob('smoke_batch_*/episodes.jsonl')))!=2:raise ValueError('missing behavior batches')
    start=time.monotonic()
    children=[]
    try:
        for gpu in range(4):
            children.append(subprocess.Popen([sys.executable,'-m',__spec__.name,str(run),'--rank',str(gpu)],env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)}))
        if any(code!=0 for code in [p.wait() for p in children]):raise RuntimeError('SCORER_FAILURE; see worker output')
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:p.wait()
        append(run/'preflight_cost_ledger.jsonl',dict(stage='scoring',gpus=4,gpu_hours=4*(time.monotonic()-start)/3600))

if __name__=='__main__':
    os.umask(0o077)
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--rank',type=int);a=p.parse_args()
    if a.rank is None:score(a.run)
    else:worker(a.run,a.rank,4)
