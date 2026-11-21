"""Full-vocabulary movement on every native response position of the shared D batch."""
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .c_targets import actor, response
from .d_data import folder, snapshot
from .parser import command_parts
from .s21_data import SmokeBatch
from .s21_store import ARMS, RUN as S21
from .s22_measure import matched_null, restore
from exploratory_shop_audit.grpo_forced_audit.common import MODEL, read, save, digest_file

RUN=S21.parent/'shop_visible_opd_e25_20261121'
PUBLIC=Path(__file__).parent/'evidence/section25_20261121'
GROUPS=('click','option','search','verb','syntax','other')
SEED=20261121

def configure():
    os.umask(0o077)
    rank=int(os.environ['LOCAL_RANK'])
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(rank)
    return rank

def labels(tok,row):
    parts,m=command_parts(tok,row['answer'])
    offset=0;out=[]
    for bs,c in zip(parts,row['classes'],strict=True):
        end=offset+len(bs)
        if c=='format': c='verb' if m and end>m.start(1) and offset<m.end(1) else 'syntax'
        out.append(c if c in GROUPS else 'other');offset=end
    return out

def pooled(batch,rank,owners=None):
    all_ids=range(len(batch.rows))
    mapping=defaultdict(list)
    for i in all_ids:
        row=batch.rows[i];owner=owners[row['goal']] if owners is not None else None
        mapping[(owner,tuple(row['prompt']),tuple(row['answer']))].append(i)
    first=[v[0] for v in mapping.values()]
    local=set(batch.local(first,rank)) if owners is None else {i for i in first if owners[batch.rows[i]['goal']]==rank}
    return [v for v in mapping.values() if v[0] in local]

def add(totals,name,rows,indices,w):
    if not indices:return
    t=totals[name];t[0]+=len(indices);t[1]+=len(indices)*w
    for j,value in enumerate(rows[indices].sum(0).tolist(),2):t[j]+=value*w

def full_metrics(base,end):
    a=base.double().log_softmax(-1);b=end.double().log_softmax(-1)
    p=a.exp();q=b.exp()
    return torch.stack(((p*(a-b)).sum(-1),-(q*b).sum(-1),q.max(-1).values),-1).cpu()

def reduce_rows(dst,prefix):
    shards=[read(dst/f'{prefix}_{i}.json') for i in range(4)]
    sums={key:[sum(r['stats'][key][i] for r in shards) for i in range(5)] for key in ('all',*GROUPS)}
    assert sums['all'][0]==4448 and abs(sums['all'][1]-1)<1e-10 and sum(v[0] for k,v in sums.items() if k!='all')==4448
    return {key:dict(positions=n,mass=w,kl=kl/w if w else None,entropy=h/w if w else None,top1=z/w if w else None)
            for key,(n,w,kl,h,z) in sums.items()}

def baseline_logits(model,batch,chunks,rank,dtype,root=RUN):
    dest=root/'base_logits'/str(dtype);dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'rank_{rank}.pt'
    if path.exists():return torch.load(path,mmap=True,map_location='cpu',weights_only=True)
    cache={}
    for ii in chunks:
        i=ii[0];row=batch.rows[i]
        cache[i]=response(model,row['prompt'],row['answer'])[1].to(dtype).cpu()
    tmp=path.with_suffix('.tmp');torch.save(cache,tmp);tmp.replace(path)
    return cache

@torch.inference_mode()
def preflight(rank):
    b=SmokeBatch();chunks=pooled(b,rank);tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    cats=[labels(tok,row) for row in b.rows];dst=RUN/'movement';dst.mkdir(parents=True,exist_ok=True)
    initial=torch.load(S21/'original/initial.pt',map_location='cpu',weights_only=True)
    for name,dtype in [('bf16',torch.bfloat16),('fp32',torch.float32)]:
        start=time.monotonic()
        model=actor(dtype=dtype).eval();params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        sha,_=restore(params,initial,initial);assert sha==read(folder(1,'P')/'optimizer.json')['before_sha']
        base=baseline_logits(model,b,chunks,rank,dtype)
        for arm in (*ARMS,'null'):
            path=dst/f'{name}_{arm}_{rank}.json'
            if path.exists():continue
            if arm=='null':
                snapshot(model,10,'P')
                norm=sum((p.cpu().double()-initial[n].double()).square().sum().item() for n,p in params)**.5
                endsha,actual=restore(params,initial,matched_null(initial,norm,SEED),)
                assert abs(actual/norm-1)<1e-5
            else:
                snapshot(model,10,arm);endsha=read(folder(10,arm)/'optimizer.json')['after_sha'];actual=norm=None
            totals={key:[0,0.,0.,0.,0.] for key in ('all',*GROUPS)}
            for ii in chunks:
                i=ii[0];r=b.rows[i];end=response(model,r['prompt'],r['answer'])[1]
                values=full_metrics(base[i].cuda(),end)
                for j in ii:
                    group=cats[j];add(totals,'all',values,list(range(len(group))),b.weights[j])
                    for c in set(group):add(totals,c,values,[k for k,x in enumerate(group) if x==c],b.weights[j])
            save(path,dict(stats=totals,actor_sha=endsha,norm=actual,p_norm=norm,initial_sha=sha,seconds_since_precision_start=time.monotonic()-start))
            print(rank,name,arm,flush=True)
        del base,params,model
        torch.cuda.empty_cache()

def movement_report():
    scores={precision:{a:reduce_rows(RUN/'movement',f'{precision}_{a}') for a in (*ARMS,'null')} for precision in ('bf16','fp32')}
    rho=(scores['bf16']['R']['all']['kl']/scores['bf16']['P']['all']['kl'])**.5
    result=dict(scores=scores,rho=rho,needs_matched=not .8<=rho<=1.25,initial_matched_lr=1e-6/rho,
        null_seed=SEED,common_records=282,positions=4448,signed_batch=SmokeBatch().signatures,
        matched_null=read(RUN/'movement/bf16_null_0.json')['norm'])
    save(PUBLIC/'movement.json',result)
    save(RUN/'movement.json',result)
    print('rho',rho,'R-matched',result['needs_matched'],flush=True)

if __name__=='__main__':
    if sys.argv[1:] == ['report']:movement_report()
    else:preflight(configure())
