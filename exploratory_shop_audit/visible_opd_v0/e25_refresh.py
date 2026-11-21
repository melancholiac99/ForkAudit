"""Re-score the entire selected historical window at one current policy snapshot and cross-fit."""
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from .c_smoke import state_hash
from .c_targets import SEED,STAGE_B,actor,response
from .content_target import content_correction
from .d_data import PARENT,SMOKE,folder as d_folder,snapshot as d_snapshot
from .e25_probe import configure,RUN,PUBLIC
from .e25_projector import FitData
from .projection import candidate_target
from .score import support
from exploratory_shop_audit.grpo_forced_audit.common import read,save,stable_seed,digest_file,digest_value

def fold_of(asin,routing):
    return routing.get(asin,stable_seed(SEED,'projection_fold',asin)%4)

def tape(path):
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as f:
        for line in f:yield json.loads(line)

def source_dir(run,k):
    if k==1:return SMOKE
    return d_folder(k,'P') if run=='benchmark' else RUN/run/'pilot'/f'{k:02}'/'P'

@torch.inference_mode()
def score(rank,run,k,window,dst):
    out=dst/f'features_{rank}.pt';done=dst/f'score_{rank}.json'
    if done.exists():return
    start=time.monotonic();model=actor().eval()
    if run=='benchmark':d_snapshot(model,k,'P')
    else:
        from .e25_layout import Layout
        Layout(run).snapshot(model,k,'P')
    sha=state_hash([(n,p) for n,p in model.named_parameters() if p.requires_grad])
    split=read(PARENT/'asin_split_manifest.json')
    allowed={t['goal_index']:t['asin'] for t in split['T_opd']}
    held={t['goal_index'] for name in ('T_pre','T_cal','V','T_opd_eval') for t in split[name]}
    features=defaultdict(list);goals=[];classes=[];files={};skipped=0;count=0
    for step in range(max(1,k-window+1),k+1):
        root=source_dir(run,step)
        for rep in range(2):
            p=root/f'rollout_{rep}.jsonl'
            if not p.exists():p=p.with_suffix('.jsonl.gz')
            files[f'{step}/{rep}']=digest_file(p)
            for rec in tape(p):
                count+=1
                if (count-1)%4!=rank:continue
                goal=rec['goal'];assert goal in allowed and goal not in held
                if any(rec['na'].values()):skipped+=1;continue
                answer=rec['response'];views=rec['views']
                h,raw=response(model,views['none'],answer)
                ids,mask=support(raw,torch.tensor(answer,device='cuda'))
                s=raw.gather(1,ids);q=candidate_target(s,torch.zeros_like(s),1,mask)
                def lp(name):return response(model,views[name],answer)[1].log_softmax(-1).gather(1,ids)
                d=content_correction(lp('full'),lp('placeholder'),q)
                for key,val in dict(h=h,ids=ids,mask=mask,s=s,q=q,d=d).items():features[key].append(val.cpu())
                goals.extend([goal]*len(answer));classes.extend(rec['classes'])
    data={key:torch.cat(v) for key,v in features.items()};data.update(goals=goals,classes=classes)
    torch.save(data,out.with_suffix('.tmp'));out.with_suffix('.tmp').replace(out)
    save(done,dict(actor_sha=sha,files=files,positions=len(goals),total_records=count,na_local=skipped,
                   goals=sorted(set(goals)),feature_sha=digest_file(out),seconds=time.monotonic()-start))

def fit(rank,dst):
    done=dst/f'fit_{rank}.json'
    if done.exists():return
    start=time.monotonic();parts=[];digests=[]
    for worker in range(4):
        p=dst/f'features_{worker}.pt';m=read(dst/f'score_{worker}.json')
        assert digest_file(p)==m['feature_sha']
        digests.append((m['actor_sha'],digest_value(m['files'])))
        parts.append(torch.load(p,weights_only=True,map_location='cpu',mmap=True))
    assert len(set(digests))==1
    split=read(PARENT/'asin_split_manifest.json');asins={t['goal_index']:t['asin'] for t in split['T_opd']}
    goals=np.array([g for v in parts for g in v['goals']]);classes=np.array([c for v in parts for c in v['classes']])
    bygoal={int(g):fold_of(asins[g],split['cal_folds']) for g in np.unique(goals)}
    folds=np.array([bygoal[g] for g in goals]);tr=np.flatnonzero(folds!=rank)
    gs=set(goals[tr].tolist());vs=set(goals[folds==rank].tolist())
    assert gs and vs and not ({asins[g] for g in gs}&{asins[g] for g in vs})
    ts={key:torch.cat([v[key] for v in parts]) for key in ('q','s','d','h','mask','ids')}
    fitdata=FitData(ts,dict(goal=goals,cls=classes),torch.load(STAGE_B/'head.pt',weights_only=True,mmap=True),'cuda')
    net=fitdata.fit(tr,rank)
    path=dst/f'content_{rank}.pt';torch.save({k:v.cpu() for k,v in net.state_dict().items()},path)
    save(done,dict(actor_sha=digests[0][0],history_sha=digests[0][1],train_goals=sorted(gs),holdout_goals=sorted(vs),
         training_asins=sorted({asins[g] for g in gs}),routing=bygoal,trained_positions=len(tr),
         total_positions=len(folds),projector_sha=digest_file(path),seconds=time.monotonic()-start))

if __name__=='__main__':
    phase,run,step,window=sys.argv[1:];step,window=int(step),int(window)
    if run=='benchmark':assert (RUN/'lora_deadzone.json').exists() and step==window==10
    rank=configure();dst=RUN/run/'refresh'/f'{step:03}';dst.mkdir(parents=True,exist_ok=True)
    if phase=='score':score(rank,run,step,window,dst)
    elif phase=='fit':fit(rank,dst)
    else:raise ValueError(phase)
