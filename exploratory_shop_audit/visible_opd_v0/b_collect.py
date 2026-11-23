"""Native on-policy trajectories from T_cal, with disjoint student and scorer views."""
import argparse,os,time
from pathlib import Path
from .parser import categories
from .packets import packet,render
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner
from exploratory_shop_audit.grpo_forced_audit.common import ROOT,read,save,append,stable_seed
from exploratory_shop_audit.runtime import make_input,parse_generation
PARENT=ROOT/'.runtime/runs/shop_visible_opd_v04_s16_20261120'
RUN=ROOT/'.runtime/runs/shop_visible_opd_content_b_20261120'

def batch(runner,tasks,goals,path,rep=0,stream='T_cal',include_legal=False):
    return native_batches(runner,tasks,goals,{rep:path},stream,include_legal)


def native_batches(runner,tasks,goals,paths,stream,include_legal,view_modes=('none','full','placeholder')):
    active=[]
    for rep in paths:
        for task in tasks:
            i=task['goal_index'];key=f'cal_{i}' if stream=='T_cal' else f'{stream}_{i}_{rep}'
            active.append((i,rep,key,[],runner.engine.call(cmd='open',key=key,goal_index=i)))
    while active:
        requests=[];rows=[]
        for i,rep,key,h,s in active:
            pub=s['public'];native=make_input(runner.tokenizer,s['task'],h,s['observation'])
            views={};reasons={}
            for mode in view_modes:
                fields,why=packet(mode,pub,{},goals[i] if mode=='full' else None)
                ids,why,_,_=render(runner.tokenizer,native,fields,why)
                views[mode]=ids;reasons[mode]=why
            seed=stable_seed(20261120,stream,i,rep,len(h))
            args=[a for a in pub['actions'] if str(a).lower() in s['available']['clickables'] and str(a).lower()!='search']
            rows.append(dict(goal=i,rep=rep,step=len(h),page=pub['page_type'],views=views,na=reasons,
                             legal_clicks=args,products=[p['asin'] for p in pub.get('products',[])],seed=seed))
            requests.append(dict(token_ids=views['none'],temperature=.7,seed=seed))
        keep=[]
        for (i,rep,key,h,s),out,row in zip(active,runner.model.call(requests),rows,strict=True):
            action=parse_generation(out['text'])
            result=runner.engine.call(cmd='step',key=key,action=action)
            row.update(response=out['token_ids'],classes=categories(runner.tokenizer,out['token_ids'],s['public']),
                       action=action,done=result['status']['done'])
            if include_legal:row["legal"]=result["legal"]
            append(paths[rep],row)
            if row['done']:runner.engine.call(cmd='close',key=key)
            else:
                if len(h)>=34:raise ValueError('step cap')
                keep.append((i,rep,key,h+[(s['observation'],out['text'])],result['state']))
        active=keep

def collect(pilot):
    os.umask(0o077)
    assert read(RUN/'packet_selection_v2.json')['selected']=='full'
    if not pilot:assert read(RUN/'p0.json')['pass']
    link=RUN/'snapshots'
    if not link.exists():link.symlink_to(PARENT/'snapshots',target_is_directory=True)
    split=read(PARENT/'asin_split_manifest.json');tasks=split['T_cal']
    assert len({r['goal_index'] for r in tasks})==len({r['asin'] for r in tasks})==512
    goals=read(link/'goals_frozen.json');os.environ['SHOP_GPU_IDS']='0,1,2,3'
    start=time.monotonic();r=Runner(RUN)
    try:
        for j in ([0] if pilot else range(8,512,64)):
            path=RUN/f'tape_{j:03d}.jsonl'
            if path.exists():continue
            tmp=path.with_suffix(f'.{time.time_ns()}.partial')
            batch(r,tasks[j:j+(8 if pilot else 64)],goals,tmp);tmp.rename(path)
    finally:
        r.close();append(RUN/'b_cost.jsonl',dict(stage='collection',pilot=pilot,gpu_hours=4*(time.monotonic()-start)/3600))
if __name__=='__main__':
    import sys
    collect('--pilot' in sys.argv)
