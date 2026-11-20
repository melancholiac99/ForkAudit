"""Run the bounded P-pre pilot. Each complete batch is an atomic directory."""
import argparse
import json
import os
from pathlib import Path
import time
from .packets import MATCH, POLICIES, packet, render, categories
from exploratory_shop_audit.grpo_forced_audit.common import read, save, append, MODEL, SOURCE, stable_seed, digest_file
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner
from exploratory_shop_audit.runtime import make_input, parse_generation

def batch(runner, catalog, goals, jobs, folder):
    folder.mkdir()
    active = []
    for task, mode, rep in jobs:
        i = task['goal_index']
        key = f'vc_{i}_{mode}_{rep}'
        active.append(dict(i=i,mode=mode,rep=rep,key=key,history=[],state=runner.engine.call(cmd='open',key=key,goal_index=i)))
    while active:
        requests = []
        for e in active:
            s = e['state']; i = e['i']; mode = e['mode']; t = len(e['history'])
            native = make_input(runner.tokenizer,s['task'],e['history'],s['observation'])
            modes = [*POLICIES, 'placeholder_product','placeholder_evidence'] if mode=='none' else [mode]
            views = {}
            for m in modes:
                fields,why = packet(m,s['public'],catalog,goals[i] if m in ('full','product') else None)
                ids,why,counts,length = render(runner.tokenizer,native,fields,why)
                views[m]=dict(ids=ids,reason=why,counts=counts,length=length)
            e['views']=views
            requests.append(dict(token_ids=views[mode]['ids'],temperature=0 if e['rep']==2 else .7,seed=stable_seed(20261120,i,e['rep'],t)))
        outputs = runner.model.call(requests)
        keep = []
        for e,req,gen in zip(active,requests,outputs,strict=True):
            t = len(e['history']); mode=e['mode']; public=e['state']['public']
            key=dict(goal=e['i'],rep=e['rep'],mode=mode,step=t)
            if mode=='none':
                append(folder/'score_inputs.jsonl',dict(**key,views=e['views'],response=gen['token_ids'],classes=categories(runner.tokenizer,gen['token_ids'],public),page=public['page_type']))
            action=parse_generation(gen['text'])
            out=runner.engine.call(cmd='step',key=e['key'],action=action)
            view=e['views'][mode]
            append(folder/'steps.jsonl',dict(**key,page=public['page_type'],legal=out['legal'],action=action,seed=req['seed'],prompt_ids=req['token_ids'],response_ids=gen['token_ids'],text=gen['text'],packet_status=view['reason'],field_counts=view['counts'],length=view['length']))
            if out['status']['done']:
                st=out['status']
                append(folder/'episodes.jsonl',dict(**key,reward=st['reward'],reason=st['termination_reason'],valid=st.get('reward_valid'),length=t+1))
                runner.engine.call(cmd='close',key=e['key'])
            else:
                if t>=34: raise RuntimeError('Native step cap violated')
                e['history'].append((e['state']['observation'],gen['text']))
                e['state']=out['state'];keep.append(e)
        active=keep

def collect(run):
    os.umask(0o077)
    assert read(run/'teacher_packet_manifest.json')['template_source_sha']==digest_file(Path(__file__).with_name('packets.py'))
    assert read(run/'preflight_report.json')['smoke_tasks']==8
    catalog={p['asin']:p for p in read(SOURCE/'community_products.json')}
    goals=read(run/'snapshots/goals_frozen.json')
    tasks=read(run/'asin_split_manifest.json')['T_pre'][:8]
    jobs=[(t,m,r) for t in tasks for m in POLICIES for r in range(3)]
    os.environ['SHOP_GPU_IDS']='0,1,2,3'
    start=time.monotonic()
    runner=None
    try:
        runner=Runner(run)
        for j in range(0,len(jobs),60):
            target=run/f'smoke_batch_{j}'
            if target.exists():continue
            tmp=run/f'partial_{j}_{time.time_ns()}'
            batch(runner,catalog,goals,jobs[j:j+60],tmp)
            tmp.rename(target)
    finally:
        if runner:runner.close()
        append(run/'preflight_cost_ledger.jsonl',dict(stage='behavior',gpus=4,gpu_hours=4*(time.monotonic()-start)/3600))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);collect(p.parse_args().run)
