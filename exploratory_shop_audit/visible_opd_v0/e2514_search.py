"""Reconstruct archived evaluation actions without changing batching or decode seeds."""
from collections import Counter, defaultdict
import os
from pathlib import Path
import sys

from .acceptance import ALL_REQUIREMENTS
from .d_eval import episodes, event
from .e25_paths import RUN, PUBLIC
from .e2512_routing import SERIAL, ROUTING, owners
from .e2512_native import checkpoint, native_rank, snapshot_dir
from .e25_layout import Layout
from .d_data import PARENT
from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.grpo_forced_audit.common import read, save, digest_file, digest_value
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner

DATA = RUN/'migration_2514/search'
OUT = PUBLIC/'migration_2514'
POOLS = ('V', 'T_opd_eval')


def original(k):
    arm = 'P' if k else 'B'
    rows = {}
    for path in sorted((RUN/SERIAL/'evaluation'/f'{k:03}'/arm).glob('rank*/episodes_*.json')):
        for row in read(path):
            key = (row['split'], row['goal'], row['rep'])
            assert key not in rows
            rows[key] = row
    assert len(rows) == 2000
    return rows


def selection():
    trials = {k:original(k) for k in (20,30)}
    selected = []
    counts = {}
    for pool in POOLS:
        subset = []
        for goal in sorted({g for s,g,r in trials[20] if s==pool}):
            wins = {k:[i for i in range(4) if trials[k][pool,goal,i]['reason'] in ALL_REQUIREMENTS] for k in trials}
            if len(wins[20]) > len(wins[30]):
                subset.append(dict(split=pool,goal=goal,good_reps=wins[20],
                    failed_reps=[i for i in range(4) if i not in wins[30]],strict_subset=not wins[30]))
        selected.extend(subset)
        counts[pool] = dict(total_tasks=200,eligible=len(subset),strict=sum(r['strict_subset'] for r in subset))
    save(OUT/'search_selection.json',dict(counts=counts,tasks=selected))
    return selected


def observer():
    logs=defaultdict(list)
    def record(e,action,generation,classes,out):
        name,param=parse_action_fixed(action)
        kind=event(e['state']['public'],action,out['legal'])
        row=dict(action=action,event=kind)
        if kind=='search':
            row.update(query=param,arg_tokens=classes.count('search'),
                       generation_ids=generation['token_ids'],
                       products=[p['asin'] for p in out['state']['public'].get('products',[])])
        if kind=='other_click':row['target']=param
        logs[e['key']].append(row)
    return logs,record


def replay(k,rank):
    arm = 'P' if k else 'B'
    l = Layout(SERIAL)
    cp, signature = checkpoint(l,k,arm)
    native_rank(rank)
    path = DATA/'reconstruction'/f'{k:03}'/f'rank{rank}'
    snapshot_dir(path)
    os.environ['OPD_ENTROPY_TAPE'] = str(path/'eval_entropy.bin')
    archive = original(k)
    split = read(PARENT/'asin_split_manifest.json')
    runner = Runner(path,cp,parallel_start=True)
    try:
        for pool in POOLS:
            goals = [t['goal_index'] for t in split[pool] if owners()[t['goal_index']]==rank]
            jobs = [(pool,g,rep) for g in goals for rep in range(5)]
            for start in range(0,len(jobs),32):
                dest=path/f'{pool}_{start:04}.json'
                if dest.exists():continue
                logs,record=observer()
                result=episodes(runner,jobs[start:start+32],action_observer=record)
                recovered=[]
                for r in result:
                    key=(pool,r['goal'],r['rep'])
                    actions=logs['_'.join(map(str,key))]
                    assert digest_value([a['action'] for a in actions])==r['actions_sha']
                    recovered.append(dict(split=pool,goal=r['goal'],rep=r['rep'],actions=actions,
                        actions_sha=r['actions_sha'],matches_original=r['actions_sha']==archive[key]['actions_sha'],
                        reason=r['reason'],original_reason=archive[key]['reason']))
                save(dest,recovered)
    finally:
        runner.close()
    save(path/'complete.json',dict(step=k,rank=rank,checkpoint=signature, routing_sha=digest_file(ROUTING)))


if __name__=='__main__':
    if sys.argv[1]=='selection':selection()
    else:replay(int(sys.argv[1]),int(os.environ['LOCAL_RANK']))
