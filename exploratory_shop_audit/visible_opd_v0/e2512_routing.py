"""Freeze one task-to-device assignment using the initial visible length, before generation."""
from functools import lru_cache
from pathlib import Path

from .d_data import PARENT
from .e25_paths import RUN, PUBLIC, SERIAL
from exploratory_shop_audit.grpo_forced_audit.common import MODEL, digest_file, read, save
from exploratory_shop_audit.runtime import environment, make_input

ROOT=RUN/SERIAL
ROUTING=ROOT/'task_routing.json'


def balanced(goals, lengths):
    if len(goals)%4 or len(set(goals))!=len(goals):
        raise ValueError('Expected a cohort of distinct tasks divisible over four ranks')
    loads=[0]*4
    counts=[0]*4
    owners={}
    for g in sorted(goals,key=lambda g:(-lengths[g],g)):
        rank=min((r for r in range(4) if counts[r]<len(goals)//4),key=lambda r:(loads[r],counts[r],r))
        owners[g]=rank;counts[rank]+=1;loads[rank]+=lengths[g]
    return owners,loads


def cohorts():
    split=read(PARENT/'asin_split_manifest.json')
    excluded={t['goal_index'] for pool in ('T_pre','V','T_cal','T_opd_eval') for t in split[pool]}
    all_train=[t['goal_index'] for t in split['T_opd'] if t['goal_index'] not in excluded][:3200]
    assert len(set(all_train))==3200
    result={f'train_{i//32+1:03}':all_train[i:i+32] for i in range(0,3200,32)}
    for pool in ('V','T_opd_eval'):
        result[pool]=[t['goal_index'] for t in split[pool]]
        assert len(result[pool])==200
    assert len(set(g for row in result.values() for g in row))==3600
    return result


@lru_cache(maxsize=1)
def owners():
    assert digest_file(ROUTING)==read(PUBLIC/'routing_2512_signature.json')['sha256']
    return {int(g):v['rank'] for g,v in read(ROUTING)['tasks'].items()}


def prepare():
    if ROUTING.exists():
        assert len(owners())==3600
        return
    from transformers import AutoTokenizer
    ROOT.mkdir(exist_ok=True,parents=True)
    if not (ROOT/'snapshots').exists():
        (ROOT/'snapshots').symlink_to(PARENT/'snapshots',target_is_directory=True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    pools=cohorts();tasks={};stats={}
    worker=environment(ROOT,'routing_reset_only')
    try:
        for pool,goals in pools.items():
            lengths={}
            for g in goals:
                key=f'routing_{g}'
                obs=worker.call(cmd='open',key=key,goal_index=g)
                lengths[g]=len(make_input(tok,obs['task'],[],obs['observation'])['token_ids'])
                worker.call(cmd='close',key=key)
            assigned,loads=balanced(goals,lengths)
            tasks.update({g:dict(rank=assigned[g],length=lengths[g],cohort=pool) for g in goals})
            stats[pool]=dict(count=len(goals),token_loads=loads)
    finally:
        worker.close()
    save(ROUTING,dict(tasks=tasks,cohorts=stats,length='tokens of initial unprivileged native prompt with empty history',
         rule='within each 32-task update or 200-task evaluation pool, decreasing length, then goal ID; lowest load then count then rank, with equal task counts',
         split_sha256=digest_file(PARENT/'asin_split_manifest.json')))
    public=PUBLIC/'task_routing_2512.json'
    public.write_bytes(ROUTING.read_bytes())
    save(PUBLIC/'routing_2512_signature.json',dict(sha256=digest_file(public),tasks=len(tasks),source='reset_only_no_generation'))
    assert len(owners())==3600


if __name__=='__main__':
    prepare()
