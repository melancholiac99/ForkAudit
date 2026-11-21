"""Force the same visible first action in both policies and sample four fresh continuations."""
from collections import Counter
import os
import sys
from .e2514_search import DATA, OUT
from .e2512_native import checkpoint,native_rank,snapshot_dir
from .e2512_routing import SERIAL,owners
from .e25_layout import Layout
from exploratory_shop_audit.runtime import make_input,parse_generation
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner
from exploratory_shop_audit.grpo_forced_audit.common import read,save,stable_seed,digest_value


def continuations(runner,jobs):
    active=[];done=[]
    for pool,g,query,rep in jobs:
        key=digest_value([pool,g,query,rep])
        init=runner.engine.call(cmd='open',key=key,goal_index=g)
        forced=f'search[{query}]'
        after=runner.engine.call(cmd='step',key=key,action=forced)
        assert after['legal'] and not after['status']['done']
        active.append(dict(split=pool,goal=g,query=query,rep=rep,key=key,
             state=after['state'],history=[(init['observation'],forced)],actions=[forced]))
    while active:
        requests=[]
        for e in active:
            s=e['state'];step=len(e['history'])
            ids=make_input(runner.tokenizer,s['task'],e['history'],s['observation'])['token_ids']
            # Common random numbers for both policies/queries on a task, independent
            # of the D_eval stream that generated the selected trajectories.
            seed=stable_seed(20261122,'search_swap_2514',e['split'],e['goal'],e['rep'],step)
            requests.append(dict(token_ids=ids,temperature=.7,seed=seed))
        keep=[]
        for e,gen in zip(active,runner.model.call(requests),strict=True):
            action=parse_generation(gen['text']);out=runner.engine.call(cmd='step',key=e['key'],action=action)
            e['actions'].append(action);st=out['status']
            if st['done']:
                done.append(dict(split=e['split'],goal=e['goal'],query=e['query'],rep=e['rep'],reason=st['termination_reason'],
                                 valid=st['reward_valid'],length=len(e['actions']),actions_sha=digest_value(e['actions'])))
                runner.engine.call(cmd='close',key=e['key'])
            else:
                e['history'].append((e['state']['observation'],gen['text']))
                e['state']=out['state'];keep.append(e)
        active=keep
    assert len(done)==len(jobs)
    return done


def run(k,rank):
    native_rank(rank);l=Layout(SERIAL);cp,sha=checkpoint(l,k,'P')
    root=DATA/'swap'/f'{k:03}'/f'rank{rank}';snapshot_dir(root)
    tasks=read(OUT/'intervention_tasks.json')
    jobs=[]
    for task in tasks:
        if owners()[task['goal']]!=rank:continue
        for q in sorted(set(task['queries']['20'])|set(task['queries']['30'])):
            jobs.extend((task['split'],task['goal'],q,r) for r in range(4))
    if (root/'complete.json').exists():return
    runner=Runner(root,cp,parallel_start=True)
    try:
        for i in range(0,len(jobs),32):
            f=root/f'part_{i:04}.json'
            if not f.exists():save(f,continuations(runner,jobs[i:i+32]))
            assert {(r['split'],r['goal'],r['query'],r['rep']) for r in read(f)}==set(jobs[i:i+32])
    finally:runner.close()
    save(root/'complete.json',dict(checkpoint=sha,jobs=len(jobs),job_sha=digest_value(jobs)))


if __name__=='__main__':run(int(sys.argv[1]),int(os.environ['LOCAL_RANK']))
