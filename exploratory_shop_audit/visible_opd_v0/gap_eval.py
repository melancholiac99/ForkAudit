"""Native free interaction and continuations from audited B greedy result-page states."""
import os
import time

from .e2512_native import native_rank, snapshot_dir
from .e2512_routing import owners
from .gap_score import RUN, split_tasks
from exploratory_shop_audit.env_worker import parse_action_fixed
from exploratory_shop_audit.grpo_forced_audit.common import digest_value, read, save
from exploratory_shop_audit.grpo_forced_audit.rollout import Runner
from exploratory_shop_audit.runtime import make_input, parse_generation

SEEDS = (2201,2202,2203,2204)


def run_batch(runner, pool, goals, rep, anchor_dir, mode, create_anchors=False):
    active, ended = [], []
    for g in goals:
        key = f'g26_{g}'
        ap = anchor_dir/f'{pool}_{g}.json'
        src = read(ap) if mode=='page' else None
        if src is not None and src['state'] is None:
            ended.append(dict(goal=g,rep=rep,no_common_page=True))
            continue
        acts = list(src['actions']) if src else []
        hist = list(src['history']) if src else []
        state = runner.engine.call(cmd='open',key=key,goal_index=g,prefix=acts)
        if src is not None and digest_value(state)!=digest_value(src['state']):
            raise ValueError(f'Native page restore mismatch for {g}')
        active.append(dict(goal=g,key=key,state=state,h=hist,acts=acts,found=False))
    while active:
        req = [dict(token_ids=make_input(runner.tokenizer,e['state']['task'],e['h'],e['state']['observation'])['token_ids'],
                    temperature=0. if rep==4 else .7,seed=SEEDS[0 if rep==4 else rep]) for e in active]
        pending = []
        for e, gen in zip(active,runner.model.call(req),strict=True):
            action = parse_generation(gen['text'])
            after = runner.engine.call(cmd='step',key=e['key'],action=action)
            e['h'].append([e['state']['observation'],gen['text']])
            e['acts'].append(action)
            stop = after['status']['done']
            if create_anchors and not e['found']:
                name,_ = parse_action_fixed(action)
                if name=='search' and after['legal'] and after['state']['public']['page_type']=='search_results':
                    assert not stop
                    save(anchor_dir/f"{pool}_{e['goal']}.json",dict(actions=e['acts'],history=e['h'],state=after['state'],
                         product_order=[v['asin'] for v in after['state']['public'].get('products',[])]))
                    e['found'] = True
                elif stop:
                    save(anchor_dir/f"{pool}_{e['goal']}.json",dict(actions=e['acts'],history=e['h'],state=None,reason=after['status']['termination_reason']))
            if stop:
                ended.append(dict(goal=e['goal'],rep=rep,reason=after['status']['termination_reason'],
                                  reward_valid=after['status']['reward_valid'],length=len(e['acts'])))
                runner.engine.call(cmd='close',key=e['key'])
            else:
                e['state']=after['state']
                pending.append(e)
        active=pending
    assert len(ended)==len(goals)
    return ended


def page_bench(rank):
    native_rank(rank)
    folder=RUN/'page_timing'/f'rank{rank}'
    snapshot_dir(folder)
    split=split_tasks()
    # Fixed training-only tasks selected before sampling, independent of page/reward.
    goals=[t['goal_index'] for t in split['T_opd'] if t['goal_index'] in owners() and owners()[t['goal_index']]==rank][:8]
    start=time.monotonic()
    runner=Runner(folder,'base',parallel_start=True)
    ready=time.monotonic()
    try:
        run_batch(runner,'training_budget',goals,4,folder/'anchors','free',True)
        greedy=time.monotonic()
        for rep in range(4):
            run_batch(runner,'training_budget',goals,rep,folder/'anchors','free')
        free_end=time.monotonic()
        completed=0
        for rep in range(5):
            rows=run_batch(runner,'training_budget',goals,rep,folder/'anchors','page')
            completed+=sum(not r.get('no_common_page',False) for r in rows)
        end=time.monotonic()
    finally:
        runner.close()
    save(folder/'timing.json',dict(rank=rank,tasks=len(goals),goals=goals,requested=len(goals)*5,completed=completed,
         start_seconds=ready-start,greedy_seconds=greedy-ready,free_seconds=free_end-ready,continuation_seconds=end-free_end,
         close_seconds=time.monotonic()-end,seeds=SEEDS))


if __name__=='__main__':
    page_bench(int(os.environ['LOCAL_RANK']))
