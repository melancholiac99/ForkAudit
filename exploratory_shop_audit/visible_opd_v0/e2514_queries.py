"""Authenticate archived actions, then construct the two fresh query conditions."""
from collections import Counter, defaultdict
from itertools import product
import csv

from exploratory_shop_audit.env_worker import parse_action_fixed
from .e2514_search import DATA, OUT, original, selection
from .d_eval import event
from .d_data import PARENT
from .e2512_native import snapshot_dir
from exploratory_shop_audit.grpo_forced_audit.checks import environment
from exploratory_shop_audit.grpo_forced_audit.common import digest_value, read, save


def native_purchase_candidates(traces,orig,goal):
    # Test strings compatible with an observed first page and a short native purchase.
    # Candidates are never accepted without the archived ordered action SHA.
    events=orig['events']
    num_options=events.get('select_visible_option',0)
    if (set(events)-{'search','open_listed_product','select_visible_option','buy'} or
            any(events.get(k)!=1 for k in ('search','open_listed_product','buy')) or num_options>3):
        return
    first={r[0]['action']:r[0] for r in traces if r and r[0]['event']=='search'}
    options={r['action'] for t in traces for r in t if r['event']=='select_visible_option'}
    for val in goal.get('goal_options') or ():
        options.add(f'click[{val}]')
        options.add(f'click[{str(val).lower()}]')
    buy_commands={r['action'] for t in traces for r in t if r.get('event')=='buy'} | {'click[buy now]', 'click[Buy Now]'}
    for search in first.values():
        products=search['products']
        if orig['reason']=='gold_purchase':products=[a for a in products if a==goal['asin']]
        for asin in products:
            for suffix in product(sorted(options),repeat=num_options):
                for buy in sorted(buy_commands):
                    commands=[f'click[{asin}]',*suffix,buy]
                    yield [search,*[dict(action=cmd) for cmd in commands]]


def recover():
    archives={k:original(k) for k in (0,10,20,30)}
    goals=read(PARENT/'snapshots/goals_frozen.json')
    bank=defaultdict(dict)
    direct={}
    for k in archives:
        for rank in range(4):
            path=DATA/'reconstruction'/f'{k:03}'/f'rank{rank}'
            assert (path/'complete.json').is_file(),path
            retry=DATA/'exact_retry'/f'{k:03}'/f'rank{rank}'
            files=sorted(path.glob('*.json'))
            if retry.is_dir():
                assert (retry/'complete.json').is_file()
                files+=sorted(retry.glob('*.json'))
            for f in files:
                if f.name=='complete.json':continue
                for row in read(f):
                    key=(row['split'],row['goal'],row['rep'])
                    actions=row['actions']
                    sha=row['actions_sha']
                    assert sha==digest_value([a['action'] for a in actions])
                    bank[key[:2]][sha]=actions
                    if (k,key) not in direct or sha==archives[k][key]['actions_sha']:direct[k,key]=actions
    result=[]
    recovered_engine=None
    scratch=DATA/'assembled_cpu_replay'
    snapshot_dir(scratch)
    def correct_labels(goal,original,actions):
        nonlocal recovered_engine
        if recovered_engine is None:recovered_engine=environment(scratch,'cpu')
        eng=recovered_engine;key=f'recover_{goal}'
        state=eng.call(cmd='open',key=key,goal_index=goal)
        checked=[]
        for item in actions:
            action=item['action']
            step=eng.call(cmd='step',key=key,action=action)
            kind=event(state['public'],action,step['legal']);new=dict(action=action,event=kind)
            if kind=='search':
                new.update(query=parse_action_fixed(action)[1],arg_tokens=item['arg_tokens'],
                           products=[p['asin'] for p in step['state']['public'].get('products',[])])
            if kind=='other_click':new['target']=parse_action_fixed(action)[1]
            checked.append(new);state=step['state']
        assert step['status']['done'] and step['status']['termination_reason']==original['reason']
        assert Counter(a['event'] for a in checked)==original['events']
        eng.call(cmd='close',key=key)
        return checked

    for k,archive in archives.items():
        for key,orig in sorted(archive.items()):
            pool,g,rep=key; expected=orig['actions_sha']; related=bank[pool,g]
            own=direct[k,key]
            if digest_value([r['action'] for r in own])==expected:
                rows=own;source='same_episode_replay'
            elif expected in related:
                rows=related[expected];source='same_task_digest'
            else:
                rows=None;source='unrecovered'
                # Try observed same-task prefixes/suffixes and one substituted action.
                # An exact match authenticates all strings; the original episode's
                # native legality and terminal reason are independently replayed below.
                traces=list(related.values());n=orig['length']
                choices={a['action']:a for trace in traces for a in trace}
                seqs=[trace for trace in traces if len(trace)==n]
                def candidates():
                    for t in seqs:
                        for i,a in enumerate(t):
                            for alt in choices.values():
                                if alt['action']!=a['action']:
                                    yield [*t[:i],alt,*t[i+1:]]
                    for t in traces:
                        for u in traces:
                            for cut in range(1,min(len(t)+1,n)):
                                if len(u)>=n-cut:
                                    yield [*t[:cut],*u[-(n-cut):]]
                for candidate in candidates():
                    if digest_value([x['action'] for x in candidate])==expected:
                        rows=correct_labels(g,orig,candidate)
                        source='assembled_digest';related[expected]=rows;break
                if rows is None:
                    for candidate in native_purchase_candidates(traces,orig,goals[g]):
                        if digest_value([x['action'] for x in candidate])==expected:
                            rows=correct_labels(g,orig,candidate)
                            source='native_page_digest';related[expected]=rows;break
            if rows is not None:
                assert digest_value([r['action'] for r in rows])==expected
                # The archived class count includes every response, including illegal commands.
                searches=[dict(step=i,query=r['query'],tokens=r['arg_tokens'],products=r['products'])
                          for i,r in enumerate(rows) if r['event']=='search']
                token_match=sum(s['tokens'] for s in searches)==orig['tokens'].get('search',0)
                if len(searches)==1 and not orig['events'].get('illegal'):
                    searches[0]['tokens']=orig['tokens'].get('search',0);token_match=True
                other=[r['target'] for r in rows if r['event']=='other_click']
            else:
                searches=None;token_match=None;other=None
            purchased=None
            if rows and rows[-1]['event']=='buy':
                purchased=next((parse_action_fixed(r['action'])[1] for r in reversed(rows) if r['event']=='open_listed_product'),None)
                assert purchased is not None
            result.append(dict(step=k,split=pool,goal=g,rep=rep,reason=orig['reason'],source=source,
                  actions_sha=expected,purchased_asin=purchased,legal_search_count=orig['events'].get('search',0),searches=searches,
                  search_token_total_matches_archive=token_match,other_click=other))
    assert len(result)==8000
    if recovered_engine is not None:recovered_engine.close()
    save(DATA/'authenticated.json',result)
    out=OUT/'search_archive';out.mkdir(parents=True,exist_ok=True)
    # Separate sampled/greedy and preserve original episode IDs, missingness and counts.
    for k in archives:
        with (out/f'checkpoint_{k:03}.tsv').open('w') as h:
            wr=csv.writer(h,delimiter='\t',lineterminator='\n')
            wr.writerow(['split','goal','rep','action_sha','source','legal_searches','first_query','each_arg_token_count','total_tokens_match','other_click'])
            for row in result:
                if row['step']!=k:continue
                searches=row['searches']
                wr.writerow([row['split'],row['goal'],row['rep'],row['actions_sha'],row['source'],row['legal_search_count'],
                    searches[0]['query'] if searches else '',','.join(str(s['tokens']) for s in searches) if searches is not None else '',
                    row['search_token_total_matches_archive'],','.join(row['other_click'] or [])])
    return result


def plan(result):
    trials={(r['step'],r['split'],r['goal'],r['rep']):r for r in result}
    tasks=[]
    for selected in selection():
        pool=selected['split'];g=selected['goal']
        groups={};missing={}
        for k,field in ((20,'good_reps'),(30,'failed_reps')):
            observed=Counter();reasons=Counter()
            for rep in selected[field]:
                row=trials[k,pool,g,rep]
                if not row['legal_search_count']:reasons['no_search']+=1
                elif row['searches'] is None:reasons['unrecovered']+=1
                else:observed[row['searches'][0]['query']]+=1
            groups[str(k)]=dict(observed);missing[str(k)]=dict(reasons)
        tasks.append(dict(**selected,queries=groups,missing=missing))
    save(OUT/'intervention_tasks.json',tasks)
    return tasks


if __name__=='__main__':
    import sys
    if sys.argv[1:]==['seal']:
        from .e2514_search import digest_file
        import fcntl
        with (DATA/'seal.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if not (DATA/'plan_sealed.json').exists():
                assert not list((DATA/'swap').glob('*/*/part_*.json'))
                tasks=plan(recover())
                save(DATA/'plan_sealed.json',dict(tasks=len(tasks),sha256=digest_file(OUT/'intervention_tasks.json'),
                     archive_sha256=digest_file(DATA/'authenticated.json')))
    elif sys.argv[1:]==['initial']:
        save(DATA/'authenticated_initial.json',recover())
    else:
        import os,subprocess
        monitor=None
        if os.environ.get('OPD_MIGRATION_GROUP')=='B' and not (OUT/'P-frozen/behavior/010.json').exists():
            # This is evaluation of the already-trained D checkpoint, with no optimizer step.
            monitor=subprocess.Popen([sys.executable,'-m',__package__+'.e2514_driver','frozen-probe'])
        try:plan(recover())
        finally:
            if monitor is not None and monitor.wait()!=0:raise RuntimeError('P-frozen starting checkpoint evaluation failed')
