"""Authenticate archived actions, then construct the two fresh query conditions."""
from collections import Counter, defaultdict
import csv

from exploratory_shop_audit.env_worker import parse_action_fixed
from .e2514_search import DATA, OUT, POOLS, original, selection
from exploratory_shop_audit.grpo_forced_audit.common import digest_value, read, save


def recover():
    archives={k:original(k) for k in (0,10,20,30)}
    bank=defaultdict(dict)
    direct={}
    for k in archives:
        for rank in range(4):
            path=DATA/'reconstruction'/f'{k:03}'/f'rank{rank}'
            assert (path/'complete.json').is_file(),path
            for f in sorted(path.glob('*.json')):
                if f.name=='complete.json':continue
                for row in read(f):
                    key=(row['split'],row['goal'],row['rep'])
                    actions=row['actions']
                    sha=row['actions_sha']
                    assert sha==digest_value([a['action'] for a in actions])
                    bank[key[:2]][sha]=actions
                    direct[k,key]=actions
    result=[]
    counts=Counter()
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
                # A query and an observed suffix can recover a missing sequence if and
                # only if the assembled ordered action strings match the archived SHA.
                options={r['action']:r for trace in related.values() for r in trace if r['event']=='search'}
                for trace in list(related.values()):
                    if len(trace)!=orig['length'] or trace[0]['event']!='search':continue
                    if Counter(a['event'] for a in trace)!=orig['events']:continue
                    for a in options.values():
                        candidate=[a,*trace[1:]]
                        if digest_value([x['action'] for x in candidate])==expected:
                            rows=candidate;source='assembled_digest';break
                    if rows is not None:break
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
            counts[k,pool,source]+=1
            result.append(dict(step=k,split=pool,goal=g,rep=rep,reason=orig['reason'],source=source,
                  actions_sha=expected,purchased_asin=purchased,legal_search_count=orig['events'].get('search',0),searches=searches,
                  search_token_total_matches_archive=token_match,other_click=other))
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


if __name__=='__main__':plan(recover())
