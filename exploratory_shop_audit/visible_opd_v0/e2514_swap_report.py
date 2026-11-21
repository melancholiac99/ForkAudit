"""Within-task fresh-control contrasts with the two prespecified task populations."""
from collections import Counter
import numpy as np
from .acceptance import ALL_REQUIREMENTS
from .e2514_search import DATA,OUT,POOLS
from exploratory_shop_audit.grpo_forced_audit.common import read,save,stable_seed,digest_file


def report(directory=None, tapes=None):
    directory=OUT if directory is None else directory
    tapes=[DATA/'swap'] if tapes is None else tapes
    tasks=read(directory/'intervention_tasks.json')
    pages={(r['split'],r['goal'],r['query']):r for r in read(directory/'intervention_result_pages.json')}
    trials={}
    for tape in tapes:
        for k in (20,30):
            for rank in range(4):
                root=tape/f'{k:03}'/f'rank{rank}'
                receipt=read(root/'complete.json')
                n=0
                for f in root.glob('part_*.json'):
                    for r in read(f):
                        key=k,r['split'],r['goal'],r['query'],r['rep']
                        assert key not in trials;trials[key]=int(r['reason'] in ALL_REQUIREMENTS);n+=1
                assert n==receipt['jobs']
    expected={(actor,t['split'],t['goal'],q,r) for t in tasks
              for q in set(t['queries']['20'])|set(t['queries']['30']) for actor in (20,30) for r in range(4)}
    assert set(trials)==expected
    task_rows=[]
    for task in tasks:
        pool=task['split'];g=task['goal'];cells={};page={};query_counts={}
        for source,queries in task['queries'].items():
            total=sum(queries.values())
            for actor in (20,30):
                wins={q:sum(trials[actor,pool,g,q,r] for r in range(4)) for q in queries}
                query_counts[f'{actor}/{source}']=wins
                cells[f'{actor}/{source}']=sum(wins[q]*w for q,w in queries.items())/(4*total) if total else None
            page[source]=sum(w*pages[pool,g,q]['contains_qualifying'] for q,w in queries.items())/total if total else None
        task_rows.append(dict(split=pool,goal=g,strict=task['strict_subset'],rates=cells,first_page_qualifying=page,
             wins_per_four_fresh=query_counts,complete_queries=not any(v.get('unrecovered') for v in task['missing'].values())))
    summary={}
    for pool in POOLS:
        for sub in ('all','strict'):
            selected=[r for r in task_rows if r['split']==pool and (sub=='all' or r['strict'])]
            for coverage in ('both_sources','all_source_queries_recovered'):
                included=[r for r in selected if all(v is not None for v in r['rates'].values()) and
                         (coverage=='both_sources' or r['complete_queries'])]
                cells=['30/20','30/30','20/20','20/30']
                if not included:continue
                values=np.array([[r['rates'][c] for c in cells] for r in included])
                rng=np.random.default_rng(stable_seed(20261122,pool,sub,coverage))
                draws=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
                def contrasts(arr):
                    p30good,p30own,p20own,p20bad=arr.T
                    span=p20own-p30own
                    return {'P30_gain_good_minus_own':p30good-p30own,'P20_loss_own_minus_bad':p20own-p20bad,
                            'reference_span_P20own_minus_P30own':span,
                            'P30_recovery_fraction':np.divide(p30good-p30own,span,out=np.full_like(span,np.nan),where=span>0),
                            'P20_loss_fraction':np.divide(p20own-p20bad,span,out=np.full_like(span,np.nan),where=span>0)}
                observed=contrasts(values.mean(0,keepdims=True));boot=contrasts(draws)
                metrics={}
                for name,x in observed.items():
                    valid=np.isfinite(boot[name])
                    metrics[name]=dict(estimate=float(x[0]) if np.isfinite(x[0]) else None,
                        ci95=np.quantile(boot[name][valid],[.025,.975]).tolist() if valid.all() and np.isfinite(x[0]) else None,
                        undefined_draws=int((~valid).sum()))
                summary[f'{pool}/{sub}/{coverage}']=dict(selected_tasks=len(selected),paired_tasks=len(included),
                         fresh_rates=dict(zip(cells,values.mean(0))),effects=metrics,
                         first_page_qualifying={s:np.mean([r['first_page_qualifying'][s] for r in included]) for s in ('20','30')})
    save(directory/'search_swap_report.json',dict(summary=summary,plan_sha256=digest_file(directory/'intervention_tasks.json'),
         pages_sha256=digest_file(directory/'intervention_result_pages.json'),fresh_episodes=len(trials),distinct_task_queries=len(expected)//8,unit='equal task weight; within query source, original episode frequency',
         estimator='All four cells use four fresh continuations per distinct task/query/recipient. Fractions divide by P20 own good minus P30 own failed-query fresh rates. Nonpositive denominators are undefined.',
         initial_history='Canonical search[query] at the empty task start, receiver handles all following decisions. Fresh seed by split, task, query text, recipient, repeat and action step.',
         interpretation='Selected tasks only; original-seed outcomes are used only for selection.',tasks=task_rows))


if __name__=='__main__':report()
