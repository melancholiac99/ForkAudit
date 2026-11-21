"""Validate complete native E evaluation endpoints and compute paired outcomes."""
from itertools import combinations
import numpy as np
from .d_behavior import readout,cluster_draws,interval,POOLS,MODES,METRICS
from .d_data import PARENT
from .e25_probe import PUBLIC
from exploratory_shop_audit.grpo_forced_audit.common import read,save,stable_seed

def outcomes(l,k,a,expected):
    root=l.root/'evaluation'/f'{k:03}'/a
    out={}
    for rank in sorted(root.glob('rank*')):
        meta=read(rank/'complete.json');n=0
        assert meta['k']==k and meta['arm']==a
        for file in sorted(rank.glob('episodes_*.json')):
            for e in read(file):
                key=e['split'],e['goal'],e['rep']
                assert key not in out and sum(e['events'].values())==e['length']
                out[key]=e;n+=1
        assert meta['count']==n
    assert set(out)==expected
    return out

def report(l,k,endpoints):
    split=read(PARENT/'asin_split_manifest.json')
    expected={(p,t['goal_index'],r) for p in POOLS for t in split[p] for r in range(5)}
    seen={t['asin'] for p in ('T_cal','T_opd') for t in split[p]}
    sets={p:split[p] for p in POOLS}
    sets.update(V_seen=[t for t in split['V'] if t['asin'] in seen],V_unseen=[t for t in split['V'] if t['asin'] not in seen])
    stats={};arrays={};pairs={}
    for a,step in endpoints.items():
        ep=outcomes(l,step,a,expected)
        for pool,tasks in sets.items():
            source=pool if pool in POOLS else 'V'
            for decode,reps in MODES.items():
                key=f'{a}/{pool}/{decode}'
                if not tasks:stats[key]=dict(tasks=0,status='EMPTY_STRATUM');continue
                groups=[[ep[source,t['goal_index'],rep] for rep in reps] for t in tasks]
                info,arr=readout(groups);stats[key]=info;arrays[a,pool,decode]=arr
    gap={};paired_gap={}
    for a in endpoints:
        for mode in MODES:
            gap[f'{a}/{mode}']={m:stats[f'{a}/T_opd_eval/{mode}'][m]-stats[f'{a}/V/{mode}'][m] for m in METRICS}
    # Orient P comparisons as registered.
    order=[a for a in ('P','P-KL','R-matched','R','B') if a in endpoints]
    for left,right in combinations(order,2):
        for mode in MODES:
            for cluster,key in [('asin','asin'),('task','goal_index')]:
                draws_by_pool={};name=f'{left}-{right}/{mode}/{cluster}'
                for pool,tasks in sets.items():
                    if not tasks:continue
                    dif=arrays[left,pool,mode]-arrays[right,pool,mode]
                    draws=cluster_draws(dif,[t[key] for t in tasks],stable_seed(20261120,'stage_d_bootstrap',mode,pool,cluster))
                    pairs[f'{name}/{pool}']=interval(dif.mean(0),draws);draws_by_pool[pool]=draws
                point=np.array([gap[f'{left}/{mode}'][m]-gap[f'{right}/{mode}'][m] for m in METRICS])
                paired_gap[name]=interval(point,draws_by_pool['T_opd_eval']-draws_by_pool['V'])
    result=dict(update=k,endpoints=endpoints,summary=stats,paired=pairs,train_eval_minus_V=gap,paired_gap=paired_gap,
        primary=['P-R-matched' if 'R-matched' in l.config['rates'] else 'P-R','P-B'],seeds=1)
    dest=l.root/'behavior'/f'{k:03}.json';save(dest,result)
    if k==100 and l.name!='formal_2512':save(PUBLIC/'stage_e_final_behavior.json',result)
