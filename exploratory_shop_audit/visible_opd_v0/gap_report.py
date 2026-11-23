"""Paired ASIN-cluster endpoint tests; called only after both arms are complete."""
from collections import defaultdict

import numpy as np

from .acceptance import ALL_REQUIREMENTS
from .gap_score import PUBLIC, split_tasks
from exploratory_shop_audit.grpo_forced_audit.common import digest_file, read, save


def contrasts(left,right,asins):
    keys=sorted(set(left)&set(right))
    if set(left)!=set(right) or not keys:
        raise ValueError('Paired models need the same tasks')
    vals=np.array([left[g]-right[g] for g in keys])
    tags,inverse=np.unique([asins[g] for g in keys],return_inverse=True)
    sums=np.bincount(inverse,weights=vals);sizes=np.bincount(inverse)
    rng=np.random.default_rng(20261123)
    counts=rng.multinomial(len(tags),np.full(len(tags),1/len(tags)),size=10000)
    draws=(counts@sums)/(counts@sizes)
    point=float(vals.mean())
    p=(1+np.count_nonzero(np.abs(draws-point)>=abs(point)-1e-14))/(1+len(draws))
    return dict(tasks=len(keys),asins=len(tags),difference=point,
                ci95=np.quantile(draws,[.025,.975]).tolist(),p_two_sided=float(p))


def holm(p):
    order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*np.array(p)[order]))
    return out.tolist()


def load_rates(l,pool,mode,arm,k,which,goals):
    sample=range(4) if which=='sample' else [4]
    values=defaultdict(dict)
    for rank in range(4):
        src=l.root/'evaluation'/pool/mode/f'{arm}_{k:02}'/f'rank{rank}'
        assert (src/'complete.json').exists()
        for f in sorted(src.glob('part_*.json')):
            for row in read(f):
                rep=row['rep'];g=row['goal']
                if rep in sample:
                    assert rep not in values[g]
                    values[g][rep]=None if row.get('no_common_page') else int(row['reason'] in ALL_REQUIREMENTS)
    assert set(values)==set(goals) and all(set(reps)==set(sample) for reps in values.values())
    available={}
    for g,reps in values.items():
        v=list(reps.values())
        if None in v:
            assert mode=='page' and all(x is None for x in v)
        else:
            available[g]=float(np.mean(v))
    return available


def report(l,ends):
    arms=('R-gap','P-gap')
    assert set(ends)==set(arms)
    checkpoints={'B':0,**ends}
    result=dict(bootstrap=dict(cluster='ASIN',draws=10000,seed=20261123,alternative='two-sided centered null'))
    primary=[];behavior={};pages={};stable={a:{} for a in arms}
    collect=lambda r: {'rate':sum(r.values())/len(r),'tasks':len(r)} if r else {'rate':None,'tasks':0}
    for pool in ('T_opd_eval','V'):
        asins={t['goal_index']:t['asin'] for t in split_tasks()[pool]}
        behavior[pool]={};pages[pool]={}
        for which in ('sample','greedy'):
            rates={};common={}
            for a,k in checkpoints.items():
                points={0} if a=='B' else {j for j in ((10,20,30,50) if pool=='T_opd_eval' else (20,30,50)) if j<=k}|{k}
                for j in sorted(points):
                    rates[a,j]=load_rates(l,pool,'free',a,j,which,asins)
                common[a]=load_rates(l,pool,'page',a,k,which,asins)
            behavior[pool][which]={f'{a}@{j}':collect(v) for (a,j),v in rates.items()}
            pages[pool][which]=dict(arms={a:collect(r) for a,r in common.items()},
                                    P_minus_R=contrasts(common['P-gap'],common['R-gap'],asins))
            if which=='sample':
                for a in arms:
                    for j in (30,50):
                        if (a,j) in rates and (a,20) in rates:
                            stable[a][f'{pool}_{j}-20']=contrasts(rates[a,j],rates[a,20],asins)
                if pool=='V':
                    for other in ('R-gap','B'):
                        primary.append(dict(comparator=other,**contrasts(rates['P-gap',ends['P-gap']],rates[other,checkpoints[other]],asins)))
        masses={}
        for a,k in checkpoints.items():
            vals={}
            for r in range(4):
                for row in read(l.root/'mass'/pool/f'{a}_{k:02}'/f'rank{r}.json'):
                    assert row['goal'] not in vals
                    vals[row['goal']]=row['mass']
            assert set(vals)==set(asins)
            masses[a]={g:v for g,v in vals.items() if v is not None}
        pages[pool]['legal_click_mass']=dict(arms={a:collect(r) for a,r in masses.items()},
                                             P_minus_R=contrasts(masses['P-gap'],masses['R-gap'],asins))
    for row,adjusted in zip(primary,holm([a['p_two_sided'] for a in primary]),strict=True):
        row.update(p_holm=adjusted,passes=bool(row['difference']>=.02-1e-12 and adjusted<.05))
    stability={}; monitors={}
    for a in arms:
        end=read(l.root/'endpoints'/f'{a}.json')
        values=list(stable[a].values())
        detected=any(r['difference']<=-.03+1e-12 and r['ci95'][1]<0 for r in values)
        label='塌陷' if end['early_stop'] or detected else ('稳定' if len(values)==4 and all(r['ci95'][0]>-.03 for r in values) else '不确定')
        stability[a]=dict(label=label,endpoint=end,comparisons=stable[a])
        monitors[a]=[]
        for k in range(1,ends[a]+1):
            p=l.folder(k,a)
            row=read(p/'monitor.json')
            if k%5==0:row['window']=read(p/'window.json')
            if k%10==0:
                probes=[read(p/f'probe_{r}.json') for r in range(4)]
                assert abs(sum(x['mass'] for x in probes)-1)<1e-9
                row['probe_bf16_full_vocab_kl']=sum(x['kl_sum'] for x in probes)
            monitors[a].append(row)
    result.update(primary=primary,stability=stability,behavior=behavior,common_pages=pages,
                  training=monitors,training_seeds=1,registration='post hoc after Stage E including V')
    save(PUBLIC/'endpoint_report.json',result)
