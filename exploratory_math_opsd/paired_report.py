"""Registered question-paired bootstrap and fixed-question decoding SE, in percentage points."""
import json
from pathlib import Path

import numpy as np

BENCHMARKS=('aime24','aime25','hmmt25')
SEEDS=(1001,2002,1500,4004)
ARMS=('B','R','T0')
CONTRASTS=(('R','T0'),('R','B'),('T0','B'))


def flags(bounds):
    lo,hi=bounds
    return dict(positive=lo>0,negative=hi<0,small_precise=lo>=-2 and hi<=2)


def decision(intervals):
    rb,tb,rt=[flags(intervals[k]) for k in ('R-B','T0-B','R-T0')]
    if rb['positive'] and rt['positive']:return 'P_COMPARISON'
    if rb['positive'] and tb['positive'] and rt['small_precise']:return 'SMALL_PRECISE'
    if rt['negative']:return 'REFERENCE_HURTS'
    return 'INCONCLUSIVE'


def compare(scores):
    """Input [benchmark, seed, question] per model, scaled to fractions."""
    seed=20261124
    rng=np.random.default_rng(seed)
    draws=rng.integers(30,size=(10000,3,30))
    contrasts={}
    bounds={}
    for a,b in CONTRASTS:
        diff=100*(scores[a]-scores[b])
        per_question=diff.mean(axis=1)
        sampled=per_question[np.arange(3)[None,:,None],draws].mean(axis=2)
        by_seed=diff.mean(axis=2)
        def stats(observed,resamples):
            out={'mean_pp':float(observed.mean()),'paired_ci95_pp':np.percentile(resamples,[2.5,97.5]).tolist(),
                 'by_seed_pp':observed.tolist(),'decoding_se_pp':None}
            if len(observed)>1:out['decoding_se_pp']=float(observed.std(ddof=1)/np.sqrt(len(observed)))
            return out
        key=a+'-'+b
        macro=stats(by_seed.mean(axis=0),sampled.mean(axis=1))
        macro['interval_flags']=flags(macro['paired_ci95_pp'])
        bounds[key]=macro['paired_ci95_pp']
        contrasts[key]={'macro':macro,'benchmarks':{c:stats(by_seed[i],sampled[:,i]) for i,c in enumerate(BENCHMARKS)}}
    return contrasts,decision(bounds)


def summarize(folder):
    quick=json.loads((folder/'quick.json').read_text())
    report={'quick':quick,'bootstrap':dict(seed=20261124,draws=10000,cluster='question within benchmark'),'checkpoints':{}}
    for step in (50,100):
        seeds=(1001,) if step==50 and quick['reduced_at_50'] else SEEDS
        scores={}
        details={}
        for arm in ARMS:
            model='B' if arm=='B' else f'{arm}@{step}'
            s=np.empty((3,len(seeds),30),dtype=np.float64)
            details[arm]={}
            for b,bench in enumerate(BENCHMARKS):
                token_sum=length_caps=count=0
                for j,seed in enumerate(seeds):
                    rows=json.loads((folder/model/bench/f'{seed}.json').read_text())['results']
                    indexed={r['problem']:r for r in rows}
                    if arm=='B' and j==0:
                        if b==0:question_sets={}
                        question_sets[bench]=sorted(indexed)
                    if sorted(indexed)!=question_sets[bench] or len(indexed)!=30:
                        raise ValueError(f'Question mismatch at {model} {bench} {seed}')
                    for i,q in enumerate(question_sets[bench]):
                        row=indexed[q]
                        assert len(row['generations'])==12
                        s[b,j,i]=row['num_correct']/12
                        for g in row['generations']:
                            token_sum+=g['generated_tokens']
                            length_caps+=g['generated_tokens']==38912
                            count+=1
                means=s[b].mean(axis=1)*100
                details[arm][bench]=dict(avg12_percent=float(means.mean()),by_seed_percent=means.tolist(),
                                         decoding_se_pp=float(means.std(ddof=1)/np.sqrt(len(seeds))) if len(seeds)>1 else None,
                                         mean_tokens=token_sum/count,length_cap_rate=length_caps/count,responses=count)
            scores[arm]=s
            macro=s.mean(axis=(0,2))*100
            details[arm]['macro']=dict(avg12_percent=float(macro.mean()),by_seed_percent=macro.tolist(),
                                       decoding_se_pp=float(macro.std(ddof=1)/np.sqrt(len(seeds))) if len(seeds)>1 else None)
        contrasts,label=compare(scores)
        report['checkpoints'][step]=dict(seeds=list(seeds),models=details,contrasts=contrasts,
                                          decision=label if step==100 else 'TREND_ONLY')
    return report
