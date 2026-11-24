"""Describe the four arms; no success criterion is inferred from observed outcomes."""
import json
from pathlib import Path
import numpy as np
from common import ROOT,WORK,json_save

def report():
    scores={}
    rates={}
    questions={}
    for arm in ('B','R@100','P@100'):
        means=[]
        rates[arm]={}
        for bench in ('aime24','aime25','hmmt25'):
            rows=json.loads((ROOT/'run/evaluation'/arm/bench/'1001.json').read_text())['results']
            indexed={r['problem']:r for r in rows}
            keys=sorted(indexed)
            assert len(keys)==30 and keys==questions.setdefault(bench,keys)
            by_question=np.array([indexed[q]['num_correct']/12 for q in keys])
            rates[arm][bench]=100*by_question.mean()
            means.append(by_question)
        scores[arm]=np.stack(means)
    draws=np.random.default_rng(20261124).integers(30,size=(10000,3,30))
    estimates={}
    for a,b in (('P@100','R@100'),('P@100','B'),('R@100','B')):
        delta=100*(scores[a]-scores[b])
        samples=delta[np.arange(3)[None,:,None],draws].mean(axis=(1,2))
        estimates[f'{a} minus {b}']=dict(difference_pp=delta.mean(),paired_ci95_pp=np.percentile(samples,[2.5,97.5]).tolist())
    primary_keys=('P@100 minus R@100','R@100 minus B')
    json_save(WORK/'descriptive_results.json',dict(seed=1001,rates_percent=rates,
        primary={k:estimates[k] for k in primary_keys},
        secondary={k:v for k,v in estimates.items() if k not in primary_keys},
        bootstrap=dict(unit='question within benchmark',draws=10000,seed=20261124)))
