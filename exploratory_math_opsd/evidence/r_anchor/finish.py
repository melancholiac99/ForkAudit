"""Run after verified phase-one process exit. Preserve conditional endpoints in final.json."""
import json
import os
from pathlib import Path
import sys
import numpy as np

from train import HERE, ROOT, OUT, train
from launch import evaluate, read, save, sha, validate_checkpoint


def endpoint(name, bench):
    receipt = evaluate(ROOT, os.environ.copy(), name, bench, 1001)
    data = read(ROOT/'run/evaluation'/name/bench/'1001.json')['results']
    assert len(data) == 30 and len({q['problem'] for q in data}) == 30
    assert all(len(q['generations']) == 12 for q in data)
    scores = {q['problem']: 100*sum(g['correct'] for g in q['generations'])/12 for q in data}
    return receipt, scores


def differences(pairs):
    rng = np.random.default_rng(20261124)
    means, draws = [], []
    for a, b in pairs:
        assert a.keys() == b.keys()
        diffs = np.array([v-b[key] for key,v in a.items()])
        means.append(diffs.mean())
        draws.append(rng.choice(diffs, (10000,len(diffs))).mean(axis=1))
    return dict(pp=float(np.mean(means)), ci95=np.quantile(np.mean(draws,axis=0), [.025,.975]).tolist())


def finish():
    results = dict(beta=.1,seed=42,windows=[],evaluations={},paired={},decision={})
    def flush():
        results['windows'] = [json.loads(line) for line in (OUT/'anchor_windows.jsonl').read_text().splitlines()]
        results['preflight_by_rank'] = {p.name:[json.loads(line) for line in p.read_text().splitlines()]
                                        for p in sorted(OUT.glob('initial-rank*.jsonl'))}
        save(HERE/'final.json', results)
    def get(name, bench='aime24'):
        receipt, scores = endpoint(name,bench)
        results['evaluations'][name+'/'+bench] = receipt
        return scores
    def stopped():
        flush()
        if results['windows'][-1]['stopped']:
            results['decision']['stop']='natural_end_rate_below_15_percent'
            flush()
            return True
        return False
    if stopped():
        return
    a100 = get('R-anchor@100')
    r100 = get('R@100')
    results['paired']['Anchor100-R100'] = differences([(a100,r100)])
    if np.mean(list(a100.values())) <= 48.6:
        results['decision']['stop']='AIME24_at_100_at_or_below_48.6'
        flush()
        return
    flush()
    if results['windows'][-1]['step'] == 100:
        train(200)
    if stopped():
        return
    a200 = get('R-anchor@200')
    r200 = get('R-extension@200')
    get('B')
    decline = differences([(a200,r200)])
    primary = differences([(a200,r100)])
    results['paired'].update({'Anchor200-R200':decline,'Anchor200-R100':primary})
    verdict = results['decision']
    verdict['decline']='止住了下滑' if decline['ci95'][0]>0 else '未止住'
    superior = primary['ci95'][0]>0
    unclear = primary['pp']>=3 and primary['ci95'][0]<=0<=primary['ci95'][1]
    verdict['aime24']='超过 R' if superior else ('未分辨' if unclear else '结案')
    if superior or unclear:
        flush()
        pairs = [(a200,r100)]
        for bench in ('aime25','hmmt25'):
            pair = (get('R-anchor@200',bench), get('R@100',bench))
            pairs.append(pair)
            results['paired'][bench]=differences([pair])
            flush()
        combined=differences(pairs)
        results['paired']['three_benchmark_equal_weight']=combined
        verdict['confirmed']=combined['ci95'][0]>0
    verdict['early_stopping_interpretation']=verdict['aime24']=='结案' and results['paired']['Anchor100-R100']['pp']<0
    flush()


if __name__ == '__main__':
    finish()
