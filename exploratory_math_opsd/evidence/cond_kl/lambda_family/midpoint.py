"""Pipeline registered AIME cells on their original four lanes."""
import fcntl
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import numpy as np
from train_lambda import ROOT, HERE
import launch

MODELS = ('R@50','R-lambda15@50')
RUN = ROOT/'run'
OUT = RUN/'evaluation'


def summarize(receipts):
    rates = {}
    ids = None
    for model in (*MODELS,'R@100','R-lambda15@100'):
        folder = OUT/model/'aime24'
        rec = launch.read(folder/'1001.receipt.json')
        assert rec['sha256']==launch.sha(folder/'1001.json')
        rows = launch.read(folder/'1001.json')['results']
        keys = [q['problem'] for q in rows]
        assert len(set(keys))==30 and (ids is None or ids==keys)
        assert all(len(q['generations'])==12 for q in rows)
        ids=keys
        rates[model]=np.array([sum(g['correct'] for g in q['generations']) for q in rows])
    weights=np.random.default_rng(20261124).multinomial(30,np.full(30,1/30),10000)
    diffs={}
    for a,b in [(MODELS[0],'R@100'),(MODELS[1],'R-lambda15@100'),(MODELS[1],MODELS[0])]:
        d=rates[a]-rates[b]
        diffs[a+' minus '+b]=dict(pp=float(d.sum()*100/360),ci95=np.quantile(weights@d*100/360,[.025,.975]).tolist())
    launch.save(HERE/'midpoint_aime24.json',dict(receipts=receipts,accuracy_pct={a:float(v.sum()*100/360) for a,v in rates.items()},differences=diffs,
        r_early_peak_rule=diffs['R@50 minus R@100']['ci95'][0]>0,
        lambda_early_peak_rule=bool(rates[MODELS[1]].sum()>=rates[MODELS[0]].sum()),bootstrap_seed=20261124))


def run():
    with (RUN/'midpoint-aime.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (HERE/'midpoint_aime24.json').exists():
            return
        original=launch.call
        failed=Event()
        def dispatch(root,cmd,log,env):
            lane=env['CUDA_VISIBLE_DEVICES']
            folder=(RUN/'math500-long-seed1001-n2/nonthinking-R-lambda15@100/shards' if log.startswith('R@50-')
                    else OUT/'R@50/aime24/1001-lanes')
            while not (folder/f'lane{lane}.json').exists():
                if failed.wait(1):
                    raise RuntimeError('Previous cell failed')
            return original(root,cmd,log,env)
        def evaluate(model):
            try:
                result=launch.evaluate(ROOT,os.environ.copy(),model,'aime24',1001)
                launch.save(HERE/('midpoint_'+model+'.json'),result)
                return result
            except BaseException:
                failed.set()
                raise
        launch.call=dispatch
        try:
            with ThreadPoolExecutor(2) as pool:
                receipts=dict(zip(MODELS,pool.map(evaluate,MODELS)))
            summarize(receipts)
        finally:
            launch.call=original


if __name__=='__main__':
    run()
