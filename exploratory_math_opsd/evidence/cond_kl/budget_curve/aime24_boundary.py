"""Evaluate the archived boundary adapter with the original four-engine runner."""
import fcntl
import os
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
from launch import evaluate, read, save, sha

ROOT=Path('/mnt/data/ForkAudit/math-opsd')
ARM='R-clip-boundary@100'


def main():
    with (ROOT/'run/cond-kl/forcing/all-gpus.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        ck=ROOT/'run/R-clip-boundary'
        assert sha(ck/'checkpoint-100/adapter_model.safetensors')==read(ck/'complete.json')['adapters']['100']
        ids=None; per_question={}; receipts={}
        for arm in ('B','R@100',ARM):
            receipts[arm]=evaluate(ROOT,os.environ.copy(),arm,'aime24',1001)
            data=read(ROOT/'run/evaluation'/arm/'aime24/1001.json')
            questions=[q['problem'] for q in data['results']]
            assert (ids is None or questions==ids) and len(set(questions))==30
            assert all(len(q['generations'])==12 for q in data['results'])
            ids=questions
            per_question[arm]=np.array([np.mean([g['correct'] for g in q['generations']]) for q in data['results']])*100
        ix=np.random.default_rng(20261124).integers(30,size=(10000,30))
        diff={}
        for other in ('B','R@100'):
            d=per_question[ARM]-per_question[other]
            diff[other]=dict(pp=d.mean(),ci95=np.quantile(d[ix].mean(1),[.025,.975]).tolist())
        save(HERE/'aime24_results.json',dict(receipts=receipts,boundary_minus=diff))


if __name__=='__main__': main()
