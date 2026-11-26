"""Run on finished shards during priority cells, or on all GPUs after the priority queue."""
import os
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path('/mnt/data/ForkAudit/math-opsd')
RAW=ROOT/'run/latent-confidence'
FORCE=ROOT/'run/cond-kl/forcing'
ARMS=['B','T0@100','R@100','R-noclip@100','R-cond@100','P@100']
def finished(mode):
    d=RAW/mode
    f,m=(33,33) if mode=='pilot' else (750,500)
    return (d/'fit.json').exists() and all(len(list((d/(k+'-'+a)).glob('*.npz')))==n for a in ARMS for k,n in [('fixed',f),('math',m)])
def windows():
    for a in ('B','R-cond@100'):
        counts=[sum((FORCE/a/f'{i}.json').exists() for i in range(k,500,4)) for k in range(4)]
        if min(counts)<125:
            return {k:'budget-'+a for k,c in enumerate(counts) if c==125}
    return {k:'after-budget' for k in range(4)}

def main():
    tasks={};finished_workers=set()
    while True:
        mode='full' if (HERE/'timing.json').exists() else 'pilot'
        allowed=windows()
        for k,(p,tag,md) in list(tasks.items()):
            if p.poll() is not None:
                if p.returncode:raise RuntimeError(f'{md} GPU {k}: exit {p.returncode}')
                finished_workers.add((k,tag,md));del tasks[k]
            elif allowed.get(k)!=tag or md!=mode:
                p.terminate();p.wait();del tasks[k]
        if finished(mode):
            if mode=='pilot':
                subprocess.run([sys.executable,'timing.py'],check=True)
                continue
            for p,_,_ in tasks.values():p.wait()
            subprocess.run([sys.executable,'summarize.py'],check=True)
            return
        if allowed:
            used=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).split()]
            for k,tag in allowed.items():
                if k in tasks or used[k]>1000 or (k,tag,mode) in finished_workers:continue
                with (RAW/f'worker-{k}.log').open('a') as log:
                    tasks[k]=(subprocess.Popen([sys.executable,'dispatch.py',mode],stdout=log,stderr=log,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(k))),tag,mode)
        time.sleep(2)
if __name__=='__main__':main()
