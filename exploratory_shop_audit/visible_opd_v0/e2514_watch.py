"""Finish CPU deliverables concurrently with the two GPU lanes and queue verified Git trees."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .e25_paths import PUBLIC,RUN,SERIAL
from . import e2514_search_extension as extension
from exploratory_shop_audit.grpo_forced_audit.common import ROOT,CPU_PYTHON,read,save,utc

MOD='exploratory_shop_audit.visible_opd_v0.'
OUT=PUBLIC/'migration_2514'
CONTROL=RUN/'migration_2514'


def call(mod,*args,cpu=False):
    subprocess.run([str(CPU_PYTHON) if cpu else sys.executable,'-m',MOD+mod,*map(str,args)],cwd=ROOT,check=True)


def commit(label,files):
    receipt=CONTROL/'publication_requests'/(label+'.json')
    if receipt.exists():return
    paths=[str(p.relative_to(ROOT)) for p in files]
    subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet','--',*paths],cwd=ROOT).returncode:
        subprocess.run(['git','-c','user.name=Codex','-c','user.email=codex@localhost','commit','--only','-m',
                        f'Publish section 25.14 {label} results','--',*paths],cwd=ROOT,check=True)
    rev=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    tree=subprocess.check_output(['git','rev-parse',rev+'^{tree}'],cwd=ROOT,text=True).strip()
    save(receipt,dict(label=label,commit=rev,tree=tree,at=utc()))


def publish_extension(wave, terminal):
    directory=extension.paths(wave)[0]
    label='search-coverage-'+wave
    if (CONTROL/'publication_requests'/(label+'.json')).is_file():return
    if not extension.finished_sampling(wave):
        if not terminal:return
        call('e2514_search_extension','seal',wave)
        if not extension.finished_sampling(wave):
            subprocess.run(['torchrun','--standalone','--nproc-per-node=4','-m',MOD+'e2514_search_extension','rank',wave],
                cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'4,5,6,7','OPD_MIGRATION_GROUP':'B'},check=True)
    if not (directory/'intervention_result_pages.json').is_file():call('e2514_search_extension','pages',wave,cpu=True)
    if not (directory/'coverage.json').is_file():call('e2514_search_extension','report',wave)
    commit(label,[directory])


def watch():
    if (CONTROL/'publication_requests/final.json').is_file():return
    while True:
        seal=CONTROL/'search/plan_sealed.json'
        if seal.exists():
            signature=read(seal)
            archived=CONTROL/'search/metrics_source.json'
            if not archived.exists() or read(archived)!=signature:
                call('e2514_pages',cpu=True)
                save(archived,signature)
            pages=CONTROL/'search/pages_source.json'
            if not pages.exists() or read(pages)['sha256']!=signature['sha256']:call('e2514_pages','pages',cpu=True)
            swap=CONTROL/'search/swap'
            if all((swap/f'{k:03}'/f'rank{r}/complete.json').exists() for k in (20,30) for r in range(4)):
                if not (OUT/'search_swap_report.json').exists():call('e2514_swap_report')
                commit('search-swap',[OUT/f for f in ('archived_search_metrics.json','intervention_tasks.json','intervention_result_pages.json',
                    'search_archive','search_swap_report.json')])
        for arm in ('R-matched','P-frozen','R'):
            folder=OUT/arm
            for f in sorted((folder/'behavior').glob('*.json')):
                k=int(f.stem)
                if not (folder/'search'/f.name).exists():call('e2514_pages',arm,k,cpu=True)
            if (folder/'endpoint.json').exists() and not (CONTROL/'publication_requests'/(arm+'.json')).exists():
                call('e2514_delivery',arm)
                commit(arm,[folder])
        terminal=all((CONTROL/'publication_requests'/f'{name}.json').is_file() for name in ('search-swap','R-matched','P-frozen','R'))
        publish_extension('handoff',terminal)
        if terminal:
            publish_extension('final',True)
            call('e2514_delivery','final')
            # The final signed manifest also covers cross-lane probe receipts outside arm folders.
            commit('final',[OUT])
            return
        time.sleep(10)


if __name__=='__main__':
    with (CONTROL/'reporter.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        watch()
