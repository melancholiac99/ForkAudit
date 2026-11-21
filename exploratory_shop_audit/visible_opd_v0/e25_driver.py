"""Restartable phase-E calibration and 100-update execution; no evaluation-driven settings."""
import csv
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .d_data import PARENT,SMOKE
from .d_report import class_stats
from .e25_behavior import report as behavior_report
from .e25_budget import budget
from .e25_layout import Layout
from .e25_measure import merge_update,merge_common
from .e25_probe import RUN,PUBLIC
from .s23_cross import RUN as D
from exploratory_shop_audit.grpo_forced_audit.common import append,digest_file,read,save,utc

PKG='exploratory_shop_audit.visible_opd_v0.'
REPO=Path(__file__).resolve().parents[2]
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True).strip()
def freeze(path,data):
    if path.exists():assert read(path)==data,(path,'frozen inputs differ')
    else:save(path,data)
def commit_file(path,message):
    rel=str(path.relative_to(REPO))
    git('add','--',rel)
    if subprocess.run(['git','-C',str(REPO),'diff','--cached','--quiet','--',rel]).returncode:
        git('-c','user.name=Codex','-c','user.email=codex@localhost','commit','--only','-m',message,'--',rel)

def launch(l,stage,k,arms):
    mark=l.root/'stages'/f'{k:03}_{stage}.json'
    if mark.exists():return
    start=time.monotonic()
    if stage.startswith('refresh_'):
        phase=stage.split('_')[1];window=l.config['window_updates']
        cmd=['-m',PKG+'e25_refresh',phase,l.name,str(k),str(window)]
    else:cmd=['-m',PKG+'e25_worker',stage,l.name,str(k),*arms]
    print(utc(),l.name,k,stage,','.join(arms),flush=True)
    with (l.root/f'{k:03}_{stage}.log').open('a') as log:
        proc=subprocess.run(['torchrun','--standalone','--nproc-per-node=4',*cmd],stdout=log,stderr=log)
    status=dict(run=l.name,update=k,phase=stage,arms=arms,elapsed=time.monotonic()-start,allocated_gpus=4,
                exit_code=proc.returncode,finished=utc())
    append(RUN/'allocation.jsonl',status);proc.check_returncode();save(mark,status)

def baseline_legal():
    with (D/'reports/training_steps.tsv').open() as f:
        rows=[r for r in csv.DictReader(f,delimiter='\t') if r['update']=='1']
    assert len(rows)==4 and len({r['legal_rate'] for r in rows})==1
    return float(rows[0]['legal_rate'])

def trajectory_stats(l,k,a):
    path=l.folder(k,a);out=path/'train_statistics.json'
    if out.exists():return read(out)
    batch,q=l.batch(k,a);stats,cover=class_stats(batch,q)
    if k==1:
        legal=baseline_legal();n=batch.total_records if hasattr(batch,'total_records') else len(batch.rows)
    else:
        good=n=0
        for rep in range(2):
            src=path/f'rollout_{rep}.jsonl';packed=src.with_suffix('.jsonl.gz')
            if not packed.exists():
                temp=packed.with_suffix('.pending')
                with src.open('rb') as inf,gzip.open(temp,'wb',compresslevel=3) as outf:shutil.copyfileobj(inf,outf)
                with gzip.open(temp,'rb') as check:
                    assert hashlib.file_digest(check,'sha256').hexdigest()==read(path/'collection.json')['rollout_sha256'][rep]
                temp.replace(packed)
            with gzip.open(packed,'rt') as f:
                for line in f:
                    row=json.loads(line);n+=1;good+=row['legal']
            src.unlink(missing_ok=True)
        legal=good/n
    import torch
    from collections import defaultdict
    mass_file=path/'candidate_mass.pt'
    assert digest_file(mass_file)==read(path/'target_manifest.json')['mass_sha256']
    mass=torch.load(mass_file,weights_only=True)
    bins=defaultdict(lambda:torch.zeros(4,dtype=torch.float64))
    for row,values,w in zip(batch.rows,mass,batch.weights,strict=True):
        assert len(row['answer'])==len(values) and values.min()>=0 and values.max()<=1.00001
        for j,c in enumerate(row['classes']):
            bins[c][:3]+=w*values[j].double();bins[c][3]+=w
    masses={c:{key:float(x[j]/x[3]) for j,key in enumerate(('student','selected','placeholder'))} for c,x in bins.items()}
    opts=read(path/'optimizer.json')
    result=dict(update=k,arm=a,legal_rate=legal,records=n,scored=len(q),by_class=stats,candidate_mass=masses,
                optimizer_steps=opts['optimizer_steps'],clip_count=sum(x>1 for x in opts['grad_norms']),
                before_sha=opts['before_sha'],after_sha=opts['after_sha'],projection_version=l.version(k,a))
    save(out,result);return result

def cleanup(l,k,a):
    if k<2:return
    old=l.state(k-1,a);moment=old/'adam.pt'
    if moment.exists():
        assert digest_file(l.state(k,a)/'adam.pt')==read(l.folder(k,a)/'optimizer.json')['adam_sha256']
        moment.unlink()
    if (k-1)%10 and (old/'adapter').exists():
        assert (l.folder(k-1,a)/'measurement.json').exists() and (l.folder(k,a)/'measurement.json').exists()
        shutil.rmtree(old/'adapter')

def run(l,steps):
    done=l.root/'complete.json'
    if done.exists():return read(done)
    arms=list(l.config['rates']);l.initialize(steps)
    stopped={}
    if l.name!='calibration':
        l.folder(0,'B').mkdir(parents=True,exist_ok=True)
        launch(l,'eval',0,['B'])
        launch(l,'common',0,['B']);merge_common(l,0,['B'])
        behavior_report(l,0,{'B':0})
    for k in range(1,steps+1):
        end=l.root/'updates'/f'{k:03}.json'
        if end.exists():
            finished=read(end);stopped=finished['stopped']
            if len(stopped)==len(arms):break
            continue
        active=[a for a in arms if a not in stopped]
        if k>1:launch(l,'collect',k,active)
        launch(l,'targets',k,active)
        launch(l,'train',k,active)
        launch(l,'per_update',k,active);merge_update(l,k,active)
        info={a:trajectory_stats(l,k,a) for a in active}
        if k%10==0:
            if 'P' in active:
                launch(l,'refresh_score',k,['P']);launch(l,'refresh_fit',k,['P'])
            launch(l,'common',k,active);probe=merge_common(l,k,active)
            if l.name!='calibration':
                ref=baseline_legal()
                for a in active:
                    now=ref-info[a]['legal_rate']
                    prev=ref-read(l.folder(k-10,a)/'train_statistics.json')['legal_rate'] if k>10 else 0.
                    if min(now,prev)>=.1-1e-12:stopped[a]=k
                launch(l,'eval',k,active)
                behavior_report(l,k,{'B':0,**{a:stopped.get(a,k) for a in arms}})
                if k==10 and 'R-matched' in active:
                    ratio=(probe['R-matched']['stats']['all']['kl']/probe['P']['stats']['all']['kl'])**.5
                    save(l.root/'rematch_at_10.json',dict(ratio=ratio,within_band=.8<=ratio<=1.25,
                        final_lr_was_adjusted=l.config['match_adjusted'],feedback_to_lr=False))
        for a in active:cleanup(l,k,a)
        save(end,dict(update=k,active=active,stopped=stopped,time=utc()))
        if len(stopped)==len(arms):break
    result=dict(run=l.name,last_step=k,endpoints={a:stopped.get(a,k) for a in arms},
                stop_labels={a:'EARLY_STOP' if a in stopped else 'COMPLETED' for a in arms},timestamp=utc())
    save(done,result);return result

def main():
    os.umask(0o077)
    with (RUN/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not (RUN/'gpu_budget.json').exists():budget()
        from .e25_preflight_report import render
        render()
        costs=read(RUN/'gpu_budget.json');move=read(RUN/'movement.json')
        assert costs['benchmark']['tasks']==320 and costs['benchmark']['positions']>0
        rates={'R':1.e-6,'P':1.e-6}
        match=dict(adjusted=False)
        if move['needs_matched']:
            l=Layout('calibration');lr=move['initial_matched_lr']
            freeze(l.root/'settings.json',dict(rates={'R-matched':lr},rho=move['rho'],window_updates=0))
            run(l,10)
            actual=read(l.root/'monitor/010.json')['R-matched']['stats']['all']['kl']
            ratio=(actual/move['scores']['bf16']['P']['all']['kl'])**.5
            adjusted=not .8<=ratio<=1.25
            rates['R-matched']=lr/ratio if adjusted else lr
            match=dict(rho_prime=ratio,matched_movement=actual,first_lr=lr,adjusted=adjusted,final_lr=rates['R-matched'])
            freeze(PUBLIC/'calibration.json',match)
        l=Layout('formal')
        settings=dict(rates=rates,window_updates=costs['window_updates'],match_adjusted=match['adjusted'],
            calibration=match,primary=['P-R-matched' if move['needs_matched'] else 'P-R','P-B'],
            learning_rate_frozen=True,updates=100,seed=20261120,budget_sha=digest_file(RUN/'gpu_budget.json'))
        freeze(l.root/'settings.json',settings);freeze(PUBLIC/'formal_settings.json',settings)
        commit_file(PUBLIC/'formal_settings.json','Freeze measured Stage E learning rates and refresh window before official training')
        run(l,100)
        offline=read(RUN/'offline/complete.json')
        if offline['p_kl_pilot_after_E']:
            l=Layout('p_kl');freeze(l.root/'settings.json',dict(rates={'P-KL':1e-6},window_updates=0))
            run(l,10)
        from .e25_delivery import final_report
        final_report()
if __name__=='__main__':main()
