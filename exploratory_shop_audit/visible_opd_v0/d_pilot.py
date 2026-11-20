"""Lockstep stage-D driver, with independently restartable stage/arm outputs."""
import fcntl
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from .d_data import RUN,SCRATCH,ARMS,folder,state_dir,initialize
from .s23_cross import EVIDENCE
from exploratory_shop_audit.grpo_forced_audit.common import append,read,save,utc,digest_file

PKG='exploratory_shop_audit.visible_opd_v0.'

def launch(stage,k,distributed=True):
    start=time.monotonic()
    args=['torchrun','--standalone','--nproc-per-node=4'] if distributed else [sys.executable]
    with (RUN/f'{stage}_{k}.log').open('a') as log:
        result=subprocess.run(args+['-m',PKG+stage,str(k)],stdout=log,stderr=log)
    append(RUN/'allocation.jsonl',dict(stage=stage,k=k,time=utc(),seconds=time.monotonic()-start,
        allocated_gpus=4,exit_code=result.returncode))
    result.check_returncode()

def pilot():
    os.umask(0o077)
    with (RUN/'pilot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert len((EVIDENCE/'cross_alignment.tsv').read_text().splitlines())==17
        initialize()
        for k in range(1,11):
            phases=([('d_collect','collection.json'),('d_targets','target_manifest.json')] if k>1 else [])
            phases += [('d_train','optimizer.json'),('d_measure','fp32.json')]
            for stage,manifest in phases:
                if all((folder(k,a)/manifest).exists() for a in ARMS):continue
                print(utc(),k,stage,flush=True)
                launch(stage,k)
            if k>1:
                for a in ARMS:
                    src=folder(k,a)
                    for rep,sha in enumerate(read(src/'collection.json')['rollout_sha256']):
                        tape=src/f'rollout_{rep}.jsonl'
                        packed=tape.with_suffix('.jsonl.gz')
                        if tape.exists():
                            temp=packed.with_suffix('.pending')
                            with tape.open('rb') as inf, gzip.open(temp,'wb',compresslevel=3) as out:
                                shutil.copyfileobj(inf,out)
                            with gzip.open(temp,'rb') as inf:
                                assert hashlib.file_digest(inf,'sha256').hexdigest()==sha
                            temp.replace(packed)
                            tape.unlink()
                # Discard only superseded, temporary Adam moments after a verified replacement.
                for a in ARMS:
                    previous=state_dir(k-1,a)/'adam.pt'
                    if not previous.exists():continue
                    new=state_dir(k,a)/'adam.pt'
                    assert digest_file(new)==read(folder(k,a)/'optimizer.json')['adam_sha256']
                    previous.unlink()
        save(RUN/'training_complete.json',dict(updates=10,arms=list(ARMS),time=utc()))

if __name__=='__main__':pilot()
