"""Adopt signed old-layout steps and resume on the frozen four-engine map at update six."""
import errno
import gzip
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

from .e2514_gate import control_authorized
from .d_data import SMOKE
from .e25_probe import RUN,PUBLIC
from .e2511_schedule import CAL,MATCH
from exploratory_shop_audit.grpo_forced_audit.common import append,digest_file,read,save,utc

RESTORE=RUN/'matched_state_reconstruction_2513'


def place(src,dst):
    if dst.exists():
        assert os.path.samefile(src,dst) or digest_file(src)==digest_file(dst),(src,dst)
        return
    dst.parent.mkdir(parents=True,exist_ok=True)
    try:
        dst.hardlink_to(src)
    except OSError as e:
        if e.errno!=errno.EXDEV:raise
        tmp=dst.with_suffix(dst.suffix+'.copying')
        shutil.copyfile(src,tmp);assert digest_file(src)==digest_file(tmp);tmp.replace(dst)


def verify_state(root,rec):
    weights=root/'adapter/adapter_model.safetensors';adam=root/'adam.pt'
    if not (weights.is_file() and adam.is_file()):
        return False
    assert digest_file(weights)==rec['safetensors_sha256'] and digest_file(adam)==rec['adam_sha256']
    return True


def copy_state(src,dst,rec):
    assert verify_state(src,rec)
    for f in (src/'adapter').iterdir():
        if f.is_file():place(f,dst/'adapter'/f.name)
    place(src/'adam.pt',dst/'adam.pt')
    assert not dst.is_symlink() and not (dst/'adapter').is_symlink()


def recover_matched(l):
    control_authorized(l)
    source=RUN/CAL
    # A later verified archive covers all earlier optimizer steps.
    receipts=list(RUN.glob('*/pilot/*/*/optimizer.json'))
    resume=0
    for k in range(5,0,-1):
        original=source/'pilot'/f'{k:02}'/MATCH/'optimizer.json'
        expected=read(original)
        dst=RESTORE/'state'/f'{k:02}'/MATCH
        candidates=[dst,source/'state'/f'{k:02}'/MATCH]
        for f in receipts:
            m=read(f)
            if (m.get('after_sha'),m.get('adam_sha256'))==(expected['after_sha'],expected['adam_sha256']):
                candidates.append(f.parents[3]/'state'/f.parent.parent.name/f.parent.name)
        for candidate in candidates:
            if verify_state(candidate,expected):
                if candidate!=dst:copy_state(candidate,dst,expected)
                optpath=RESTORE/'pilot'/f'{k:02}'/MATCH/'optimizer.json'
                if not optpath.exists():place(original,optpath)
                resume=k
                break
        if resume:break
    for k in range(resume+1,6):
        old=source/'pilot'/f'{k:02}'/MATCH
        opt=read(old/'optimizer.json')
        dst=RESTORE/'state'/f'{k:02}'/MATCH
        log=RESTORE/f'{k:03}_restore.log';log.parent.mkdir(parents=True,exist_ok=True)
        start=time.monotonic()
        with log.open('a') as f:
            proc=subprocess.run(['torchrun','--standalone','--nproc-per-node=4','-m',
                'exploratory_shop_audit.visible_opd_v0.e2513_restore',str(k)],stdout=f,stderr=f)
        append(RUN/'allocation.jsonl',dict(run='matched_state_reconstruction_2513',phase='signed_optimizer_replay',
               update=k,arms=[MATCH],allocated_gpus=4,elapsed=time.monotonic()-start,exit_code=proc.returncode,finished=utc()))
        proc.check_returncode()
        assert verify_state(dst,opt)
    return RESTORE/'state/05'/MATCH


def adopt_prefix(l,a):
    from .e25_driver import freeze
    out=l.root/'adoption'/f'{a}.json'
    if out.is_file():return read(out)
    if a!='P':control_authorized(l)
    root=RUN/(CAL if a==MATCH else 'formal')
    status=[];previous=None;count=0
    for k in range(1,6):
        old=root/'pilot'/f'{k:02}'/a;dst=l.folder(k,a)
        rec=read(old/'optimizer.json');stats=read(old/'train_statistics.json')
        assert rec['arm']==stats['arm']==a and rec['update']==stats['update']==k
        assert rec['before_sha']==stats['before_sha'] and rec['after_sha']==stats['after_sha']
        if previous is not None:assert rec['before_sha']==previous
        previous=rec['after_sha'];count+=rec['optimizer_steps']
        assert count==rec['cumulative_steps']
        assert read(dst.parent/'tasks.json')==read(old.parent/'tasks.json')
        for fname in ('optimizer.json','measurement.json','train_statistics.json','target_manifest.json'):
            place(old/fname,dst/fname)
        files=[]
        tape_root=SMOKE if k==1 else old
        signed=read(old/'collection.json')['rollout_sha256'] if k>1 else None
        for rep in range(2):
            f=tape_root/f'rollout_{rep}.jsonl'
            if not f.is_file():f=f.with_suffix('.jsonl.gz')
            if f.suffix=='.gz':
                with gzip.open(f,'rb') as stream:
                    raw=hashlib.file_digest(stream,'sha256').hexdigest()
            else:raw=digest_file(f)
            if signed is not None:assert raw==signed[rep]
            place(f,dst/f.name);files.append(raw)
        marker=dict(arm=a,update=k,stopped=False,before_sha=rec['before_sha'],after_sha=previous,
                    optimizer_steps=rec['optimizer_steps'],cumulative_steps=count,source=root.name,
                    provenance='adopted',native_layout='interleaved_before_section_25.12',rollout_sha256=files,
                    optimizer_receipt_sha=digest_file(old/'optimizer.json'))
        freeze(l.root/'arms'/a/'updates'/f'{k:03}.json',marker);status.append(marker)
    rec=read(root/'pilot/05'/a/'optimizer.json')
    state=root/'state/05'/a
    if not verify_state(state,rec):
        assert a==MATCH,'A signed P/R checkpoint must never be reconstructed or substituted'
        state=recover_matched(l)
    copy_state(state,l.state(5,a),rec)
    initial_legal=read(root/'pilot/01'/a/'train_statistics.json')['legal_rate']
    info=dict(arm=a,adopted_steps=5,source=root.name,resumed_from_state=str(state.relative_to(RUN)),
        before_six_sha=previous,adam_sha256=rec['adam_sha256'],safetensors_sha256=rec['safetensors_sha256'],
        cumulative_optimizer_steps=count,initial_training_legal_rate=initial_legal,layout_switch_at=6,steps=status)
    freeze(out,info);freeze(PUBLIC/f'adopted_2513_{a}.json',info)
    return info
