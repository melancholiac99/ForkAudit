"""Aggregate ten fixed-target updates and unprivileged paired evaluation."""
import csv
import gzip
import json
import sys
from collections import Counter,defaultdict
from itertools import chain
import numpy as np
import torch
from .d_data import RUN,ARMS,folder,load_batch,SMOKE,PARENT,state_dir
from .c_targets import STAGE_B
from .s21_data import SmokeBatch
from .s22_monitor import cumulative_window
from .s22_report import tsv
from .s23_cross import EVIDENCE
from .section16 import DECISION_CLASSES,p2_coverage
from exploratory_shop_audit.grpo_forced_audit.common import read,read_rows,save,digest_file,stable_seed


def archive_final(arm):
    """Losslessly store the final adapter against the shared §21 initialization."""
    from safetensors.torch import load_file
    from .s21_store import pack,unpack_member,RUN as PRIOR
    import shutil,zipfile
    from pathlib import Path
    src=state_dir(10,arm)/'adapter'
    dest=RUN/'final_adapters'/arm
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    initial=torch.load(PRIOR/'original/initial.pt',weights_only=True)
    values=load_file(src/'adapter_model.safetensors')
    pairs=[(n,values[n.replace('.default.weight','.weight')]) for n in initial]
    local=state_dir(10,arm)/'durable.zip'
    pack(local,pairs,initial)
    with zipfile.ZipFile(local) as z:
        for n,p in pairs:assert np.array_equal(unpack_member(z,n,initial).view(np.uint32),p.numpy().view(np.uint32))
    size=local.stat().st_size
    if shutil.disk_usage(dest).free<size+128*1024**2:
        raise OSError(f'{arm} needs {size} durable bytes plus 128 MiB; verified archive retained at {local}')
    dst=dest/'adapter.zip'
    tmp=dest/'adapter.pending'
    shutil.copyfile(local,tmp)
    assert digest_file(tmp)==digest_file(local)
    tmp.replace(dst)
    save(dest/'complete.json',dict(sha256=digest_file(dst),bytes=size,source_sha256=digest_file(src/'adapter_model.safetensors')))
    local.unlink()


def first_legal():
    from exploratory_shop_audit.grpo_forced_audit.checks import environment
    engine=environment(SMOKE,'s23_replay_legality')
    legal=0;total=0
    try:
        for rep in range(2):
            for r in read_rows(SMOKE/f'rollout_{rep}.jsonl'):
                key=f"D_legal_{r['goal']}_{rep}"
                if not r['step']:engine.call(cmd='open',key=key,goal_index=r['goal'])
                out=engine.call(cmd='step',key=key,action=r['action'])
                assert out['status']['done']==r['done']
                total+=1;legal+=out['legal']
                if r['done']:engine.call(cmd='close',key=key)
    finally:engine.close()
    return legal,total


def class_stats(batch,targets):
    stats=defaultdict(Counter);cover=defaultdict(set)
    for r,raw in zip(batch.rows,targets,strict=True):
        q=raw.double();s=r['snapshot'].double();mask=r['mask']
        q=q/q.sum(-1,keepdim=True);s=s/s.sum(-1,keepdim=True)
        kl=torch.special.xlogy(q,q/s.clamp_min(1e-300)).sum(-1)
        entropy=-torch.special.xlogy(s,s).sum(-1)
        grad=(q-s).abs().sum(-1)
        same=(raw==r['snapshot']).all(1)
        for j,c in enumerate(r['classes']):
            z=stats[c];cover[c].add(r['goal'])
            z.update(dict(n=1,kl=float(kl[j]),entropy=float(entropy[j]),gradient=float(grad[j]),
                          exact_equal=int(same[j]),signal=int(kl[j]>1e-5)))
    return {c:dict(tokens=v['n'],task_count=len(cover[c]),target_kl=v['kl']/v['n'],
        student_entropy_candidate=v['entropy']/v['n'],gradient_l1=v['gradient']/v['n'],
        exactly_zero_fraction=v['exact_equal']/v['n'],kl_above_1e5_fraction=v['signal']/v['n'])
        for c,v in stats.items()},cover


def training():
    minima=read(STAGE_B/'reports/p2_minima.json');probe=SmokeBatch();summary={};rows=[];byclass={}
    costs=read_rows(RUN/'allocation.jsonl')
    checks=dict(t_cal_calibration_sha256=digest_file(STAGE_B/'reports/calibration_strength_report.json'),
        calibration={c:{'b':v['b'],'beta':v['beta']['scale'],'gamma':v['gamma']['scale']}
                     for c,v in read(STAGE_B/'reports/calibration_strength_report.json')['classes'].items()},
        projector_sha256={p.name:digest_file(p) for p in sorted((STAGE_B/'projector_versions').glob('*.pt'))},
        P2_minima=minima,workers=4,lr=1e-6,weight_decay=0,clip=1,epoch=1)
    base_legal=first_legal()
    for arm in ARMS:
        updates=[];counts=Counter();covered=defaultdict(set);last_sha=None
        for k in range(1,11):
            path=folder(k,arm);opt=read(path/'optimizer.json');mon=read(path/'fp32.json')
            assert opt['after_sha']==mon['actor_after_sha'] and opt['before_sha']==mon['actor_before_sha']
            if k>1:assert last_sha==opt['before_sha']==read(path/'target_manifest.json')['scorer_sha']
            last_sha=opt['after_sha'];tape=torch.load(path/'probes.pt',weights_only=True)
            updates.append(dict(**mon,probe_before=tape['before'],probe_after=tape['after']))
            batch,q=load_batch(k,arm);stats,cover=class_stats(batch,q);byclass[f'{arm}/{k}']=stats
            for c in DECISION_CLASSES:
                counts[c]+=stats.get(c,{}).get('tokens',0);covered[c].update(cover[c])
            if k==1:legal,total=base_legal
            else:
                legal=total=0
                for rep in range(2):
                    gz=path/f'rollout_{rep}.jsonl.gz'
                    with (gzip.open(gz,'rt') if gz.exists() else (path/f'rollout_{rep}.jsonl').open()) as f:
                        for line in f:
                            total+=1;legal+=json.loads(line)['legal']
            assert total==len(batch.rows)
            rows.append(dict(arm=arm,update=k,optimizer_steps=opt['optimizer_steps'],minibatch_rows=opt['minibatch_rows'],
                 records=len(batch.rows),positions=sum(len(r['answer']) for r in batch.rows),
                 legal_rate=legal/total,projection_age=k-1,actor_sha=opt['before_sha'],scorer_sha=mon['actor_before_sha'],
                 updated_sha=opt['after_sha'],**mon['fixed_target'],probe_movement=mon['probe_movement'],
                 bf16_delta_kl=opt['bf16_fixed_target']['delta_kl']))
        window=cumulative_window(updates,probe,probe.probes,probe.targets(arm))
        tasks={c:len(covered[c]) for c in DECISION_CLASSES}
        summary[arm]=dict(**window,improving_updates=sum(r['delta']>0 for r in window['rows']),
             counts=counts,covered_tasks=tasks,coverage_pass=p2_coverage(minima,counts,tasks))
    EVIDENCE.mkdir(exist_ok=True,parents=True)
    save(EVIDENCE/'training_report.json',dict(checks=checks,arms=summary,by_class=byclass,
        allocation_gpu_hours=sum(r['seconds']*r['allocated_gpus'] for r in costs if r['stage']!='d_eval')/3600,
        fixed_projector=True));tsv(EVIDENCE/'training_steps.tsv',rows)
    for a in ARMS:archive_final(a)

if __name__=='__main__':
    if sys.argv[1]=='training':
        training()
    elif sys.argv[1]=='behavior':
        from .d_behavior import report
        report()
    else:
        raise ValueError('Expected training or behavior')
