"""Continue R/P while calibrating and then launching a new formal R-matched from zero."""
import fcntl
import json
import os
import subprocess
import time
from collections import defaultdict

from .e25_driver import baseline_legal, cleanup, commit_file, freeze, launch, trajectory_stats
from .e25_behavior import report as behavior_report
from .e25_layout import Layout
from .e25_measure import merge_common, merge_update
from .e25_probe import RUN, PUBLIC
from .e2511_schedule import ARMS, CAL, MATCH, classify, interpolate, scheduled
from exploratory_shop_audit.grpo_forced_audit.common import append, read, save, utc

WORK = RUN/'section_2511'
PKG = 'exploratory_shop_audit.visible_opd_v0.'


def native(phase, round_number, jobs):
    if not jobs:
        return
    mark = WORK/'stages'/f'{round_number:03}_{phase}.json'
    if mark.exists():
        assert read(mark)['jobs'] == [list(j) for j in jobs]
        return
    payload=jobs
    if phase=='targets':
        from .e2511_targets import plan,merge
        manifest=WORK/'partition'/f'{round_number:03}.json'
        if not manifest.exists():
            save(manifest,plan(jobs))
        payload=read(manifest)
    start = time.monotonic()
    print(utc(), phase, jobs, flush=True)
    with (WORK/f'{round_number:03}_{phase}.log').open('a') as log:
        result = subprocess.run(['torchrun','--standalone','--nproc-per-node=4','-m',
                                 PKG+'e2511_worker',phase,json.dumps(payload)],stdout=log,stderr=log)
    names = {name for name, _, _ in jobs}
    data = dict(run=next(iter(names)) if len(names)==1 else 'shared_2511', phase=phase,
                round=round_number,jobs=jobs,elapsed=time.monotonic()-start,allocated_gpus=4,
                exit_code=result.returncode,finished=utc())
    append(RUN/'allocation.jsonl',data)
    result.check_returncode()
    if phase=='targets':
        merge(jobs,payload)
    save(mark,data)


def grouped_phase(layout, phase, k, arms):
    # At a common formal index the delayed matched arm arrives ten scheduling rounds later.
    # A separate receipt keeps the original R/P work authoritative on restart.
    key = '_'.join(arms)
    mark = WORK/'stages'/layout.name/f'{k:03}_{phase}_{key}.json'
    if mark.exists():
        return
    start = time.monotonic()
    print(utc(), layout.name, k, phase, arms, flush=True)
    with (mark.parent.parent.parent/f'{layout.name}_{k:03}_{phase}_{key}.log').open('a') as log:
        result = subprocess.run(['torchrun','--standalone','--nproc-per-node=4','-m',
                                 PKG+'e25_worker',phase,layout.name,str(k),*arms],stdout=log,stderr=log)
    data=dict(run=layout.name,update=k,phase=phase,arms=arms,elapsed=time.monotonic()-start,
              allocated_gpus=4,exit_code=result.returncode,finished=utc())
    append(RUN/'allocation.jsonl',data)
    result.check_returncode()
    save(mark,data)


def progress(layout,k,arm):
    return layout.root/'progress_2511'/f'{k:03}'/f'{arm}.json'


def setup():
    archive=read(PUBLIC/'superseded_rmatched_2511.json')
    assert archive['status']=='SUPERSEDED' and not archive['included_in_comparisons']
    formal=Layout('formal')
    for arm,kept in archive['continued_R_P'].items():
        original=read(formal.folder(1,arm)/'optimizer.json')
        assert all(original[key]==val for key,val in kept.items())
    form0=read(formal.root/'settings.json')
    old=read(PUBLIC/'calibration.json')
    move=read(PUBLIC/'movement.json')['scores']['bf16']
    m_r,m_p=(move[a]['all']['kl'] for a in ('R','P'))
    lr,power=interpolate(form0['rates']['R'],m_r,old['first_lr'],old['matched_movement'],m_p)
    cal=Layout(CAL)
    settings=dict(rates={MATCH:lr},window_updates=0,source='section_25.11',
                  rate1=form0['rates']['R'],movement1=m_r,rate2=old['first_lr'],
                  movement2=old['matched_movement'],target=m_p,exponent=power,
                  previous_rate_predicted_ratio=(m_r*(old['final_lr']/form0['rates']['R'])**power/m_p)**.5)
    freeze(cal.root/'settings.json',settings)
    freeze(PUBLIC/'calibration_2511_settings.json',settings)
    cal.initialize(10)
    formal.initialize(100)
    return settings


def freeze_matched(calsettings):
    layout=Layout(CAL)
    probe=read(layout.root/'monitor/010.json')[MATCH]
    stats=probe['stats']['all']
    ratio,label=classify(stats['kl'],calsettings['target'])
    assert stats['positions']==4448 and abs(stats['mass']-1)<1e-10
    for k in range(1,11):
        assert progress(layout,k,MATCH).is_file()
    form=Layout('formal')
    settings=read(form.root/'settings.json')
    lr=calsettings['rates'][MATCH]
    settings['rates'][MATCH]=lr
    result=dict(rho_double_prime=ratio,label=label,learning_rate=lr,frozen=True,readout=probe,
                reference_movement=calsettings['target'],null_reference_file='movement.json',
                original_calibration_file='calibration.json',
                calibration_final_sha=read(layout.folder(10,MATCH)/'optimizer.json')['after_sha'])
    settings.update(section='25.11',matching_2511={'rho_double_prime':ratio,'label':label,'frozen_lr':lr})
    freeze(PUBLIC/'calibration_2511_result.json',result)
    freeze(form.root/'settings_2511.json',settings)
    freeze(PUBLIC/'formal_settings_2511.json',settings)
    commit_file(PUBLIC/'formal_settings_2511.json','Freeze Section 25.11 R-matched learning rate after its second calibration')
    return result


def record_checkpoint(layout,k,arms):
    # Rebuild the combined table from raw shards even if a previous merge was interrupted.
    ready=[a for a in ARMS if all((layout.folder(k,a)/f'common_{rank}.json').exists() for rank in range(4))]
    assert set(arms)<=set(ready)
    return merge_common(layout,k,ready)


def finish_comparable(form,round_number,stopped):
    for k in range(1,min(100,round_number)+1):
        mark=form.root/'updates'/f'{k:03}.json'
        if mark.exists():
            continue
        active=[a for a in ARMS if stopped.get(a,101)>=k]
        if not active or not all(progress(form,k,a).exists() for a in active):
            break
        if k%10==0:
            endpoints={'B':0,**{a:min(k,stopped.get(a,k)) for a in ARMS}}
            behavior_report(form,k,endpoints)
            if k==10:
                vals=read(form.root/'monitor/010.json')
                ratio,label=classify(vals[MATCH]['stats']['all']['kl'],vals['P']['stats']['all']['kl'])
                save(form.root/'rematch_at_10.json',dict(ratio=ratio,within_band=label=='MATCHED',
                     label=label,final_lr_was_adjusted=True,feedback_to_lr=False,section='25.11'))
        save(mark,dict(update=k,active=active,stopped={a:v for a,v in stopped.items() if v<=k},time=utc()))


def run_round(round_number,jobs,stopped):
    layouts={name:Layout(name) for name,_,_ in jobs}
    if any(name=='formal' and arm==MATCH for name,_,arm in jobs):
        assert read(PUBLIC/'calibration_2511_result.json')['frozen']
        assert layouts['formal'].config['section']=='25.11'
    native('collect',round_number,[j for j in jobs if j[1]>1])
    native('targets',round_number,jobs)
    groups=defaultdict(list)
    for name,k,arm in jobs:
        groups[name,k].append(arm)
    eval_jobs=[]
    for (name,k),arms in groups.items():
        layout=layouts[name]
        if not all((layout.folder(k,a)/'optimizer.json').exists() for a in arms):
            grouped_phase(layout,'train',k,arms)
        if not all((layout.folder(k,a)/'measurement.json').exists() for a in arms):
            grouped_phase(layout,'per_update',k,arms)
            merge_update(layout,k,arms)
        for arm in arms:
            trajectory_stats(layout,k,arm)
        if k%10==0:
            if 'P' in arms:
                launch(layout,'refresh_score',k,['P'])
                launch(layout,'refresh_fit',k,['P'])
            grouped_phase(layout,'common',k,arms)
            record_checkpoint(layout,k,arms)
            if name=='formal':
                eval_jobs.extend((name,k,a) for a in arms)
    native('eval',round_number,eval_jobs)
    for name,k,a in jobs:
        layout=layouts[name]
        stats=read(layout.folder(k,a)/'train_statistics.json')
        if name=='formal' and k>=20 and k%10==0:
            prev=read(layout.folder(k-10,a)/'train_statistics.json')
            if baseline_legal()-max(prev['legal_rate'],stats['legal_rate'])>=.1-1e-12:
                stopped[a]=k
        if name=='formal' and a==MATCH and k==1:
            signed=read(layout.folder(k,a)/'optimizer.json')
            reference=read(layout.folder(1,'R')/'optimizer.json')
            assert signed['before_sha']==reference['before_sha'] and signed['cumulative_steps']==signed['optimizer_steps']
            assert signed['before_sha']!=read(PUBLIC/'calibration_2511_result.json')['calibration_final_sha']
        cleanup(layout,k,a)
        freeze(progress(layout,k,a),dict(run=name,update=k,arm=a,before_sha=stats['before_sha'],
                                        after_sha=stats['after_sha'],stopped=stopped.get(a)==k))


def main():
    if (RUN/'formal/SUPERSEDED_2512.json').exists():
        raise SystemExit('The interleaved formal run is archived under protocol 25.12')
    os.umask(0o077)
    WORK.mkdir(exist_ok=True)
    with (RUN/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        calsettings=setup()
        stopped={}
        for n in range(1,111):
            mark=WORK/'rounds'/f'{n:03}.json'
            if mark.exists():
                stopped=read(mark)['stopped']
                continue
            if n==11:
                freeze_matched(calsettings)
            jobs=scheduled(n,stopped)
            if jobs:
                run_round(n,jobs,stopped)
            finish_comparable(Layout('formal'),n,stopped)
            save(mark,dict(round=n,jobs=jobs,stopped=stopped,time=utc()))
        form=Layout('formal')
        ends={a:stopped.get(a,100) for a in ARMS}
        last=max(ends.values())
        assert (form.root/'behavior'/f'{last:03}.json').exists()
        save(form.root/'complete.json',dict(run='formal',last_step=last,endpoints=ends,
            stop_labels={a:'EARLY_STOP' if a in stopped else 'COMPLETED' for a in ARMS},timestamp=utc()))
        from .e25_delivery import final_report
        final_report()


if __name__=='__main__':
    main()
