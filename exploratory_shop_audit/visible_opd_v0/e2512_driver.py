"""Serial four-card P, conditional R-matched and R, with a training-domain-only endpoint gate."""
import fcntl
import os
import subprocess
import time

from .e25_behavior import report as behavior_report
from .e25_driver import REPO, cleanup, commit_file, freeze, trajectory_stats
from .e25_layout import Layout
from .e25_measure import merge_common, merge_update
from .e25_probe import PUBLIC, RUN
from .e2511_targets import merge
from .e2512_gate import decision
from .e2512_native import merge_tapes
from .e2512_routing import ROUTING, SERIAL, owners
from .e2512_worker import target_shards
from exploratory_shop_audit.grpo_forced_audit.common import append, digest_file, read, save, utc

ORDER=('P','R-matched','R')


def launch(l,phase,k,a,env=None):
    stage=l.root/'stages'/a/f'{k:03}_{phase}.json'
    if stage.exists():
        m=read(stage)
        assert (m['phase'],m['update'],m['arms'],m['routing_sha'])==(phase,k,[a],digest_file(ROUTING))
        return
    start=time.monotonic()
    logfile=l.root/'logs'/a/f'{k:03}_{phase}.log'
    logfile.parent.mkdir(exist_ok=True,parents=True)
    print(utc(),SERIAL,a,k,phase,flush=True)
    with logfile.open('a') as f:
        proc=subprocess.run(['torchrun','--standalone','--nproc-per-node=4','-m',
             'exploratory_shop_audit.visible_opd_v0.e2512_worker',phase,str(k),a],stdout=f,stderr=f,env=env)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    status=dict(run=SERIAL,update=k,phase=phase,arms=[a],source_commit=revision,routing_sha=digest_file(ROUTING),elapsed=time.monotonic()-start,
                allocated_gpus=4,exit_code=proc.returncode,finished=utc())
    physical = (os.environ if env is None else env).get('CUDA_VISIBLE_DEVICES')
    if physical:status['physical_devices']=physical
    append(RUN/'allocation.jsonl',status);proc.check_returncode()
    save(stage,status)


def setup(l):
    signed=read(PUBLIC/'calibration_2511_result.json')
    assert signed['frozen'] and read(RUN/'calibration_2511/complete_2512.json')['final_step']==10
    from .d_data import PARENT
    assert len(owners())==3600 and read(ROUTING)['split_sha256']==digest_file(PARENT/'asin_split_manifest.json')
    assert read(RUN/'formal/SUPERSEDED_2512.json')['status']=='SUPERSEDED'
    settings=dict(rates={'P':1e-6,'R-matched':signed['learning_rate'],'R':1e-6},section='25.13',adopted_old_layout_updates=5,four_engine_from_update=6,
        order=list(ORDER),window_updates=read(RUN/'gpu_budget.json')['window_updates'],seed=20261120,
        updates=100,primary=['P-R-matched','P-B'],learning_rate_frozen=True,
        matching_2511=dict(rho_double_prime=signed['rho_double_prime'],label=signed['label'],frozen_lr=signed['learning_rate']),
        routing_sha=digest_file(ROUTING),calibration_sha=digest_file(PUBLIC/'calibration_2511_result.json'),
        baseline='new four-engine B@0',training_legal_reference='B batch used for adopted update 1',control_gate='P@100 > B@0 count on the 800 T_opd_eval samples')
    freeze(l.root/'settings.json',settings);freeze(PUBLIC/'formal_settings_2512.json',settings)
    commit_file(PUBLIC/'formal_settings_2512.json','Freeze adopted-prefix serial settings before the new B baseline')


def ended(l,a):return l.root/'arms'/a/'complete.json'


def endpoints_at(l,k):
    result={'B':0}
    for a in ORDER:
        end=ended(l,a)
        step=min(k,read(end)['step']) if end.is_file() else k
        if all((l.root/'evaluation'/f'{step:03}'/a/f'rank{i}'/'complete.json').exists() for i in range(4)):
            result[a]=step
    return result


def gate(l,k):
    result=decision(l,k)
    freeze(l.root/'controls_gate.json',result);freeze(PUBLIC/'serial_controls_gate.json',result)
    commit_file(PUBLIC/'serial_controls_gate.json','Record the serial P endpoint decision before starting any control')
    return result


def train_arm(l,a):
    if ended(l,a).exists():return read(ended(l,a))
    if a!='P':
        assert read(l.root/'controls_gate.json')['decision']=='RUN_CONTROLS'
        assert ended(l,ORDER[ORDER.index(a)-1]).is_file()
    l.initialize(100,arms=[a])
    from .e2513_adopt import adopt_prefix
    adopt_prefix(l,a)
    for k in range(6,101):
        receipt=l.root/'arms'/a/'updates'/f'{k:03}.json'
        if receipt.exists():
            if read(receipt)['stopped']:break
            continue
        if k!=1 or not (l.root/'shared_first/collection.json').is_file():
            launch(l,'collect',k,a)
        merge_tapes(l,k,a)
        launch(l,'targets',k,a)
        merge([(SERIAL,k,a)],target_shards(l,k,a))
        launch(l,'train',k,a)
        opt=read(l.folder(k,a)/'optimizer.json')
        if k==1:
            assert opt['optimizer_steps']==opt['cumulative_steps']
            if a!='P':
                assert opt['before_sha']==read(l.folder(1,'P')/'optimizer.json')['before_sha']
                assert read(l.folder(k,a)/'collection.json')==read(l.folder(1,'P')/'collection.json')
        launch(l,'per_update',k,a);merge_update(l,k,[a])
        stats=trajectory_stats(l,k,a)
        stopped=False
        if k%10==0:
            if a=='P':
                launch(l,'refresh_score',k,a);launch(l,'refresh_fit',k,a)
            launch(l,'common',k,a)
            ready=[arm for arm in ORDER if all((l.folder(k,arm)/f'common_{i}.json').exists() for i in range(4))]
            monitors=merge_common(l,k,ready)
            if k==10 and a=='R-matched':
                from .e2511_schedule import classify
                ratio,label=classify(monitors[a]['stats']['all']['kl'],monitors['P']['stats']['all']['kl'])
                save(l.root/'rematch_at_10.json',dict(ratio=ratio,within_band=label=='MATCHED',label=label,feedback_to_lr=False,section='25.13'))
            if k>=20:
                prev=read(l.folder(k-10,a)/'train_statistics.json')
                baseline=read(l.root/'adoption'/f'{a}.json')['initial_training_legal_rate']
                stopped=baseline-max(prev['legal_rate'],stats['legal_rate'])>=.1-1e-12
            launch(l,'eval',k,a)
            if a=='P' and (stopped or k==100):gate(l,k)
            behavior_report(l,k,endpoints_at(l,k))
        save(receipt,dict(arm=a,update=k,stopped=stopped,before_sha=stats['before_sha'],
             after_sha=stats['after_sha'],optimizer_steps=stats['optimizer_steps'],native_layout='fixed_four_engines',
             provenance='continued',finished=utc()))
        cleanup(l,k,a)
        if stopped:break
    status=dict(arm=a,step=k,status='EARLY_STOP' if read(receipt)['stopped'] else 'COMPLETED',time=utc())
    save(ended(l,a),status)
    return status


def commit_delivery():
    report_names=('STAGE_E.md','stage_e_steps.tsv','stage_e_monitors.tsv','stage_e_category_metrics.json',
        'stage_e_behavior_by_checkpoint.tsv','stage_e_final_behavior.json','stage_e_completion.json')
    paths=[str((PUBLIC/name).relative_to(REPO)) for name in report_names]
    assert all((PUBLIC/name).stat().st_size>0 for name in report_names)
    subprocess.run(['git','-C',str(REPO),'add','--',*paths],check=True)
    if subprocess.run(['git','-C',str(REPO),'diff','--cached','--quiet','--',*paths]).returncode:
        subprocess.run(['git','-C',str(REPO),'-c','user.name=Codex','-c','user.email=codex@localhost',
                        'commit','--only','-m','Record serial Stage E outcomes, gate and checkpoint reports','--',*paths],check=True)


def main():
    os.umask(0o077)
    l=Layout(SERIAL)
    with (l.root/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        setup(l)
        l.folder(0,'B').mkdir(exist_ok=True,parents=True)
        launch(l,'eval',0,'B');launch(l,'common',0,'B')
        if not (l.root/'behavior/000.json').exists():
            merge_common(l,0,['B']);behavior_report(l,0,{'B':0})
        p=train_arm(l,'P')
        outcome=gate(l,p['step']);finished={'P':p};not_run=[]
        if outcome['decision']=='RUN_CONTROLS':
            for a in ORDER[1:]:finished[a]=train_arm(l,a)
        else:
            not_run=list(ORDER[1:])
            for a in not_run:
                assert not (l.root/'arms'/a/'updates').exists()
                freeze(l.root/'arms'/a/'not_run.json',dict(arm=a,status='not run',gate_sha=digest_file(l.root/'controls_gate.json')))
        last=max(v['step'] for v in finished.values())
        for k in range(10,last+1,10):behavior_report(l,k,endpoints_at(l,k))
        save(l.root/'complete.json',dict(run=SERIAL,last_step=last,endpoints={a:v['step'] for a,v in finished.items()},
             stop_labels={**{a:v['status'] for a,v in finished.items()},**{a:'not run' for a in not_run}},
             not_run=not_run,gate=outcome,section='25.13',adopted_steps=5,timestamp=utc()))
        from .e25_delivery import final_report
        final_report(l)
        commit_delivery()


if __name__=='__main__':
    main()
