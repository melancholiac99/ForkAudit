"""Model four reserved GPUs explicitly, including idle time and the separate calibration."""
import math
import numpy as np
from .e25_probe import RUN,PUBLIC
from .s23_cross import RUN as D
from exploratory_shop_audit.grpo_forced_audit.common import read,read_rows,save,digest_file

def budget():
    rows=[r for r in read_rows(D/'allocation.jsonl') if r['exit_code']==0]
    def mean(stage):return np.mean([r['seconds'] for r in rows if r['stage']==stage and 2<=r['k']<=10])
    score,fit=(int(t.split()[2])-int(t.split()[1]) for t in (RUN/'refresh_timing.txt').read_text().strip().splitlines()[-2:])
    refreshed=RUN/'benchmark/refresh/010'
    fits=[read(refreshed/f'fit_{i}.json') for i in range(4)]
    scored=[read(refreshed/f'score_{i}.json') for i in range(4)]
    assert score>0 and fit>0 and len({v['actor_sha'] for v in fits+scored})==1
    block=(score+fit)*4/3600
    collect,target,train,measure=map(mean,('d_collect','d_targets','d_train','d_measure'))
    standalone=(collect+target+train/4+measure)*10/3600
    window=10*min(10,max(1,math.floor(.5*standalone/block)))
    movement=read(RUN/'movement.json');n=2+int(movement['needs_matched'])
    def policy_hours(updates,n):
        return updates*(collect+target+(train+measure)*n/4)*4/3600
    behavior={r['k']:r['seconds'] for r in rows if r['stage']=='d_eval'}
    # 4 dedicated devices, one per arm in the end evaluations; unused devices remain reserved.
    evaluation=(behavior[0]+10*behavior[10])*4/3600
    refresh=sum(min(k,window)/10*block for k in range(10,101,10))
    full=sum(range(1,11))*block
    # Four independent row shards for all arms. First baseline bank is already available.
    probe=RUN/'movement'
    def time_per_arm(prec):
        return max(read(probe/f'{prec}_null_{i}.json')['seconds_since_precision_start'] for i in range(4))/6
    common=(time_per_arm('bf16')+time_per_arm('fp32'))*4/3600
    core=policy_hours(100,n)
    cal=(policy_hours(10,1)+common) if n==3 else 0.
    total=core+refresh+10*n*common+evaluation
    out=dict(window_updates=window,active_training_groups=n,calibration_gpu_h=cal,E_gpu_h=total,
        E_plus_calibration_gpu_h=total+cal,wall_hours_if_four_reserved=(total+cal)/4,
        all_history_refresh_gpu_h=full,selected_refresh_gpu_h=refresh,ten_update_P_occupied_gpu_h=standalone,
        baseline_phase_mean_seconds=dict(collect=collect,target=target,train_four=train,measure_four=measure),
        cost_components_E_gpu_h=dict(rollout_target_training_measurement=core,behavior=evaluation,
            refresh=refresh,common_probe=10*n*common),benchmark=dict(actor_sha=fits[0]['actor_sha'],cutoff=10,
            tasks=len(set(g for s in scored for g in s['goals'])),records=scored[0]['total_records'],
            positions=sum(s['positions'] for s in scored),seconds_score=score,seconds_fit=fit,
            allocated_gpu_hours=block,folds=4,raw_tape_manifest_sha=fits[0]['history_sha']),
        assumptions='Four reserved A100s; D phase means and measured 10-update refresh scale linearly; no early stop; native replay/IO and trajectory lengths can change.',
        movement_sha=digest_file(RUN/'movement.json'))
    save(PUBLIC/'gpu_budget.json',out);save(RUN/'gpu_budget.json',out);return out
