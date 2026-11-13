"""Independent full-P1 counts/probabilities/outcomes and descriptive paired comparisons."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'verl_agent_migration'))
from audit_run import recompute
from native_p1 import read,write
from prepare_campaign import sha


def audit_arm(run):
    assert read(run/'exit.json')['returncode']==0
    checks=[]
    for line in (run/'mismatch.jsonl').read_text().splitlines():
        reported=json.loads(line);step=reported['step'];phase=reported['phase']
        name=f'probabilities_step_{step}.npz' if phase=='pre_update' else f'probabilities_post_update_step_{step}.npz'
        with np.load(run/name) as d:
            actual,extra=recompute(d['actor'],d['rollout'],d['mask'])
            _,ids=np.unique(d['row_ids'],return_index=True)
            unique,_=recompute(d['actor'][ids],d['rollout'][ids],d['mask'][ids])
        assert all(math.isclose(v,reported[k],rel_tol=1e-9,abs_tol=1e-10) for k,v in actual.items())
        assert actual['sampled_behavior_k3']<=.02 and actual['clip_fraction']<=.30
        checks.append(dict(step=step,phase=phase,**actual,diagnostics=extra,unique_rows=unique,sha256=sha(run/name)))
    assert {(r['step'],r['phase']) for r in checks}=={(s,p) for s in range(1,13) for p in ['pre_update','post_update']}
    tb=EventAccumulator(str(ROOT/'.runtime/tensorboard/ctha-a100-r2'/run.name),size_guidance={'scalars':0}).Reload()
    scalars={tag:[dict(step=e.step,value=e.value) for e in tb.Scalars(tag)] for tag in tb.Tags()['scalars']}
    gradients=scalars.get('actor/grad_norm',[])
    assert len(gradients)==12 and all(math.isfinite(e['value']) for e in gradients)
    schedule=read(Path(read(run/'registration.json')['env']['FORKAUDIT_P1_CAMPAIGN'])/'tasks.json')
    rows=[]
    lookup=lambda key,step:next((e['value'] for e in scalars.get(key,[]) if e['step']==step),None)
    for iteration in range(12):
        eps=read(run/f'episodes_train_{iteration:03d}.json')
        expected={g['game_file'] for g in schedule['train'][iteration]}
        assert len(eps)==64 and {e['game_file'] for e in eps}==expected
        assert all(sum(e['game_file']==g for e in eps)==4 for g in expected)
        assert all(e['provenance']['split']=='train' for e in eps)
        aux=read(run/f'applied_auxiliary_step_{iteration+1}.json')
        row=dict(round=iteration+1,train_success=float(np.mean([e['won'] for e in eps])),
            episodes=64,env_steps=sum(len(e['steps']) for e in eps),aux_unique_rows=aux['effective_unique_rows'])
        for key in ['timing_s/gen','timing_s/old_log_prob','timing_s/update_actor','timing_s/testing','timing_s/save_checkpoint','timing_s/step','actor/grad_norm','actor/pg_loss']:
            row[key]=lookup(key,iteration+1)
        rows.append(row)
    evals=[]
    for i in range(7):
        record=read(run/f'tune_evaluation_{i:02d}.json')
        eps=[e for j in range(i*5,(i+1)*5) for e in read(run/f'episodes_tune_{j:03d}.json')]
        assert len(eps)==80 and all(e['provenance']['split']=='tune' for e in eps)
        assert {e['game_file'] for e in eps}=={g['game_file'] for g in schedule['tune']}
        assert math.isclose(record['success_rate'],float(np.mean([e['won'] for e in eps])))
        record['after_round']=i*2;evals.append(record)
    for path in sorted(run.glob('validation_alignment_*.npz')):
        with np.load(path) as d:actual,_=recompute(d['actor'],d['rollout'],d['mask'])
        assert actual['sampled_behavior_k3']<=.02 and actual['clip_fraction']<=.30
    assert len(list(run.glob('validation_alignment_*.npz')))==35
    checkpoint=run/'checkpoints/global_step_12/actor'
    manifests=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(checkpoint.rglob('*')) if p.is_file()]
    assert manifests and (checkpoint/'live_fingerprint_rank_0.json').exists()
    write(run/'checkpoint_files.json',manifests)
    with (run/'training_table.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(execution='PASS',Engineering='PENDING_NATIVE_FINAL_RESTORE',Learning='RUN; full one-seed development P1',
        Generalization='NOT_TESTED_ON_HELDOUT_SPLITS',CTHA_online_efficacy='NOT_TESTED' if not sum(r['aux_unique_rows'] for r in rows) else 'MEASURED; requires paired comparison',
        checks=checks,training=rows,tune=evals,precision=read(run/'precision_rank_0.json'),
        checkpoint_live=read(checkpoint/'live_fingerprint_rank_0.json'),
        steps_to_80pct_tune=next((sum(r['env_steps'] for r in rows[:e['after_round']]) for e in evals if e['success_rate']>=.8),None))
    write(run/'full_p1_audit.json',result)
    return result


def compare(campaign):
    runs=read(campaign/'run_names.json')
    reports={arm:read(ROOT/'.runtime/runs'/name/'full_p1_audit.json') for arm,name in runs.items()}
    assert len({r['precision']['initial_trainable_sha256'] for r in reports.values()})==1
    output={}
    for arm,reference in [('gigpo','grpo'),('gigpo_ctha','gigpo')]:
        a=reports[arm]['tune'][-1]['per_game'];b=reports[reference]['tune'][-1]['per_game']
        assert set(a)==set(b) and len(a)==40
        delta=np.array([np.mean(a[g])-np.mean(b[g]) for g in sorted(a)])
        rng=np.random.default_rng(20261113)
        ci=np.percentile(delta[rng.integers(0,40,size=(10000,40))].mean(1),[2.5,97.5]).tolist()
        output[f'{arm}_minus_{reference}']=dict(mean=float(delta.mean()),ci95=ci,games=40)
    write(campaign/'comparison.json',dict(comparisons=output,interpretation='Paired development-Tune game bootstrap, one training seed; not heldout generalization or multi-seed replication',reports=reports))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--campaign',type=Path);a=p.parse_args()
    if a.run:audit_arm(a.run)
    else:compare(a.campaign)
