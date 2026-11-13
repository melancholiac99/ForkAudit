"""Independent audit for the bounded four-round native baseline, not efficacy."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from audit_run import recompute

ROOT = Path(__file__).resolve().parents[2]


def audit(run):
    registration = json.loads((run / 'registration.json').read_text())
    checks = []
    for line in (run / 'mismatch.jsonl').read_text().splitlines():
        reported = json.loads(line)
        suffix = (f"step_{reported['step']}" if reported['phase'] == 'pre_update'
                  else f"post_update_step_{reported['step']}")
        path = run / f'probabilities_{suffix}.npz'
        with np.load(path) as data:
            actual, extra = recompute(data['actor'], data['rollout'], data['mask'])
            for k, v in actual.items():
                assert math.isclose(v, reported[k], rel_tol=1e-9, abs_tol=1e-10), (path, k)
            _, idx = np.unique(data['row_ids'], return_index=True)
            unique, unique_extra = recompute(data['actor'][idx], data['rollout'][idx], data['mask'][idx])
        checks.append(dict(step=reported['step'], phase=reported['phase'],
            gates_passed=actual['sampled_behavior_k3'] <= .02 and actual['clip_fraction'] <= .30,
            **actual, **extra, unique_rows=dict(**unique, **unique_extra),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    tb = EventAccumulator(str(ROOT / '.runtime/tensorboard/ctha-a100-r2' / run.name),
                          size_guidance={'scalars': 0}).Reload()
    scalars = {tag: [dict(step=e.step,value=e.value,wall_time=e.wall_time) for e in tb.Scalars(tag)]
               for tag in tb.Tags().get('scalars', [])}
    with (run / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(['metric','step','value','wall_time'])
        for key, events in scalars.items():
            for e in events:
                writer.writerow([key,e['step'],e['value'],e['wall_time']])
    table = []
    lookup = lambda tag, step: next((e['value'] for e in scalars.get(tag,[]) if e['step'] == step), None)
    for step in range(1,5):
        phases = {r['phase']:r for r in checks if r['step']==step}
        row = dict(round=step, training_success_rate=lookup('episode/success_rate',step),
                   rollout_seconds=lookup('timing_s/gen',step), scoring_seconds=lookup('timing_s/old_log_prob',step),
                   actor_seconds=lookup('timing_s/update_actor',step), total_seconds=lookup('timing_s/step',step),
                   grad_norm=lookup('actor/grad_norm',step))
        for phase in ['pre_update','post_update']:
            for field in ['sampled_behavior_k3','clip_fraction','token_ESS_fraction','response_ESS_fraction',
                          'token_ratio_min','token_ratio_max']:
                row[phase+'_'+field] = phases.get(phase,{}).get(field)
        table.append(row)
    with (run / 'training_table.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    required={(s,p) for s in range(1,5) for p in ['pre_update','post_update']}
    got={(r['step'],r['phase']) for r in checks}
    finite = all(e['value'] is not None and math.isfinite(e['value']) for e in scalars.get('actor/grad_norm',[]))
    execution = (json.loads((run/'exit.json').read_text())['returncode']==0 and got==required and
                 all(r['gates_passed'] for r in checks) and len(scalars.get('actor/grad_norm',[]))==4 and finite)
    checkpoint=run/'checkpoints/global_step_4/actor'
    manifests=[]
    if checkpoint.exists():
        for path in sorted(checkpoint.rglob('*')):
            if path.is_file():
                h=hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
                manifests.append(dict(path=str(path.resolve()),bytes=path.stat().st_size,sha256=h.hexdigest()))
    (run/'checkpoint_files.json').write_text(json.dumps(manifests,indent=2)+'\n')
    result=dict(run=run.name, configuration_sha256=registration['configuration_sha256'],
        four_round_execution='PASS' if execution else 'FAIL_OR_INCOMPLETE',
        Engineering='PASS' if execution else 'FAIL_OR_INCOMPLETE',
        scope='Native BF16 statistical-baseline runtime gates; final checkpoint restore audited separately',
        Learning='RUN; descriptive four TRAIN tasks only',Generalization='NOT_RUN',CTHA_online_efficacy='NOT_TESTED',
        exact_trajectory_reproduction='Not required by authorized statistical protocol; historical NOT_REPRODUCED retained',
        probability_checks=checks,training_table=table,
        greedy_training_task_evaluation=scalars.get('val/success_rate',[]),
        precision=json.loads((run/'precision_rank_0.json').read_text()),
        checkpoint_live_state=json.loads((checkpoint/'live_fingerprint_rank_0.json').read_text()) if checkpoint.exists() else None,
        checkpoint_files='checkpoint_files.json',tensorboard=scalars)
    (run/'baseline_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args()
    result=audit(args.run.resolve())
    print(json.dumps({k:v for k,v in result.items() if k not in ['probability_checks','tensorboard']},indent=2))
    assert result['four_round_execution']=='PASS', 'Bounded baseline incomplete or failed'


if __name__=='__main__':main()
