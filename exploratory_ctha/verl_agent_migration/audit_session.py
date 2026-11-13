"""Independently recompute saved session-probe probability evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from audit_run import recompute

p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();run=args.run
r=json.loads((run/'session_benchmark.json').read_text()); checks=[];responses=[]
for i,b in enumerate(r['blocks']):
    path=run/f"probabilities_{i}_{b['mode']}.npz"
    with np.load(path) as d:
        actual,extra=recompute(d['actor'],d['rollout'],d['mask']);responses.append(d['responses'])
    for k,v in actual.items():assert math.isclose(v,b['raw_model_score_checks'][k],rel_tol=1e-9,abs_tol=1e-10),(path,k)
    assert actual['sampled_behavior_k3']<=.02 and actual['clip_fraction']<=.30
    assert b['manager_enters']==b['manager_exits']==(1 if b['mode']=='session' else 6)
    checks.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**actual,**extra))
for b in r['sampled_checks']:
    path=run/f"sampled_{b['mode']}.npz"
    with np.load(path) as d:actual,extra=recompute(d['actor'],d['rollout'],d['mask'])
    for k,v in actual.items():assert math.isclose(v,b[k],rel_tol=1e-9,abs_tol=1e-10)
    assert actual['sampled_behavior_k3']<=.02 and actual['clip_fraction']<=.30
    checks.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**actual,**extra))
equal=all(np.array_equal(responses[0],x) for x in responses[1:]);assert equal==r['exact_greedy_responses_equal']
assert r['native_full_response_loss_before']==r['native_full_response_loss_after']
assert r['live_state']==json.loads((run/'restore_rank_0.json').read_text())['actual']
assert json.loads((run/'exit.json').read_text())['returncode']==0
memory=list(csv.DictReader((run/'gpu_memory.csv').open()))
result=dict(passed=True,scope='Frozen real request tape; not complete ALFWorld rollout or learning',
    checks=checks,exact_greedy_responses_equal=equal,loss_and_state_checks='Runtime assertions and source checkpoint fingerprint verified',
    median_wall_speedup=r['median_wall_speedup'],peak_nvml_mib=max(float(x['used_mib']) for x in memory),
    Learning='NOT_RUN',Generalization='NOT_RUN',CTHA_online_efficacy='NOT_TESTED')
(run/'independent_session_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
