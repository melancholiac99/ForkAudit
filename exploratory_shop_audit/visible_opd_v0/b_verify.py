"""Verify provenance, paired labels, OOF separation and delivery before signing B."""
from collections import Counter
import hashlib,json,re
from pathlib import Path
import numpy as np
import torch
from .b_collect import RUN,PARENT
from .action_gate import REGISTRATION
from .b_metrics import strength
from .section16 import freeze_p2_minima
from exploratory_shop_audit.grpo_forced_audit.common import ROOT,MODEL,read,save,digest_file,digest_value

def verify():
    rep=RUN/'reports';splits=read(PARENT/'asin_split_manifest.json');env=read(PARENT/'env_manifest.json')
    for n,h in env['model_files'].items():
        assert digest_file(MODEL/n)==h
    selected=read(RUN/'packet_selection_v2.json')
    assert selected['selected']=='full' and selected['original_verdict']=='PRIVILEGE_NOT_USEFUL'
    registration=(Path(__file__).parent/'evidence/content_b_20261120/protocol_registration.gitcommit').read_bytes()
    assert hashlib.sha1(b'commit '+str(len(registration)).encode()+b'\x00'+registration).hexdigest()==REGISTRATION
    commit_time=int(re.search(rb'^committer .* (\d+) [+-]\d{4}$',registration,re.MULTILINE)[1])
    recount_time=(RUN/'packet_selection_v2.json').stat().st_mtime_ns
    assert commit_time*10**9 < recount_time
    goals=[r['goal_index'] for r in splits['T_cal']];ids={g:i for i,g in enumerate(goals)}
    hold={r['goal_index']:splits['cal_folds'][r['asin']] for r in splits['T_cal']}
    views=Counter();steps=Counter();terminal=Counter();classes=Counter();identity=0.;positions=0;max_center=0.;min_mass=1.
    for p in sorted(RUN.glob('tape_*.jsonl')):
        for line in p.open():
            row=json.loads(line);g=row['goal'];i=ids[g];t=row['step']
            assert row['rep']==0 and steps[g]==t and not terminal[g] and not any(row['na'].values())
            steps[g]+=1;terminal[g]+=int(row['done'])
            v=torch.load(RUN/'features'/f'{i:03d}_{t:02d}.pt',weights_only=True)
            assert v['goal']==g and v['step']==t and v['classes']==row['classes']
            mask=v['mask'].numpy();q=v['q'].numpy();cand=v['ids'].numpy();y=np.array(row['response']);d=v['d'].numpy()
            assert ((cand==y[:,None])&mask).sum(1).min()==1 and ((cand==y[:,None])&mask).sum(1).max()==1
            assert np.allclose(q.sum(1),1,atol=1e-5) and np.all(q[~mask]==0) and np.isfinite(d).all()
            identity=max(identity,v['identity_error']);max_center=max(max_center,float(abs((q*d).sum(1)).max()))
            for m in ('none','full','placeholder'):
                mass=v['mass_'+m].numpy();assert (mass>0).all() and (mass<1.00002).all()
                views[m]+=1;min_mass=min(min_mass,float(mass.min()))
            classes.update(v['classes']);positions+=len(y)
    assert set(steps)==set(goals) and set(terminal.values())=={1} and identity<=1e-5 and max_center<=1e-5
    idxs=[]
    for f in range(4):
        fitted=read(RUN/f'fit_{f}.json');o=np.load(RUN/f'fold_{f}.npz')
        ix=o['idx'];idxs.extend(ix.tolist())
        train=sorted(g for g in goals if hold[g]!=f);valid=sorted(g for g in goals if hold[g]==f)
        assert fitted['train_ids_sha256']==digest_value(train) and fitted['heldout_ids_sha256']==digest_value(valid)
        assert np.isfinite(o['content']).all() and np.isfinite(o['original']).all() and np.isfinite(o['synthetic']).all()
    assert sorted(idxs)==list(range(positions))
    assert read(rep/'p2_minima.json')==freeze_p2_minima(classes,512)
    files=['projection_report','original_label_control','power_injection_report','capacity_action_ridge_report','candidate_mass_report','calibration_strength_report','equivalent_product_bias_report','P1_decision','p2_minima']
    evidence={f:digest_file(rep/(f+'.json')) for f in files}
    outputs={p.name:digest_file(p) for p in sorted(RUN.glob('fold_*.npz'))}
    led=[json.loads(l) for l in (RUN/'b_cost.jsonl').read_text().splitlines()]
    costs=dict(collection=sum(x['gpu_hours'] for x in led if x['stage']=='collection'),three_view_scoring=sum(x['gpu_hours'] for x in led if x['stage']=='three_view_scoring'),projectors=sum(read(RUN/f'fit_{f}.json')['gpu_hours'] for f in range(4)),ridge_and_equivalence=sum(x['gpu_hours'] for x in led if x['stage'] in ('action_ridge','bias_audit')))
    save(rep/'acceptance.json',dict(passed=True,registered_before_recount=True,registered_commit=REGISTRATION,registered_unix_seconds=commit_time,recount_unix_ns=recount_time,original_a5_sha=selected['original_sha256'],recount_sha=digest_file(RUN/'packet_selection_v2.json'),raw_rollout_hashes={p.name:digest_file(p) for p in RUN.glob('tape_*.jsonl')},oof_hashes=outputs,reports=evidence,model_files=env['model_files'],seed=20261120,tasks=len(goals),steps=sum(steps.values()),positions=positions,classes=classes,scorer_views=views,max_target_identity_error=identity,max_center_residual=max_center,min_candidate_mass=min_mass,planned_score_factor=1.5,measured_gpu_hours=costs,total_measured_gpu_hours=sum(costs.values()),actor_updates=0,eval_V_tasks=0))
if __name__=='__main__':verify()
