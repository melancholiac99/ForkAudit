"""One registered read-only look; outputs are isolated from original results."""
from pathlib import Path
import hashlib, json, sys
import numpy as np
F=Path(__file__).resolve().parent;H=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(H))
import exp2_common as C
import exp5_token_slot as S
import token_slot_backend as K

T2='T2_b1_clip5_full';T1='T1_b1'

def interval(values,clusters):
    keep=[i for i,v in enumerate(values) if v is not None and np.isfinite(v)]
    if not keep:return dict(n=0,n_trajectories=0,estimate=None,ci95=None)
    vals=np.asarray([values[i] for i in keep],dtype=float)
    labels=[clusters[i] for i in keep];names=sorted(set(labels))
    ix=np.array([names.index(c) for c in labels])
    w=np.random.default_rng(20261106).multinomial(len(names),[1/len(names)]*len(names),size=4000)
    sums=np.bincount(ix,weights=vals,minlength=len(names));counts=np.bincount(ix,minlength=len(names))
    boot=(w@sums)/(w@counts)
    return dict(n=len(vals),n_trajectories=len(names),estimate=float(vals.mean()),ci95=np.percentile(boot,[2.5,97.5]).tolist())

def decision(stat):
    enough=stat['n']>=30 and stat['n_trajectories']>=10
    valid=stat['ci95'] is not None and all(np.isfinite(stat['ci95']))
    return 'stop_for_futility' if enough and valid and stat['ci95'][1]<3.0 else 'continue_to_full'

def point_contrast(record):
    q=np.asarray(record['q'],dtype=float);assert q.shape==(5,) and np.isfinite(q).all()
    values=[]
    for p in record['probes']:
        assert p['phase']==S.phase(p['char'],record['k_char'])
        if p['phase']!='pre_k':continue
        assert len(p['arms'])==5
        probs=[]
        for target in (T2,T1):
            ll=[]
            for arm in p['arms']:
                score=arm['sequence_logp'][target]
                rebuilt=sum(t['target_logp'][target] for t in arm['tokens'])
                assert np.isfinite(score) and abs(score-rebuilt)<1e-5
                ll.append(score)
            x=np.asarray(ll);pr=np.exp(x-x.max());pr/=pr.sum();probs.append(pr)
        values.append(float((probs[0]-probs[1])@q*100))
    return float(np.mean(values)) if values else None

def checked_registration():
    reg=C.read_json(F/'registration.json')
    for p,h in reg['frozen_inputs'].items():assert C.sha_file(Path(p))==h,p
    for p,h in reg['implementation'].items():assert C.sha_file(Path(p))==h,p
    assert S.fingerprint()==reg['scoring_source']
    return reg

def load_snapshot(reg):
    out=Path(reg['output']);full=out/'32b/full';run=C.read_json(full/'run.json')
    assert run['fingerprint']['name']=='32b' and run['fingerprint']['limit']==0
    assert run['fingerprint']['source']==reg['scoring_source']
    assert run['fingerprint']['gpu_uuid']==reg['gpu_uuid']
    gates=C.read_json(full/'numerical_gate.json');assert gates and all(x['passed'] and x['tv']<.005 for x in gates)
    pilot=out/'32b/first32';v=C.read_json(pilot/'verification.json');assert v['passed'] and v['analysis_sha256']==C.sha_file(pilot/'analysis.json')
    groups,_=C.decode_rows(out/'groups.jsonl.gz');expected=groups[:reg['checkpoint_points']]
    assert [g['id'] for g in expected]==reg['checkpoint_ids']
    refs=C.read_json(out/'7b/full/manifest.json');records=[];manifest={}
    for g in expected:
        p=full/'records'/(g['id']+'.json.gz');h=C.sha_file(p);r=S.read(p)
        assert r['id']==g['id'] and r['cluster']==g['cluster'] and r['q']==g['q']
        assert r['group_sha256']==g['sha256'] and r['fingerprint_sha256']==K.digest(run['fingerprint'])
        ref=out/'7b/full/records'/p.name;assert C.sha_file(ref)==refs[g['id']];old=S.read(ref)
        for k in ('commitment_curve','k','k_char','positions','student_logp'):assert r[k]==old[k],(g['id'],k)
        assert r['k']==S.stable_k(r['commitment_curve']) and r['positions']==S.native_positions(g,r['k'])
        assert C.sha_file(p)==h;records.append(r);manifest[str(p)]=h
    assert len(records)>=190 and len({r['cluster'] for r in records})>=30
    return records,manifest

def analyze():
    reg=checked_registration();target=F/'results';target.mkdir(exist_ok=True)
    existing=target/'decision.json'
    if existing.exists():
        old=C.read_json(existing);assert old['registration_sha256']==C.sha_file(F/'registration.json')
        return old
    records,manifest=load_snapshot(reg)
    values=[point_contrast(r) for r in records];stat=interval(values,[r['cluster'] for r in records])
    out=dict(registration_sha256=C.sha_file(F/'registration.json'),decision=decision(stat),one_look=True,
        primary_target=T2,baseline=T1,checkpoint_points=len(records),checkpoint_trajectories=len({r['cluster'] for r in records}),
        pre_k_T2_minus_T1_value_pp=stat,points_without_pre_k=sum(x is None for x in values),
        paired_point_values=[dict(id=r['id'],cluster=r['cluster'],pre_k_difference_pp=v) for r,v in zip(records,values)],
        record_hashes=manifest,bootstrap=dict(draws=4000,seed=20261106,unit='trajectory after within-point averaging'),
        rule='One look: sufficient support and two-sided 95% CI upper < +3pp stops for futility; never early success.',
        interpretation='Futility rule, not mathematical impossibility of later reversal. Stopped-sample CIs are exploratory.')
    assert all(C.sha_file(Path(p))==h for p,h in manifest.items())
    C.save_json(existing,out);return out

if __name__=='__main__':
    d=analyze();print(json.dumps({k:d[k] for k in ('decision','checkpoint_points','checkpoint_trajectories','pre_k_T2_minus_T1_value_pp')},indent=2))
