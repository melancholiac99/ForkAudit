"""Independent E2M score/point-estimate/bootstrap audit and plain-text report.
No scientific definitions or thresholds are changed by this reporting layer.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import gzip, hashlib, json, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exploratory_P4';AUDIT=OUT/'e2m_run'
sys.path.insert(0,str(OUT/'post_e2'))
from post_e2_mechanism import arm_classes
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 result=json.loads((OUT/'e2m.json').read_text())
 for p,h in result['inputs'].items():assert sha(p)==h,p
 frozen=json.loads((AUDIT/'protected_inputs.json').read_text())
 for p,h in frozen.items():assert sha(ROOT/p)==h,p
 requests={r['request_id']:r for r in map(json.loads,gzip.decompress((OUT/'e2m_requests.jsonl.gz').read_bytes()).splitlines())}
 assert len(requests)==1746
 s=pd.read_parquet(OUT/'strata.parquet');d=s[s.live & s.stratum.isin(['I','U'])].sort_values(['traj_id','turn']).reset_index(drop=True)
 pids=[f'{r.traj_id}_t{r.turn:02d}' for r in d.itertuples()];index={p:i for i,p in enumerate(pids)}
 arms=pd.read_parquet(ROOT/'exploratory_deep_discovery/exploratory_arms.parquet').set_index(['traj_id','turn','j'])
 keys=[(r.traj_id,r.turn,j) for r in d.itertuples() for j in range(5)]
 a=arms.loc[keys];actions=a.action.to_numpy().reshape(873,5)
 q={src:a['q_'+src].to_numpy(float).reshape(873,5) for src in ['12','gold']}
 episodes={}
 for line in gzip.decompress((OUT/'e2_requests.jsonl.gz').read_bytes()).splitlines():
  row=json.loads(line)
  if row['condition']=='L':episodes[row['traj_id']]=json.JSONDecoder().raw_decode(row['text'][row['text'].find('{"task":'):])[0]
 classes=np.array([arm_classes(actions[i],[(st['executed_action'],st['observation_after']) for st in episodes[r.traj_id]['steps'][:r.turn-1]]) for i,r in enumerate(d.itertuples())])
 probs={};max_prob_error=0.;models={}
 for m in ['7b','32b']:
  original=json.loads((OUT/f'e2_{m}/run.json').read_text());run=json.loads((OUT/f'e2m_{m}/run.json').read_text())
  assert run['complete'] and run['contexts']==1746
  for k in ['revision','model_sha256','scorer_sha256']:assert run[k]==original[k],(m,k)
  checks=json.loads((OUT/f'e2m_{m}/numerical_check.json').read_text());assert len(checks)==3
  assert all(c['passed'] and 0<=c['tv']<.005 for c in checks)
  models[m]=dict(contexts=1746,max_tv=max(c['tv'] for c in checks),revision=run['revision'])
  for prefix,conditions in [('e2',['S','F']),('e2m',['M','MF'])]:
   rows=[json.loads(line) for line in (OUT/f'{prefix}_{m}/scores.jsonl').read_text().splitlines()]
   assert len({r['request_id'] for r in rows})==len(rows)
   if prefix=='e2m':assert {r['request_id'] for r in rows}==set(requests)
   for c in conditions:
    chosen=[r for r in rows if r['condition']==c];assert len(chosen)==873
    values=np.full((873,5),np.nan)
    for r in chosen:
     i=index[r['point_id']];p=np.asarray(r['candidate_prob']);lp=np.asarray(r['sequence_logp']);expected=np.exp(lp-lp.max());expected/=expected.sum()
     assert p.shape==(5,) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-12
     max_prob_error=max(max_prob_error,float(abs(p-expected).max()));assert int(p.argmax())==r['argmax']
     if prefix=='e2m':
      req=requests[r['request_id']];assert req['actions']==list(actions[i]);assert req['request_sha256']==r['request_sha256']
     values[i]=p
    assert np.isfinite(values).all();probs[m,c]=values
 assert max_prob_error<1e-12
 # Regenerate the frozen multinomial trajectory bootstrap, using a separate aggregation path.
 trajs=np.unique(s.loc[s.live,'traj_id']);rng=np.random.default_rng(30102026)
 weights=rng.multinomial(len(trajs),np.full(len(trajs),1/len(trajs)),size=10000).astype(np.float32)
 vectors={};contrast_vectors={}
 for src in ['12','gold']:
  for m in ['7b','32b']:
   for c in ['S','F','M','MF']:
    vectors[f'{src}/{m}/{c}/G_u']=(probs[m,c]*q[src]).sum(1)-q[src].mean(1)
   for l,r in [('M','S'),('F','M'),('MF','F'),('MF','M')]:contrast_vectors[f'{src}/{m}/{l}-{r}/G_u']=vectors[f'{src}/{m}/{l}/G_u']-vectors[f'{src}/{m}/{r}/G_u']
  contrast_vectors[f'{src}/32b-7b/M/G_u']=vectors[f'{src}/32b/M/G_u']-vectors[f'{src}/7b/M/G_u']
  contrast_vectors[f'{src}/interaction_(32b-7b)_M_minus_S/G_u']=contrast_vectors[f'{src}/32b-7b/M/G_u']-(vectors[f'{src}/32b/S/G_u']-vectors[f'{src}/7b/S/G_u'])
 for m in ['7b','32b']:
  for k in ['E_new','R_seen']:
   for c in ['S','F','M','MF']:vectors[f'{m}/{c}/mass_{k}']=(probs[m,c]*(classes==k)).sum(1)
   contrast_vectors[f'{m}/M-S/mass_{k}']=vectors[f'{m}/M/mass_{k}']-vectors[f'{m}/S/mass_{k}']
 errors=[];ci_errors=[]
 for block,vv in [('table',vectors),('contrasts',contrast_vectors)]:
  for key,record in result[block].items():
   stem,layer=key.rsplit('/',1);v=vv[stem];mask=d.stratum.eq(layer).to_numpy() & d.het.to_numpy() & np.isfinite(v)
   assert int(mask.sum())==record['n']
   errors.append(abs(float(v[mask].mean())-record['estimate']))
   sums=np.array([v[mask & d.traj_id.eq(t).to_numpy()].sum() for t in trajs]);counts=np.array([(mask & d.traj_id.eq(t).to_numpy()).sum() for t in trajs])
   with np.errstate(invalid='ignore',divide='ignore'):dist=(weights@sums)/(weights@counts)
   ci=np.percentile(dist[np.isfinite(dist)],[2.5,97.5]);ci_errors.append(float(abs(ci-record['ci95']).max()))
 assert max(errors)<1e-12 and max(ci_errors)<1e-12
 c=result['contrasts'];lo=lambda key:c[key]['ci95'][0]>0
 pred={'M-1':c['32b/M-S/mass_R_seen/I']['ci95'][1]<0 and all(lo(f'{m}/M-S/mass_E_new/I') for m in ['7b','32b']),
       'M-2':{m:lo(f'12/{m}/M-S/G_u/I') for m in ['7b','32b']},
       'M-3':{m:lo(f'12/{m}/F-M/G_u/I') for m in ['7b','32b']},
       'M-4':lo('12/interaction_(32b-7b)_M_minus_S/G_u/I')}
 assert pred==result['predictions']
 reading='R-M-NULL' if not any(pred['M-2'].values()) else 'R-M-TWO-GAPS' if pred['M-1'] and all(pred['M-2'].values()) and all(pred['M-3'].values()) else 'R-M-PARTIAL'
 assert reading==result['reading']
 report=dict(passed=True,models=models,protected_files_unchanged=len(frozen),max_probability_error=max_prob_error,max_point_error=max(errors),max_ci_error=max(ci_errors),statistics_checked=len(errors),predictions_match=True,reading_match=True)
 (AUDIT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['P4-E2M server report', '', 'Reading: '+reading,'Predictions: '+json.dumps(pred,ensure_ascii=False),
 'Post_e2: selftest passed; all scientific outputs exactly reproduced, E2 G_u max error 2.7755575615628914e-17.',
 'Post_e2 mechanism analysis is post-hoc; E2M M-1..M-4 follow PREREG_E2M.md (cefd377).',
 '873 existing I/U Live points; 1746 contexts per model. Primary Fork population: I=184, U=202.',
 'M=self-history before t; MF=M plus aligned one-step feedback. S/F reused from original E2.',
 'No new trajectories, training, threshold changes or altered E1/E2/P4-X readings.',
 'All effects below in percentage points with 95% trajectory-bootstrap confidence intervals.',
 'P4 remains R-MIXED; P4-X remains R-X-MISDIRECT. E2M is a separate exploratory extension.',
 'Non-significance is not equivalence or proof that history has zero effect.', '', 'Primary I-layer contrasts:']
 def fmt(v):return f"{v['estimate']*100:+.3f} [{v['ci95'][0]*100:+.3f}, {v['ci95'][1]*100:+.3f}], n={v['n']}"
 for k,v in result['contrasts'].items():
  if k.endswith('/I') and not k.startswith('gold/'):lines.append(k+': '+fmt(v))
 for block in ['table','contrasts']:
  lines+=['',block+' (all, including Gold/U reporting items):']
  lines += [k+': '+fmt(v) for k,v in result[block].items()]
 lines+=['','Verification:',json.dumps(report,indent=2),'','Execution: see e2m_run/driver.log; commands use /root/miniconda3/envs/m0/bin/python.']
 (OUT/'E2M_REPORT.txt').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(reading=reading,predictions=pred,verification=report),indent=2))
if __name__=='__main__':main()
