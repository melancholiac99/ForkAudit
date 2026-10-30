"""Independent E2 integrity and point-estimate audit, without changing scientific outputs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
KEY=['traj_id','turn']
def main():
 result=json.loads((OUT/'e2.json').read_text())
 for path,digest in result['inputs'].items():assert sha((ROOT/path).read_bytes())==digest,path
 e1=json.loads((OUT/'e1.json').read_text())
 for path,digest in e1['inputs'].items():assert sha((ROOT/path).read_bytes())==digest,path
 prep=json.loads((OUT/'e2_preparation.json').read_text())
 for path,digest in prep['inputs'].items():assert sha((ROOT/path).read_bytes())==digest,path
 raw=(OUT/'e2_requests.jsonl').read_bytes();assert sha(raw)==prep['requests_sha256']
 jobs=[json.loads(line) for line in raw.decode().splitlines()]
 for row in jobs:
  row=dict(row);digest=row.pop('request_sha256');assert sha(json.dumps(row,sort_keys=True,ensure_ascii=False).encode())==digest
 s=pd.read_parquet(OUT/'strata.parquet');d=s[s.live & s.stratum.isin(['I','U'])].reset_index(drop=True)
 assert len(d)==873 and int(d.het.sum())==386
 arms=pd.read_parquet(ROOT/'exploratory_deep_discovery/exploratory_arms.parquet').set_index(KEY+['j'])
 q={source:[] for source in ['12','gold']};checked=0
 for row in d.to_dict('records'):
  tid,t=row['traj_id'],row['turn'];rec=[]
  for folder in ['replay','replay_b1','replay_gold']:
   path=ROOT/f'data/{folder}/{tid}/t{t:02d}.json'
   if not path.exists():assert folder=='replay_gold';rec.append(None);continue
   r=json.loads(path.read_text());actions=[r['factual_action']]+r['alternatives']
   assert actions==[arms.loc[(tid,t,j),'action'] for j in range(5)]
   rec.append(np.array([r['factual_outcomes']]+r['alt_outcomes'],float));checked+=1
  qs={'12':np.concatenate(rec[:2],axis=1).mean(1),'gold':rec[2].mean(1) if rec[2] is not None else np.full(5,np.nan)}
  for src,Q in qs.items():
   old=np.array([arms.loc[(tid,t,j),'q_'+src] for j in range(5)])
   assert np.allclose(Q,old,equal_nan=True),(tid,t,src);q[src].append(Q)
 q={src:np.asarray(rows) for src,rows in q.items()}
 max_error=0.;gains={};models={}
 for name in ['7b','32b']:
  run=json.loads((OUT/f'e2_{name}/run.json').read_text());assert run['complete'] and run['contexts']==3492
  checks=json.loads((OUT/f'e2_{name}/numerical_check.json').read_text());assert len(checks)==5 and all(c['passed'] and c['tv']<.005 for c in checks)
  rows=[json.loads(line) for line in (OUT/f'e2_{name}/scores.jsonl').read_text().splitlines()]
  assert len(rows)==len(set(r['request_id'] for r in rows))==3492
  for row in rows:
   lp=np.array(row['sequence_logp']);p=np.exp(lp-lp.max());p/=p.sum()
   assert np.allclose(p,row['candidate_prob'],atol=1e-12) and int(p.argmax())==row['argmax']
  frame=pd.DataFrame(rows)
  for condition in ['S','L','F','Fx']:
   part=frame[frame.condition.eq(condition)].set_index(KEY).reindex(pd.MultiIndex.from_frame(d[KEY]));p=np.stack(part.candidate_prob)
   for src,Q in q.items():
    for mode,prob in [('soft',p),('argmax',np.eye(5)[p.argmax(1)])]:
     J=np.einsum('ij,ij->i',prob,Q)
     for metric,values in [('G_u',J-Q.mean(1)),('G_c',J-Q[:,0])]:
      gains[src,name,condition,mode,metric]=values
      for layer in ['I','U']:
       mask=d.stratum.eq(layer).to_numpy() & d.het.to_numpy() & np.isfinite(values)
       expected=result['table'][src]['/'.join([name,condition,mode,metric,layer])]
       assert int(mask.sum())==expected['n'];max_error=max(max_error,abs(float(values[mask].mean())-expected['estimate']))
  models[name]=dict(max_numerical_tv=max(c['tv'] for c in checks),contexts=len(rows),peak_gb=run['peak_gb'],revision=run['revision'])
  if name=='7b':expected={line.split()[1]:line.split()[0] for line in (ROOT/'env/qwen_sha256.txt').read_text().splitlines() if len(line.split())==2 and line.split()[1].endswith('.safetensors')}
  else:
   meta=json.loads((OUT/'e2_32b_model_source.json').read_text());assert run['revision']==meta['sha']
   expected={item['rfilename']:item['lfs']['sha256'] for item in meta['siblings'] if item['rfilename'].endswith('.safetensors')}
  assert run['model_sha256']==expected
 assert max_error<1e-12
 # Independently build the paired trajectory bootstrap for the P4-5 interaction.
 trajs=sorted(s.loc[s.live,'traj_id'].unique());inv={v:i for i,v in enumerate(trajs)}
 weights=np.random.default_rng(30102026).multinomial(len(trajs),np.ones(len(trajs))/len(trajs),size=10000)
 max_ci_error=0.
 for src in ['12','gold']:
  x=gains[src,'32b','S','soft','G_u']-gains[src,'7b','S','soft','G_u'];dist={}
  for layer in ['I','U']:
   mask=d.stratum.eq(layer).to_numpy() & d.het.to_numpy() & np.isfinite(x)
   sums=np.zeros(len(trajs));counts=np.zeros(len(trajs))
   for i in np.flatnonzero(mask):sums[inv[d.traj_id.iloc[i]]]+=x[i];counts[inv[d.traj_id.iloc[i]]]+=1
   with np.errstate(invalid='ignore',divide='ignore'):dist[layer]=(weights@sums)/(weights@counts)
  ci=np.nanpercentile(dist['U']-dist['I'],[2.5,97.5]);max_ci_error=max(max_ci_error,float(abs(ci-result['contrasts'][src]['P4-5_interaction']['ci95']).max()))
 assert max_ci_error<1e-12
 report=dict(passed=True,input_hashes_checked=len(result['inputs']),E1_input_hashes_unchanged=len(e1['inputs']),preparation_inputs_unchanged=len(prep['inputs']),raw_replay_files_checked=checked,
             point_estimate_max_error=max_error,paired_bootstrap_interaction_ci_max_error=max_ci_error,models=models)
 (OUT/'e2_verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
