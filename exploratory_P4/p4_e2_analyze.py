"""Analyze complete paired E2 scores only; reject incomplete model runs."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,hashlib,sys,subprocess
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from evalcore import Boot
KEY=['traj_id','turn']
def main():
 inputs={}
 def track(path):
  b=(ROOT/path).read_bytes();inputs[path]=hashlib.sha256(b).hexdigest();return b
 for model in ['7b','32b']:
  p=OUT/f'e2_{model}/run.json'
  if not p.exists() or not json.loads(p.read_text())['complete']:raise SystemExit(f'E2 incomplete: {model}; no verdict issued.')
 s=pd.read_parquet(ROOT/'exploratory_P4/strata.parquet');track('exploratory_P4/strata.parquet')
 d=s[s.live & s.stratum.isin(['I','U'])].copy().reset_index(drop=True)
 assert len(d)==873
 b=Boot(s.loc[s.live,'traj_id'],B=10000,seed=30102026)
 b.inv=np.searchsorted(b.t,d.traj_id.to_numpy());keys=pd.MultiIndex.from_frame(d[KEY])
 arms=pd.read_parquet(ROOT/'exploratory_deep_discovery/exploratory_arms.parquet');track('exploratory_deep_discovery/exploratory_arms.parquet')
 track('exploratory_deep_discovery/exploratory_input_manifest.json')
 q={source:arms.pivot(index=KEY,columns='j',values='q_'+source).reindex(keys).to_numpy(float) for source in ['12','gold']}
 prep=json.loads(track('exploratory_P4/e2_preparation.json'))
 raw=track('exploratory_P4/e2_requests.jsonl');assert hashlib.sha256(raw).hexdigest()==prep['requests_sha256']
 requests={j['request_id']:j for j in map(json.loads,raw.decode().splitlines())}
 probabilities={}
 for model in ['7b','32b']:
  run=json.loads(track(f'exploratory_P4/e2_{model}/run.json'));assert run['requests_sha256']==prep['requests_sha256']
  checks=json.loads(track(f'exploratory_P4/e2_{model}/numerical_check.json'));assert all(v['passed'] for v in checks)
  scores=[json.loads(line) for line in track(f'exploratory_P4/e2_{model}/scores.jsonl').decode().splitlines()]
  assert len(scores)==3492 and set(r['request_id'] for r in scores)==set(requests)
  for row in scores:
   req=requests[row['request_id']];assert row['request_sha256']==req['request_sha256']
   for k in ['point_id','traj_id','turn','condition']:assert row[k]==req[k]
   act=arms[(arms.traj_id==row['traj_id']) & (arms.turn==row['turn'])].sort_values('j').action.tolist();assert act==req['actions']
  frame=pd.DataFrame(scores)
  for condition in ['S','L','F','Fx']:
   f=frame[frame.condition.eq(condition)].set_index(KEY).reindex(keys)
   p=np.stack(f.candidate_prob);assert p.shape==(873,5) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1)
   probabilities[model,condition]=p
 def stat(v,mask):
  mask=np.asarray(mask)&np.isfinite(v);n=int(mask.sum())
  if not n:return dict(n=0,estimate=None,ci95=None),np.full(10000,np.nan)
  sums=np.bincount(b.inv[mask],weights=v[mask],minlength=len(b.t));counts=np.bincount(b.inv[mask],minlength=len(b.t))
  with np.errstate(divide='ignore',invalid='ignore'):dist=(b.W@sums)/(b.W@counts)
  return dict(n=n,n_traj=int(d.loc[mask,'traj_id'].nunique()),estimate=float(v[mask].mean()),ci95=Boot.ci(dist)),dist
 gains={};table={};contrasts={};dists={}
 for source,Q in q.items():
  table[source]={};contrasts[source]={}
  for model in ['7b','32b']:
   for condition in ['S','L','F','Fx']:
    p=probabilities[model,condition]
    for mode,pred in [('soft',p),('argmax',np.eye(5)[p.argmax(1)])]:
     j=(pred*Q).sum(1)
     for metric,values in [('G_u',j-Q.mean(1)),('G_c',j-Q[:,0])]:
      gains[source,model,condition,mode,metric]=values
      for layer in ['I','U']:
       res,_=stat(values,d.het & d.stratum.eq(layer));table[source]['/'.join([model,condition,mode,metric,layer])]=res
  for layer in ['I','U']:
   mask=d.het & d.stratum.eq(layer)
   x=gains[source,'32b','S','soft','G_u']-gains[source,'7b','S','soft','G_u']
   res,dist=stat(x,mask);contrasts[source]['32b-7b_S_'+layer]=res;dists[source,layer]=dist
   for model in ['7b','32b']:
    for left,right in [('F','S'),('F','Fx'),('L','S')]:
     diff=gains[source,model,left,'soft','G_u']-gains[source,model,right,'soft','G_u']
     contrasts[source][model+'_'+left+'-'+right+'_'+layer]=stat(diff,mask)[0]
     if layer=='I' and left=='L':
      for future in [False,True]:contrasts[source][model+'_L-S_I_future'+str(int(future))]=stat(diff,mask & d.future_reveals.eq(future))[0]
  contrasts[source]['P4-5_interaction']=dict(estimate=contrasts[source]['32b-7b_S_U']['estimate']-contrasts[source]['32b-7b_S_I']['estimate'],ci95=Boot.ci(dists[source,'U']-dists[source,'I']))
 c=contrasts['12'];positive=lambda x:x['ci95'] is not None and x['ci95'][0]>0
 zero=lambda x:x['ci95'] is not None and x['ci95'][0]<=0<=x['ci95'][1]
 p5=positive(c['32b-7b_S_U']) and zero(c['32b-7b_S_I']) and positive(c['P4-5_interaction'])
 p6={m:positive(c[m+'_F-S_I']) and positive(c[m+'_F-Fx_I']) for m in ['7b','32b']}
 p7={m:positive(c[m+'_L-S_I_future1']) and zero(c[m+'_L-S_I_future0']) for m in ['7b','32b']}
 e1=json.loads(track('exploratory_P4/e1.json'));p=e1['predictions']
 # Positive-control failure explicitly makes negative E1 uninterpretable (§3 E1).
 if not p['positive_control']:verdict='R-MIXED'
 elif not any(p6.values()):verdict='R-NULL'
 elif zero(c['P4-5_interaction']) and positive(table['12']['32b/S/soft/G_u/I']):verdict='R-PRIOR'
 elif p['P4-1'] and p['P4-2'] and p5 and all(p6.values()):verdict='R-BLIND'
 elif p['P4-1'] and not p['P4-2'] and p5 and all(p6.values()):verdict='R-BLIND-7B'
 else:verdict='R-MIXED'
 track('exploratory_P4/p4_e2_analyze.py');track('scripts/evalcore.py')
 out=dict(table=table,contrasts=contrasts,predictions={'P4-5':p5,'P4-6':p6,'P4-7':p7},verdict=verdict,E3_allowed=verdict in ['R-BLIND','R-BLIND-7B'],inputs=inputs,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
 (OUT/'e2.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
