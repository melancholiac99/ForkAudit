"""Independent checks against sklearn and pre-existing deep outcome extraction."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,sys,hashlib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from evalcore import Boot
KEY=['traj_id','turn']
def main():
 r=json.loads((OUT/'e1.json').read_text());s=pd.read_parquet(OUT/'strata.parquet');lab=pd.read_parquet(ROOT/'data/labels_final.parquet');d=lab[lab.live].merge(s[KEY+['stratum']],on=KEY).merge(pd.read_parquet(ROOT/'eval/scores_B2.parquet'),on=KEY)
 b=Boot(d.traj_id,B=40,seed=30102026);auc_error=0.;boot_error=0.
 for layer in ['I','U','N']:
  mask=d.stratum.eq(layer).to_numpy()
  for name,res in r['state'][layer].items():
   y=d.loc[mask,'het'].to_numpy();x=d.loc[mask,name].to_numpy();ok=np.isfinite(x)
   auc_error=max(auc_error,abs(roc_auc_score(y[ok],x[ok])-res['auroc']))
   dist=b.auroc_dist(d.het,np.where(mask,d[name],np.nan))
   for j in range(40):
    w=b.W[j,b.inv[mask]][ok];v=roc_auc_score(y[ok],x[ok],sample_weight=w)
    boot_error=max(boot_error,abs(v-dist[j]))
 assert auc_error<1e-12 and boot_error<1e-6
 a=pd.read_parquet(ROOT/'exploratory_deep_discovery/exploratory_arms.parquet');a=a.merge(d[KEY+['stratum','het']],on=KEY,validate='many_to_one')
 score=pd.read_parquet(ROOT/'data/scores_fp32.parquet')[KEY+['action','logp_sum']];a=a.merge(score,on=KEY+['action'],validate='one_to_one')
 records=[]
 for key,g in a.groupby(KEY):
  g=g.sort_values('j');p=softmax(g.logp_sum.to_numpy());h=g.target_mention.to_numpy(float);w=h/h.sum() if h.sum() else np.array([1.,0,0,0,0])
  row=dict(stratum=g.stratum.iloc[0],het=bool(g.het.iloc[0]))
  for src in ['12','gold']:
   q=g['q_'+src].to_numpy()
   for name,prob in [('policy',p),('world',w)]:
    row[src+name+'G_u']=prob@q-q.mean();row[src+name+'G_c']=prob@q-q[0]
  records.append(row)
 x=pd.DataFrame(records);error=0.
 for layer in ['I','U','N']:
  for src in ['12','gold']:
   for name in ['policy','world']:
    for metric in ['G_u','G_c']:
     col=x.loc[x.het & x.stratum.eq(layer),src+name+metric].dropna();expected=r['action'][layer][src][name][metric]
     assert len(col)==expected['n'];error=max(error,abs(col.mean()-expected['estimate']))
 assert error<1e-12
 # Verify all E1 input hashes after execution, including every replay file.
 for path,digest in r['inputs'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
 prep=json.loads((OUT/'e2_preparation.json').read_text());raw=(OUT/'e2_requests.jsonl').read_bytes();assert hashlib.sha256(raw).hexdigest()==prep['requests_sha256']
 jobs=[json.loads(t) for t in raw.decode().splitlines()];assert len(jobs)==3492 and len({j['request_id'] for j in jobs})==3492
 for j in jobs:
  h=j.pop('request_sha256');assert hashlib.sha256(json.dumps(j,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==h
  assert len(j['actions'])==len(set(j['actions']))==5
 for _,group in pd.DataFrame(jobs).groupby('point_id'):
  assert set(group.condition)=={'S','L','F','Fx'}
  assert len({tuple(a) for a in group.actions})==1
 out=dict(passed=True,sklearn_auroc_max_error=auc_error,weighted_bootstrap_max_error=boot_error,independent_deep_action_max_error=error,
          input_hashes_checked=len(r['inputs']),e2_contexts_checked=len(jobs),gpu_numerical_check='NOT RUN: no GPU')
 (OUT/'cpu_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
