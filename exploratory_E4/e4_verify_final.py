"""Final independent completeness/numeric audit. Does not change E4 results or thresholds."""
from pathlib import Path
import hashlib,json,math
import numpy as np
import e4_common as E
H=Path(E.HERE);D=Path(E.DATA);R=H/'run'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 protected=read(R/'protected_inputs.json')
 for p,h in protected.items():assert sha(Path(E.ROOT)/p)==h,p
 games=read(E.GAMES_PATH);result=read(H/'e4.json');pairs=read(D/'pairs/manifest.json');lr=read(R/'learning_rate.json')['selected']
 assert read(D/'points_manifest.json')['total_refeed_div']==0
 assert read(D/'replay_manifest.json')['refeed_div']==0
 for c in ['F','L']:
  folder=D/'judge'/('32b_'+c);run=read(folder/'run.json');assert run['complete']
  assert all(v['passed'] and v['tv']<.005 for v in read(folder/'numerical_check.json'))
  rows=[json.loads(l) for l in (folder/'scores.jsonl').read_text().splitlines()]
  assert len(rows)==len({v['request_id'] for v in rows})==run['contexts']
  for row in rows:
   p=np.array(row['candidate_prob']);assert p.shape==(5,) and np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-10
 for arm in E.ARMS_TRAINED:
  file=D/'pairs'/f'{arm}.jsonl';assert sha(file)==pairs['sha256'][arm]
  assert len(file.read_text().splitlines())==pairs['counts'][arm]>0
  for seed in [0,1,2]:
   folder=D/'adapters'/f'{arm}_s{seed}';log=read(folder/'train_log.json');assert log['lr']==lr and log['epochs']==2 and not log['smoke']
   assert log['n_examples']==pairs['counts'][arm] and log['steps']==math.ceil(2*pairs['counts'][arm]/16)
   assert all(math.isfinite(v['loss']) for v in log['log']);assert (folder/'adapter_model.safetensors').is_file()
 assert pairs['counts']['Rnd']==pairs['counts']['A']==pairs['counts']['Esft']
 episodes=0;err=0.
 for split in ['valid_unseen','valid_seen']:
  arrays={}
  for arm in E.ARMS_ALL:
   columns=[]
   for tag in ['base'] if arm=='base' else [f'{arm}_s{s}' for s in [0,1,2]]:
    folder=D/'rollout'/split/tag;summary=read(folder/'summary.json');assert summary['n']==len(games[split])*3
    matrix=[]
    for g in games[split]:
     values=[]
     for rep in range(3):
      episode=read(folder/f"s{g['seed']}_r{rep}.json");episodes+=1
      assert episode['game_file']==g['game_file'] and episode['seed']==g['seed'] and episode['rep']==rep
      values.append(int(episode['won']))
     assert values==summary['per_game'][str(g['seed'])];matrix.append(values)
    columns.append(np.array(matrix,float))
   arrays[arm]=np.concatenate(columns,axis=1).mean(axis=1)
   assert abs(100*arrays[arm].mean()-result['splits'][split]['success'][arm])<1e-10
  for key,reported in result['splits'][split]['diff'].items():
   a,b=key.split('-');v=arrays[a]-arrays[b]
   # Same frozen resampling seed; independent point-level reconstruction from episodes.
   idx=np.random.default_rng(31102026).integers(0,len(v),size=(10000,len(v)))
   ci=np.percentile(v[idx].mean(1),[2.5,97.5])*100
   err=max(err,abs(100*v.mean()-reported['diff']),float(np.max(abs(ci-reported['ci95']))))
 assert err<1e-10,err
 c=result['splits']['valid_unseen']['diff'];lo=lambda k:c[k]['ci95'][0]
 predictions={'PL-1':lo('A-base')>0,'PL-2':lo('A-Bmatch')>-2 and lo('A-Cmatch')>0,'PL-3':lo('A-D')>0 and lo('A-Rnd')>0,'PL-4':lo('A-H')>0}
 mech=read(D/'mechanism.json');assert len(mech['points'])==386 and len(set(mech['points']))==386
 select=np.array(mech['layer'])=='I';assert select.sum()==184
 vals=np.mean([np.array(mech['models'][f'A_s{s}'])[select] for s in [0,1,2]],axis=0)-np.array(mech['models']['base'])[select]
 ids=np.array([p.split('_t')[0] for p in mech['points']])[select];groups=np.unique(ids)
 w=np.random.default_rng(30102026).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=10000)
 sums=np.array([vals[ids==g].sum() for g in groups]);counts=np.array([(ids==g).sum() for g in groups]);ci=np.percentile((w@sums)/(w@counts),[2.5,97.5])*100
 assert np.max(abs(ci-result['mechanism_I']['A_minus_base']['ci95']))<1e-10
 predictions['PL-5']=bool(ci[0]>0);assert predictions==result['predictions']
 reading='R-L-NULL' if not predictions['PL-1'] else 'R-L-HEURISTIC' if not predictions['PL-3'] else 'R-L-COSTLY' if not predictions['PL-2'] else 'R-L-WIN'
 assert reading==result['reading']
 report=dict(passed=True,episodes_reconstructed=episodes,max_numeric_error=err,protected_files_unchanged=len(protected),reading=reading,predictions=predictions,mechanism_limit='Mechanism source retains per-point gains, not raw generated drafts; audit checks aggregation, not regenerating draft counts.')
 E.write_json(str(R/'final_verification.json'),report)
 manifest={str(p.relative_to(H)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in D.rglob('*') if p.is_file()}
 E.write_json(str(R/'final_artifact_manifest.json'),manifest);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
