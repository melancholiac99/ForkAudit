"""Keep original gate thresholds. P0 is an explicit train-only cold-start pool."""
import argparse,json,sys
from pathlib import Path
from util import *
from habits import load_episodes
from habits_v2 import make_batch,oof_audit
from prepare import validate_train

def prepare(folder,out,base='grpo',mode='grpo',cold_start=False,history=()):
 summary=load(folder/'summary.json');iteration=summary['iteration']
 current=load_episodes(folder,iteration);validate_train(current,iteration)
 pool=[];pool_sources={}
 if cold_start:
  for i,d in enumerate(sorted(Path('/root/autodl-tmp/ctha_v2/p0_base').glob('round_*'))):
   episodes=load_episodes(d/'rollouts',-1000+i)
   assert all(e['provenance']['split']=='train' and e['provenance']['adapter_sha256'] is None for e in episodes)
   pool.extend(episodes)
   for p in (d/'rollouts').glob('s*_r*.json'):pool_sources[str(p)]=sha(p)
 for path in history:
  s=load(path/'summary.json');past=load_episodes(path,s['iteration'])
  validate_train(past,s['iteration']);pool.extend(past)
  for p in path.glob('s*_r*.json'):pool_sources[str(p)]=sha(p)
 opts=dict(family='habit',feature_window=8,train_horizon=0,aux_horizon=8,
  alpha=5.,min_mixed=20,min_pairs=200,permutations=12,bootstraps=300,
  min_delta=0.,sign_fraction=.75,omega=.3,clip=2.,baseline='legal',gamma=1.,local_weight=1.)
 rows,report=make_batch(current,pool,mode=mode,base=base,**opts)
 index={(e['episode_id'],s['turn']):s for e in current for s in e['steps']}
 for row in rows:
  step=index[row['episode_id'],row['turn']]
  for k in ['behavior_logprobs','prompt_token_ids','finish_reason']:
   if k not in step:raise ValueError('Cannot train v3 from old samples without behavior logprobs')
   row[k]=step[k]
 report.update(schema='ctha-v3',train_only=True,source_hashes={p.name:sha(p) for p in folder.glob('s*_r*.json')},
  cold_start=cold_start,prior_sources=pool_sources,current_policy_gate=oof_audit(current,family='habit',min_mixed=20,min_pairs=200,permutations=12,bootstraps=300),code_sha=source_hashes())
 out.mkdir(parents=True,exist_ok=False)
 data=''.join(json.dumps(row)+'\n' for row in rows);(out/'samples.jsonl').write_text(data)
 report['samples_sha256']=sha(out/'samples.jsonl');save(out/'advantage_report.json',report)
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--rollouts',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--base',choices=['grpo','gigpo'],default='grpo');p.add_argument('--mode',default='grpo');p.add_argument('--cold-start',action='store_true');a=p.parse_args()
 prepare(a.rollouts,a.out,a.base,a.mode,a.cold_start)
