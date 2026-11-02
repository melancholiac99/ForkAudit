"""Sequential execution of the registered E4 README after the active rollout.
Stops on any failed gate. No retries, rule changes, publication or shutdown.
"""
from pathlib import Path
import fcntl,json,os,shutil,subprocess,sys,time,traceback
import e4_common as E
HERE=Path(E.HERE);ROOT=Path(E.ROOT);RUN=HERE/'run';DATA=Path(E.DATA)
def write(stage,**kw):
 E.write_json(str(RUN/'pipeline_status.json'),dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
 print(stage,kw,flush=True)
def identity(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return None if fields[0]=='Z' else fields[19]
 except FileNotFoundError:return None

def call(stage,script,*args):
 write(stage)
 assert shutil.disk_usage(HERE).free>3*1024**3,'Less than 3 GiB free: stop before next stage'
 cmd=[sys.executable,'-u',script,*map(str,args)]
 print(json.dumps(dict(command=cmd)),flush=True)
 with (RUN/(stage+'.log')).open('a') as f:subprocess.run(cmd,cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
def read(p):return json.loads(Path(p).read_text())
def validate_rollout(split,tag,reps):
 games=read(E.GAMES_PATH)[split];folder=DATA/'rollout'/split/tag;summary=read(folder/'summary.json')
 assert summary['n']==len(games)*reps
 for g in games:
  for r in range(reps):
   ep=read(folder/f"s{g['seed']}_r{r}.json")
   assert ep['seed']==g['seed'] and ep['rep']==r and ep['game_file']==g['game_file']
   assert 1<=len(ep['steps'])<=50

def main():
 with (RUN/'pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   pid=int(sys.argv[1]);ident=identity(pid);assert ident
   write('waiting_for_training_rollout',pid=pid)
   while identity(pid)==ident:time.sleep(15)
   validate_rollout('train','base',1)
   call('points','e4_02_points.py')
   points=read(DATA/'points_manifest.json');assert points['n_trajectories']==400 and points['total_refeed_div']==0 and points['n_cf_defined']>0
   call('judge_prepare','e4_03_judge.py','prepare')
   for c in ['F','L']:
    call('judge_'+c,'e4_03_judge.py','score','--cond',c,'--model','/root/autodl-tmp/models/Qwen2.5-32B-Instruct','--name','32b','--revision','5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd')
    d=DATA/'judge'/('32b_'+c);assert read(d/'run.json')['complete']
    assert all(x['passed'] and x['tv']<.005 for x in read(d/'numerical_check.json'))
   call('replay','e4_04_replay.py')
   replay=read(DATA/'replay_manifest.json');assert replay['refeed_div']==0 and replay['n_points']==points['n_cf_defined']
   call('pairs','e4_05_pairs.py')
   pairs=read(DATA/'pairs/manifest.json');assert all(pairs['counts'][a]>0 for a in E.ARMS_TRAINED)
   assert pairs['counts']['Esft']==pairs['counts']['A'],'Preregistered Esft/A sample-count equality failed'
   call('smoke_train','e4_06_train.py','--arm','Aprime','--seed',0,'--lr','2e-5','--smoke')
   call('smoke_eval','e4_07_eval.py','--split','tune','--adapters','Aprime_s0_smoke','--no_base','--reps',1)
   validate_rollout('tune','Aprime_s0_smoke',1)
   # Smoke outcomes never influence model/learning-rate choice; retain only smoke logs.
   for p in [DATA/'adapters/Aprime_s0_smoke',DATA/'rollout/tune/Aprime_s0_smoke']:shutil.rmtree(p)
   tags=[]
   for lr in E.LR_GRID:
    tag=f'Aprime_s0_lr{lr}';tags.append(tag)
    call('lr_train_'+str(lr),'e4_06_train.py','--arm','Aprime','--seed',0,'--lr',lr,'--out',DATA/'adapters'/tag)
   call('lr_eval','e4_07_eval.py','--split','tune','--no_base','--adapters',*tags)
   scores={}
   for lr,tag in zip(E.LR_GRID,tags):
    validate_rollout('tune',tag,3);scores[lr]=read(DATA/'rollout/tune'/tag/'summary.json')['success']
   lr=E.select_lr(scores)
   E.write_json(str(RUN/'learning_rate.json'),dict(grid_scores=scores,selected=lr,rule='e4_common.select_lr: highest tune success, ties smaller LR'))
   with (ROOT/'DECISIONS.md').open('a') as f:
    f.write('\nD-061 E4 automatic registered learning-rate selection (tune only): '+json.dumps(dict(scores=scores,selected=lr))+'. All arms use this LR; no validation outcomes used.\n')
   for arm in E.ARMS_TRAINED:
    for seed in [0,1,2]:
     call(f'train_{arm}_{seed}','e4_06_train.py','--arm',arm,'--seed',seed,'--lr',lr)
     log=read(DATA/f'adapters/{arm}_s{seed}/train_log.json');assert log['n_examples']==pairs['counts'][arm] and log['epochs']==2
   for split in ['valid_unseen','valid_seen']:
    call('eval_'+split,'e4_07_eval.py','--split',split,'--all')
    for tag in ['base']+[f'{a}_s{s}' for a in E.ARMS_TRAINED for s in [0,1,2]]:validate_rollout(split,tag,3)
   call('mechanism','e4_08_mechanism.py','--models','base','A_s0','A_s1','A_s2','Rnd_s0','Rnd_s1','Rnd_s2')
   call('analysis','e4_09_analyze.py')
   result=read(HERE/'e4.json');write('completed_pending_review',reading=result['reading'],predictions=result['predictions'])
  except Exception as exc:
   write('failed',error_type=type(exc).__name__,message=str(exc));traceback.print_exc();raise
if __name__=='__main__':main()
