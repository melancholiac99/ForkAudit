"""Bounded, fail-closed v3 validation. No shutdown or full P1 expansion."""
import fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from util import *
from run import select_games
PYTHON='/root/miniconda3/envs/m0/bin/python'

def state(stage,**kw):save(DATA/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw));print(stage,kw,flush=True)
def notify(msg):
 try:
  with (DATA/'wake.log').open('a') as log:subprocess.run(['codex','queue','--thread',os.environ['CODEX_THREAD_ID'],'--resume','--message',msg],stdout=log,stderr=subprocess.STDOUT,timeout=60,check=True)
 except Exception:(DATA/'wake_error.log').write_text(traceback.format_exc())
def job(name,args):
 manifest=load(DATA/'registration.json')
 assert source_hashes()==manifest['source_sha'],'Source changed during queue; use a new registration/output namespace'
 state('running',job=name)
 with (DATA/(name+'.log')).open('a') as log:
  p=subprocess.Popen([PYTHON,'-u',*map(str,args)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
  save(DATA/(name+'_launch.json'),dict(pid=p.pid,args=list(map(str,args)),time=time.time()))
  state('running',job=name,child=p.pid)
  rc=p.wait()
 if rc:raise RuntimeError(f'{name} exited {rc}; see {DATA/name}.log')
def collect(name,games,out,iteration,split='train',reps=4,adapter=None,namespace='train'):
 args=[HERE/'collect.py','--model',MODEL,'--games',games,'--split',split,'--out',out,'--iteration',iteration,'--reps',reps,'--workers',16,'--chunk-games',16,'--gpu-util',.70,'--seed-namespace',namespace]
 if adapter:args+=['--adapter',adapter]
 job(name,args)
def prepare(name,raw,out,base):job(name,[HERE/'prepare_v3.py','--rollouts',raw,'--out',out,'--base',base])
def train(name,raw,prepared,out,seed,adapter=None,probe=False):
 args=[HERE/'trainer.py','--data',prepared/'samples.jsonl','--summary',raw/'summary.json','--out',out,'--seed',seed]
 if adapter:args+=['--adapter',adapter]
 if probe:args+=['--probe']
 job(name,args)
def compare_repeat(a,b):
 rows=[]
 for p in sorted(a.glob('s*_r*.json')):
  x=load(p);y=load(b/p.name)
  fields=['response_token_ids','prompt_token_ids','sample_seed','action','obs_before','obs_after','won_after']
  exact=len(x['steps'])==len(y['steps']) and all(all(s[k]==t[k] for k in fields) for s,t in zip(x['steps'],y['steps'])) and x['won']==y['won']
  rows.append({'file':p.name,'exact':exact})
 result={'passed':all(x['exact'] for x in rows),'episodes':len(rows),'samples':rows};save(DATA/'repeat_verification.json',result)
 if not result['passed']:raise RuntimeError('Frozen-policy repeat failed; quantify and repair before scaling')
def paired(a,b):
 games={}
 for p in a.glob('s*_r*.json'):
  x=load(p);y=load(b/p.name);assert x['game_file']==y['game_file'] and x['rep']==y['rep']
  assert x['steps'][0]['sample_seed']==y['steps'][0]['sample_seed']
  games.setdefault(x['game_file'],[]).append(int(y['won'])-int(x['won']))
 vals=np.array([np.mean(v) for _,v in sorted(games.items())]);rng=np.random.default_rng(20261113)
 boot=vals[rng.integers(0,len(vals),size=(10000,len(vals)))].mean(axis=1)
 return {'paired_change':float(vals.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'games':len(vals)}
def main():
 DATA.mkdir(exist_ok=True)
 with (DATA/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   cpu=load(DATA/'cpu_validation.json');assert cpu['cold_start_gate']['passed']
   assert cpu['source_sha']==source_hashes(),'CPU checks must cover current frozen sources'
   if (DATA/'registration.json').exists():raise RuntimeError('Queue already registered; no duplicate restart')
   immutable(DATA/'registration.json',dict(source_sha=source_hashes(),plan=load(HERE/'PLAN.json'),time=time.time(),environment={k:os.getenv(k) for k in ['CUDA_VISIBLE_DEVICES','VLLM_BATCH_INVARIANT','CUBLAS_WORKSPACE_CONFIG','OMP_NUM_THREADS']},gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True)))
   g=load(ROOT/'exploratory_E4/games.json');selected=select_games(g['train'],0,0,16)
   save(DATA/'train_games.json',selected);save(DATA/'repeat_games.json',selected[:4]);save(DATA/'tune_games.json',g['tune'])
   collect('repeat_a',DATA/'repeat_games.json',DATA/'repeat_a',0,reps=2)
   collect('repeat_b',DATA/'repeat_games.json',DATA/'repeat_b',0,reps=2)
   compare_repeat(DATA/'repeat_a',DATA/'repeat_b')
   collect('initial_train',DATA/'train_games.json',DATA/'initial_train',0)
   prepare('initial_prepare',DATA/'initial_train',DATA/'initial_prepared','grpo')
   train('direction_probe',DATA/'initial_train',DATA/'initial_prepared',DATA/'direction_probe',0,probe=True)
   collect('baseline_tune',DATA/'tune_games.json',DATA/'baseline_tune',0,split='tune',reps=2,namespace='eval')
   results={}
   for arm in ['grpo','gigpo']:
    root=DATA/arm;root.mkdir();prior=None
    for i in range(4):
     d=root/f'round_{i:03d}';d.mkdir()
     if i==0:shutil.copytree(DATA/'initial_train',d/'rollouts')
     else:collect(f'{arm}_{i}_collect',DATA/'train_games.json',d/'rollouts',i,adapter=prior)
     prepare(f'{arm}_{i}_prepare',d/'rollouts',d/'prepared',arm)
     train(f'{arm}_{i}_train',d/'rollouts',d/'prepared',d/'adapter',i,adapter=prior)
     prior=d/'adapter'
     if (i+1)%2==0:
      collect(f'{arm}_{i}_tune',DATA/'tune_games.json',d/'eval_tune',0,split='tune',reps=2,adapter=prior,namespace='eval')
      collect(f'{arm}_{i}_train_eval',DATA/'train_games.json',d/'eval_train',0,adapter=prior)
    results[arm]={'train':paired(DATA/'initial_train',d/'eval_train'),'tune':paired(DATA/'baseline_tune',d/'eval_tune'),'final_train':load(d/'eval_train/summary.json')['success_rate'],'final_tune':load(d/'eval_tune/summary.json')['success_rate']}
   save(DATA/'analysis.json',{'baseline_train':load(DATA/'initial_train/summary.json')['success_rate'],'baseline_tune':load(DATA/'baseline_tune/summary.json')['success_rate'],'results':results,'interpretation':'Engineering pilot on repeated training tasks. Tune gains require positive paired lower confidence bound; no formal CTHA efficacy claim.'})
   state('completed',results=results)
   notify(f'【CTHA v3验收结束】读取 {DATA}/analysis.json、status.json、direction_probe/direction_check.json 和各轮train_log.json。工程正确性与训练任务学习、tune收益分别解释。未重跑完整P1/P2，未关机。')
  except Exception:
   error=traceback.format_exc();(DATA/'failure.log').write_text(error);state('failed',error=error);os.sync()
   notify(f'【CTHA v3验收异常】读取 {DATA}/status.json、failure.log及对应日志，尽力修复。旧P1不改，门槛不放宽；先查健康进程，修订需新注册及目录，禁止混配置。当前无自动关机。')
   raise
if __name__=='__main__':main()
