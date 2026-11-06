"""Two independent GPU workers; user authorized parallel experiments after replacing GPUs."""
from pathlib import Path
import argparse,fcntl,os,subprocess,sys,time,traceback
RUN=Path(__file__).resolve().parent;H=RUN.parent;ROOT=H.parent
sys.path.insert(0,str(H));import exp2_common as C;import exp5_token_slot as S

def state(worker,stage,**kw):
 with (RUN/'status.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  C.save_json(RUN/(worker+'_status.json'),dict(worker=worker,stage=stage,pid=os.getpid(),time=time.time(),**kw))
  states={w:C.read_json(RUN/(w+'_status.json')) if (RUN/(w+'_status.json')).exists() else dict(stage='not_started') for w in ('token','math')}
  C.save_json(RUN/'status.json',dict(stage='parallel_workers',time=time.time(),workers=states,no_shutdown=True))
 print(worker,stage,kw,flush=True)
def wake(message):
 if not os.environ.get('CODEX_THREAD_ID'):return
 with (RUN/'wake.log').open('a') as f:
  try:subprocess.run(['codex','queue','--thread',os.environ['CODEX_THREAD_ID'],'--resume','--message',message],stdout=f,stderr=subprocess.STDOUT,timeout=45)
  except Exception:traceback.print_exc(file=f)
def execute(worker,key,args):
 assert not (RUN/'USER_PAUSED.json').exists(),'User paused the queue'
 with (RUN/(key+'.log')).open('a') as log:
  p=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
  state(worker,'running',task=key,child_pid=p.pid)
  while p.poll() is None:time.sleep(15)
  assert p.returncode==0,f'{key} failed, exit {p.returncode}'
def worker(name):
 with (RUN/(name+'_driver.lock')).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=C.read_json(RUN/'execution.json');assert plan['mode']=='parallel' and plan['user_parallel_authorized']
  assert os.environ.get('CUDA_VISIBLE_DEVICES')==plan['gpu_assignments'][name]
  try:
   if name=='token':
    for limit in (32,0):
     for model in ('7b','32b'):
      for command in ('score','analyze','verify'):
       assert plan['source']==S.fingerprint(),'Experiment 1 code changed during scoring'
       execute(name,f'token_{model}_{limit}_{command}',[H/'exp5_token_slot.py',command,'--name',model,'--limit',limit,'--model',f'/root/autodl-tmp/models/Qwen2.5-{model.upper()}-Instruct'])
      if limit:
       a=C.read_json(S.OUT/model/'first32/analysis.json')
       C.save_json(RUN/f'token_{model}_pilot_timing.json',dict(points=32,seconds=a['seconds'],seconds_per_point=a['seconds']/32,extrapolated_full_hours=a['seconds']/32*386/3600,model_calls=a['model_calls']))
     if limit:wake('【逐 token 槽位试跑通过】P4 first32两模型已核验，按授权继续386点。请读 '+str(RUN)+'/token_*_pilot_timing.json 更新时长。数学实验在另一卡独立运行，禁止改统计门槛或混配置。')
    for model in ('7b','32b'):assert C.read_json(S.OUT/model/'full/verification.json')['passed']
   else:
    ready=C.read_json(RUN/'math_ready.json')
    assert ready['cpu_verified'] and all(C.sha_file(Path(p))==sha for p,sha in ready['source'].items())
    execute(name,'math_experiment',[H/'exp6_math_slot.py','run'])
   state(name,'completed_verified')
   wake('【'+name+' 实验完成】请读 '+str(RUN)+'/'+name+'_status.json，核验、汇报并发布 research/hindsight-judges。两实验已获双卡并行授权；另一实验可继续，不关机。')
  except Exception:
   error=traceback.format_exc();(RUN/(name+'_failure.log')).write_text(error);state(name,'failed',error=error)
   wake('【双卡槽位实验异常 '+name+'】请读 '+str(RUN)+'/'+name+'_status.json、'+name+'_failure.log 和对应日志，尽力修复失败的工作队列。用户已改为两卡并行；不要停止另一正常实验，不混分数/配置，不放宽TV或统计门槛，无关机授权。');raise

def launch():
 assert not (RUN/'USER_PAUSED.json').exists(),'User paused the queue'
 plan=C.read_json(RUN/'execution.json');assert plan['mode']=='parallel'
 assert len(set(plan['gpu_assignments'].values()))==2
 children={}
 for name,gpu in plan['gpu_assignments'].items():
  lock=(RUN/(name+'_driver.lock')).open('a')
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
  env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
  with (RUN/(name+'_driver.log')).open('a') as f:
   p=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',name],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  children[name]=dict(pid=p.pid,gpu=gpu)
 C.save_json(RUN/'parallel_launch.json',dict(time=time.time(),children=children,thread=os.environ.get('CODEX_THREAD_ID')));print(children)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--worker',choices=['token','math']);a=p.parse_args()
 worker(a.worker) if a.worker else launch()
