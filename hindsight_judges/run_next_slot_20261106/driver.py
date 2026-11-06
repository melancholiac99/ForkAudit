"""Single-card queue: finish and verify P4 token-slot experiment before MATH GPU work."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time,traceback
RUN=Path(__file__).resolve().parent;H=RUN.parent;ROOT=H.parent
sys.path.insert(0,str(H));import exp2_common as C;import exp5_token_slot as S
GPU='GPU-2d103f24-c0fc-18b0-c2f3-9edd2710692a'
ENV=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
def state(stage,**kw):
 C.save_json(RUN/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw));print(stage,kw,flush=True)
def wake(message):
 if not os.environ.get('CODEX_THREAD_ID'):return
 with (RUN/'wake.log').open('a') as f:
  try:subprocess.run(['codex','queue','--thread',os.environ['CODEX_THREAD_ID'],'--resume','--message',message],stdout=f,stderr=subprocess.STDOUT,timeout=45)
  except Exception:traceback.print_exc(file=f)
def execute(key,args):
 with (RUN/(key+'.log')).open('a') as log:
  p=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT)
  state('running',task=key,child_pid=p.pid)
  while p.poll() is None:time.sleep(15)
  assert p.returncode==0,f'{key} failed, exit {p.returncode}'
def main():
 with (RUN/'driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  plan=C.read_json(RUN/'execution.json');assert plan['source']==S.fingerprint()
  try:
   for limit in (32,0):
    for name in ('7b','32b'):
     for command in ('score','analyze','verify'):
      assert plan['source']==S.fingerprint(),'Experiment 1 code changed during scoring'
      execute(f'token_{name}_{limit}_{command}',[H/'exp5_token_slot.py',command,'--name',name,'--limit',limit,'--model',f'/root/autodl-tmp/models/Qwen2.5-{name.upper()}-Instruct'])
     if limit:
      a=C.read_json(S.OUT/name/'first32/analysis.json')
      C.save_json(RUN/f'token_{name}_pilot_timing.json',dict(points=32,seconds=a['seconds'],seconds_per_point=a['seconds']/32,
                  extrapolated_full_hours=a['seconds']/32*386/3600,model_calls=a['model_calls']))
    if limit:wake('【逐 token 槽位试跑通过】P4 first32 两模型已核验，按授权继续386点。请读 '+str(RUN)+'/token_*_pilot_timing.json 更新时长；主T2固定，禁止改门槛/混配置。实验2必须等待实验1完成。')
   for name in ('7b','32b'):assert C.read_json(S.OUT/name/'full/verification.json')['passed']
   state('experiment1_completed_verified')
   wake('【实验1完成】P4逐token槽位诊断两模型全量已核验。请读 '+str(S.OUT)+' 下 analysis.json/verification.json，汇报T2相对T1和pre-k可用性；实验2准备好后按单卡顺序继续。此新队列无关机授权。')
   while not (RUN/'math_ready.json').exists():time.sleep(30)
   math=C.read_json(RUN/'math_ready.json')
   assert math['cpu_verified'] and all(C.sha_file(Path(p))==sha for p,sha in math['source'].items())
   execute('math_experiment',[H/'exp6_math_slot.py','run'])
   state('both_experiments_completed')
   wake('【两实验完成】请核验实验2数学结果和实验1结果，发布到 research/hindsight-judges 并汇报。无训练、无新ALFWorld轨迹、无自动关机。')
  except Exception:
   error=traceback.format_exc();(RUN/'failure.log').write_text(error);state('failed',error=error)
   wake('【新槽位实验异常】请读 '+str(RUN)+'/status.json、failure.log和对应日志，尽力修复续跑。先实验1后实验2；保护旧结果与协议，不放宽TV或统计门槛，不静默混配置。无关机授权。');raise
if __name__=='__main__':main()
