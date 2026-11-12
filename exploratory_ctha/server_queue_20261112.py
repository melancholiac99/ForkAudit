"""P0 -> P0 audit -> GPU integration smoke, with persistent status and recovery wakeup."""
import argparse,fcntl,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent;D=H/'server_20261112';DATA=Path('/root/autodl-tmp/ctha_v2');MODEL='/root/autodl-tmp/models/Qwen2.5-7B-Instruct'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(j,indent=2)+'\n');t.replace(p)
def state(stage,**kw):save(D/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw));print(stage,kw,flush=True)
def live(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
 except FileNotFoundError:return False
def notify(message):
 for attempt in range(2):
  try:
   with (D/'wake.log').open('a') as f:subprocess.run(['codex','queue','--thread',os.environ['CODEX_THREAD_ID'],'--resume','--message',message],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
   return
  except Exception:
   (D/'wake_error.log').write_text(traceback.format_exc())
def preflight():
 r=json.loads((D/'registration.json').read_text())
 for path,digest in r['source'].items():assert sha(ROOT/path)==digest,path
 inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip();assert inventory==r['gpu_inventory']
 assert json.loads((D/'doctor.json').read_text())['ready']
 assert json.loads((D/'native_check.json').read_text())['passed']
 return r
def job(key,args):
 preflight()
 with (D/(key+'.log')).open('a') as log:p=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
 state('running',job=key,child=p.pid);save(D/(key+'_launch.json'),dict(pid=p.pid,args=list(map(str,args)),time=time.time()))
 rc=p.wait();assert rc==0,f'{key} exited {rc}'
def failed(error):
 (D/'failure.log').write_text(error);state('failed',error=error);os.sync()
 notify(f'【CTHA v2队列异常】读取 {D}/status.json、failure.log、当前任务日志。先写 recovery_claim.json 后尽力修复；保留旧E4/E4e及新CTHA已有记录，不放宽门槛、不静默改配置，修改需新目录。用户要求开始最新CTHA v2；P0后GPU smoke，再核验后推进P1/P2。当前无关机授权。恢复入口 {H}/server_queue_20261112.py；检查健康子进程，不重复启动。')
def main():
 with (D/'queue.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   r=preflight();state('preflight')
   # Frozen base files verified against the existing E4e manifest; no GPU library changes.
   for path,digest in r['model_files'].items():
    h=hashlib.sha256()
    with open(path,'rb') as f:
     for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    assert h.hexdigest()==digest,path
   job('p0_collect',[H/'run.py','--collect-only','--run-dir',DATA/'p0_base','--model',MODEL,'--rounds','6','--tasks-per-round','32','--reps','4','--eval-every','0','--workers','16','--chunk-games','16'])
   files=[];fingerprints=[]
   for i in range(6):
    folder=DATA/'p0_base'/f'round_{i:03d}'/'rollouts';s=json.loads((folder/'summary.json').read_text())
    assert s['n_episodes']==128 and s['n_games']==32 and s['split']=='train'
    assert s['provenance']['adapter_sha256'] is None and not s['provenance']['dry']
    fs=sorted(folder.glob('s*_r*.json'));assert len(fs)==128;files+=fs
    for p in fs:
     ep=json.loads(p.read_text());assert ep['provenance']==s['provenance']
     assert ep['n_turns']==len(ep['steps'])<=50 and all(x['response_token_ids'] for x in ep['steps'])
    fingerprints.append(s['provenance']['fingerprint'])
   save(D/'p0_collection_verification.json',dict(passed=True,episodes=len(files),rounds=6,fingerprints=fingerprints,raw_sha256={str(p):sha(p) for p in files}))
   job('p0_audit',[H/'p0_audit.py','--run-dir',DATA/'p0_base','--out',DATA/'p0_report.json','--feature-window','8','--min-pairs','200','--min-mixed','20'])
   report=json.loads((DATA/'p0_report.json').read_text());assert report['n_episodes']==768 and report['source']=='train-only'
   save(D/'p0_report.json',report)
   job('smoke',[H/'run.py','--run-dir',DATA/'smoke','--model',MODEL,'--mode','ctha','--base-advantage','gigpo','--rounds','1','--tasks-per-round','4','--reps','2','--feature-window','2','--train-horizon','2','--aux-horizon','2','--min-mixed','2','--min-pairs','2','--permutations','2','--bootstraps','20','--workers','8','--chunk-games','4','--eval-every','1','--eval-games','6'])
   assert (DATA/'smoke/round_000/adapter/train_log.json').is_file()
   assert (DATA/'smoke/round_000/eval_tune/summary.json').is_file()
   state('awaiting_independent_smoke_review',p0_report=str(DATA/'p0_report.json'),smoke=str(DATA/'smoke'));os.sync()
   notify(f'【CTHA v2 P0与GPU试跑完成】读取 {DATA}/p0_report.json、{DATA}/smoke/round_000/prepared/advantage_report.json、adapter/train_log.json及eval_tune/summary.json，独立核验后按README_V2.md推进P1五个对照及P2消融。P0固定768回合train-only，未改门槛；不要把零辅助奖励smoke当失败或科研结论。若试跑没有实际非零更新需补工程核验。保留旧结果，当前无关机授权。状态 {D}/status.json。')
  except Exception:failed(traceback.format_exc())
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--watch',type=int);a=ap.parse_args()
 if a.watch:
  while live(a.watch):time.sleep(20)
  s=json.loads((D/'status.json').read_text())
  if s['stage'] not in ('failed','awaiting_independent_smoke_review'):failed('Queue exited unexpectedly: '+json.dumps(s))
 else:main()
