"""Authorized E2M queue: existing 7B -> 32B -> analysis -> audit/report -> GitHub.
No training, new trajectories, automatic conflict resolution, force push or shutdown.
"""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exploratory_P4';AUDIT=OUT/'e2m_run'
def state(stage,**extra):
 p=AUDIT/'status.json';tmp=p.with_suffix('.tmp');data=dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
 tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p);print(json.dumps(data),flush=True)
def run(script,args,log):
 command=[sys.executable,'-u',str(OUT/script),*args];print(json.dumps(dict(command=command)),flush=True)
 with (AUDIT/log).open('a') as f:subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
def identity(pid):
 try:
  fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();return None if fields[0]=='Z' else fields[19]
 except FileNotFoundError:return None

def main():
 with (AUDIT/'driver.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   pid=int(sys.argv[1]);start=identity(pid)
   assert start and b'p4_e2m_score.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
   state('waiting_7b',pid=pid)
   while identity(pid)==start:time.sleep(10)
   first=json.loads((OUT/'e2m_7b/run.json').read_text());assert first['complete'] and first['contexts']==1746
   state('scoring_32b')
   run('p4_e2m_score.py',['--model','/root/autodl-tmp/models/Qwen2.5-32B-Instruct','--name','32b','--revision','5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'],'32b.log')
   state('analysis');run('p4_e2m_analyze.py',[],'analysis.log')
   state('verification_and_report');run('verify_render_e2m.py',[],'verification.log')
   result=json.loads((OUT/'e2m.json').read_text());assert json.loads((AUDIT/'verification.json').read_text())['passed']
   files=['exploratory_P4/e2m.json','exploratory_P4/E2M_REPORT.txt']
   files += [f'exploratory_P4/e2m_{m}/{f}' for m in ['7b','32b'] for f in ['run.json','scores.jsonl','numerical_check.json']]
   files += [f'exploratory_P4/e2m_run/{f}' for f in ['preflight.json','post_e2_server.json','protected_inputs.json','verification.json','7b.log','32b.log','analysis.log','verification.log']]
   # Snapshot the driver log before publication so the committed log is immutable.
   (AUDIT/'execution.log').write_bytes((AUDIT/'driver.log').read_bytes());files.append('exploratory_P4/e2m_run/execution.log')
   manifest={}
   for f in files:
    p=ROOT/f;assert p.is_file() and p.stat().st_size<95*1024**2
    manifest[f]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
   (AUDIT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');files.append('exploratory_P4/e2m_run/artifact_manifest.json')
   assert git('branch','--show-current')=='main'
   assert git('remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
   assert not git('diff','--cached','--name-only'),'Unrelated staged files'
   subprocess.run(['git','add','--',*files],cwd=ROOT,check=True)
   subprocess.run(['git','commit','-m','Complete preregistered E2M and independently verify post-E2 results'],cwd=ROOT,check=True)
   head=git('rev-parse','HEAD');state('publishing',commit=head,reading=result['reading'],predictions=result['predictions'])
   net=['git','-c','http.version=HTTP/1.1'];proxy=os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
   if proxy:net+=['-c','http.proxy='+proxy]
   for attempt in range(3):
    try:
     subprocess.run(net+['push','origin','HEAD:refs/heads/main'],cwd=ROOT,check=True,timeout=300)
     remote=subprocess.check_output(net+['ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True,timeout=90).split()
     assert remote and remote[0]==head;break
    except (subprocess.CalledProcessError,subprocess.TimeoutExpired,AssertionError):
     if attempt==2:raise
     time.sleep(30)
   state('complete_pushed',commit=head,reading=result['reading'],predictions=result['predictions'],report='exploratory_P4/E2M_REPORT.txt')
  except Exception as exc:
   state('failed',error_type=type(exc).__name__);traceback.print_exc();raise
if __name__=='__main__':main()
