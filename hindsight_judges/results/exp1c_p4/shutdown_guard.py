"""Success-only shutdown after the independently verified E1c publication."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path('/root/autodl-tmp/forkaudit_hindsight')
RUN=ROOT/'hindsight_judges/run_exp1c'
OUT=ROOT/'hindsight_judges/results/exp1c_p4'
BRANCH='research/hindsight-judges'

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def command(*args):return subprocess.check_output(args,cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def state(stage,**kw):
    value=dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    tmp=RUN/'shutdown_status.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(RUN/'shutdown_status.json')
    print(json.dumps(value),flush=True)

def ready():
    status=read(RUN/'status.json')
    return status.get('stage')=='completed_published' and (RUN/'publication.json').exists()

def verify():
    policy=read(OUT/'shutdown_policy.json')
    assert policy['authorized'] and policy['only_after_verified_publication']
    assert sha(Path(__file__))==policy['guard_sha256']
    assert not (RUN/'shutdown_cancelled_1c.json').exists(),'shutdown was cancelled'
    assert ready(),'experiment has not completed publication'
    pub=read(RUN/'publication.json');status=read(RUN/'status.json')
    assert pub['verified'] and pub['branch']==BRANCH and status['publication']==pub
    assert command('git','branch','--show-current')==BRANCH
    assert command('git','remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    commit=pub['commit'];assert command('git','rev-parse','HEAD')==commit
    assert command('git','ls-remote','--heads','origin',BRANCH).split()[0]==commit
    run=read(OUT/'32b/run.json');check=read(OUT/'verification.json')
    assert run['complete'] and run['contexts']==3474
    assert check['passed'] and check['contexts']==3474 and check['points']==386
    assert check['primary_endpoints_independently_verified']==3
    assert check['source_and_prior_E1_unchanged'] and check['hardware_unchanged']
    assert sha(OUT/'32b/scores.jsonl')==run['scores_sha256']==check['scores_sha256']
    assert sha(OUT/'32b/analysis.json')==check['analysis_sha256']
    gates=read(OUT/'32b/numerical_check.json')
    assert len(gates)==19 and all(g['passed'] and g['tv']<.005 for g in gates)
    manifest=read(OUT/'artifact_manifest.json')
    files={str(p.relative_to(OUT)):p for p in OUT.rglob('*') if p.is_file()}
    assert set(files)==set(manifest)|{'artifact_manifest.json'}
    for name,entry in manifest.items():
        assert files[name].stat().st_size==entry['bytes'] and sha(files[name])==entry['sha256'],name
    for path in files.values():
        blob=command('git','hash-object','--',str(path))
        assert command('git','rev-parse',commit+':'+str(path.relative_to(ROOT)))==blob,str(path)
    for path,digest in read(RUN/'protected_inputs.json').items():assert sha(Path(path))==digest,path
    return commit

def main():
    if '--check-only' in sys.argv:
        print(json.dumps(dict(ready=ready(),experiment_stage=read(RUN/'status.json')['stage'],shutdown_executed=False)))
        return
    with (RUN/'shutdown_guard.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            state('waiting_for_verified_publication')
            while not ready():
                if (RUN/'shutdown_cancelled_1c.json').exists():state('cancelled');return
                time.sleep(30)
            state('checking_published_artifacts')
            # Allow brief network failures without touching the experiment or its results.
            for attempt in range(3):
                try:commit=verify();break
                except Exception:
                    if attempt==2:raise
                    time.sleep(20)
            for _ in range(20):
                if not command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'):break
                time.sleep(30)
            else:raise RuntimeError('GPU remains busy after publication; refusing shutdown')
            assert not (RUN/'shutdown_cancelled_1c.json').exists()
            state('verified_shutting_down',commit=commit,command=['/bin/bash','/usr/bin/shutdown'])
            os.sync()
            with (RUN/'shutdown_command.log').open('a') as log:
                result=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdout=log,stderr=subprocess.STDOUT,timeout=60)
            state('shutdown_command_returned',returncode=result.returncode,commit=commit)
            os.sync()
            assert result.returncode==0
        except BaseException as exc:
            state('blocked',error=str(exc));traceback.print_exc();os.sync()
            thread=os.environ.get('CODEX_THREAD_ID')
            if thread:
                subprocess.run(['codex','queue','--thread',thread,'--resume','--message',
                    '【Hindsight 1c 关机核验异常】用户已授权：实验完成、核验和上传成功后关机。请读 '+str(RUN/'shutdown_status.json')+
                    ' 和 shutdown_guard.log，尽力修复并核验 GitHub 完整结果后关机；不得跳过核验或强推。关机命令 /bin/bash /usr/bin/shutdown。'],timeout=60)
            raise

if __name__=='__main__':main()
