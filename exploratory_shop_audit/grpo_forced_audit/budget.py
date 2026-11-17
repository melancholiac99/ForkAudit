"""Per-device leases and one shared GPU-hour budget for concurrent stages."""
import fcntl
import subprocess
import time
import uuid
from contextlib import contextmanager
from .common import *


def spent(run):return sum(r.get('gpu_hours',0) for r in read_rows(run/'cost_ledger.jsonl') if r['event']=='end')


@contextmanager
def accounting_lock(run):
    with (run/'budget_accounting.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def occupancy(run,now=None):
    now=time.time() if now is None else now
    active=[read(p) for p in (run/'active_gpu_sessions').glob('*.json')]
    return spent(run)+sum(max(0,now-r['started_unix'])*r['gpu_count']/3600 for r in active)


def uncapped(run):
    path=run/'resource_authorization.json'
    return path.exists() and read(path).get('gpu_hour_limit') is None and read(path).get('researcher_authorized') is True


def execute(run,stage,command,env=None,limit_hours=None):
    import psutil
    env=dict(os.environ if env is None else env)
    ids=[int(x) for x in env.get('CUDA_VISIBLE_DEVICES',env.get('SHOP_GPU_ID','0')).split(',')]
    locks=[]
    try:
        for gpu in sorted(ids):
            lock=(run/f'gpu_{gpu}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
        (run/'active_gpu_sessions').mkdir(exist_ok=True)
        token=uuid.uuid4().hex;active_path=run/'active_gpu_sessions'/f'{token}.json'
        with accounting_lock(run):
            unlimited=uncapped(run)
            previous=occupancy(run);remaining=24-previous
            if not unlimited and limit_hours is not None:remaining=min(remaining,limit_hours)
            if not unlimited and remaining<=.02:raise RuntimeError('GPU budget exhausted; preserve artifacts')
            start=time.time();deadline=None if unlimited else start+remaining*3600/len(ids)-15
            save(active_path,{'stage':stage,'session_id':token,'started_unix':start,'gpu_ids':ids,'gpu_count':len(ids)})
            append(run/'cost_ledger.jsonl',{'event':'start','time':utc(),'stage':stage,'session_id':token,
                'gpu_ids':ids,'gpu_count':len(ids),'previous_gpu_hours':previous,'deadline_unix':deadline,'command':[str(x) for x in command]})
        code=None;reason=None;tracked={};proc=None;log=(run/f'{stage}.log').open('a')
        try:
            proc=subprocess.Popen([str(x) for x in command],env=env,cwd=ROOT,stdout=log,stderr=log,start_new_session=True)
            while proc.poll() is None:
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):tracked[(child.pid,child.create_time())]=child
                except psutil.NoSuchProcess:pass
                with accounting_lock(run):total=occupancy(run)
                state={'stage':stage,'pid':proc.pid,'session_id':token,'gpu_ids':ids,'time':utc(),'gpu_hours_cumulative':total,'deadline_unix':deadline}
                save(run/f'live_status_gpu_{ids[0]}.json',state)
                if ids[0]==0:save(run/'live_status.json',state)
                if not unlimited and (time.time()>=deadline or total>=24-.02):
                    reason='gpu_budget_deadline'
                    children=psutil.Process(proc.pid).children(recursive=True)
                    for child in reversed(children):
                        try:child.terminate()
                        except psutil.NoSuchProcess:pass
                    proc.terminate();_,alive=psutil.wait_procs(children,timeout=5)
                    for child in alive:
                        try:child.kill()
                        except psutil.NoSuchProcess:pass
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:proc.kill()
                    break
                time.sleep(2)
            code=proc.wait()
        finally:
            alive=[p for p in tracked.values() if p.is_running()]
            for p in reversed(alive):
                try:p.terminate()
                except psutil.NoSuchProcess:pass
            _,alive=psutil.wait_procs(alive,timeout=5)
            for p in alive:
                try:p.kill()
                except psutil.NoSuchProcess:pass
            with accounting_lock(run):
                append(run/'cost_ledger.jsonl',{'event':'end','time':utc(),'stage':stage,'session_id':token,'pid':proc.pid if proc else None,
                    'gpu_ids':ids,'gpu_count':len(ids),'gpu_hours':(time.time()-start)*len(ids)/3600,'exit_code':code,'reason':reason})
                active_path.unlink()
            log.close()
        if code:raise RuntimeError(f'{stage} exited {code}; inspect {run/stage}.log')
    finally:
        for lock in locks:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
