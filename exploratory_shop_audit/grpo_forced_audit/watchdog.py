"""Read-only experiment monitor; confirmed incidents enqueue one Codex repair."""
import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path
from .common import DEFAULT_RUN, save, read, utc


def command(args, timeout=15):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return subprocess.CompletedProcess(args, 124, '', str(exc))


def tail(path, count=65536):
    if not path.exists():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-count))
        return f.read().decode('utf-8',errors='replace')


def issues(snapshot):
    """Operational thresholds only; never use reward or significance to restart."""
    out={}
    if snapshot['complete']:return out
    if snapshot['service'] not in ['active','activating','deactivating']:out['service_down']='Training supervisor is '+snapshot['service']
    if snapshot['disk_free_gib']<20:out['disk_critical']='Less than 20 GiB free; a full checkpoint needs about 17 GiB'
    elif snapshot['disk_free_gib']<30:out['disk_low']='Less than 30 GiB free; inspect checkpoint/cache growth'
    if snapshot['memory_available_gib']<8:out['memory_low']='Less than 8 GiB host memory available'
    if snapshot['gpu_error']:out['gpu_error']=snapshot['gpu_error']
    if snapshot['kernel_fault']:out['kernel_fault']='Current boot has NVIDIA/NULL-pointer/OOM kernel fault evidence'
    if snapshot['service']=='active' and snapshot['progress_age_seconds']>2400:out['progress_stalled']='No saved experiment progress for over 40 minutes'
    if snapshot['service']=='active' and snapshot['heartbeat_age_seconds']>180:out['heartbeat_stalled']='Supervisor heartbeat is over 3 minutes old'
    if snapshot['new_error']:out['new_error']='New exception appended to experiment errors.jsonl'
    if not snapshot['tensorboard_ok']:out['tensorboard_down']='TensorBoard HTTP check failed'
    return out


def confirmations(state, current, now):
    """Two successive samples; one incident per persistent fault, rearm on recovery."""
    active=state.setdefault('faults',{})
    for key in list(active):
        if key not in current:del active[key]
    ready=[]
    for key,message in current.items():
        row=active.setdefault(key,{'first_seen':now,'count':0,'dispatched':False})
        row['count']+=1
        if row['count']>=2 and not row['dispatched']:ready.append(key)
    return ready


def snapshot(run,state,now):
    import psutil
    live=read(run/'live_status.json') if (run/'live_status.json').exists() else {}
    status=read(run/'campaign_status.json') if (run/'campaign_status.json').exists() else {}
    service=command(['systemctl','is-active','forkaudit-shop-resume.service']).stdout.strip()
    # Heartbeat is not evidence of actual training progress.
    roots=[run/'rollouts',run/'audit_base',run/'audit_post',run/'evaluation']
    files=[run/'training_metrics.jsonl',run/'weight_updates.jsonl']
    for root in roots:
        if root.exists():files.extend(root.rglob('*.jsonl'))
    last=max((p.stat().st_mtime for p in files if p.exists()),default=0)
    session=live.get('session_id')
    if 'stage_seen' not in state or state.get('session')!=session:
        state.update(session=session,stage_seen=now)
    last=max(last,state.get('stage_seen',now))
    gpu=command(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'])
    gpu_rows=[];gpu_error=None
    if gpu.returncode:gpu_error='nvidia-smi failed: '+gpu.stderr[-300:]
    else:
        for line in gpu.stdout.splitlines():
            fields=[s.strip() for s in line.split(',')]
            if len(fields)==3:gpu_rows.append(fields)
        if len(gpu_rows)!=4:gpu_error=f'Expected 4 GPUs, observed {len(gpu_rows)}'
    kernel=command(['dmesg']).stdout
    kernel_fault=any(s in kernel for s in ['BUG: kernel NULL pointer dereference','reboot is needed','NVRM: Xid','Out of memory: Killed process'])
    error_path=run/'errors.jsonl';size=error_path.stat().st_size if error_path.exists() else 0
    previous=state.setdefault('error_size',size)
    # Retain a newly observed error until dispatched or a new stage starts.
    if size>previous:state['error_pending']=True
    state['error_size']=size
    http=command(['curl','--max-time','5','-s','-o','/dev/null','-w','%{http_code}','http://127.0.0.1:6006/'])
    metrics=tail(run/'training_metrics.jsonl').strip().splitlines();update=None
    if metrics:
        try:update=json.loads(metrics[-1])['update']
        except (ValueError,KeyError):pass
    return {'time':utc(),'unix_time':now,'stage':live.get('stage'),'update':update,'service':service,
        'complete':status.get('status')=='complete','disk_free_gib':shutil.disk_usage(run).free/2**30,
        'memory_available_gib':psutil.virtual_memory().available/2**30,'gpus':gpu_rows,'gpu_error':gpu_error,
        'kernel_fault':kernel_fault,'progress_age_seconds':max(0,now-last),
        'heartbeat_age_seconds':max(0,now-(run/'live_status.json').stat().st_mtime) if (run/'live_status.json').exists() else 9999,
        'new_error':state.get('error_pending',False),'tensorboard_ok':http.stdout=='200'}


def poll(run):
    folder=run/'watchdog';folder.mkdir(exist_ok=True)
    with (folder/'monitor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=read(folder/'state.json') if (folder/'state.json').exists() else {}
        if (folder/'DISABLED').exists():return
        now=time.time();snap=snapshot(run,state,now);current=issues(snap)
        ready=confirmations(state,current,now)
        save(folder/'status.json',{**snap,'issues':current,'repair_running':command(['systemctl','is-active','forkaudit-shop-repair.service']).stdout.strip()=='active'})
        # Retry broker start if it failed, but never enqueue overlapping repairs.
        if ready and not (folder/'pending.json').exists() and now-state.get('last_dispatch',0)>=1800:
            identity=f'{int(now)}';incident={'id':identity,'time':utc(),'codes':ready,
                'messages':{k:current[k] for k in ready},'snapshot':snap,'probe':False}
            save(folder/'incidents'/f'{identity}.json',incident);save(folder/'pending.json',incident)
            for key in ready:state['faults'][key]['dispatched']=True
            state['last_dispatch']=now
            if 'new_error' in ready:state['error_pending']=False
        save(folder/'state.json',state)
        if (folder/'pending.json').exists():
            result=command(['systemctl','start','--no-block','forkaudit-shop-repair.service'])
            if result.returncode:save(folder/'dispatch_error.json',{'time':utc(),'error':result.stderr})


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,default=DEFAULT_RUN);args=p.parse_args()
    poll(args.run.resolve())


if __name__=='__main__':main()
