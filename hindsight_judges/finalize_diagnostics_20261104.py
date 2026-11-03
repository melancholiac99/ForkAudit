"""All requested diagnostics -> independent audit -> complete Git publication -> shutdown.

Separate from immutable scoring code. Failure requires a best-effort recovery handoff.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback

H=Path(__file__).resolve().parent;ROOT=H.parent
FIN=H/'finalization_diagnostics_20261104'
RUNS=[H/'run_diagnostics_20261104',H/'run_token_diagnostics_20261104']
OUTPUTS=[H/'results'/name for name in ('action_diagnostic','think_localization','token_diagnostic_p4','token_diagnostic_e4')]
BRANCH='research/hindsight-judges'

def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    with tmp.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def state(stage,**kw):
    x=dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    save(FIN/'status.json',x);print(json.dumps(x),flush=True)
def command(args,timeout=180,input=None):
    p=subprocess.run(list(map(str,args)),cwd=ROOT,text=True,input=input,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:raise RuntimeError(f'{args}: exit {p.returncode}\n{p.stdout}')
    return p.stdout.strip()
def git(*args,**kw):return command(['git',*args],**kw)
def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False
def queue_states():return [read(r/'status.json') for r in RUNS]
def classify(states):
    if any(s['stage'] in ('failed','scoring_with_failures') for s in states):return 'recover'
    if all(s['stage']=='completed_verified' for s in states):return 'audit'
    if any(not alive(s['pid']) and s['stage']!='completed_verified' for s in states):return 'recover'
    return 'wait'
def gpu_idle():return not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
def drained():return all(not alive(s['pid']) for s in queue_states()) and gpu_idle()

def artifacts():
    files=[]
    for root in OUTPUTS+RUNS:
        for p in root.rglob('*'):
            if not p.is_file():continue
            parts=p.relative_to(root).parts
            if '__pycache__' in parts or any(x.startswith('rejected_') for x in parts):continue
            if p.suffix in ('.tmp','.lock','.npz','.pyc'):continue
            # Student hidden states are derived local caches. Publish their manifests
            # and numerical provenance, not multi-GB hidden-state payloads.
            if any(x.startswith('student_cache') for x in parts) and p.suffix!='.json':continue
            files.append(p)
    return sorted(set(files+[Path(__file__),FIN/'policy.json']))

def audit():
    states=queue_states();assert classify(states)=='audit' and drained(),'Queues not complete and drained'
    assert read(FIN/'policy.json')['authorized']
    sys.path.insert(0,str(H))
    # Re-execute original candidate/localization verifier without changing scientific code.
    path=RUNS[0]/'driver.py';spec=importlib.util.spec_from_file_location('original_diagnostic_verifier',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify()
    v=read(RUNS[0]/'verification.json')
    assert v['passed'] and v['counts']==dict(localization=1544,**{'7b':3088,'32b':3088})
    records=[]
    for cohort,name,points,responses in [('p4','7b',386,1544),('p4','32b',386,1544),('e4','32b',1586,31720)]:
        out=H/('results/token_diagnostic_'+cohort)
        command([sys.executable,H/'exp3_token_diagnostic.py','verify','--cohort',cohort,'--name',name,'--out',out],timeout=1800)
        check=read(out/name/'full/verification.json')
        assert check['passed'] and check['points']==points and check['responses']==responses
        records.append(check)
    save(FIN/'local_verification.json',dict(passed=True,all_requested_diagnostics=True,candidate_and_localization=v,full_response=records))
    return artifacts()+[FIN/'local_verification.json']

def verify_tree(commit,files):
    # Batch Git operations, rather than launching a process for every token-record shard.
    names=[str(p.relative_to(ROOT)) for p in files]
    blobs=git('hash-object','--stdin-paths',input='\n'.join(names)+'\n').splitlines()
    assert len(blobs)==len(names)
    tree=git('-c','core.quotePath=false','ls-tree','-r',commit,'--','hindsight_judges')
    remote={line.split('\t',1)[1]:line.split('\t',1)[0].split()[2] for line in tree.splitlines()}
    assert all(remote.get(n)==b for n,b in zip(names,blobs)),'Published Git tree does not contain complete local artifacts'

def publish(files):
    assert git('branch','--show-current')==BRANCH
    assert git('remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only'),'Review existing tracked edits before publication'
    assert all(p.stat().st_size<50*1024**2 for p in files),'Large artifact needs reviewed splitting, not omission'
    inventory={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    save(FIN/'artifact_manifest.json',inventory);files=files+[FIN/'artifact_manifest.json']
    git('add','-f','--',*[str(p.relative_to(ROOT)) for p in files])
    if git('diff','--cached','--name-only'):git('commit','-m','Publish verified action, Think-localization and full-response Hindsight diagnostics')
    head=git('rev-parse','HEAD');verify_tree(head,files);errors=[]
    for attempt in range(3):
        try:
            git('-c','http.version=HTTP/1.1','-c','http.postBuffer=524288000','push','origin',f'HEAD:refs/heads/{BRANCH}',timeout=600)
            git('fetch','--no-tags','origin',BRANCH,timeout=180)
            remote=git('rev-parse','FETCH_HEAD');verify_tree(remote,files)
            receipt=dict(verified=True,branch=BRANCH,local_commit=head,remote_commit=remote,files=len(files),prior_errors=errors)
            save(FIN/'publication.json',receipt);return receipt
        except Exception:
            errors.append(traceback.format_exc());save(FIN/'publication_errors.json',errors)
            if attempt<2:time.sleep(20)
    raise RuntimeError('Normal push/remote verification failed after three attempts. No force push; review divergence in detached worktree.')

def snapshot(error):
    dest=FIN/'failures'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());dest.mkdir(parents=True,exist_ok=False)
    save(dest/'error.json',dict(error=error,queues=queue_states(),head=git('rev-parse','HEAD')))
    files=artifacts()+[dest/'error.json']
    # Stream a point-in-time archive. Active tasks may still write their logs;
    # hashes below describe the captured archive bytes, not a racing live file.
    archive=dest/'results_and_logs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    inventory={}
    with tarfile.open(archive,'r:gz') as tar:
        for member in tar:
            assert member.isfile();stream=tar.extractfile(member);h=hashlib.sha256();size=0
            for b in iter(lambda:stream.read(1024**2),b''):h.update(b);size+=len(b)
            assert size==member.size;inventory[member.name]=dict(bytes=size,sha256=h.hexdigest())
    assert len(inventory)==len(set(files))
    save(dest/'inventory.json',inventory)
    save(dest/'archive_verified.json',dict(passed=True,sha256=sha(archive),bytes=archive.stat().st_size,
        retained_student_cache_directories=[str(o/'student_cache') for o in OUTPUTS if (o/'student_cache').exists()],
        note='Derived hidden-state cache payloads remain intact on persistent disk; their metadata is archived.'))
    os.sync();return dest

def shutdown(outcome,**kw):
    assert read(FIN/'policy.json')['authorized'] and drained()
    assert not (FIN/'cancelled.json').exists()
    state('shutting_down',outcome=outcome,**kw);os.sync()
    with (FIN/'shutdown_command.log').open('a') as f:
        p=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdout=f,stderr=subprocess.STDOUT,timeout=60)
    state('shutdown_command_returned',returncode=p.returncode,outcome=outcome,**kw);os.sync()
    assert p.returncode==0

def recover(error,thread):
    try:dest=snapshot(error)
    except Exception:
        dest=FIN/'snapshot_error.json';save(dest,dict(error=error,snapshot_error=traceback.format_exc(),original_results_retained=True));os.sync()
    claim=FIN/'recovery_claim.json'
    if claim.exists():claim.rename(FIN/f'recovery_claim_previous_{time.time_ns()}.json')
    state('requesting_recovery',snapshot=str(dest),error=error)
    message=('【Hindsight 全诊断异常修复与最终关机】最新用户明确授权：按以前要求，所有诊断做好核验上传后关机；任何阶段失败先尽力修复，实在修不好保存日志/已有结果、sync后关机。'
        f'先读 {FIN}/policy.json、status.json 和 {dest}，并写 {claim} 接管。'
        '两个队列是 run_diagnostics_20261104（动作7B/32B和Think定位）、run_token_diagnostics_20261104（P4完整回复7B/32B，E4 32B加冻结7B Student）。'
        '必须全部完成且核验GitHub完整结果后才成功关机，不能只等旧队列完成。旧脚本里“无关机授权”已被本次用户授权覆盖。'
        '只跑诊断，不启动HLN/OPD训练或新轨迹。先检查其他GPU进程，保留有效结果，TV<0.005及统计门槛不得放宽，必要配置修订先记录，不静默混用。'
        '修复后重启相应队列及 finalize_diagnostics_20261104.py --thread 当前线程。原分支 research/hindsight-judges；普通push可重试，远端分歧用独立detached worktree检查，不强推不覆盖研究结果。'
        '实在修不好时保存完整错误日志和结果、sync，再 /bin/bash /usr/bin/shutdown；不要少试一次就放弃，也不要无限重试。'
        '监控在15分钟无人接管后，仅当两个队列进程均结束且GPU空闲时保存最终快照并关机。')
    try:
        response=command(['codex','queue','--thread',thread,'--resume','--message',message],timeout=60)
        (FIN/'recovery_queue.log').write_text(response+'\n')
    except Exception:
        (FIN/'recovery_queue.log').write_text(traceback.format_exc())
    started=time.monotonic()
    while True:
        if (FIN/'cancelled.json').exists():state('cancelled');return
        if claim.exists():state('recovery_handoff',claim=read(claim),snapshot=str(dest));return
        if time.monotonic()-started>=900 and drained():
            try:last=snapshot(error+'\nRecovery agent did not claim; all queue processes ended and GPUs idle.')
            except Exception:
                last=FIN/'final_snapshot_error.json'
                save(last,dict(error=traceback.format_exc(),earlier_snapshot=str(dest),
                               original_results_retained=[str(p) for p in OUTPUTS],original_logs_retained=[str(p) for p in RUNS]))
                os.sync()
            shutdown('failed_recovery_unavailable',snapshot=str(last));return
        time.sleep(30)

def main():
    p=argparse.ArgumentParser();p.add_argument('--thread');p.add_argument('--check-only',action='store_true');a=p.parse_args()
    if a.check_only:
        states=queue_states();print(json.dumps(dict(classification=classify(states),queues=[s['stage'] for s in states],shutdown_executed=False)));return
    assert a.thread and read(FIN/'policy.json')['authorized']
    with (FIN/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state('watching_all_diagnostics')
        while True:
            if (FIN/'cancelled.json').exists():state('cancelled');return
            states=queue_states();decision=classify(states)
            if decision=='recover':recover('Diagnostic queue failed or exited unexpectedly: '+json.dumps(states),a.thread);return
            if decision=='audit' and drained():break
            time.sleep(30)
        try:
            state('independent_audit');files=audit()
            state('publishing');receipt=publish(files)
            save(FIN/'completion_verified.json',dict(passed=True,**receipt))
            shutdown('all_verified_and_uploaded',**receipt)
        except Exception:recover(traceback.format_exc(),a.thread)

if __name__=='__main__':main()
