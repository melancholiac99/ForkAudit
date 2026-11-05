"""Eight-condition prefix v2 diagnostic -> independent audit -> complete Git publication -> shutdown.

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
FIN=H/'finalization_prefix_v2_20261105'
RUNS=[H/'run_prefix_attribution_v2_20261105']
OUTPUTS=[H/'results/prefix_attribution_v2_20261105']
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
def drained():
    jobs=read(RUNS[0]/'jobs.json') if (RUNS[0]/'jobs.json').exists() else {}
    return (all(not alive(s['pid']) for s in queue_states())
            and all(not alive(j['pid']) for j in jobs.values()) and gpu_idle())

def artifacts():
    files=[]
    for root in OUTPUTS+RUNS:
        for p in root.rglob('*'):
            if not p.is_file():continue
            parts=p.relative_to(root).parts
            if '__pycache__' in parts or any(x.startswith(('rejected_', 'pre_gpu_')) for x in parts):continue
            if p.suffix in ('.tmp','.lock','.npz','.pyc'):continue
            # Student hidden states are derived local caches. Publish their manifests
            # and numerical provenance, not multi-GB hidden-state payloads.
            if any(x.startswith('student_cache') for x in parts) and p.suffix!='.json':continue
            files.append(p)
    return sorted(set(files+[Path(__file__),FIN/'policy.json']))

def audit():
    states=queue_states();assert classify(states)=='audit' and drained(),'Queue not complete and drained'
    assert read(FIN/'policy.json')['authorized']
    records=[]
    for name in ('7b','32b'):
        command([sys.executable,H/'exp4_prefix_attribution.py','verify','--name',name,'--out',OUTPUTS[0]],timeout=1800)
        check=read(OUTPUTS[0]/name/'full/verification.json')
        assert check['passed'] and check['points']==386 and check['contexts']==6176 and check['clusters']==60
        assert check['historical_anchors_compatible'] and check['centered_slopes_recomputed']
        records.append(check)
    save(FIN/'local_verification.json',dict(passed=True,models=records,total_contexts=12352,
        conditions=['AS','AO','QS','QO','QN','AP','KEEP','DROP'],prior_results_read_only=True))
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
    allowed={str(p.relative_to(ROOT)) for p in files}
    changed=set(git('diff','--name-only').splitlines()) | set(git('diff','--cached','--name-only').splitlines())
    assert changed<=allowed,'Unrelated tracked/staged edits require review before publication'
    # Lossless publication for any large raw score files; local originals remain intact.
    import gzip
    published=[]; compressed=[]
    for p in files:
        if p.stat().st_size >= 40*1024**2:
            target=p.with_name(p.name+'.gz')
            if p.suffix=='.gz':raise RuntimeError('Large compressed artifact requires review: '+str(p))
            with p.open('rb') as source,target.open('wb') as raw:
                with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as dest:
                    for chunk in iter(lambda:source.read(1024**2),b''):dest.write(chunk)
            h=hashlib.sha256()
            with gzip.open(target,'rb') as f:
                for chunk in iter(lambda:f.read(1024**2),b''):h.update(chunk)
            assert h.hexdigest()==sha(p)
            compressed.append(dict(original=str(p.relative_to(ROOT)),published=str(target.relative_to(ROOT)),
                original_sha256=sha(p),gzip_sha256=sha(target),restore='gzip -dk '+target.name))
            published.append(target)
        else:published.append(p)
    save(FIN/'publication_format.json',dict(lossless=True,compressed=compressed,
         note='Large raw files are published as verified gzip; restore with gzip -dk for the frozen verifier. Local originals retained.'))
    files=sorted(set(published+[FIN/'publication_format.json']))
    assert all(p.stat().st_size<50*1024**2 for p in files),'Compressed artifact still too large; preserve and review'
    inventory={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    save(FIN/'artifact_manifest.json',inventory);files=files+[FIN/'artifact_manifest.json']
    git('add','-f','--',*[str(p.relative_to(ROOT)) for p in files])
    if git('diff','--cached','--name-only'):git('commit','-m','Publish verified eight-condition prefix attribution experiments and anchor reproduction')
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
    files=sorted(set(artifacts()+[dest/'error.json']+
                     [p for p in FIN.iterdir() if p.is_file() and p.suffix in ('.json','.log')]))
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
    message=('【Prefix v2 自动异常修复与关机】最新用户明确要求：实验跑完按原要求核验上传后关机；任何阶段失败先尽力修复，实在修不好保全日志结果、sync后关机。'
        f'先读 {FIN}/policy.json、status.json 和 {dest}，并写 {claim} 接管时间。'
        '工作区 /root/autodl-tmp/forkaudit_hindsight，分支 research/hindsight-judges；只处理八条件 prefix v2（7B和32B全量各386点），原实验已完成不要重跑。'
        '评分队列 run_prefix_attribution_v2_20261105，结果 results/prefix_attribution_v2_20261105。旧队列消息中的无关机授权由最新 policy.json 覆盖。'
        '保留当前有效结果；原数据、AP/QN结构、TV<0.005、精度和统计门槛不得改动或静默混用。修复后重启队列及 finalize_prefix_v2_20261105.py --thread 当前线程。'
        '上传只到原research分支，普通push可重试；远端分歧用独立detached worktree核查，不强推或覆盖研究结论。'
        '如果确实无法修复，保存完整日志/结果、sync后 /bin/bash /usr/bin/shutdown。不要一次尝试就放弃，不无限重试。'
        '若15分钟无人接管，监控只在实验队列进程已结束且GPU空闲时保存最终快照并关机。')

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
        state('watching_prefix_v2')
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
