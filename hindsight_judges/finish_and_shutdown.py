"""User-authorized completion audit, failure recovery handoff, and shutdown.

Independent of frozen scoring code. Never changes experimental gates or results.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

H=Path(__file__).resolve().parent
ROOT=H.parent
RUN=H/'run'
FIN=H/'finalization'
BRANCH='research/hindsight-judges'
E4=Path('/root/autodl-tmp/idea/exploratory_E4b')


def read(path):return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def state(stage,**extra):
    value=dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    save(FIN/'status.json',value);print(json.dumps(value),flush=True)


def identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if s[0]=='Z' else s[19]
    except FileNotFoundError:return None


def command(args,cwd=ROOT,timeout=180):
    result=subprocess.run(list(map(str,args)),cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    if result.returncode:raise RuntimeError(f'{args!r}: exit {result.returncode}\n{result.stdout}')
    return result.stdout.strip()


def git(*args):return command(['git',*args])


def gpu_idle():
    return not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])


def classify(status,alive):
    if alive:return 'wait'
    return 'verify_success' if status.get('stage')=='completed_published' else 'recover'


def local_audit():
    audit=read(RUN/'final_verification.json')
    assert audit['passed'] and audit['protected_inputs_unchanged']
    assert set(audit['models'])=={'32b','7b'}
    for name in ('32b','7b'):
        p=H/'results/exp1_p4'/name;v=read(p/'verification.json');r=read(p/'run.json')
        assert v==audit['models'][name] and v['passed'] and v['points']==386 and v['contexts']==3860
        assert r['complete'] and r['contexts']==3860 and r['attention_backend']=='sdpa_math'
        if name=='7b':
            assert r['compute_precision']=='float32' and v['precision_provenance_verified']
        assert sha(p/'scores.jsonl')==v['scores_sha256']==r['scores_sha256']
        assert sha(p/'analysis.json')==v['analysis_sha256']
    files=read(RUN/'artifact_manifest.json')
    for name,info in files.items():
        p=ROOT/name;assert p.resolve().is_relative_to(H.resolve()) and p.is_file()
        assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],name
    required=list(files)+[str(p.relative_to(ROOT)) for p in [RUN/'final_verification.json',RUN/'artifact_manifest.json']]
    assert read(E4/'run/final_verification.json')['passed']
    assert read(E4/'run/publication_status.json')['stage']=='published_verified'
    return required


def verify_tree(commit,files):
    for name in files:
        assert git('rev-parse',f'{commit}:{name}')==git('hash-object',str(ROOT/name)),name


def verify_remote(files):
    assert git('remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    git('fetch','--no-tags','origin',BRANCH)
    remote=git('rev-parse','FETCH_HEAD');verify_tree(remote,files)
    git('fetch','--no-tags','origin','main')
    main=git('rev-parse','FETCH_HEAD');e4commit=read(E4/'run/publication_status.json')['commit']
    git('merge-base','--is-ancestor',e4commit,main)
    return dict(hindsight_commit=remote,hindsight_branch=BRANCH,e4b_commit=e4commit,e4b_branch='main')


def try_publication_repair(files):
    # Network/transient push failures are retried without rewriting history.
    # Divergence/conflicts are handed to the recovery reviewer, not auto-overwritten.
    assert git('branch','--show-current')==BRANCH
    assert not git('diff','--cached','--name-only') and not git('diff','--name-only')
    verify_tree('HEAD',files)
    errors=[]
    for attempt in range(3):
        try:
            git('push','origin',f'HEAD:refs/heads/{BRANCH}')
            result=verify_remote(files)
            save(FIN/'publication_repair.json',dict(passed=True,attempts=attempt+1,previous_errors=errors,**result))
            return result
        except Exception:
            errors.append(traceback.format_exc());save(FIN/'publication_repair.json',dict(passed=False,errors=errors))
            if attempt<2:time.sleep(20)
    raise RuntimeError('Publication repair exhausted; see finalization/publication_repair.json')


def snapshot_failure(error):
    dest=FIN/'failures'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());dest.mkdir(parents=True,exist_ok=False)
    save(dest/'error.json',dict(error=error,queue_state=read(RUN/'status.json'),head=git('rev-parse','HEAD')))
    files=sorted({p for folder in (RUN,H/'results',H/'diagnostics') for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.lock','.tmp')}
                 |{p for p in H.iterdir() if p.is_file() and p.suffix in ('.py','.txt','.md')})
    inventory={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    save(dest/'inventory.json',inventory)
    archive=dest/'results_and_logs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    # Read the archive back before treating it as a durable recovery copy.
    with tarfile.open(archive,'r:gz') as tar:
        names=set()
        for member in tar:
            assert member.isfile();stream=tar.extractfile(member);h=hashlib.sha256()
            for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
            assert h.hexdigest()==inventory[member.name]['sha256'];names.add(member.name)
        assert names==set(inventory)
    save(dest/'archive_verified.json',dict(passed=True,files=len(files),sha256=sha(archive),bytes=archive.stat().st_size))
    os.sync();return dest


def shutdown(outcome,**extra):
    if (FIN/'shutdown_cancelled.json').exists():
        state('shutdown_cancelled',reason=read(FIN/'shutdown_cancelled.json'));return
    # Do not interrupt any newly started GPU work, including a recovery run.
    assert gpu_idle(),'GPU is busy: shutdown deferred'
    state('shutdown_requested',outcome=outcome,**extra);os.sync()
    with (FIN/'shutdown_command.log').open('a') as f:
        result=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdout=f,stderr=subprocess.STDOUT)
    state('shutdown_command_returned',returncode=result.returncode,outcome=outcome,**extra);os.sync()


def recovery(thread,error):
    try:dest=snapshot_failure(error)
    except Exception:
        # Retain originals even if a secondary archive cannot be created.
        error+='\nSnapshot error (original files retained):\n'+traceback.format_exc()
        dest=FIN/'snapshot_failure.json';save(dest,dict(error=error,original_results=str(H/'results'),original_logs=str(RUN)));os.sync()
    state('requesting_recovery',snapshot=str(dest),error=error)
    message=(
        '【Hindsight 自动异常修复】用户已要求：任何阶段失败先检查并尝试修复；修不好保存错误日志和结果、sync 后关机。'
        f'工作区 {ROOT}，原分支 research/hindsight-judges。失败快照：{dest}。'
        '先读取 finalization/status.json、run/status.json 和最新日志，并在 finalization/recovery_claim.json 写入接管时间。'
        'E4B 已完成并推 main，不重跑。Hindsight 使用 SDPA MATH；32B 已验证保留，7B 已授权 FP32 修复。不得放宽 TV<0.005、改统计门槛或混用运行配置。用户明确要求尽力修复，不要因少量诊断尝试失败就过早放弃。'
        '上传失败可重试；远端分歧用独立 detached worktree 检查合并，冲突保留，不强推或覆盖研究结果。'
        '若能修复续跑，重新启动 finish_and_shutdown.py 监控新队列；所有结果核验并确认上传后执行 /bin/bash /usr/bin/shutdown。'
        '若无法修复，完整保存错误日志和已有结果并 sync，再执行同一关机命令，不无限重试。'
        '本监控将在 15 分钟内未收到接管文件且 GPU 空闲时按用户授权关机。')
    try:
        response=command(['codex','queue','--thread',thread,'--message',message,'--resume'],timeout=60)
        (FIN/'recovery_queue_response.log').write_text(response+'\n');os.sync()
        state('recovery_queued',snapshot=str(dest),claim_deadline_seconds=900)
        for _ in range(30):
            if (FIN/'recovery_claim.json').exists():
                state('recovery_handoff',snapshot=str(dest),claim=read(FIN/'recovery_claim.json'));return
            time.sleep(30)
        error+='\nRecovery reviewer did not claim within 15 minutes.'
    except Exception:error+='\n'+traceback.format_exc()
    save(FIN/'unrecoverable.json',dict(error=error,snapshot=str(dest)))
    shutdown('failed_recovery_unavailable',snapshot=str(dest),error=error)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--thread',required=True);args=parser.parse_args()
    FIN.mkdir(exist_ok=True)
    if (FIN/'shutdown_cancelled.json').exists():
        state('shutdown_cancelled',reason=read(FIN/'shutdown_cancelled.json'));return
    with (FIN/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # A prior claimed handoff must be retained, not mistaken for a new one.
        claim=FIN/'recovery_claim.json'
        if claim.exists():claim.rename(FIN/f'recovery_claim_previous_{time.time_ns()}.json')
        launch=read(RUN/'launch.json');pid=launch['pid'];start=identity(pid)
        state('watching',queue_pid=pid,queue_identity=start,thread=args.thread)
        while classify(read(RUN/'status.json'),start is not None and identity(pid)==start)=='wait':time.sleep(30)
        try:
            files=local_audit()
            try:published=verify_remote(files)
            except Exception:
                state('repairing_publication',error=traceback.format_exc())
                published=try_publication_repair(files)
            save(FIN/'final_check.json',dict(passed=True,files=len(files),**published))
            shutdown('verified_and_uploaded',**published)
        except Exception:
            recovery(args.thread,traceback.format_exc())


if __name__=='__main__':main()
