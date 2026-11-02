"""E4c: audit published results, request best-effort recovery, then authorized shutdown.
This independent guard never edits the running scientific pipeline.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import traceback

H=Path(__file__).resolve().parent;ROOT=H.parent;RUN=H/'run';FIN=H/'finalization'

def read(p):return json.loads(Path(p).read_text())
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def state(stage,**kw):
    save(FIN/'status.json',dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
    print(stage,kw,flush=True)
def command(args,timeout=180):
    r=subprocess.run(list(map(str,args)),cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=timeout)
    if r.returncode:raise RuntimeError(f'{args!r}: exit {r.returncode}\n{r.stdout}')
    return r.stdout.strip()
def git(*args):return command(['git',*args])
def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except FileNotFoundError:return None
def idle():return not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])
def authorized():return read(FIN/'policy.json').get('shutdown_authorized') is True

def audit():
    assert read(RUN/'status.json')['stage']=='completed_published'
    check=read(RUN/'final_verification.json')
    assert check['passed'] and check['episodes']==4110 and check['contrasts_verified']==32
    assert check['original_inputs_unchanged'] and check['all_step_constraints_and_seeds_verified'] and check['behaviour_verified']
    for name,info in read(RUN/'artifact_manifest.json').items():
        p=H/name;assert p.resolve().is_relative_to(H.resolve()) and p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],name
    archive=H/'results_archive/e4c_results.tar.gz';am=read(archive.parent/'archive_manifest.json')
    assert am['roundtrip_verified'] and sha(archive)==am['sha256'] and archive.stat().st_size==am['bytes']
    publication=read(RUN/'publication.json');assert publication['verified'] and publication['branch']=='main'
    assert git('remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    git('fetch','--no-tags','origin','main');remote=git('rev-parse','FETCH_HEAD')
    git('merge-base','--is-ancestor',publication['commit'],remote)
    files=[archive,archive.parent/'archive_manifest.json',H/'e4c.json',H/'E4c_REPORT.txt',RUN/'artifact_manifest.json',RUN/'final_verification.json']
    files+=list((H/'data/rollout').glob('*/*/summary.json'))
    for p in files:
        assert git('rev-parse',f'{remote}:{p.relative_to(ROOT)}')==git('hash-object',p),str(p)
    result=dict(passed=True,remote_main=remote,published_commit=publication['commit'],episodes=4110,files_verified=len(files))
    save(FIN/'completion_verified.json',result);return result

def preserve(error):
    dest=FIN/'failures'/time.strftime('%Y%m%dT%H%M%SZ',time.gmtime());dest.mkdir(parents=True,exist_ok=False)
    save(dest/'error.json',dict(error=error,queue_state=read(RUN/'status.json'),head=git('rev-parse','HEAD')))
    files=sorted({p for folder in (H/'data',RUN) for p in folder.rglob('*') if p.is_file() and p.suffix not in ('.tmp','.lock')}
                 |{p for p in H.iterdir() if p.is_file() and p.suffix in ('.py','.sh','.md','.json','.txt')})
    manifest={str(p.relative_to(H)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    save(dest/'manifest.json',manifest);archive=dest/'data_and_logs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(H)),recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        seen=set()
        for member in tar:
            assert member.isfile();f=tar.extractfile(member);digest=hashlib.sha256()
            for b in iter(lambda:f.read(1024**2),b''):digest.update(b)
            assert digest.hexdigest()==manifest[member.name]['sha256'];seen.add(member.name)
        assert seen==set(manifest)
    save(dest/'archive_verified.json',dict(passed=True,files=len(files),sha256=sha(archive)))
    os.sync();return dest

def shutdown(reason,**kw):
    if not authorized():state('shutdown_cancelled');return
    assert idle(),'New GPU workload is active; do not shut down'
    launch=read(RUN/'launch.json');assert identity(launch['pid']) is None,'E4c supervisor is still active'
    state('shutdown_requested',reason=reason,**kw);os.sync()
    with (FIN/'shutdown_command.log').open('a') as log:
        r=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdout=log,stderr=subprocess.STDOUT)
    state('shutdown_command_returned',returncode=r.returncode,reason=reason,**kw);os.sync()

def recover(thread,error):
    try:dest=preserve(error)
    except Exception:
        error+='\nArchive error; original files retained:\n'+traceback.format_exc()
        dest=FIN/'archive_error.json';save(dest,dict(error=error,original_data=str(H/'data'),logs=str(RUN)));os.sync()
    state('requesting_recovery',snapshot=str(dest),error=error)
    message=(
        f'【E4C 尽力修复与最终关机】最新用户明确要求：E4C 出错唤醒本会话尽力修好并跑完；不要过早放弃；成功核验上传 main 后关机，确实修不好则完整保存错误和结果、sync 后关机。'
        f'这覆盖旧 supervisor 消息中的“不得关机”，以 {FIN}/policy.json 最新授权为准。工作区 {ROOT}，main 分支。'
        f'先读 {RUN}/status.json、pipeline.log、{FIN}/status.json、失败快照 {dest}，并写 {FIN}/recovery_claim.json 接管时间。'
        '检查根因，尽力修复；不得放宽预注册判定阈值或覆盖原始数据。输入/科学方案必要变化须明确记录，不得伪造结果。'
        '上传失败可重试；分支冲突先检查，必要时 detached worktree 合并，不强推或自动覆盖研究结论。'
        '如续跑，重新启动 completion_guard.py --thread 当前线程 来监控新 supervisor，继续执行核验、上传和关机。'
        '若确实无法修复，保存完整错误和已有数据、确认落盘后 /bin/bash /usr/bin/shutdown。不要无限重试或仅报告失败后留下空闲机器。'
        '本守护程序等待接管 15 分钟；若无人接管且没有 GPU 任务，会保存唤醒失败日志后按授权关机。')
    try:
        response=command(['codex','queue','--thread',thread,'--message',message,'--resume'],timeout=60)
        (FIN/'wake_response.log').write_text(response+'\n');os.sync();state('recovery_queued',snapshot=str(dest))
        for _ in range(30):
            if (FIN/'recovery_claim.json').exists():
                state('recovery_handoff',snapshot=str(dest),claim=read(FIN/'recovery_claim.json'));return
            time.sleep(30)
        error+='\nRecovery thread did not claim within 15 minutes.'
    except Exception:error+='\n'+traceback.format_exc()
    save(FIN/'recovery_unavailable.json',dict(error=error,snapshot=str(dest)));os.sync()
    shutdown('recovery_unavailable_after_preservation',snapshot=str(dest))

def main():
    p=argparse.ArgumentParser();p.add_argument('--thread',required=True);args=p.parse_args()
    FIN.mkdir(exist_ok=True)
    with (FIN/'guard.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        claim=FIN/'recovery_claim.json'
        if claim.exists():claim.rename(FIN/f'previous_claim_{time.time_ns()}.json')
        launch=read(RUN/'launch.json');pid=launch['pid'];start=identity(pid)
        state('watching',supervisor_pid=pid,identity=start,thread=args.thread)
        while start is not None and identity(pid)==start:time.sleep(20)
        try:
            result=audit();shutdown('verified_uploaded_main',verification=result)
        except Exception:recover(args.thread,traceback.format_exc())

if __name__=='__main__':main()
