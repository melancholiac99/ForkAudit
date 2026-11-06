"""Publish credit-grok separately, verify both remote branches, then shut down."""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import e4d as E
H=Path(__file__).resolve().parent;ROOT=H.parent;R=H/'run';FIN=H/'finalization'
HIN=Path('/root/autodl-tmp/forkaudit_hindsight/hindsight_judges')
HP=HIN/'finalization_7b_math_publish_only_20261107'
BRANCH='credit-grok'

def state(stage,**kw):
    E.save_json(FIN/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw));print(stage,kw,flush=True)

def command(args,timeout=180,input=None,cwd=ROOT):
    p=subprocess.run(list(map(str,args)),cwd=cwd,input=input,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:raise RuntimeError(str(args)+'\n'+p.stdout)
    return p.stdout.strip()

def git(*args,**kw):return command(['git',*args],**kw)

def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ',1)[1].split()[0]!='Z'

def gpu_idle():return not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])

def queue_states():return [E.load(R/'status.json'),E.load(HP/'status.json')]

def classify(states):
    done=('completed_verified','completed_published_shutdown_deferred')
    if any(s.get('stage')=='failed' for s in states):return 'recover'
    if all(s.get('stage')==expected for s,expected in zip(states,done)):return 'audit'
    if any(s.get('stage')!=expected and not alive(s['pid']) for s,expected in zip(states,done)):return 'recover'
    return 'wait'

def live_experiment_processes():
    found=[]
    needles=('run_server.py','e4d.py','finish_7b_then_dual_math.py','math_two_gpu_worker.py','exp5_token_slot.py','exp6_math_slot.py')
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            args=p.joinpath('cmdline').read_bytes().split(b'\0');words=[x.decode(errors='replace') for x in args if x]
            if words and 'python' in Path(words[0]).name and any(Path(w).name in needles for w in words[1:]):
                if alive(int(p.name)):found.append(int(p.name))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return found

def artifacts():
    files=[]
    for root in (E.DATA,R):
        for p in root.rglob('*'):
            if p.is_file() and not p.is_symlink() and p.suffix not in ('.lock','.tmp','.pyc','.pending-target') and '__pycache__' not in p.parts:files.append(p)
    files += [p for p in H.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.txt')]
    files += [ROOT/'DECISIONS.md',FIN/'policy.json']
    return sorted(set(files))

def verify_tree(commit,files,base=ROOT):
    names=[str(p.relative_to(base)) for p in files]
    blobs=command(['git','hash-object','--stdin-paths'],cwd=base,input='\n'.join(names)+'\n').splitlines()
    tree=git('-c','core.quotePath=false','ls-tree','-r',commit)
    remote={x.split('\t',1)[1]:x.split('\t',1)[0].split()[2] for x in tree.splitlines()}
    assert len(blobs)==len(names) and all(remote.get(n)==b for n,b in zip(names,blobs)),'Remote artifacts differ from local complete results'

def verify_hindsight():
    receipt=E.load(HP/'publication.json');assert receipt['verified'] and receipt['branch']=='research/hindsight-judges'
    manifest=E.load(HP/'artifact_manifest.json');files=[HIN.parent/n for n in manifest]
    assert all(p.is_file() and E.sha_file(p)==manifest[str(p.relative_to(HIN.parent))]['sha256'] for p in files)
    git('fetch','--no-tags','origin','refs/heads/research/hindsight-judges:refs/remotes/origin/research/hindsight-judges')
    remote=git('rev-parse','origin/research/hindsight-judges');verify_tree(remote,files+[HP/'artifact_manifest.json'],HIN.parent)
    return dict(branch='research/hindsight-judges',commit=remote,files=len(files)+1)

def publish():
    assert git('branch','--show-current')==BRANCH
    assert git('remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    command([sys.executable,H/'run_server.py','--verify'],timeout=1800)
    assert E.load(R/'verification.json')['passed']
    files=artifacts();allowed={str(p.relative_to(ROOT)) for p in files}
    changes=set(git('diff','--name-only').splitlines())|set(git('diff','--cached','--name-only').splitlines())
    assert changes<=allowed,'Unrelated edits must be reviewed before publication'
    assert all(p.stat().st_size<90*1024**2 for p in files),'Large artifact requires verified lossless packaging, never omission'
    E.save_json(FIN/'artifact_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=E.sha_file(p)) for p in files})
    files=sorted(set(files+[FIN/'artifact_manifest.json']))
    paths=FIN/'publication_paths.nul';paths.write_bytes(b''.join(str(p.relative_to(ROOT)).encode()+b'\0' for p in files))
    git('add','-f','--pathspec-from-file='+str(paths),'--pathspec-file-nul')
    if git('diff','--cached','--name-only'):git('commit','-m','Publish verified independent E4d and credit-grok dual-GPU results')
    head=git('rev-parse','HEAD');verify_tree(head,files);errors=[]
    for attempt in range(3):
        try:
            git('-c','http.version=HTTP/1.1','push','origin','HEAD:refs/heads/'+BRANCH,timeout=900)
            git('fetch','--no-tags','origin','refs/heads/'+BRANCH+':refs/remotes/origin/'+BRANCH)
            remote=git('rev-parse','origin/'+BRANCH);verify_tree(remote,files)
            receipt=dict(verified=True,branch=BRANCH,commit=remote,local_commit=head,files=len(files))
            E.save_json(FIN/'publication.json',receipt);return receipt
        except Exception:
            errors.append(traceback.format_exc());E.save_json(FIN/'publication_errors.json',errors)
            if attempt<2:time.sleep(20)
    raise RuntimeError('Publication failed after three attempts; detached-worktree divergence review required, no force push')

def snapshot(error):
    dest=FIN/'failures'/str(time.time_ns());dest.mkdir(parents=True)
    E.save_json(dest/'error.json',dict(error=error,states=queue_states(),time=time.time()))
    files={p:'credit-grok/'+str(p.relative_to(ROOT)) for p in artifacts()+[dest/'error.json']}
    for directory in (HIN/'run_next_slot_20261106',HIN/'results/token_slot_20261106',HIN/'results/math_slot_20261106',HIN/'results/next_slot_supplement_20261106',HP):
        for p in directory.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.gz','.log','.txt','.py','.md') and '__pycache__' not in p.parts:
                files[p]='hindsight/'+str(p.relative_to(HIN.parent))
    archive=dest/'results_and_logs.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p,name in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
    inventory={}
    with tarfile.open(archive,'r:gz') as tar:
        for item in tar:
            assert item.isfile();stream=tar.extractfile(item);h=hashlib.sha256();size=0
            for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk);size+=len(chunk)
            assert size==item.size;inventory[item.name]=dict(bytes=size,sha256=h.hexdigest())
    assert len(inventory)==len(files)
    E.save_json(dest/'archive_verified.json',dict(passed=True,sha256=E.sha_file(archive),inventory=inventory));os.sync();return dest

def shutdown(outcome,**kw):
    assert E.load(FIN/'policy.json')['authorized'] and not (FIN/'cancelled.json').exists()
    assert gpu_idle() and not live_experiment_processes(),'Active experiments must never be interrupted by shutdown'
    state('shutting_down',outcome=outcome,**kw);os.sync()
    with (FIN/'shutdown_command.log').open('a') as log:p=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdout=log,stderr=subprocess.STDOUT,timeout=60)
    state('shutdown_command_returned',returncode=p.returncode,outcome=outcome);os.sync();assert p.returncode==0

def recover(error,thread):
    try:dest=snapshot(error)
    except Exception:
        dest=FIN/'snapshot_error.json';E.save_json(dest,dict(error=error,snapshot_error=traceback.format_exc(),original_results_retained=True));os.sync()
    claim=FIN/'recovery_claim.json'
    if claim.exists():claim.rename(FIN/f'recovery_claim_previous_{time.time_ns()}.json')
    state('requesting_recovery',error=error,snapshot=str(dest))
    message=('【credit-grok + hindsight 联合异常修复】用户要求两项独立idea都完成核验、分别上传各自分支之后关机；任何失败尽力修复续跑，实在修不好保存完整日志结果、sync后关机。'
      f'先读 {FIN}/policy.json、status.json 和 {dest}，写 {claim} 接管时间。'
      f'credit-grok目录 {ROOT}，只推credit-grok；hindsight目录 {HIN.parent}，只推research/hindsight-judges；不改main或旧原始数据。'
      '逐token32B全量已取消，只做7B全量；数学继续双卡。credit-grok为E4d主2x2加72点次要探针，不训练、不新增RL checkpoint、不按pi1结果换样本。'
      '不得放宽TV或统计门槛，不混配置。保持另一正常GPU工作，旧日志不得当新错误。'
      '修复后续跑 exploratory_E4d/run_server.py，hindsight保留其独立队列；重启 exploratory_E4d/finalize_server.py --thread 当前线程。'
      '普通push可重试；远端分歧用独立detached worktree检查，不强推或覆盖结论。确实修不好保全两个分支日志结果后sync，再 /bin/bash /usr/bin/shutdown。'
      '本监控在30分钟无人接管且所有实验进程结束、GPU空闲时保存最终快照并关机；不要一次尝试就放弃。')
    for attempt in range(3):
        try:
            response=command(['codex','queue','--thread',thread,'--resume','--message',message],timeout=60)
            (FIN/'recovery_queue.log').write_text(response+'\n');break
        except Exception:
            with (FIN/'recovery_queue.log').open('a') as log:log.write(traceback.format_exc())
            if attempt<2:time.sleep(20)
    start=time.monotonic()
    while True:
        if (FIN/'cancelled.json').exists():state('cancelled');return
        if claim.exists():state('recovery_handoff',claim=E.load(claim));return
        if time.monotonic()-start>1800 and gpu_idle() and not live_experiment_processes():
            try:dest=snapshot(error+'\nRecovery unavailable; all GPU work ended.')
            except Exception:
                E.save_json(FIN/'final_snapshot_error.json',dict(error=traceback.format_exc(),original_results_retained=True));os.sync()
            shutdown('failed_recovery_unavailable',snapshot=str(dest));return
        time.sleep(30)

def main():
    p=argparse.ArgumentParser();p.add_argument('--thread');p.add_argument('--check-only',action='store_true');args=p.parse_args()
    if args.check_only:print(dict(classification=classify(queue_states()),shutdown_executed=False));return
    policy=E.load(FIN/'policy.json');assert policy['authorized'] and args.thread
    assert all(E.sha_file(Path(p))==s for p,s in policy['source'].items())
    with (FIN/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);state('waiting_for_two_separate_experiments')
        try:
            while True:
                if (FIN/'cancelled.json').exists():state('cancelled');return
                decision=classify(queue_states())
                if decision=='recover':raise RuntimeError('A queue failed or exited: '+str(queue_states()))
                if decision=='audit' and gpu_idle() and not live_experiment_processes():break
                time.sleep(30)
            state('verifying_hindsight_remote');hindsight=verify_hindsight()
            state('publishing_credit_grok');credit=publish()
            hindsight=verify_hindsight()
            E.save_json(FIN/'completion_verified.json',dict(passed=True,hindsight=hindsight,credit_grok=credit))
            shutdown('both_branches_verified_and_uploaded',hindsight=hindsight,credit_grok=credit)
        except Exception:recover(traceback.format_exc(),args.thread)

if __name__=='__main__':main()
