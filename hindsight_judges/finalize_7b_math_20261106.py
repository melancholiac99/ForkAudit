"""Verify, publish and shut down the user-selected 7B + mathematics scope."""
import argparse,fcntl,os,sys,time,traceback,subprocess
from pathlib import Path
import finalize_prefix_v2_20261105 as F
H=Path(__file__).resolve().parent;ROOT=H.parent;R=H/'run_next_slot_20261106'
A=R/'math_two_gpu_after_7b';FIN=H/'finalization_7b_math_20261106'
OUTPUTS=[H/'results/token_slot_20261106',H/'results/math_slot_20261106',H/'results/next_slot_supplement_20261106']

def states():return [F.read(A/'controller_status.json'),F.read(R/'supplement_status.json')]

def classify(ss):
    expected=('completed_verified','completed')
    if any(s.get('stage')=='failed' for s in ss):return 'recover'
    if all(s.get('stage')==end for s,end in zip(ss,expected)):return 'audit'
    if any(s.get('stage')!=end and not F.alive(s['pid']) for s,end in zip(ss,expected)):return 'recover'
    return 'wait'

def drained():
    # GPU-idle alone is insufficient while a controller is loading or reviewing.
    controller=F.read(A/'controller_status.json')
    launch=F.read(A/'launch.json')
    pids=[controller.get('pid',launch['controller_pid']),launch['controller_pid']]
    supplement=F.read(R/'supplement_status.json')
    if supplement.get('pid'):pids.append(supplement['pid'])
    for q in ('token','math'):
        s=F.read(R/(q+'_status.json'))
        if s.get('child_pid'):pids.append(s['child_pid'])
        pids.extend(s.get('children',{}).values())
    return all(not F.alive(p) for p in pids) and F.gpu_idle()

def artifacts():
    files=[]
    for root in OUTPUTS+[A,R/'reboot_resume_20261106']:
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.lock','.tmp','.pyc'):files.append(p)
    files += [p for p in R.iterdir() if p.is_file() and p.suffix in ('.py','.json','.log','.txt')]
    files += [Path(__file__),H/'finalize_prefix_v2_20261105.py',H/'test_finalize_7b_math_20261106.py',FIN/'policy.json']
    return sorted(set(files))

def audit():
    assert classify(states())=='audit' and drained()
    policy=F.read(FIN/'policy.json');assert policy['authorized']
    for p,s in policy['finalizer_source'].items():assert F.sha(Path(p))==s
    target=F.read(A/'plan.json')['target']
    for args in ([H/'exp5_token_slot.py','verify','--name','7b','--limit','0'],
                 [H/'exp6_math_slot.py','verify','--stage','full','--target',str(target)]):
        F.command([sys.executable,*args],timeout=1800)
    token=OUTPUTS[0]/'7b/full';math=OUTPUTS[1]/'full'
    assert F.read(token/'verification.json')['passed'] and F.read(token/'run.json')['complete']
    assert len(F.read(token/'manifest.json'))==386
    assert F.read(math/'verification.json')['passed']
    assert F.read(R/'token_status.json')['stage']=='completed_7b_verified_32b_full_cancelled'
    for key,original in (('token_7b_full',token),('math_full',math)):
        report=OUTPUTS[2]/(key+'.json');check=F.read(OUTPUTS[2]/(key+'_execution_check.json'));x=F.read(report)
        assert check['passed'] and check['output_sha256']==F.sha(report)
        assert x['original_verification_sha256']==F.sha(original/'verification.json')
        if key=='math_full':assert x['blind_review']['status']=='complete'
    for stage in ('screen','evaluate'):
        assignment=A/(stage+'_assignment.json')
        if not assignment.exists():
            assert stage=='screen';continue
        ids=[]
        for role in ('primary','helper'):
            folder=A/stage/role;done=F.read(folder/'completion.json')
            assert done['passed'] and F.read(folder/'cross_card_check.json')['passed']
            assert all(F.sha(Path(p))==s for p,s in done['outputs'].items())
            ids+=done['questions']
        assert len(ids)==len(set(ids)) and set(ids)==set(F.read(assignment)['all_ids'])
    assert all(F.sha(Path(p))==s for p,s in F.read(A/'preserved_math_hashes.json').items())
    F.save(FIN/'local_verification.json',dict(passed=True,scope=['token_7b_full386','math_full','supplements'],
        cancelled='token_32b_full',preserved_32b_pilot=True,time=time.time()))
    return artifacts()+[FIN/'local_verification.json']

def publish(files):
    assert F.git('branch','--show-current')==F.BRANCH
    assert F.git('remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    allowed={str(p.relative_to(ROOT)) for p in files}
    changed=set(F.git('diff','--name-only').splitlines())|set(F.git('diff','--cached','--name-only').splitlines())
    assert changed<=allowed,'Unrelated edits require recovery review'
    assert all(p.stat().st_size<50*1024**2 for p in files),'Large artifact needs lossless publication review'
    F.save(FIN/'artifact_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=F.sha(p)) for p in files})
    files=sorted(set(files+[FIN/'artifact_manifest.json']))
    F.git('add','-f','--',*[str(p.relative_to(ROOT)) for p in files])
    if F.git('diff','--cached','--name-only'):F.git('commit','-m','Publish verified 7B token and dual-GPU mathematics experiments')
    head=F.git('rev-parse','HEAD');F.verify_tree(head,files);errors=[]
    for attempt in range(3):
        try:
            F.git('-c','http.version=HTTP/1.1','push','origin',f'HEAD:refs/heads/{F.BRANCH}',timeout=600)
            F.git('fetch','--no-tags','origin',F.BRANCH)
            remote=F.git('rev-parse','FETCH_HEAD');F.verify_tree(remote,files)
            receipt=dict(verified=True,local_commit=head,remote_commit=remote,branch=F.BRANCH,files=len(files))
            F.save(FIN/'publication.json',receipt);return receipt
        except Exception:
            errors.append(traceback.format_exc());F.save(FIN/'publication_errors.json',errors)
            if attempt<2:time.sleep(20)
    raise RuntimeError('Publication attempts failed; inspect divergence in detached worktree, never force push')

def recover(error,thread):
    try:snapshot=F.snapshot(error)
    except Exception:
        snapshot=FIN/'snapshot_error.json';F.save(snapshot,dict(error=error,snapshot_error=traceback.format_exc(),original_results_retained=True));os.sync()
    claim=FIN/'recovery_claim.json'
    if claim.exists():claim.rename(FIN/f'recovery_claim_previous_{time.time_ns()}.json')
    F.state('requesting_recovery',error=error,snapshot=str(snapshot))
    message=('【7B与数学异常修复及关机】最新用户要求做完核验上传后关机；任何失败先尽力修复，确实修不好保存日志和结果、sync后关机。'
        f'先读 {FIN}/policy.json、status.json、{snapshot}，在 {claim} 写接管时间。'
        f'工作区 {ROOT}，仅 research/hindsight-judges。当前范围：逐token仅7B完整386点，32B全量已取消；数学剩余工作双卡按题分工。'
        '不要启动已取消的32B全量，不改冻结科学代码、种子、TV<0.005或统计门槛，不混配置。另一正常工作队列保留。'
        '控制器 run_next_slot_20261106/finish_7b_then_dual_math.py；补充监控 watch_supplement_7b_math.py。'
        '旧脚本中的无关机授权已被最新policy覆盖，但不得提前关机。尽力排查并修复后续跑，同时重启 finalize_7b_math_20261106.py --thread 当前线程。'
        '上传可重试；远端分歧用独立detached worktree检查合并，不强推或覆盖研究结果。'
        '实在无法修复，完整保存日志及已有结果、sync，然后 /bin/bash /usr/bin/shutdown。'
        '若30分钟无人接管，仅在控制器及GPU工作均已结束时保存最终快照并关机。')
    for attempt in range(3):
        try:
            response=F.command(['codex','queue','--thread',thread,'--resume','--message',message],timeout=60)
            (FIN/'recovery_queue.log').write_text(response+'\n');break
        except Exception:
            with (FIN/'recovery_queue.log').open('a') as f:f.write(traceback.format_exc())
            if attempt<2:time.sleep(20)
    started=time.monotonic()
    while True:
        if (FIN/'cancelled.json').exists():F.state('cancelled');return
        if claim.exists():F.state('recovery_handoff',claim=F.read(claim),snapshot=str(snapshot));return
        if time.monotonic()-started>=1800:
            # An idle CPU watcher can otherwise wait forever for a failed GPU queue.
            supplement=F.read(R/'supplement_status.json');controller=F.read(A/'controller_status.json')
            pid=supplement.get('pid');controller_pid=controller.get('pid',F.read(A/'launch.json')['controller_pid'])
            if pid and F.alive(pid) and not F.alive(controller_pid) and F.gpu_idle() and supplement['stage']=='waiting_for_verified_inputs_or_blind_review':
                children=Path(f'/proc/{pid}/task/{pid}/children')
                cmd=Path(f'/proc/{pid}/cmdline')
                if children.exists() and not children.read_text().strip() and cmd.exists() and 'watch_supplement_7b_math.py' in cmd.read_text():
                    import signal
                    F.save(FIN/'stopped_idle_supplement.json',dict(pid=pid,time=time.time(),reason='Recovery unavailable, GPU queue ended, no active CPU child'))
                    os.kill(pid,signal.SIGTERM)
        if time.monotonic()-started>=1800 and drained():
            # Preserve any completed work from the other queue before fallback.
            try:snapshot=F.snapshot(error+'\nRecovery unavailable; workers ended and GPUs idle.')
            except Exception:
                F.save(FIN/'final_snapshot_error.json',dict(error=traceback.format_exc(),original_results_retained=True));os.sync()
            F.shutdown('failed_recovery_unavailable',snapshot=str(snapshot));return
        time.sleep(30)

def main():
    p=argparse.ArgumentParser();p.add_argument('--thread');p.add_argument('--check-only',action='store_true');args=p.parse_args()
    if args.check_only:
        print(dict(classification=classify(states()),states=states(),shutdown_executed=False));return
    policy=F.read(FIN/'policy.json');assert args.thread and policy['authorized']
    assert all(F.sha(Path(p))==s for p,s in policy['finalizer_source'].items())
    with (FIN/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        F.state('watching_7b_and_mathematics')
        try:
            while True:
                if (FIN/'cancelled.json').exists():F.state('cancelled');return
                decision=classify(states())
                if decision=='recover':raise RuntimeError('Experiment/supplement failed or exited: '+str(states()))
                if decision=='audit' and drained():break
                time.sleep(30)
            F.state('independent_audit');files=audit()
            F.state('publishing');receipt=publish(files)
            F.save(FIN/'completion_verified.json',dict(passed=True,**receipt))
            F.shutdown('all_verified_and_uploaded',**receipt)
        except Exception:recover(traceback.format_exc(),args.thread)

# Reuse only generic I/O, snapshot, Git-tree checks and guarded platform shutdown.
F.FIN=FIN;F.RUNS=[R];F.OUTPUTS=OUTPUTS;F.queue_states=states;F.artifacts=artifacts;F.drained=drained
if __name__=='__main__':main()
