"""Single-GPU E4e queue, independent verification/review, bounded publication and shutdown."""
import argparse,fcntl,hashlib,importlib.metadata,json,os,subprocess,sys,tarfile,time,traceback
from pathlib import Path
import analyze_server as V
H,ROOT,RUN,RESULT=V.H,V.ROOT,V.RUN,V.RESULT
PLAN=RUN/'plan.json';CONTROL=RUN/'control_manifest.json'

def state(stage,**kw):V.save(RUN/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw));print(stage,kw,flush=True)
def command(args,timeout=180,cwd=ROOT,input=None):
    p=subprocess.run(list(map(str,args)),cwd=cwd,input=input,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:raise RuntimeError(str(args)+'\n'+p.stdout)
    return p.stdout.strip()
def git(*args,**kw):return command(['git',*args],**kw)
def live(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]!='Z'
    except FileNotFoundError:return False

def preflight(deep=False):
    p=V.load(PLAN);c=V.load(CONTROL)
    for file,sha in {**p['source'],**c['source']}.items():assert V.sha(file)==sha,file
    for k,v in c['versions'].items():assert importlib.metadata.version(k)==v,k
    assert V.sha(PLAN)==c['plan_sha256'] and c['shutdown_authorized']
    assert git('branch','--show-current')=='credit-grok'
    assert git('remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    if deep:
        for file,sha in p['model_files'].items():assert V.sha(file)==sha,file
        assert V.sha(RUN/'input_manifest.json')==p['input_manifest_sha256']
        for file,sha in V.load(RUN/'input_manifest.json').items():assert V.sha(file)==sha,file
    return p

def env():
    p=V.load(PLAN)
    return dict(os.environ,CUDA_VISIBLE_DEVICES=p['gpu_uuid'],PATH='/root/miniconda3/envs/m0/bin:/usr/local/cuda/bin:'+os.environ['PATH'],
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',VLLM_WORKER_MULTIPROC_METHOD='spawn',**p['environment'])

def eval_args(stage,tag):
    p=V.load(PLAN)
    args=[sys.executable,'-u',str(H/'server_eval.py'),'--plan',str(PLAN),'--split','valid_unseen','--policy-tag',tag,'--model-path',p['model'],
        '--conditions',*p['conditions'],'--reps',str(p['pilot_reps'] if stage=='pilot' else p['full_reps']),
        '--workers',str(p['workers']),'--chunk',str(p['chunk'])]
    if stage=='pilot':args+=['--game-seeds',*map(str,p['pilot_games'])]
    args+=['--out',str(H/'data'/f'{stage}_20261110')]
    if p['policies'][tag]:args+=['--adapter',p['policies'][tag]]
    return args

def invoke(key,args,cpu=False):
    with (RUN/(key+'.log')).open('a') as log:
        p=subprocess.Popen(args,cwd=ROOT,env=dict(env(),CUDA_VISIBLE_DEVICES='') if cpu else env(),stdout=log,stderr=subprocess.STDOUT)
    V.save(RUN/(key+'_launch.json'),dict(pid=p.pid,args=args,time=time.time()));state('running',job=key,child=p.pid)
    rc=p.wait();assert rc==0,f'{key} failed with exit {rc}'

def notify(text):
    thread=os.environ.get('CODEX_THREAD_ID');assert thread
    out=command(['codex','queue','--thread',thread,'--resume','--message',text],timeout=60)
    with (RUN/'wake.log').open('a') as f:f.write(out+'\n')

def shutdown(outcome):
    assert V.load(CONTROL)['shutdown_authorized'] and not (RUN/'cancel_shutdown.json').exists()
    assert not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']),'GPU work still active'
    state('shutting_down',outcome=outcome);os.sync()
    with (RUN/'shutdown_command.log').open('a') as log:p=subprocess.run(['/bin/bash','/usr/bin/shutdown'],stdout=log,stderr=subprocess.STDOUT,timeout=60)
    state('shutdown_command_returned',returncode=p.returncode,outcome=outcome);os.sync();assert p.returncode==0

def archive_failure(error):
    d=RUN/'failures'/str(time.time_ns());d.mkdir(parents=True)
    V.save(d/'error.json',dict(error=error,time=time.time(),original_results_retained=True))
    paths=[p for p in H.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'failures' not in p.parts and p.suffix not in ('.lock','.tmp','.pyc')]
    with tarfile.open(d/'results_and_logs.tar.gz','w:gz') as tar:
        for p in paths+[d/'error.json']:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    V.save(d/'archive.json',dict(sha256=V.sha(d/'results_and_logs.tar.gz'),files=len(paths)+1));os.sync();return d

def recover(error):
    (RUN/'failure.log').write_text(error);dest=archive_failure(error)
    claim=RUN/'recovery_claim.json'
    if claim.exists():claim.rename(RUN/f'recovery_claim_previous_{time.time_ns()}.json')
    state('requesting_recovery',error=error,snapshot=str(dest))
    text=(f'【E4e单卡队列异常】用户要求运行credit-grok最新E4e，完成核验上传后关机；任何失败先尽力修复，修不好保存日志结果sync后关机。'
        f'读取 {RUN}/status.json、failure.log、plan.json、{dest}，接管先写 {claim}。'
        '只改exploratory_E4e，保留旧E4/E4d/hindsight数据与协议。全量固定134局×3重复×4条件×4策略=6432回合，不按效果缩样本、不训练。'
        '只读主工作区Aprime权重和原轨迹，不修改main。无本地公开RL权重，已明确列为缺项。'
        '恢复入口 exploratory_E4e/run_server.py；先确认没有健康GPU子进程，不能重复启动或清空已保存回合。'
        '上传冲突用独立detached worktree审查，不强推。修复成功续跑并重启watcher。'
        '30分钟无人接管且GPU空闲时保全日志后关机。')
    for attempt in range(3):
        try:notify(text);break
        except Exception:
            with (RUN/'wake_errors.log').open('a') as f:f.write(traceback.format_exc())
            if attempt<2:time.sleep(10)
    start=time.monotonic()
    while True:
        if (RUN/'cancel_shutdown.json').exists():state('cancelled');return
        if claim.exists():state('recovery_handoff');return
        if time.monotonic()-start>=1800 and not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']):
            archive_failure(error+'\nRecovery unavailable after 30 minutes.');shutdown('unrecoverable_saved');return
        time.sleep(20)

def review():
    request=RESULT/'full_override_review_request.json';answer=RESULT/'full_override_review.json'
    if not answer.exists():
        state('awaiting_manual_review',request=str(request))
        notify(f'【E4e全量干预待核验】6432回合已完成、结构与原环境回放抽检通过。请先读 {request}，逐样本依据目标、携带反馈、合法动作和执行反馈标注rule_supported/rule_contradicted/uncertain；不要把放下成功当作有因果收益。保存 {answer}，格式 {{"request_sha256":"...","samples":[{{"id":"review_000","label":"rule_supported","reason":"..."}}]}}。只做核验，运行中的E4e控制器会汇总、上传credit-grok并关机，不要重复启动发布。')
    start=time.monotonic()
    while not answer.exists():
        assert time.monotonic()-start<1800,'Manual review unavailable after 30 minutes; request recovery'
        time.sleep(15)
    x=V.load(answer);q=V.load(request)
    assert x['request_sha256']==V.sha(request)
    ids=[s['id'] for s in x['samples']];assert len(ids)==len(set(ids)) and set(ids)=={s['id'] for s in q['samples']}
    assert all(s['label'] in q['labels'] and s.get('reason') for s in x['samples'])
    counts={label:sum(s['label']==label for s in x['samples']) for label in q['labels']}
    V.save(RESULT/'manual_review_summary.json',dict(samples=len(ids),counts=counts,request_sha256=V.sha(request),review_sha256=V.sha(answer),coverage_note='Stratified examples, not exhaustive causal or strategic-necessity labels.'))
    with (RESULT/'E4e_REPORT.txt').open('a') as f:f.write('\nManual rule review: '+json.dumps(counts)+'\n')

def artifact_files():
    return sorted(p for p in H.rglob('*') if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and 'failures' not in p.parts and p.suffix not in ('.lock','.tmp','.pyc')
        and p.name not in ['status.json','publication.json','publication_paths.nul','shutdown_command.log','wake.log','watcher.log','driver.log','artifact_manifest.json'])

def verify_tree(commit,files):
    names=[str(p.relative_to(ROOT)) for p in files]
    hashes=command(['git','hash-object','--stdin-paths'],input='\n'.join(names)+'\n').splitlines()
    tree=git('-c','core.quotePath=false','ls-tree','-r',commit)
    remote={l.split('\t',1)[1]:l.split('\t',1)[0].split()[2] for l in tree.splitlines()}
    assert len(hashes)==len(names) and all(remote.get(n)==v for n,v in zip(names,hashes)),'Remote artifacts differ'

def publish():
    preflight();files=artifact_files();allowed={str(p.relative_to(ROOT)) for p in files}
    changes=set(git('diff','--name-only').splitlines())|set(git('diff','--cached','--name-only').splitlines())
    assert changes<=allowed,'Unrelated staged/modified files must not be published'
    assert all(p.stat().st_size<90*1024**2 for p in files),'Need lossless split archive for large artifacts'
    V.save(RUN/'artifact_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=V.sha(p)) for p in files})
    files.append(RUN/'artifact_manifest.json');paths=RUN/'publication_paths.nul';paths.write_bytes(b''.join(str(p.relative_to(ROOT)).encode()+b'\0' for p in files))
    git('add','-f','--pathspec-from-file='+str(paths),'--pathspec-file-nul')
    if git('diff','--cached','--name-only'):git('commit','-m','Publish verified E4e stage funnel and four-arm inference audit')
    verify_tree('HEAD',files)
    errors=[]
    for attempt in range(3):
        try:
            git('-c','http.version=HTTP/1.1','push','origin','HEAD:credit-grok',timeout=900)
            git('fetch','--no-tags','origin','refs/heads/credit-grok:refs/remotes/origin/credit-grok')
            remote=git('rev-parse','origin/credit-grok');verify_tree(remote,files)
            receipt=dict(verified=True,branch='credit-grok',commit=remote,files=len(files));V.save(RUN/'publication.json',receipt);return receipt
        except Exception:
            errors.append(traceback.format_exc());V.save(RUN/'publication_errors.json',errors)
            if attempt<2:time.sleep(20)
    raise RuntimeError('Publication failed after bounded retries; inspect divergence without force push')

def main():
    with (RUN/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            state('preflight');p=preflight(deep=True)
            launch=RUN/'pilot_base_launch.json'
            if launch.exists():
                initial=V.load(launch);pid=initial['pid'];cmd=Path(f'/proc/{pid}/cmdline')
                while cmd.exists() and live(pid) and b'server_eval.py' in cmd.read_bytes():
                    state('waiting_initial_pilot',child=pid);time.sleep(20)
            for tag in p['pilot_policies']:
                if not all((V.data_root('pilot')/tag/c/'summary.json').exists() for c in p['conditions']):invoke('pilot_'+tag,eval_args('pilot',tag))
            invoke('verify_pilot',[sys.executable,'-u',str(H/'analyze_server.py'),'pilot'],cpu=True)
            state('pilot_passed')
            for tag in p['policies']:
                if not all((V.data_root('full')/tag/c/'summary.json').exists() for c in p['conditions']):invoke('full_'+tag,eval_args('full',tag))
            invoke('verify_full',[sys.executable,'-u',str(H/'analyze_server.py'),'full'],cpu=True)
            review();preflight(deep=True)
            state('publishing');receipt=publish();state('completed_verified_published',publication=receipt)
            try:
                notify(f'【E4e已完成并发布】请读 {RESULT}/E4e_REPORT.txt、full_verification.json、full_environment_replay.json、manual_review_summary.json 和 {RUN}/publication.json。公开RL比较缺本地权重，未训练。用户授权收尾关机，控制器将sync并关机。')
            except Exception:
                (RUN/'completion_notification_error.log').write_text(traceback.format_exc())
            shutdown('available_E4e_scope_verified_uploaded')
        except Exception:recover(traceback.format_exc())

def watch(pid):
    with (RUN/'watcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while live(pid):time.sleep(30)
        s=V.load(RUN/'status.json')
        if s['stage'] in ('shutdown_command_returned','shutting_down','recovery_handoff','cancelled'):return
        recover('E4e controller exited unexpectedly: '+json.dumps(s))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--watch',type=int);args=ap.parse_args()
    if args.watch:watch(args.watch)
    else:main()
