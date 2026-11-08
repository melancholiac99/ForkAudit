"""Frozen placement of remaining whole shards; identical science/engine configuration."""
import argparse,fcntl,os,subprocess,sys,time,traceback
from pathlib import Path
import accelerate_20261108 as A
N,B,E,D,S,H,ROOT=A.N,A.B,A.E,A.D,A.S,A.H,A.ROOT
PREV=A.R
R=H/'run_balanced_20261109'
B.R=S.R=R


def checked_plan():
    p=A.M._checked_plan()
    for name,sha in p['parent_plans'].items():assert E.sha_file(Path(name))==sha,name
    old=E.load(PREV/'plan.json')
    for k in ('environment','engine_changes','model','data','gpus','gpu_inventory','workers_per_gpu','max_concurrent_per_gpu','expected_formal','expected_probe'):
        assert p[k]==old[k],k
    assert E.sha_file(R/'handoff.json')==p['handoff_sha256']
    assert E.sha_file(Path(p['reference']))==p['reference_sha256']
    return p


def shard(point):return int(D.digest('E4d-gpu-shard:'+point['point_id']),16)%2


def key_for(path,point):
    parts=path.relative_to(E.DATA).parts
    if parts[0]=='results':return f'formal:{parts[1]}:{parts[2]}:{shard(point)}'
    assert parts[0]=='armprobe'
    return f'probe:{parts[1]}:{shard(point)}'


def job_paths(job,points):
    for p in points:
        if shard(p)!=job['shard']:continue
        if job['phase']=='formal':
            for u in range(24):yield p,D.run_path(E.DATA,job['arm'],job['cond'],p['point_id'],u)
        elif p['point_id'] in job['probe_ids']:
            for j in range(5):
                for u in range(8):yield p,E.probe_path(job['arm'],p['point_id'],j,u)


def check_smokes(gpu=None):
    p=checked_plan();ref=E.load(Path(p['reference']))
    for i in range(2) if gpu is None else [gpu]:
        x=E.load(R/f'lora_smoke_{i}.json')
        assert x['gpu']==p['gpus'][i] and x['execution_plan_sha256']==E.sha_file(R/'plan.json')
        for k in ('point_ids','prompts_sha256','seeds','outputs'):assert x[k]==ref[k],(i,k)
        assert len(x['outputs'])==13


def execution_valid(x,path,point,plan,manifest):
    rel=str(path.relative_to(ROOT));ex=x['provenance']['execution'];frozen=dict(manifest)
    for i in range(2):
        tailfile=R/f'transition_tail_{i}.json'
        if tailfile.exists():
            status=E.load(R/f'handoff_{i}.json');assert status['sha256']==E.sha_file(tailfile)
            tail=E.load(tailfile)
            if rel in tail:
                assert key_for(path,point)==plan['active_jobs'][str(i)]['key']
                assert ex['execution_plan_sha256']==E.sha_file(PREV/'plan.json') and ex['gpu_uuid']==plan['gpus'][i]
            frozen.update(tail)
    if rel in frozen:
        assert E.sha_file(path)==frozen[rel]
        parents={sha:E.load(Path(name)) for name,sha in plan['parent_plans'].items()}
        assert ex['execution_plan_sha256'] in parents
        assert ex['gpu_uuid']==parents[ex['execution_plan_sha256']]['gpus'][shard(point)]
    else:
        job=plan['jobs'][key_for(path,point)]
        assert ex['execution_plan_sha256']==E.sha_file(R/'plan.json')
        assert ex['gpu_uuid']==plan['gpus'][job['gpu']]
    assert ex['source_sha256']==E.sha_file(H/'e4d.py')


def wait_old(gpu):
    plan=checked_plan();info=E.load(R/'handoff.json')['science'][str(gpu)]
    while True:
        p=A.proc(info['pid'])
        if not p or p['start']!=info['start'] or p['state']=='Z':break
        E.save_json(R/f'worker_{gpu}_status.json',dict(stage='waiting_for_current_stage',pid=os.getpid(),science_pid=info['pid'],time=time.time()))
        time.sleep(15)
    log=Path(info['log']).read_text(errors='replace')
    job=plan['active_jobs'][str(gpu)]
    assert f"FORMAL RUN COMPLETE {job['arm']} {job['cond']}" in log,'Interrupted old stage; preserve and repair before takeover'
    if p and p['start']==info['start'] and p['state']=='Z':assert p['exitcode']==0
    manifest=E.load(R/'preserved_manifest.json');tail={}
    for point,path in job_paths(job,E.zpoints(E.load(E.LOCK))):
        x=E.load(path);E.verify_target_log(x);rel=str(path.relative_to(ROOT))
        if rel in manifest:execution_valid(x,path,point,plan,manifest)
        else:
            ex=x['provenance']['execution']
            assert ex['execution_plan_sha256']==E.sha_file(PREV/'plan.json') and ex['gpu_uuid']==plan['gpus'][gpu]
            tail[rel]=E.sha_file(path)
    dst=R/f'transition_tail_{gpu}.json'
    if dst.exists():assert E.load(dst)==tail
    else:E.save_json(dst,tail)
    E.save_json(R/f'handoff_{gpu}.json',dict(passed=True,records=len(tail),sha256=E.sha_file(dst),time=time.time()))
    receipt=E.load(R/'handoff.json');A.terminate_scheduler(receipt['workers'][gpu],receipt)


def invoke(key,command,gpu):
    env=N.env_for(gpu) if gpu is not None else dict(os.environ,CUDA_VISIBLE_DEVICES='',**N.ENV_CHANGES)
    with (R/(key+'.log')).open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    E.save_json(R/(key+'_process.json'),dict(pid=p.pid,args=command,time=time.time(),gpu=gpu))
    assert p.wait()==0,key+' failed; preserve the other worker'


def execute(key,args,gpu=None):
    checked_plan();assert not (R/'PAUSED.json').exists()
    check_smokes(gpu)
    assert str(args[0])==str(H/'e4d.py') and str(args[1]) in ('run','armprobe','analyze','probe-analyze')
    invoke(key,[sys.executable,'-u',str(Path(__file__).resolve()),'--science',*map(str,args[1:])],gpu)


def worker(gpu):
    with (R/f'gpu_{gpu}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=checked_plan();wait_old(gpu)
        if not (R/f'lora_smoke_{gpu}.json').exists():
            invoke(f'smoke_gpu{gpu}',[sys.executable,'-u',str(Path(__file__).resolve()),'--smoke',str(gpu)],gpu)
        check_smokes(gpu)
        for key in plan['queues'][str(gpu)]:
            job=plan['jobs'][key];assert job['gpu']==gpu
            E.save_json(R/f'worker_{gpu}_status.json',dict(stage='running',job=key,pid=os.getpid(),time=time.time()))
            execute(key.replace(':','_'),[H/'e4d.py',*job['args']],gpu)
        E.save_json(R/f'worker_{gpu}_status.json',dict(stage='completed',pid=os.getpid(),time=time.time()))


def wake(message):
    N._wake('【E4d均衡续跑】控制目录 '+str(R)+'；核验 balance_20261109.py --verify。'
        '最新要求是完整实验核验上传credit-grok后关机。只读主报告不代表次要实验完成。'
        '任何异常尽力修复，不动健康GPU；禁止修改旧结果、协议或门槛。详情：'+message)
S.wake=wake
B.checked_plan=S.checked_plan=checked_plan
B.check_smokes=check_smokes
B.execution_valid=execution_valid
S.execute=execute


def main():
    with (R/'driver.lock').open('a') as lock,(B.OLD/'driver.lock').open('a') as old:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(old,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            S.state('checking_preserved_inputs');B.preserved()
            children=[]
            for i in range(2):
                with (R/f'worker_{i}.log').open('a') as log:
                    children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(i)],cwd=ROOT,env=N.env_for(i),stdout=log,stderr=subprocess.STDOUT))
            S.state('formal',children=[p.pid for p in children]);E.save_json(R/'worker_launch.json',dict(pids=[p.pid for p in children]))
            while any(p.poll() is None for p in children):
                assert all(p.poll() in (None,0) for p in children),'Worker failed; healthy worker preserved'
                time.sleep(15)
            assert all(p.returncode==0 for p in children)
            S.state('analyzing')
            for cmd in ('analyze','probe-analyze'):execute(cmd,[H/'e4d.py',cmd])
            B.verify()
            primary=H/'primary_complete_20261109/analysis.json'
            if primary.exists():
                preview=E.load(primary);final=E.load(H/'e4d.json')
                assert preview['pooled']==final['pooled'] and preview['reading']==final['reading']
                E.save_json(R/'primary_report_consistency.json',dict(passed=True,primary_sha256=E.sha_file(primary),final_sha256=E.sha_file(H/'e4d.json')))
            S.state('completed_verified');wake('全量已核验，等待独立发布和关机监控。')
        except Exception:
            err=traceback.format_exc();(R/'failure.log').write_text(err);S.state('failed',error=err);wake(err);raise

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--science':
        assert all(os.environ.get(k)==v for k,v in N.ENV_CHANGES.items())
        checked_plan();assert sys.argv[2] in ('run','armprobe','analyze','probe-analyze')
        if sys.argv[2] in ('run','armprobe'):A.install_engine_configuration()
        sys.argv=[str(H/'e4d.py'),*sys.argv[2:]];E.main()
    else:
        p=argparse.ArgumentParser();p.add_argument('--worker',type=int,choices=[0,1]);p.add_argument('--smoke',type=int,choices=[0,1]);p.add_argument('--verify',action='store_true');args=p.parse_args()
        if args.worker is not None:worker(args.worker)
        elif args.smoke is not None:checked_plan();A.install_engine_configuration();B.lora_smoke(args.smoke);check_smokes(args.smoke)
        elif args.verify:B.verify()
        else:main()
