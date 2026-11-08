"""Registered execution-only acceleration; original science and records stay frozen."""
import argparse, os, signal, subprocess, sys, time
from pathlib import Path
import resume_devices_20261108 as M
N,B,E,D,S,H,ROOT=M.N,M.B,M.E,M.D,M.S,M.H,M.ROOT
PREV=H/'run_resume_devices_20261108'
R=H/'run_accelerated_20261108'
B.R=S.R=R


def checked_plan():
    p=M._checked_plan()
    for name,sha in p['parent_plans'].items():assert E.sha_file(Path(name))==sha,name
    old=E.load(PREV/'plan.json')
    for k in ('environment','model','data','gpus','gpu_inventory','expected_formal','expected_probe'):
        assert p[k]==old[k],k
    assert p['environment']==N.ENV_CHANGES
    for name,sha in p['benchmark_files'].items():assert E.sha_file(Path(name))==sha,name
    assert p['engine_changes']==E.load(H/'speedcheck_20261108/fast400.json')['engine_changes']
    assert p['workers_per_gpu']==24 and p['max_concurrent_per_gpu']==400
    return p


def check_smokes():
    p=checked_plan();gate=E.load(R/'smoke_verified.json')
    assert E.sha_file(R/'smoke_verified.json')==p['smoke_verification_sha256']
    assert gate['passed'] and gate['large_exact_responses']==800 and gate['old_exact_anchors']==26


def install_engine_configuration():
    changes=checked_plan()['engine_changes']
    # GPU imports must stay AFTER the original science stage forks EnvPool.
    from e4_policy import LoraPolicy
    original_init=LoraPolicy.__init__
    def registered_init(self,*args,**kwargs):
        import vllm
        original=vllm.LLM
        class RegisteredEngine(original):
            def __init__(self,*a,**kw):
                kw.update(changes);super().__init__(*a,**kw)
        vllm.LLM=RegisteredEngine
        try:original_init(self,*args,**kwargs)
        finally:vllm.LLM=original
    LoraPolicy.__init__=registered_init


def execution_valid(x,path,point,plan,manifest):
    rel=str(path.relative_to(ROOT));ex=x['provenance']['execution']
    shard=int(D.digest('E4d-gpu-shard:'+point['point_id']),16)%2
    tail=E.load(R/'transition_tail_manifest.json') if (R/'transition_tail_manifest.json').exists() else {}
    if rel in manifest or rel in tail:
        expected=manifest[rel] if rel in manifest else tail[rel]
        assert E.sha_file(path)==expected,'Preserved result changed: '+rel
        parents={sha:E.load(Path(name)) for name,sha in plan['parent_plans'].items()}
        assert ex['execution_plan_sha256'] in parents
        prior=parents[ex['execution_plan_sha256']]
        assert ex['gpu_uuid']==prior['gpus'][shard]
        if rel in tail:
            assert shard==0 and ex['execution_plan_sha256']==E.sha_file(PREV/'plan.json')
            assert path==D.run_path(E.DATA,'Azero','01',point['point_id'],x['continuations'][0]['rep'])
    else:
        assert ex['execution_plan_sha256']==E.sha_file(R/'plan.json')
        assert ex['gpu_uuid']==plan['gpus'][shard]
    assert ex['source_sha256']==E.sha_file(H/'e4d.py')


def proc(pid):
    try:
        q=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return dict(state=q[0],start=q[19],exitcode=int(q[49]) if len(q)>49 else None)
    except FileNotFoundError:return None


def terminate_scheduler(pid,receipt):
    p=proc(pid)
    if p and p['start']==receipt['processes'][str(pid)]['start'] and p['state']!='Z':
        assert p['state']=='T','Only paused old schedulers may be retired'
        os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)


def wait_for_gpu0():
    receipt=E.load(R/'handoff.json');pid=receipt['continuing_gpu0']
    while True:
        p=proc(pid)
        if not p or p['start']!=receipt['processes'][str(pid)]['start'] or p['state']=='Z':break
        E.save_json(R/'gpu0_handoff_status.json',dict(stage='waiting_for_original_science',pid=pid,time=time.time()))
        time.sleep(10)
    # A missing process after reboot is not considered successful without its completion log.
    log=(PREV/'formal_Azero_01_gpu0.log').read_text(errors='replace')
    assert 'FORMAL RUN COMPLETE Azero 01' in log,'Old GPU0 stage did not complete normally'
    if p and p['state']=='Z':assert p['exitcode']==0
    plan=checked_plan();manifest=E.load(R/'preserved_manifest.json');tail={}
    for point in E.zpoints(E.locked()):
        if int(D.digest('E4d-gpu-shard:'+point['point_id']),16)%2:continue
        for u in range(24):
            path=D.run_path(E.DATA,'Azero','01',point['point_id'],u)
            x=E.load(path);E.verify_target_log(x)
            rel=str(path.relative_to(ROOT))
            if rel in manifest:execution_valid(x,path,point,plan,manifest)
            else:
                ex=x['provenance']['execution']
                assert ex['execution_plan_sha256']==E.sha_file(PREV/'plan.json')
                assert ex['gpu_uuid']==plan['gpus'][0] and x['continuations'][0]['rep']==u
                tail[rel]=E.sha_file(path)
    dst=R/'transition_tail_manifest.json'
    if dst.exists():assert E.load(dst)==tail
    else:E.save_json(dst,tail)
    E.save_json(R/'gpu0_handoff_status.json',dict(stage='complete',tail_records=len(tail),tail_sha256=E.sha_file(dst),time=time.time()))
    for old in receipt['workers']:terminate_scheduler(old,receipt)


def execute(key,args,gpu=None):
    checked_plan();check_smokes();assert not (R/'PAUSED.json').exists()
    assert str(args[0])==str(H/'e4d.py') and str(args[1]) in ('run','armprobe','analyze','probe-analyze')
    env=N.env_for(gpu) if gpu is not None else dict(os.environ,CUDA_VISIBLE_DEVICES='',**N.ENV_CHANGES)
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--science',*map(str,args[1:])]
    with (R/(key+'.log')).open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    E.save_json(R/(key+'_process.json'),dict(pid=p.pid,args=command,gpu=gpu,time=time.time()))
    assert p.wait()==0,key+' failed; preserve healthy worker'


def dual(phase):
    assert phase=='formal';children=[]
    for i in range(2):
        with (R/f'formal_{i}_driver.log').open('a') as log:
            children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker','formal','--gpu',str(i)],
                cwd=ROOT,env=N.env_for(i),stdout=log,stderr=subprocess.STDOUT))
    E.save_json(R/'formal_launch.json',dict(pids=[p.pid for p in children],time=time.time()))
    S.state('formal',children=[p.pid for p in children],data=str(E.DATA),gpu0='finish old stage then accelerate')
    while any(p.poll() is None for p in children):
        assert all(p.poll() in (None,0) for p in children),'Worker failed; other worker preserved'
        time.sleep(10)
    assert all(p.returncode==0 for p in children)


B.checked_plan=S.checked_plan=checked_plan
B.check_smokes=check_smokes
B.execution_valid=execution_valid
B.dual=dual
S.execute=execute

def wake(message):
    N._wake('【E4d执行加速队列】当前控制目录 '+str(R)+'；数据 '+str(E.DATA)+
        '；核验入口 '+str(Path(__file__).resolve())+' --verify。仅发布credit-grok；H80032B仍未完成，不得关机。'+message)
S.wake=wake

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--science':
        assert sys.argv[2] in ('run','armprobe','analyze','probe-analyze')
        assert all(os.environ.get(k)==v for k,v in N.ENV_CHANGES.items())
        checked_plan();check_smokes()
        if sys.argv[2] in ('run','armprobe'):install_engine_configuration()
        sys.argv=[str(H/'e4d.py'),*sys.argv[2:]];E.main()
    else:
        ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['formal']);ap.add_argument('--gpu',type=int,choices=[0,1]);ap.add_argument('--verify',action='store_true');a=ap.parse_args()
        if a.worker:
            check_smokes()
            if a.gpu==0:wait_for_gpu0()
            S.worker('formal',a.gpu)
        elif a.verify:B.verify()
        else:check_smokes();B.main()
