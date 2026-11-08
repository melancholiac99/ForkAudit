"""Same numerical configuration, new GPU identities, immutable old records."""
import argparse, os, subprocess, sys, time
from pathlib import Path
import run_deterministic_20261107 as N
B,E,D,S,H,ROOT=N.B,N.E,N.D,N.S,N.H,N.ROOT
PREV=H/'run_deterministic_20261107'
R=H/'run_resume_devices_20261108'
B.R=S.R=R
_checked_plan=B.checked_plan


def checked_plan():
    p=_checked_plan()
    assert E.sha_file(PREV/'plan.json')==p['previous_deterministic_plan_sha256']
    old=E.load(PREV/'plan.json')
    for k in ('environment','engine_changes','model','workers_per_gpu','max_concurrent_per_gpu','data'):
        assert p[k]==old[k],k
    for path,sha in old['source'].items():assert E.sha_file(Path(path))==sha,path
    assert p['environment']==N.ENV_CHANGES and p['engine_changes']==N.ENGINE_CHANGES
    assert E.sha_file(Path(p['pause_manifest']))==p['pause_manifest_sha256']
    assert E.sha_file(Path(p['reference']))==p['reference_sha256']
    return p


def check_smokes():
    p=checked_plan();ref=E.load(Path(p['reference']))
    paths=[R/f'lora_smoke_{i}.json' for i in range(2)]
    for i,path in enumerate(paths):
        x=E.load(path)
        assert x['gpu']==p['gpus'][i] and x['execution_plan_sha256']==E.sha_file(R/'plan.json')
        assert x['point_ids']==ref['point_ids'] and x['prompts_sha256']==ref['prompts_sha256']
        assert x['seeds']==ref['seeds'] and len(x['outputs'])==13
        assert x['outputs']==ref['outputs'],f'Exact old/current generation differs on card {i}'
    result=dict(passed=True,responses_per_card=26,exact_old_and_current_match=True,
        plan_sha256=E.sha_file(R/'plan.json'),reference_sha256=p['reference_sha256'],
        files={str(path):E.sha_file(path) for path in paths})
    dst=R/'smoke_verified.json'
    if dst.exists():assert E.load(dst)==result
    else:E.save_json(dst,result)


def execution_valid(x,path,point,plan,manifest):
    rel=str(path.relative_to(ROOT));ex=x['provenance']['execution']
    shard=int(D.digest('E4d-gpu-shard:'+point['point_id']),16)%2
    if rel in manifest:
        assert E.sha_file(path)==manifest[rel]
        old=E.load(PREV/'plan.json')
        gpu,sha=old['gpus'][shard],E.sha_file(PREV/'plan.json')
    else:gpu,sha=plan['gpus'][shard],E.sha_file(R/'plan.json')
    assert ex['gpu_uuid']==gpu and ex['execution_plan_sha256']==sha
    assert ex['source_sha256']==E.sha_file(H/'e4d.py')


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
    S.state('formal',children=[p.pid for p in children],data=str(E.DATA),reused_records=6248)
    while any(p.poll() is None for p in children):
        assert all(p.poll() in (None,0) for p in children),'Worker failed; other worker preserved'
        time.sleep(10)
    assert all(p.returncode==0 for p in children)


# Original science, worker ordering, analysis, and verification remain reused.
B.checked_plan=S.checked_plan=checked_plan
B.check_smokes=check_smokes
B.execution_valid=execution_valid
B.dual=dual
S.execute=execute
def wake(message):
    N._wake('【E4d 20261108设备续跑】当前控制目录 '+str(R)+
        '；正式数据仍为 '+str(E.DATA)+'，旧6248条已冻结复用。核验入口 '+
        str(Path(__file__).resolve())+' --verify。请只发布credit-grok，'+
        '不要重启旧队列或旧finalizer；H80032B未完成，不得提前关机。详情：'+message)
S.wake=wake


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--science':
        assert sys.argv[2] in ('run','armprobe','analyze','probe-analyze')
        assert all(os.environ.get(k)==v for k,v in N.ENV_CHANGES.items())
        checked_plan();check_smokes()
        if sys.argv[2] in ('run','armprobe'):N.install_engine_configuration()
        sys.argv=[str(H/'e4d.py'),*sys.argv[2:]];E.main()
    else:
        p=argparse.ArgumentParser();p.add_argument('--smoke',type=int,choices=[0,1]);p.add_argument('--worker',choices=['formal']);p.add_argument('--gpu',type=int,choices=[0,1]);p.add_argument('--verify',action='store_true');a=p.parse_args()
        if a.smoke is not None:
            assert all(os.environ.get(k)==v for k,v in N.ENV_CHANGES.items())
            checked_plan();N.install_engine_configuration();B.lora_smoke(a.smoke)
        elif a.worker:check_smokes();S.worker('formal',a.gpu)
        elif a.verify:B.verify()
        else:
            check_smokes();B.main()
