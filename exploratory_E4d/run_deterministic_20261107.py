"""Isolated E4d continuation restart after reproducibility failure."""
import argparse, os, subprocess, sys, time
from pathlib import Path
import resume_6000d_20261107 as B
E, D, S, H, ROOT = B.E, B.D, B.S, B.H, B.ROOT
R = H / 'run_deterministic_20261107'
DATA = H / 'data_deterministic_20261107'
B.R = S.R = R
E.DATA = DATA
E.CANDIDATES = DATA / 'candidates.json'
E.SELECTED = DATA / 'selected.json'
E.LOCK = DATA / 'lock.json'
E.FIRST_MANIFEST = DATA / 'first_manifest.json'
E.PROBE_REGISTRY = DATA / 'probe_registrations'
ENV_CHANGES = {'VLLM_BATCH_INVARIANT':'1', 'PYTHONHASHSEED':'0', 'VLLM_USE_V2_MODEL_RUNNER':'0'}
ENGINE_CHANGES = {'enforce_eager':True, 'async_scheduling':False, 'enable_prefix_caching':False}


def install_engine_configuration():
    # Defer vLLM import/initialization until the original stage has forked its
    # CPU EnvPool. Importing the GPU engine before that boundary is unsafe.
    from e4_policy import LoraPolicy
    original_init=LoraPolicy.__init__
    def registered_init(self, *args, **kwargs):
        import vllm
        original=vllm.LLM
        class RegisteredEngine(original):
            def __init__(self, *engine_args, **engine_kwargs):
                engine_kwargs.update(ENGINE_CHANGES)
                super().__init__(*engine_args, **engine_kwargs)
        vllm.LLM=RegisteredEngine
        try: original_init(self, *args, **kwargs)
        finally: vllm.LLM=original
    LoraPolicy.__init__=registered_init


def env_for(i):
    return dict(B.env_for(i), **ENV_CHANGES)


def check_smokes():
    plan = B.checked_plan()
    gate = E.load(R / 'smoke_verified.json')
    assert gate['passed'] and gate['responses'] == 26 and gate['environment'] == ENV_CHANGES
    assert gate['engine_changes'] == ENGINE_CHANGES
    for name, sha in gate['files'].items(): assert E.sha_file(Path(name)) == sha
    assert E.sha_file(R / 'smoke_verified.json') == plan['smoke_verification_sha256']
    a, b, c = [E.load(Path(gate[k])) for k in ('card0','card1','repeat')]
    assert a['prompts_sha256'] == b['prompts_sha256'] == c['prompts_sha256']
    assert a['seeds'] == b['seeds'] == c['seeds']
    assert len(a['outputs']) == 13 and a['outputs'] == b['outputs'] == c['outputs']


def execute(key, args, gpu=None):
    B.checked_plan(); check_smokes()
    assert not (R / 'PAUSED.json').exists()
    assert str(args[0]) == str(H / 'e4d.py'), 'Only original science entrypoint can be routed'
    assert str(args[1]) in ('run', 'armprobe', 'analyze', 'probe-analyze')
    env = env_for(gpu) if gpu is not None else dict(os.environ, CUDA_VISIBLE_DEVICES='', **ENV_CHANGES)
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--science', *map(str,args[1:])]
    with (R / (key+'.log')).open('a') as log:
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    E.save_json(R / (key+'_process.json'), dict(pid=p.pid,args=command,gpu=gpu,time=time.time()))
    assert p.wait() == 0, key+' failed; preserve remaining worker'


def dual(phase):
    assert phase == 'formal'
    children=[]
    for i in range(2):
        with (R / f'formal_{i}_driver.log').open('a') as log:
            children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                '--worker','formal','--gpu',str(i)],cwd=ROOT,env=env_for(i),stdout=log,stderr=subprocess.STDOUT))
    E.save_json(R/'formal_launch.json',dict(pids=[p.pid for p in children],time=time.time()))
    S.state('formal',children=[p.pid for p in children],data=str(DATA),execution='batch_invariant')
    while any(p.poll() is None for p in children):
        assert all(p.poll() in (None,0) for p in children), 'Worker failed; do not stop healthy worker'
        time.sleep(10)
    assert all(p.returncode == 0 for p in children)


S.env_for=env_for
S.execute=execute
B.dual=dual
B.check_smokes=check_smokes
_wake=S.wake
def wake(message):
    _wake('【E4d确定性执行修订】当前唯一正式结果目录为 '+str(DATA)+
          '；旧data的5622条仅存档，不纳入主分析。核验命令为 '+
          str(Path(__file__).resolve())+' --verify。请阅读 DETERMINISTIC_RESTART_20261107.txt，'+
          '使用新run目录的核验清单发布credit-grok，不要启动旧finalize_server.py。'+message)
S.wake=wake


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--science':
        assert sys.argv[2] in ('run','armprobe','analyze','probe-analyze')
        assert all(os.environ.get(k)==v for k,v in ENV_CHANGES.items())
        B.checked_plan();check_smokes()
        if sys.argv[2] in ('run','armprobe'): install_engine_configuration()
        sys.argv=[str(H/'e4d.py'),*sys.argv[2:]]
        E.main()
    else:
        p=argparse.ArgumentParser();p.add_argument('--worker',choices=['formal']);p.add_argument('--gpu',type=int,choices=[0,1]);p.add_argument('--verify',action='store_true');a=p.parse_args()
        if a.worker:
            check_smokes();S.worker('formal',a.gpu)
        elif a.verify: B.verify()
        else: B.main()
