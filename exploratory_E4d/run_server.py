"""Isolated dual-card E4d queue; frozen scientific rules live in e4d.py."""
import argparse,fcntl,importlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
import e4d as E,e4d_common as D
H=Path(__file__).resolve().parent;ROOT=H.parent;R=H/'run'
HIN=Path('/root/autodl-tmp/forkaudit_hindsight/hindsight_judges')

def state(stage,**kw):
    E.save_json(R/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw));print(stage,kw,flush=True)

def wake(message):
    thread=os.environ.get('CODEX_THREAD_ID');assert thread
    with (R/'wake.log').open('a') as log:
        try:subprocess.run(['codex','queue','--thread',thread,'--resume','--message',message],stdout=log,stderr=subprocess.STDOUT,timeout=60)
        except Exception:traceback.print_exc(file=log)

def checked_plan():
    plan=E.load(R/'plan.json')
    assert all(E.sha_file(Path(p))==s for p,s in plan['source'].items()),'Execution source changed; review before resuming'
    import importlib.metadata as metadata
    assert all(metadata.version(k)==v for k,v in plan['versions'].items()),'Dependency version changed'
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip()
    assert inventory==plan['gpu_inventory'],'Hardware/driver changed; do not mix configurations'
    return plan

def env_for(i):
    p=checked_plan()
    return dict(os.environ,CUDA_VISIBLE_DEVICES=p['gpus'][i],E4D_EXECUTION_PLAN_SHA256=E.sha_file(R/'plan.json'),
                M0_MODEL=p['model'],OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
                VLLM_WORKER_MULTIPROC_METHOD='spawn')

def execute(key,args,gpu=None):
    checked_plan();assert not (R/'PAUSED.json').exists()
    with (R/(key+'.log')).open('a') as log:
        p=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,env=env_for(gpu) if gpu is not None else dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
    E.save_json(R/(key+'_process.json'),dict(pid=p.pid,args=list(map(str,args)),gpu=gpu,time=time.time()))
    rc=p.wait();assert rc==0,f'{key} failed with exit {rc}'

def smoke(i):
    plan=checked_plan();assert os.environ['CUDA_VISIBLE_DEVICES']==plan['gpus'][i]
    import alfenv,common
    from policy import Policy
    tr=next(t for t,p in E.base_episodes() if not t['won']);requests=[]
    g=alfenv.AlfGame(os.path.join(alfenv.ALFWORLD_DATA,tr['game_file']))
    try:
        for k in (0, min(29,len(tr['steps'])-1)):
            prefix,expected,before,mem=E.prefix_of(tr,k);obs,adm,won,div=g.refeed(prefix,expected)
            assert not div and obs==before
            requests.append(common.build_prompt(tr['task'],mem,obs,adm))
    finally:g.close()
    pol=Policy();start=time.time()
    try:outputs=pol.sample(requests,[D.SCREEN_OFFSET+987654+k for k in range(len(requests))],logprobs=False)
    finally:pol.close()
    E.save_json(R/f'smoke_{i}.json',dict(passed=True,gpu=plan['gpus'][i],prompts_sha256=D.digest(requests),outputs=outputs,seconds=time.time()-start))

def worker(phase,i):
    plan=checked_plan()
    with (R/f'gpu_{i}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args=['--workers',plan['workers_per_gpu'],'--max_concurrent',plan['max_concurrent_per_gpu']]
        shard=['--shard-count',2,'--shard-index',i]
        jobs=[]
        if phase=='smoke':jobs=[('smoke',[Path(__file__),'--smoke',i])]
        elif phase in ('pilot','screen'):
            jobs=[(phase,[H/'e4d.py','screen',*args,*shard,*(['--limit',20] if phase=='pilot' else [])])]
        elif phase=='first':
            for n,(arm,s) in enumerate((a,s) for a in D.ALL_ARMS for s in D.SEEDS):
                if n%2==i and not E.first_path(arm,s).exists():jobs.append((f'first_{arm}_{s}',[H/'e4d.py','first','--arm',arm,'--seed',s]))
        elif phase=='formal':
            for arm,cond in [('base','00')]+[(a,c) for a in D.ALL_ARMS for c in ('01','10','11')]:
                jobs.append((f'formal_{arm}_{cond}',[H/'e4d.py','run','--arm',arm,'--cond',cond,*args,*shard]))
            for arm in ('base',)+D.PRIMARY_ARMS:
                jobs.append((f'probe_{arm}',[H/'e4d.py','armprobe','--policy-id',arm,*args,*shard]))
        for key,command in jobs:
            E.save_json(R/f'{phase}_{i}_status.json',dict(stage='running',task=key,pid=os.getpid(),time=time.time()))
            execute(f'{key}_gpu{i}',command,i)
        E.save_json(R/f'{phase}_{i}_status.json',dict(stage='completed',pid=os.getpid(),time=time.time()))

def dual(phase):
    children=[]
    for i in range(2):
        with (R/f'{phase}_{i}_driver.log').open('a') as log:
            children.append(subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'--worker',phase,'--gpu',str(i)],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT))
    E.save_json(R/f'{phase}_launch.json',dict(pids=[p.pid for p in children],time=time.time()))
    state(phase,children=[p.pid for p in children])
    while any(p.poll() is None for p in children):
        assert all(p.poll() in (None,0) for p in children),'Credit-grok worker failed; preserve the other worker and inspect '+phase
        time.sleep(15)
    assert all(p.returncode==0 for p in children)

def pilot_budget():
    path=R/'pilot_budget.json'
    if path.exists():return
    points=E.load(E.SELECTED)['points'];pilot=sorted(points,key=lambda p:(D.digest('E4d-pi0-pilot:'+p['point_id']),p['point_id']))[:20]
    counts={};classes=[];records={}
    for p in pilot:
        f=E.DATA/'screen'/p['point_id']/'replay.json';x=E.load(f);E.verify_target_log(x)
        classes.append(D.classify([x['factual_outcomes']]+x['alt_outcomes']));records[p['point_id']]=E.sha_file(f)
    for c in ('Z3','V3','E3'):counts[c]=classes.count(c)
    timing=[]
    for i in range(2):
        rows=[json.loads(line) for line in (E.DATA/f'screen_runs_shard{i}.jsonl').read_text().splitlines()]
        timing.extend(x for x in rows if x['limit']==20)
    seconds=max(x['seconds'] for x in timing);nz=len(points)*counts['Z3']/len(pilot)
    E.save_json(path,dict(frozen_before_pi1=True,pilot_points=len(pilot),pilot_ids=[p['point_id'] for p in pilot],
        pilot_hashes=records,selected=len(points),classes=counts,screen_dual_seconds=seconds,
        rough_screen_extrapolated_hours=seconds*len(points)/len(pilot)/3600,
        expected_Z3_descriptive_only=nz,formal_continuations_per_state=312,
        expected_formal_continuations=312*nz,probe_continuations=72*5*8*3,probe_n=72,
        inclusion_rule='Every screened Z3, unchanged; no effect-dependent sample size adjustment',
        caution='Screen timing alone does not establish trained-policy throughput; no promise of 3-4 hours',time=time.time()))
    wake('【credit-grok 旧策略试跑完成】独立分支 credit-grok 的20点pi0试跑及成本估算已保存 '+str(path)+'。不改样本规则，继续全部pi0筛选、冻结Z3与72点探针，再运行pi1。不要和hindsight结果混合；最终两分支上传后才关机。')

def verify():
    plan=checked_plan();lock=E.locked();E.frozen_first_manifest(lock);pts=E.zpoints(lock)
    R0=importlib.import_module('11_replay');checked=0;raw={}
    def check_record(path,p,u,offset):
        nonlocal checked
        x=E.load(path);E.verify_target_log(x);c=x['continuations'];assert len(c)==1;c=c[0]
        assert c['rep']==u and not c['refeed_div'] and len(c['actions'])==len(c['valids'])
        assert c['len']==p['turn']+len(c['actions'])-1 and c['len']<=50
        assert len(c['seeds'])==len(c['actions'])-1
        assert c['seeds']==[R0.seed_of(p['seed'],p['turn'],1,u*100+p['turn']+k)+offset for k in range(len(c['seeds']))]
        assert x['factual_outcomes']==[int(c['won'])]
        execution=x['provenance']['execution'];shard=int(D.digest('E4d-gpu-shard:'+p['point_id']),16)%2
        assert execution['gpu_uuid']==plan['gpus'][shard] and execution['execution_plan_sha256']==E.sha_file(R/'plan.json')
        assert execution['source_sha256']==E.sha_file(H/'e4d.py')
        raw[str(path.relative_to(ROOT))]=E.sha_file(path);checked+=1
        return x
    for p in pts:
        for arm,cond in [('base','00')]+[(a,c) for a in D.ALL_ARMS for c in ('01','10','11')]:
            E.result_prob(arm,cond,p,lock)
            for u in range(24):
                x=check_record(D.run_path(E.DATA,arm,cond,p['point_id'],u),p,u,D.FORMAL_OFFSET)
                trained=cond in ('01','11');expected=lock['adapters'][f'{arm}_s{u//8}'] if trained else None
                assert x['provenance']['adapter_sha256']==expected
    by={p['point_id']:p for p in pts}
    for arm in ('base',)+D.PRIMARY_ARMS:
        E.read_probe_policy(arm,lock)
        for pid in lock['probe_ids']:
            for j in range(5):
                for u in range(8):check_record(E.probe_path(arm,pid,j,u),by[pid],u,D.PROBE_OFFSET)
    assert checked==312*len(pts)+3*5*8*len(lock['probe_ids'])
    assert all(E.sha_file(Path(p))==s for p,s in plan['protected_inputs'].items())
    E.save_json(R/'raw_manifest.json',raw)
    E.save_json(R/'verification.json',dict(passed=True,states=len(pts),games=lock['n_games_Z3'],records=checked,
        probe_points=len(lock['probe_ids']),raw_manifest_sha256=E.sha_file(R/'raw_manifest.json'),
        primary_sha256=E.sha_file(H/'e4d.json'),secondary_sha256=E.sha_file(H/'e4d_armprobe.json'),
        primary_reading=E.load(H/'e4d.json').get('reading'),secondary_reading=E.load(H/'e4d_armprobe.json')['reading_secondary'],
        paired_seeds_and_gpu_assignment_verified=True,protected_inputs_unchanged=True,source=plan['source']))

def main():
    with (R/'driver.lock').open('a') as l:
        fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=checked_plan();state('waiting_for_hindsight_gpu_work')
            while True:
                s=E.load(HIN/'run_next_slot_20261106/math_two_gpu_after_7b/controller_status.json')
                if s['stage']=='failed':raise RuntimeError('Hindsight failed; preserve both queues and request repair')
                if s['stage']=='completed_verified' and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():break
                time.sleep(30)
            assert all(E.sha_file(Path(p))==s for p,s in plan['protected_inputs'].items()),'Original input/model changed while queued'
            if not (R/'smoke_verified.json').exists():
                dual('smoke');a,b=[E.load(R/f'smoke_{i}.json') for i in range(2)]
                assert a['prompts_sha256']==b['prompts_sha256'] and a['outputs']==b['outputs'],'Cross-card base smoke differs; inspect before formal work'
                E.save_json(R/'smoke_verified.json',dict(passed=True,outputs_equal=True,model='base_only'))
            if not E.SELECTED.exists():
                state('candidates');execute('candidates',[H/'e4d.py','candidates','--workers',plan['workers_per_gpu']],0)
            if not E.LOCK.exists():
                if not (R/'pilot_budget.json').exists():dual('pilot');pilot_budget()
                dual('screen')
                state('freezing_pi0_cohort');execute('lock',[H/'e4d.py','lock','--approve','--probe-n',72])
            if not E.FIRST_MANIFEST.exists():dual('first');E.frozen_first_manifest(E.locked())
            dual('formal')
            state('analyzing')
            for cmd in ('analyze','probe-analyze'):execute(cmd,[H/'e4d.py',cmd])
            verify();state('completed_verified')
            wake('【credit-grok 完整实验已完成】请只在 credit-grok 分支读取 exploratory_E4d/e4d.json、e4d_armprobe.json 和 run/verification.json。主终点与次要O(h)探针分开解释，不混main/hindsight。联合收尾监控会分别上传两分支、核验后关机。')
        except Exception:
            error=traceback.format_exc();(R/'failure.log').write_text(error);state('failed',error=error)
            wake('【credit-grok 队列异常】请读 '+str(R/'failure.log')+' 并尽力修复续跑。仅改credit-grok，保留另一正常队列、原E4/E4b数据及门槛。用户授权两分支完成核验上传后关机；实在修不好保存日志结果并sync再关机，联合finalization监控负责。');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=('smoke','pilot','screen','first','formal'));p.add_argument('--gpu',type=int,choices=(0,1));p.add_argument('--smoke',type=int,choices=(0,1));p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.smoke is not None:smoke(a.smoke)
    elif a.worker:worker(a.worker,a.gpu)
    elif a.verify:verify()
    else:main()
