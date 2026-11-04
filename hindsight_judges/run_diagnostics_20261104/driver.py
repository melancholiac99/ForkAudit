"""Only action-correction and Think-localization diagnostics. No HLN/OPD/shutdown."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['HF_HUB_OFFLINE']='1'
from pathlib import Path
import fcntl,json,subprocess,sys,time,traceback
import numpy as np

RUN=Path(__file__).resolve().parent;H=RUN.parent;ROOT=H.parent
sys.path.insert(0,str(H));import exp2_common as C
import exp1c as E
A=H/'results/action_diagnostic';L=H/'results/think_localization'
ARMS=Path('/root/autodl-tmp/idea/exploratory_deep_discovery/exploratory_arms.parquet')
MODELS=Path('/root/autodl-tmp/models')

def command(*args):return subprocess.check_output(list(map(str,args)),cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=120).strip()
def state(stage,**kw):
    C.save_json(RUN/'status.json',dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
    print(stage,kw,flush=True)
def wake(message):
    if os.environ.get('CODEX_THREAD_ID'):
        with (RUN/'wake.log').open('a') as f:
            subprocess.run(['codex','queue','--thread',os.environ['CODEX_THREAD_ID'],'--resume','--message',message],stdout=f,stderr=subprocess.STDOUT,timeout=60)

def independent_ci(values,clusters,nboot):
    v=np.asarray(values,float);names=sorted(set(clusters));idx={n:i for i,n in enumerate(names)}
    groups=np.array([idx[n] for n in clusters]);s=np.bincount(groups,weights=v);n=np.bincount(groups)
    w=np.random.default_rng(20261103).multinomial(len(names),np.full(len(names),1/len(names)),size=nboot)
    dist=(w@s)/(w@n)
    return dict(n=len(v),n_trajectories=len(names),estimate=float(v.mean()),ci95=np.percentile(dist,[2.5,97.5]).tolist())
def same(got,want):
    assert got['n']==want['n'] and got['n_trajectories']==want['n_trajectories']
    assert abs(got['estimate']-want['estimate'])<1e-10
    assert np.max(abs(np.array(got['ci95'])-want['ci95']))<1e-10

def verify():
    aj,ah=C.decode_rows(A/'requests.jsonl.gz');lj,lh=C.decode_rows(L/'requests.jsonl.gz')
    assert len(aj)==3088 and len(lj)==1544
    assert C.read_json(A/'prepare.json')['skipped']==[]
    maps={};counts={}
    for label,base,jobs,idkey,hashkey,probkey in [
        ('7b',A/'7b/full',aj,'id','sha256','prob'),
        ('32b',A/'32b/full',aj,'id','sha256','prob'),
        ('localization',L/'32b',lj,'request_id','request_sha256','candidate_prob')]:
        run=C.read_json(base/'run.json');scores=base/'scores.jsonl'
        assert run['complete'] and C.sha_file(scores)==run['scores_sha256']
        gates=C.read_json(base/'numerical_gate.json')
        assert len(gates)==2 and all(x['passed'] and x['tv']<.005 for x in gates)
        rows=[json.loads(x) for x in scores.read_text().splitlines()]
        byid={r[idkey]:r for r in rows};assert len(byid)==len(rows)==len(jobs)
        for j in jobs:
            r=byid[j[idkey]];assert r[hashkey]==j[hashkey]
            assert r['point_id']==j['point_id'] and r['condition']==j['condition']
            if label!='localization':assert r['think']==j['think']
            ll=np.array(r['sequence_logp']);p=np.array(r[probkey]);assert ll.shape==p.shape==(5,)
            assert np.isfinite(ll).all() and np.isfinite(p).all()
            expected=np.exp(ll-ll.max());expected/=expected.sum()
            assert np.max(abs(expected-p))<1e-12
        maps[label]=byid;counts[label]=len(rows)
    arms_hash=C.read_json(ROOT/'exploratory_P4/e2.json')['inputs']['exploratory_deep_discovery/exploratory_arms.parquet']
    arms=E.read_arms(ROOT,ARMS,arms_hash)
    meta=C.read_json(H/'results/exp1_p4/pointmeta.json');pids=sorted(x['point_id'] for x in meta)
    tr={x['point_id']:x['traj'] for x in meta};clusters=[tr[p] for p in pids]
    q={p:np.array([arms[p][k]['q3456'] for k in range(5)]) for p in pids}
    assert len(pids)==386 and all(np.isfinite(x).all() for x in q.values())
    rebuilt={};analysis=C.read_json(A/'analysis_full.json')
    assert analysis['n_points']==386
    for name in ['7b','32b']:
        for cond in ['M','L','LN','F']:
            vals=[float((np.array(maps[name][p+'__'+cond+'__drop']['prob'])-
                         np.array(maps[name][p+'__'+cond+'__keep']['prob']))@q[p]) for p in pids]
            key=name+'_'+cond+'_drop_minus_keep_teacher_value';rebuilt[key]=independent_ci(vals,clusters,2000)
            same(analysis['metrics'][key],rebuilt[key])
    analysis=C.read_json(L/'analysis.json');assert analysis['n_points']==386
    for cond in ['DROP_T','KEEP_T','SWAP_T']:
        vals=[float((np.array(maps['localization'][p+'__'+cond]['candidate_prob'])-
                     np.array(maps['localization'][p+'__L_RE']['candidate_prob']))@q[p]) for p in pids]
        key=cond+'_minus_LRE_value';rebuilt[key]=independent_ci(vals,clusters,4000)
        same(analysis[key],rebuilt[key])
    for path,digest in C.read_json(RUN/'protected_inputs.json').items():assert C.sha_file(Path(path))==digest,path
    env=C.read_json(RUN/'environment.json')
    assert command('nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.total','--format=csv,noheader').splitlines()==env['gpus']
    report=dict(passed=True,points=386,counts=counts,independent_value_contrasts=rebuilt,
                prior_results_and_sources_unchanged=True,hardware_unchanged=True,
                hln_run=False,opd_run=False,shutdown=False)
    C.save_json(RUN/'verification.json',report)
    return report

def analyze(script,out):
    with (RUN/(script+'_analyze.log')).open('a') as f:
        subprocess.run([sys.executable,'-u',str(H/(script+'.py')),'analyze','--out',str(out),'--arms',str(ARMS)],
                       cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    with (RUN/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            env=C.read_json(RUN/'environment.json')
            assert command('git','rev-parse','HEAD')==env['git_commit']
            assert command('git','branch','--show-current')=='research/hindsight-judges'
            assert not command('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader')
            assert len(env['gpus'])==2
            uuids=[line.split(',')[1].strip() for line in env['gpus']]
            active={};done={};pending=True;failures=[];launches={}
            def start(name,card):
                script='think_localization.py' if name=='localization' else 'exp3_action_diagnostic.py'
                model=MODELS/('Qwen2.5-7B-Instruct' if name=='7b' else 'Qwen2.5-32B-Instruct')
                out=L if name=='localization' else A
                target=out/'32b' if name=='localization' else out/name/'full'
                assert not target.exists(),'Fresh run required; inspect existing diagnostic artifacts before resume'
                cmd=[sys.executable,'-u',str(H/script),'score','--out',str(out),'--model',str(model)]
                if name!='localization':cmd+=['--name',name]
                log=(RUN/(name+'_score.log')).open('a')
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuids[card]))
                active[name]=(proc,log,card)
                launches[name]=dict(pid=proc.pid,gpu_index=card,gpu_uuid=uuids[card],command=cmd,started=time.time())
                C.save_json(RUN/'jobs.json',launches)
            start('7b',0);start('32b',1)
            state('scoring',jobs=launches)
            while active:
                for name,(proc,log,card) in list(active.items()):
                    code=proc.poll()
                    if code is None:continue
                    log.close();del active[name];done[name]=code
                    launches[name].update(returncode=code,ended=time.time());C.save_json(RUN/'jobs.json',launches)
                    if code:
                        failures.append(name)
                        wake('【Hindsight 诊断异常】只跑诊断，不跑 HLN/OPD/新轨迹，也没有本任务关机授权。'+name+
                             ' 失败，请读 '+str(RUN)+' 的 status.json、jobs.json 和评分日志，保留其他卡正在跑的任务，尽力修复；门槛不放宽，不混用配置。')
                    if pending:
                        start('localization',card);pending=False
                    state('scoring' if not failures else 'scoring_with_failures',jobs=launches,failures=failures)
                if active:time.sleep(20)
            assert not failures,'Diagnostic scoring failures: '+','.join(failures)
            state('analyzing');analyze('exp3_action_diagnostic',A);analyze('think_localization',L)
            state('verifying');report=verify();state('completed_verified',counts=report['counts'],shutdown=False)
            os.sync()
            wake('【Hindsight 诊断完成】按用户要求只跑诊断，未运行 HLN/OPD 或新轨迹。动作诊断 7B/32B 各3088条，Think定位1544条已完成、分析并独立核验。请读 '+str(RUN/'verification.json')+
                 '、'+str(A/'analysis_full.json')+'、'+str(L/'analysis.json')+'，检查后向用户汇报结果。本次没有关机授权。')
        except BaseException as exc:
            state('failed',error=str(exc));traceback.print_exc();os.sync()
            wake('【Hindsight 诊断队列异常】用户只跑诊断、不跑HLN。请读 '+str(RUN/'status.json')+' 和日志，检查并尽力修复。先检查其他GPU进程，不要重复启动，不改变门槛或混用分数；本次无关机授权。')
            raise

if __name__=='__main__':main()
