"""Operational continuation for the user-authorized E1c run. No shutdown."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import fcntl
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
import numpy as np

RUN=Path(__file__).resolve().parent;H=RUN.parent;ROOT=H.parent
sys.path.insert(0,str(H));import exp1c as E
OUT=H/'results/exp1c_p4'
ARMS=Path('/root/autodl-tmp/idea/exploratory_deep_discovery/exploratory_arms.parquet')
MODEL=Path('/root/autodl-tmp/models/Qwen2.5-32B-Instruct')
APPROVED=E.load(RUN/'authorization.json')['execution_head']

def command(args):return subprocess.check_output(list(map(str,args)),cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=180).strip()
def state(stage,**kw):
    E.save(RUN/'status.json',dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
    print(stage,kw,flush=True)
def identity(pid):
    try:
        a=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if a[0]=='Z' else a[19]
    except FileNotFoundError:return None
def hardware():
    return dict(gpu=command(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader']),
                versions={k:importlib.metadata.version(k) for k in ('torch','transformers','numpy')})
def stage(label,*args):
    state(label,command=list(map(str,args)))
    with (RUN/(label+'.log')).open('a') as f:
        subprocess.run([sys.executable,'-u',str(H/'exp1c.py'),*map(str,args)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
                       env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false'))

def statistic(values,trajectories):
    v=np.asarray(values,float);names=sorted(set(trajectories));tr=np.asarray(trajectories)
    assert len(v) and np.isfinite(v).all()
    sums=np.array([sum(v[tr==name]) for name in names]);counts=np.array([sum(tr==name) for name in names])
    w=np.random.default_rng(20261103).multinomial(len(names),[1/len(names)]*len(names),size=4000)
    samples=np.array([float(np.dot(row,sums)/np.dot(row,counts)) for row in w])
    return dict(n=len(v),n_trajectories=len(names),estimate=float(v.mean()),ci95=np.percentile(samples,[2.5,97.5]).tolist())

def verify():
    jobs,audit,man=E.prepared(OUT);p=OUT/'32b';run=E.load(p/'run.json')
    assert run['complete'] and run['contexts']==3474 and E.file_sha(p/'scores.jsonl')==run['scores_sha256']
    assert run['script_sha256']==E.file_sha(H/'exp1c.py') and run['protocol_sha256']==E.file_sha(H/'EXPERIMENT1C_DRAFT.md')
    assert run['chunked_backend_sha256']==E.file_sha(H/'exp1c_backend.py') and run['chunk']==1024
    import exp1c_precision as precision
    assert run['precision']==precision.POLICY
    assert run['precision_backend_sha256']==E.file_sha(H/'exp1c_precision.py')
    assert run['precision_amendment_sha256']==E.file_sha(H/'EXECUTION_AMENDMENT_1C_FP16_20261103.txt')
    pc=E.load(p/'precision_check.json')
    assert pc['policy']==precision.POLICY and pc['buffers_unchanged']
    assert pc['transformer_dtype']=='torch.float16' and pc['head_dtype']=='torch.float32'
    assert not any(run['fingerprint'][k] for k in ('matmul_tf32','cudnn_tf32','fp16_reduced_precision_reduction'))
    checks=E.load(p/'numerical_check.json');assert all(c['passed'] and c['tv']<.005 for c in checks)
    assert sum(c['kind']=='chunked_shared_prefix_vs_chunked_direct' for c in checks)==10
    assert sum(c['kind']=='chunked_vs_frozen_one_pass' for c in checks)>=3
    rows=[json.loads(line) for line in (p/'scores.jsonl').read_text().splitlines()]
    scores={r['request_id']:r for r in rows};assert len(scores)==len(rows)==len(jobs)==3474
    for j in jobs:
        r=scores[j['request_id']];assert all(r[k]==j[k] for k in ('point_id','condition','request_sha256'))
        lp=np.asarray(r['sequence_logp']);probs=np.exp(lp-lp.max());probs/=probs.sum()
        assert np.isfinite(lp).all() and lp.shape==(5,) and max(abs(probs-np.asarray(r['candidate_prob'])))<1e-12
        assert r['argmax']==int(probs.argmax())
    arms=E.read_arms(ROOT,ARMS,man['source_sha256']['P4-ARMS']);vals=[[],[],[]];traj=[[],[],[]]
    for a in audit:
        pid=a['point_id'];b=a['swap_arm'];t=a['traj'];others=[j for j in range(5) if j not in (0,b)]
        s={c:scores[pid+'__'+c] for c in E.CONDITIONS};ll={c:np.asarray(s[c]['sequence_logp']) for c in s}
        vals[0].append(float(((ll['DE'][0]-ll['DE'][b])-(ll['DX'][0]-ll['DX'][b]))/2));traj[0].append(t)
        for cond,arm,outcome in [('D1F',0,a['factual_outcome']),('D1B',b,a['contrast_outcome'])]:
            if outcome==0:
                change=ll[cond]-ll['M'];vals[1].append(float(change[arm]-sum(change[o] for o in others)/3));traj[1].append(t)
        q=[arms[pid][j]['q3456'] for j in range(5)];assert np.isfinite(q).all()
        assert abs((arms[pid][b]['q12']-arms[pid][0]['q12'])-a['q12_gap'])<1e-8
        assert b==sorted(range(1,5),key=lambda j:(-arms[pid][j]['q12'],j))[0]
        if a['q12_gap']>=1/3-1e-8:
            vals[2].append(sum((s['DE']['candidate_prob'][j]-s['D0']['candidate_prob'][j])*q[j] for j in range(5)));traj[2].append(t)
    result=E.load(p/'analysis.json');names=[('all','dual_pointer_swap_nat'),('all','single_branch_pull_failed_nat'),('bad_q12_ge_third','dual_true_label_minus_anon_value')]
    rebuilt=[statistic(v,t) for v,t in zip(vals,traj)]
    for expected,(group,name) in zip(rebuilt,names):
        got=result['metrics'][group][name]
        assert expected['n']==got['n'] and expected['n_trajectories']==got['n_trajectories']
        assert abs(expected['estimate']-got['estimate'])<1e-10 and max(abs(np.array(expected['ci95'])-got['ci95']))<1e-10
    expected=dict(H1_identity_label_under_symmetric_two_branch_evidence=rebuilt[0]['ci95'][0]>0,
                  H2_pull_toward_displayed_failed_branch=rebuilt[1]['ci95'][0]>0,
                  H3_true_label_value_harm_on_q12_worse=rebuilt[2]['ci95'][1]<0,bootstrap_seed=20261103,nboot=4000)
    assert expected==result['decisions']
    for path,digest in E.load(RUN/'protected_inputs.json').items():assert E.file_sha(Path(path))==digest,path
    assert hardware()==E.load(RUN/'hardware.json')
    E.save(OUT/'verification.json',dict(passed=True,contexts=3474,points=386,primary_endpoints_independently_verified=3,
        primary_statistics=rebuilt,decisions=expected,scores_sha256=run['scores_sha256'],analysis_sha256=E.file_sha(p/'analysis.json'),
        source_and_prior_E1_unchanged=True,hardware_unchanged=True))

def publish():
    files=[p for p in OUT.rglob('*') if p.is_file()]
    E.save(OUT/'artifact_manifest.json',{str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=E.file_sha(p)) for p in files})
    files=list(OUT.rglob('*'));files=[p for p in files if p.is_file()]
    files += [RUN/x for x in ['driver.py','authorization.json','hardware.json','protected_inputs.json','cpu_gate.json','lengths.log','prepare.log','preflight.log']]
    assert all(p.stat().st_size<40*1024**2 for p in files)
    assert command(['git','branch','--show-current'])=='research/hindsight-judges'
    assert command(['git','remote','get-url','--push','origin'])=='https://github.com/melancholiac99/ForkAudit.git'
    assert not command(['git','diff','--name-only']) and not command(['git','diff','--cached','--name-only'])
    command(['git','add','-f','--',*[str(p.relative_to(ROOT)) for p in files]])
    if command(['git','diff','--cached','--name-only']):command(['git','commit','-m','Publish verified Hindsight 1c results, reconstruction and numerical audits'])
    head=command(['git','rev-parse','HEAD'])
    for attempt in range(3):
        try:
            command(['git','push','origin','HEAD:refs/heads/research/hindsight-judges'])
            assert command(['git','ls-remote','--heads','origin','research/hindsight-judges']).split()[0]==head
            E.save(RUN/'publication.json',dict(verified=True,branch='research/hindsight-judges',commit=head));return
        except Exception:
            if attempt==2:raise
            time.sleep(20)

def main():
    with (RUN/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            state('cpu_prepare_reused')
            assert (OUT/'manifest.json').exists(),'CPU prepare did not finish; inspect prepare.log'
            jobs,audit,man=E.prepared(OUT)
            common=['--repo',ROOT,'--out',OUT,'--arms',ARMS]
            stage('lengths','lengths',*common,'--model',MODEL)
            E.save(RUN/'cpu_gate.json',dict(passed=True,points=len(audit),requests=len(jobs),conditions=list(E.CONDITIONS),
                manifest_sha256=E.file_sha(OUT/'manifest.json'),requests_sha256=man['requests_sha256'],audit_sha256=man['audit_sha256'],
                source_commit=APPROVED,no_new_policy_sampling=True))
            assert command(['git','rev-parse','HEAD'])==APPROVED
            protected={str(p):E.file_sha(p) for p in [H/'exp1c.py',H/'exp1c_backend.py',H/'EXPERIMENT1C_DRAFT.md',ARMS,
                H/'exp1c_precision.py',H/'EXECUTION_AMENDMENT_1C_FP16_20261103.txt',RUN/'driver.py',
                ROOT/'exploratory_P4/p4_e2_backend.py',*sorted((H/'results/exp1_p4').rglob('*'))] if p.is_file()}
            protected.update({str(OUT/name):E.file_sha(OUT/name) for name in ('manifest.json','audit.json','requests.jsonl.gz')})
            E.save(RUN/'protected_inputs.json',protected);E.save(RUN/'hardware.json',hardware())
            assert not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']),'GPU busy'
            stage('score','score',*common,'--model',MODEL,'--approved-commit',APPROVED)
            stage('analyze','analyze',*common);state('verifying');verify()
            state('completed_verified');publish();state('completed_published',publication=E.load(RUN/'publication.json'),shutdown=False)
        except BaseException as exc:
            state('failed',error=str(exc));traceback.print_exc()
            if os.environ.get('CODEX_THREAD_ID'):
                try:subprocess.run(['codex','queue','--thread',os.environ['CODEX_THREAD_ID'],'--resume','--message',
                    f'【Hindsight 1c 异常检查】用户要求检查并运行 1c；请读 {RUN}/status.json 和最新日志并尽力修复。原分支 research/hindsight-judges。预注册门槛不放宽，原始轨迹/旧实验结果不修改，精度或分块等必要修订先记录，别静默混用分数。此新任务没有关机授权。'],cwd=ROOT,timeout=45)
                except Exception:traceback.print_exc()
            raise

if __name__=='__main__':main()
