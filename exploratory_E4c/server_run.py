"""Operational E4c preparation, sequential execution, audit and main publication.
Frozen E4c scientific scripts are unchanged. No shutdown action.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import fcntl
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tarfile
import time
import traceback
import numpy as np
import e4c_common as C

H=Path(__file__).resolve().parent;ROOT=H.parent;RUN=H/'run'
E4=ROOT/'exploratory_E4'
UP=Path('/root/autodl-tmp/forkaudit_hindsight/hindsight_judges')

def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def cmd(args,cwd=ROOT):return subprocess.check_output(list(map(str,args)),cwd=cwd,stderr=subprocess.STDOUT,text=True,timeout=240).strip()
def status(stage,**kw):save(RUN/'status.json',dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw))
def idle():return not cmd(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'])

def preflight():
    RUN.mkdir(exist_ok=True);g=read(E4/'games.json');manifest=read(E4/'run/final_artifact_manifest.json')
    frozen={};references=0
    def protect(p,expected=None):
        p=Path(p).resolve();actual=sha(p)
        if expected:assert actual==expected,str(p)
        frozen[str(p)]=actual
    old=read(ROOT/'exploratory_E4b/run/protected_inputs.json')
    for split in ('valid_unseen','valid_seen'):
        for tag in ['base']+C.TRAINED_REF['A']+C.TRAINED_REF['D']:
            folder=E4/'data/rollout'/split/tag
            paths=[folder/'summary.json']+[folder/f's{x["seed"]}_r{r}.json' for x in g[split] for r in range(3)]
            for p in paths:
                info=manifest[str(p.relative_to(E4))];protect(p,info['sha256']);references+=1
            sm=read(folder/'summary.json');assert sm['n']==len(g[split])*3
            assert set(sm['per_game'])=={str(x['seed']) for x in g[split]}
        for game in g[split]:
            folder=(Path(g['alfworld_data'])/game['game_file']).parent
            for p in folder.iterdir():
                if p.is_file():protect(p,old.get(str(p.resolve())))
    for p in [E4/'games.json',E4/'e4_common.py',E4/'e4_01_rollout.py',E4/'e4_policy.py',
              *(ROOT/'scripts'/x for x in ('common.py','policy.py','alfenv.py','envpool.py','11_replay.py')),
              ROOT/'prompts/hcapo_alfworld_template.txt']:
        protect(p,old.get(str(p.resolve())))
    model=Path(C.common.MODEL_PATH);modelrun=read(ROOT/'exploratory_P4/e2_7b/run.json')
    for name,digest in modelrun['model_sha256'].items():protect(model/name,digest)
    env=read(ROOT/'exploratory_P4/e2_environment.json')
    for name,digest in env['model_metadata_hashes']['7B'].items():protect(model/name,digest)
    for p in [*H.glob('*.py'),H/'run_e4c.sh',H/'PREREG_E4c.md']:
        if p.name!='server_run.py':
            original=subprocess.check_output(['git','show',f'1d8d6f5:{p.relative_to(ROOT)}'],cwd=ROOT)
            if p.name=='test_e4c.py':
                fix=read(RUN/'test_isolation_fix.json')
                assert hashlib.sha256(original).hexdigest()==fix['original_sha256'] and sha(p)==fix['fixed_sha256']
            else:assert hashlib.sha256(original).hexdigest()==sha(p),str(p)
        protect(p)
    with (RUN/'cpu_preflight.log').open('w') as log:
        subprocess.run([sys.executable,str(H/'test_e4c.py')],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
    goals=cmd([sys.executable,H/'s0_checks.py','goals'],cwd=H)
    assert 'unparsed games: 0' in goals
    (RUN/'s0_goals_preflight.log').write_text(goals+'\n')
    save(RUN/'protected_inputs.json',frozen)
    save(RUN/'preflight.json',dict(passed=True,prereg_commit='330b735',source_commit='1d8d6f5',reference_files=references,
        protected_files=len(frozen),expected_episodes=4110,conditions=C.CONDITIONS,
        split_games={s:len(g[s]) for s in ('valid_unseen','valid_seen')},goals_check=goals,
        versions={k:importlib.metadata.version(k) for k in ('torch','transformers','vllm','numpy','alfworld')},
        model=str(model),shutdown=False,publication_branch='main'))
    print('CPU preflight passed',references,'original reference files verified',flush=True)

def protected():
    assert read(RUN/'preflight.json')['passed']
    for p,digest in read(RUN/'protected_inputs.json').items():assert sha(p)==digest,p

def equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b),(set(a),set(b))
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(float,np.floating)):assert abs(a-b)<1e-10,(a,b)
    else:assert a==b,(a,b)

def blocked(cond,a,prev):
    norm=lambda s:' '.join(s.lower().split())
    repeat=prev is not None and norm(a)==norm(prev)
    return (repeat if cond in ('R','PR') else repeat or bool(re.match(r'^(examine|look|inventory)\b',norm(a))) if cond=='X' else False)

def behaviour(eps):
    n=inspect=goto=repeat=seen=known=won_seen=wins=failures=timeouts=0
    for ep in eps:
        actions=[' '.join(s['action'].lower().split()) for s in ep['steps']]
        n+=len(actions);inspect+=sum(bool(re.match(r'^(examine|look|inventory)\b',a)) for a in actions)
        goto+=sum(a.startswith('go to') for a in actions);repeat+=sum(a==b for a,b in zip(actions,actions[1:]))
        word=C.E.target_word(ep['task'])
        if word is not None:
            known+=1;rx=C.E.target_regex(word)
            found=any(rx.search(t.lower()) for t in [ep['initial_obs']]+[s['obs_after'] for s in ep['steps']])
            seen+=bool(found);won_seen+=bool(found and ep['won'])
        wins+=bool(ep['won']);failures+=not ep['won'];timeouts+=not ep['won'] and len(actions)>=50
    return dict(episodes=len(eps),steps=n,inspect_share=inspect/max(n,1),goto_share=goto/max(n,1),repeat_share=repeat/max(n,1),
        target_seen=seen/max(known,1),success_given_seen=won_seen/max(seen,1),success=wins/max(len(eps),1),timeout_share_of_failures=timeouts/max(failures,1))

def verify():
    protected();g=read(E4/'games.json');result=read(H/'e4c.json');total=0
    equal(result['delta_pp'],4.0);equal(result['hint'],C.HINT);equal(result['k_resample'],8)
    equal(result['trained_range'],dict(inspect_max=.25,repeat_max=.04))
    assert set(result['splits'])=={'valid_unseen','valid_seen'}
    for split in ('valid_unseen','valid_seen'):
        gs=g[split];matrices={};all_eps={};r=result['splits'][split];assert r['missing']==[]
        entries=[(c,[c],H/'data') for c in C.CONDITIONS]+[(a,tags,E4/'data') for a,tags in C.TRAINED_REF.items()]+[('base_E4',['base'],E4/'data')]
        for name,tags,data in entries:
            eps=[];cols=[]
            for tag in tags:
                folder=data/'rollout'/split/tag;sm=read(folder/'summary.json');matrix=[]
                assert sm['n']==len(gs)*3 and set(sm['per_game'])=={str(x['seed']) for x in gs}
                assert len(list(folder.glob('s*_r*.json')))==len(gs)*3
                for game in gs:
                    vals=[]
                    for rep in range(3):
                        ep=read(folder/f's{game["seed"]}_r{rep}.json');eps.append(ep);vals.append(int(ep['won']))
                        assert ep['seed']==game['seed'] and ep['rep']==rep and ep['game_file']==game['game_file']
                        assert ep['n_turns']==len(ep['steps'])<=50
                        if data==H/'data':
                            total+=1;assert ep['condition']==name
                            prov=ep['provenance'];assert prov['games_sha256']==sha(E4/'games.json') and prov['k_resample']==8
                            assert prov['hint']==(C.HINT if name in ('P','PR') else None) and prov['vllm']!='dry'
                            prev=None
                            for turn,s in enumerate(ep['steps'],1):
                                i=s['draw_index'];assert s['turn']==turn and 0<=i<=8 and s['n_draws']==i+1
                                assert s['sample_seed']==1000000*game['seed']+turn+10**11*(rep+1)+i*10**13
                                assert s['kept_blocked']==blocked(name,s['action'],prev)
                                assert len(s['blocked'])==i+int(s['kept_blocked'])
                                assert all(blocked(name,a,prev) for a in s['blocked'])
                                if s['kept_blocked']:assert i==8 and s['blocked'][-1]==s['action']
                                if name in ('base','P'):assert i==0 and s['blocked']==[]
                                prev=s['action']
                    equal(vals,sm['per_game'][str(game['seed'])]);matrix.append(vals)
                arr=np.asarray(matrix,float);equal(float(arr.mean()),sm['success']);cols.append(arr)
            matrices[name]=np.concatenate(cols,axis=1);all_eps[name]=eps
        equal({k:float(100*v.mean()) for k,v in matrices.items()},r['success'])
        equal({k:behaviour(v) for k,v in all_eps.items()},r['behaviour'])
        rng=np.random.default_rng(31102026);indices=rng.integers(0,len(gs),size=(10000,len(gs)))
        pairs=[(c,ref) for ref in ('base','A','D') for c in ('P','R','PR','X')]+[('A','base'),('D','base'),('A','D'),('base','base_E4')]
        rebuilt={}
        for a,b in pairs:
            d=matrices[a].mean(1)-matrices[b].mean(1);ci=np.percentile(d[indices].mean(1),[2.5,97.5])*100
            rebuilt[a+'-'+b]=dict(diff=float(d.mean()*100),ci95=ci.tolist(),n_games=len(gs))
        equal(rebuilt,r['diff'])
        redraw={}
        for c in C.CONTROLS:
            steps=[s for ep in all_eps[c] for s in ep['steps']]
            redraw[c]=dict(steps=len(steps),redrawn_share=sum(s['draw_index']>0 for s in steps)/len(steps),fallback_blocked_share=sum(s['kept_blocked'] for s in steps)/len(steps))
        equal(redraw,r['redraw'])
    u=result['splits']['valid_unseen'];d=u['diff']
    ok={c:u['behaviour'][c]['inspect_share']<=.25 and u['behaviour'][c]['repeat_share']<=.04 for c in C.CONTROLS}
    reading=('E4c-SUBSTITUTE' if any(d[c+'-A']['ci95'][0]>-4 for c in C.CONTROLS) else
             'E4c-HEURISTIC-ONLY' if any(d[c+'-D']['ci95'][0]>-4 for c in C.CONTROLS) else
             'E4c-NOMANIP' if not any(ok.values()) else
             'E4c-LEARNED' if all(d[c+'-D']['ci95'][1]<0 for c in C.CONTROLS) else 'E4c-MIXED')
    equal(reading,result['reading']);equal(ok,result['behaviour_in_trained_range']);assert total==4110
    save(RUN/'final_verification.json',dict(passed=True,episodes=total,reading=reading,contrasts_verified=32,
        all_step_constraints_and_seeds_verified=True,behaviour_verified=True,original_inputs_unchanged=True))
    files=[p for p in (H/'data').rglob('*.json')]+[H/'e4c.json',H/'E4c_REPORT.txt',RUN/'final_verification.json']
    save(RUN/'artifact_manifest.json',{str(p.relative_to(H)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})

def publish():
    dest=H/'results_archive';dest.mkdir(exist_ok=True);archive=dest/'e4c_results.tar.gz'
    inventory=read(RUN/'artifact_manifest.json')
    with tarfile.open(archive,'w:gz') as tar:
        for name,info in sorted(inventory.items()):
            assert sha(H/name)==info['sha256'];tar.add(H/name,arcname=name,recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        seen=set()
        for member in tar:
            f=tar.extractfile(member);digest=hashlib.sha256(f.read()).hexdigest()
            assert digest==inventory[member.name]['sha256'];seen.add(member.name)
        assert seen==set(inventory)
    assert archive.stat().st_size<40*1024**2
    save(dest/'archive_manifest.json',dict(files=len(inventory),bytes=archive.stat().st_size,sha256=sha(archive),roundtrip_verified=True))
    files=[archive,dest/'archive_manifest.json',H/'e4c.json',H/'E4c_REPORT.txt',RUN/'final_verification.json',RUN/'artifact_manifest.json']
    files+=list((H/'data/rollout').glob('*/*/summary.json'))
    assert cmd(['git','branch','--show-current'])=='main'
    assert cmd(['git','remote','get-url','--push','origin'])=='https://github.com/melancholiac99/ForkAudit.git'
    assert not cmd(['git','diff','--name-only']) and not cmd(['git','diff','--cached','--name-only'])
    cmd(['git','add','-f','--',*[str(p.relative_to(ROOT)) for p in files]])
    if cmd(['git','diff','--cached','--name-only']):cmd(['git','commit','-m','Publish verified E4c prompt and decoding control results'])
    head=cmd(['git','rev-parse','HEAD'])
    for attempt in range(3):
        try:
            cmd(['git','push','origin','HEAD:refs/heads/main'])
            assert cmd(['git','ls-remote','--heads','origin','main']).split()[0]==head
            save(RUN/'publication.json',dict(verified=True,branch='main',commit=head));return
        except Exception:
            if attempt==2:raise
            time.sleep(20)

def main():
    if '--preflight' in sys.argv:preflight();return
    RUN.mkdir(exist_ok=True)
    with (RUN/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('waiting_for_hindsight')
            while True:
                s=read(UP/'run/status.json')
                assert s['stage'] not in ('failed','completed_verified_publication_failed'),s
                if s['stage']=='completed_published' and idle():
                    audit=read(UP/'run/final_verification.json');assert audit['passed'] and all(v['passed'] for v in audit['models'].values())
                    save(RUN/'upstream_verified.json',dict(status=s,audit=audit));break
                time.sleep(20)
            status('prelaunch_verification');protected();assert idle()
            status('running',command=['bash','run_e4c.sh'],shutdown=False)
            with (RUN/'pipeline.log').open('a') as log:
                subprocess.run(['bash',str(H/'run_e4c.sh')],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+':'+os.environ['PATH'],HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false'))
            status('verifying');verify();status('completed_verified',reading=read(H/'e4c.json')['reading'])
            publish();status('completed_published',publication=read(RUN/'publication.json'),shutdown=False)
        except Exception as exc:
            status('failed',error=str(exc));traceback.print_exc()
            thread=os.environ.get('CODEX_THREAD_ID')
            if thread:
                try:
                    subprocess.run(['codex','queue','--thread',thread,'--resume','--message',
                        f'【E4C 自动异常检查】用户要求 Hindsight 后继续完成 E4C，且已明确取消关机。请检查 {RUN}/status.json 和 pipeline.log，尽力修复并续跑。保留原始实验数据与预注册，不放宽门槛，不强推研究冲突；无论如何不得关机。'],cwd=ROOT,timeout=45)
                except Exception:traceback.print_exc()
            raise

if __name__=='__main__':main()
