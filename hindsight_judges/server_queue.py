"""Sequential experiment queue: verified E4b -> 32B -> 7B -> verified publication."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import exp1 as E

H=Path(__file__).resolve().parent;ROOT=H.parent
SOURCE=Path('/root/autodl-tmp/idea');E4=SOURCE/'exploratory_E4b'
OUT=H/'results/exp1_p4';RUN=H/'run'
ARMS=SOURCE/'exploratory_deep_discovery/exploratory_arms.parquet'
MODELS={'32b':('Qwen2.5-32B-Instruct','5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'),
        '7b':('Qwen2.5-7B-Instruct','a09a35458c702b33eeacc393d103063234e8bc28')}
BRANCH='research/hindsight-judges'
DESTINATION='research/hindsight-judges'


def status(stage,**kw):
    row=dict(stage=stage,pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw)
    E.write_json(RUN/'status.json',row);print(json.dumps(row),flush=True)


def hashes_check(hashes):
    missing=[p for p in hashes if not Path(p).is_file()]
    assert not missing,'Missing files:\n'+'\n'.join(missing)
    def check(item):
        p,h=item;actual=E.digest(p)
        if p.endswith('.safetensors'):print('SHA checked',Path(p).name,flush=True)
        return p if actual!=h else None
    with ThreadPoolExecutor(max_workers=4) as pool:
        changed=[p for p in pool.map(check,hashes.items()) if p is not None]
    assert not changed,'Input/code hash mismatch:\n'+'\n'.join(changed)


def preflight():
    from transformers import AutoTokenizer
    jobs,meta,manifest=E.rows_and_meta(OUT)
    actual,pm,inputs=E.build_dataset(SOURCE,ARMS)
    assert jobs==actual and meta==pm and manifest['input_sha256']==inputs
    original=Path('/tmp/hindsight_original_prepare_2fab12d')
    parity={f:E.digest(OUT/f) for f in ('requests.jsonl.gz','pointmeta.json')}
    assert all((OUT/f).read_bytes()==(original/f).read_bytes() for f in parity)
    frozen={str(ARMS if k.startswith('exploratory_deep_discovery/') else SOURCE/k):h for k,h in inputs.items()}
    oldenv=E.load_json(SOURCE/'exploratory_P4/e2_environment.json')
    for name,(dirname,rev) in MODELS.items():
        model=SOURCE.parent/'models'/dirname;run=E.load_json(SOURCE/f'exploratory_P4/e2_{name}/run.json')
        assert run['complete'] and run['revision']==rev
        expected={str(model/k):h for k,h in run['model_sha256'].items()}
        expected.update({str(model/k):h for k,h in oldenv['model_metadata_hashes'][name.upper()].items()})
        print('Verifying immutable weights/metadata',name,flush=True);hashes_check(expected);frozen.update(expected)
        assert E.digest(SOURCE/'exploratory_P4/p4_e2_backend.py')==run['scorer_sha256']
        for file in ('run.json','scores.jsonl'):
            p=SOURCE/f'exploratory_P4/e2_{name}'/file;frozen[str(p)]=E.digest(p)
    tok=AutoTokenizer.from_pretrained(SOURCE.parent/'models/Qwen2.5-7B-Instruct',local_files_only=True)
    lengths=[E.tokenize(tok,j) for j in jobs]
    token_report={c:dict(contexts=sum(j['condition']==c for j in lengths),
                    max_full_tokens=max(len(j['prefix_ids'])+max(len(x['suffix_ids']) for x in j['candidates']) for j in lengths if j['condition']==c)) for c in E.CONDS}
    for p in [*H.glob('*.py'),H/'EXPERIMENT1_PROTOCOL.md',H/'SERVER_EXECUTION.md',H/'EXECUTION_AMENDMENT_20261102.txt',SOURCE/'exploratory_P4/p4_e2_backend.py',
              SOURCE/'exploratory_P4/e2_environment.json',*(OUT/f for f in ('manifest.json','pointmeta.json','requests.jsonl.gz'))]:
        frozen[str(p.resolve())]=E.digest(p)
    versions={k:importlib.metadata.version(k) for k in oldenv['versions']}
    assert versions==oldenv['versions'], ('Original software versions differ',versions,oldenv['versions'])
    E.write_json(RUN/'protected_inputs.json',frozen)
    E.write_json(RUN/'preflight.json',dict(passed=True,points=386,contexts_per_model=3860,models=['32b','7b'],
        layers={c:sum(m['layer']==c for m in meta) for c in ('I','U')},requests_identical_to_reviewed_branch=parity,
        tokens=token_report,versions=versions,protected_files=len(frozen),free_disk_gib=shutil.disk_usage(ROOT).free/1024**3,
        protocol_commit='2fab12da4d24223111bf83ff46d10b6015766421',gpu_started=False))
    print('CPU preflight passed; no GPU scoring started',flush=True)


def proc_identity(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if s[0]=='Z' else s[19]
    except FileNotFoundError:return None


def gpu_idle():
    return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()


def ready(state,audit,supervisor_alive,idle):
    return state.get('stage')=='completed_verified' and audit.get('passed') is True and not supervisor_alive and idle


def wait_e4():
    launch=E.load_json(E4/'run/launch.json');pid=launch['pid'];identity=proc_identity(pid)
    status('waiting_for_e4b_verified',upstream_pid=pid,upstream_identity=identity,models=['32b','7b'])
    while True:
        state=E.load_json(E4/'run/status.json');assert state['stage']!='failed','E4b failed; refusing to start new experiment'
        live=identity is not None and proc_identity(pid)==identity
        if state['stage']=='completed_verified':
            audit=E.load_json(E4/'run/final_verification.json')
            assert audit['passed'] and audit['training_runs']==15 and audit['episodes']==13152
            if ready(state,audit,live,gpu_idle()):
                # Verify final report/analysis against E4b's final archive manifest.
                inventory=E.load_json(E4/'run/final_artifact_manifest.json')
                for f in ('e4b.json','E4b_REPORT.txt','run/final_verification.json'):
                    assert E.digest(E4/f)==inventory[f]['sha256']
                E.write_json(RUN/'upstream_verified.json',dict(status=state,audit=audit,verified_utc=time.time()))
                return
        elif not live:
            raise RuntimeError('E4b supervisor exited before verified completion; inspect upstream logs')
        time.sleep(30)


def stage(label,script,*args):
    hashes_check({p:h for p,h in E.load_json(RUN/'protected_inputs.json').items() if not p.endswith('.safetensors')})
    assert gpu_idle(),'Unexpected concurrent GPU worker'
    assert shutil.disk_usage(ROOT).free>8*1024**3,'Less than 8 GiB free'
    command=[sys.executable,'-u',str(H/script),*map(str,args)];status(label,command=command)
    log=RUN/(label+'.log')
    with log.open('a') as f:
        subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,
                       env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false'))


def publish():
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=180).strip()
    assert git('branch','--show-current')==BRANCH
    assert git('remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    assert not git('diff','--cached','--name-only') and not git('diff','--name-only'),'Preserve unrelated staged/tracked edits'
    files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl','.gz','.txt')]
    files += [RUN/'upstream_verified.json',RUN/'final_verification.json',RUN/'artifact_manifest.json']
    assert all(p.stat().st_size<40*1024**2 for p in files)
    git('add','-f','--',*[str(p.relative_to(ROOT)) for p in files])
    if git('diff','--cached','--name-only'):git('commit','-m','Publish independently verified hindsight experiment 1 results')
    head=git('rev-parse','HEAD')
    for attempt in range(3):
        try:
            git('push','origin',f'HEAD:refs/heads/{DESTINATION}')
            assert git('ls-remote','--heads','origin',DESTINATION).split()[0]==head
            E.write_json(RUN/'publication.json',dict(verified=True,branch=DESTINATION,commit=head));return
        except Exception:
            if attempt==2:raise
            time.sleep(20)


def main():
    if len(sys.argv)>1 and sys.argv[1]=='preflight':preflight();return
    RUN.mkdir(exist_ok=True)
    with (RUN/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            assert E.load_json(RUN/'preflight.json')['passed'];wait_e4()
            for name,(dirname,revision) in MODELS.items():
                common=['--repo',SOURCE,'--out',OUT,'--arms',ARMS,'--name',name]
                stage('score_'+name,'exp1.py','score',*common,'--model',SOURCE.parent/'models'/dirname,'--revision',revision)
                stage('analyze_'+name,'exp1.py','analyze',*common)
                stage('verify_'+name,'verify_exp1.py',*common)
            hashes_check(E.load_json(RUN/'protected_inputs.json'))
            audits={n:E.load_json(OUT/n/'verification.json') for n in MODELS};assert all(v['passed'] for v in audits.values())
            E.write_json(RUN/'final_verification.json',dict(passed=True,models=audits,protected_inputs_unchanged=True))
            files=[p for p in OUT.rglob('*') if p.is_file()]
            E.write_json(RUN/'artifact_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=E.digest(p)) for p in files})
            status('completed_verified',reports={n:str(OUT/n/'REPORT.txt') for n in MODELS})
            try:
                publish();status('completed_published',publication=E.load_json(RUN/'publication.json'))
            except Exception as exc:
                status('completed_verified_publication_failed',error=str(exc));traceback.print_exc()
        except BaseException as exc:
            status('failed',error=str(exc));traceback.print_exc();raise


if __name__=='__main__':main()
