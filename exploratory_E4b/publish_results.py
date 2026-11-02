"""Publish verified E4b results to main, independently of hindsight experiment 1."""
from pathlib import Path
import fcntl
import gzip
import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
import time
import traceback

H=Path(__file__).resolve().parent;ROOT=H.parent;RUN=H/'run'


def read(p):return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)


def status(stage,**kw):
    save(RUN/'publication_status.json',dict(stage=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),destination='main',**kw))


def archive():
    manifest=read(H/'run/final_artifact_manifest.json');dest=H/'results_archive'
    if (dest/'archive_manifest.json').exists():
        report=read(dest/'archive_manifest.json')
        assert report['source_manifest_sha256']==sha(H/'run/final_artifact_manifest.json')
        for p in report['parts']:assert sha(dest/p['path'])==p['sha256']
        assert report['roundtrip_verified'];return
    assert not dest.exists(),'Partial archive retained; inspect before retry'
    dest.mkdir();included={};excluded={}
    for name,info in manifest.items():
        p=H/name;assert p.resolve().is_relative_to(H.resolve()) and p.is_file() and not p.is_symlink()
        assert p.stat().st_size==info['bytes'] and sha(p)==info['sha256'],name
        (excluded if p.suffix=='.safetensors' else included)[name]=info
    spec=importlib.util.spec_from_file_location('existing_archive_helpers',ROOT/'exploratory_E4/archive_results.py')
    A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A);A.OUT=dest
    writer=A.SplitWriter()
    with gzip.GzipFile(filename='',mode='wb',fileobj=writer,compresslevel=3,mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.PAX_FORMAT) as tar:
            for name,info in sorted(included.items()):
                entry=tarfile.TarInfo(name);entry.size=info['bytes'];entry.mode=0o644;entry.uid=entry.gid=entry.mtime=0
                with (H/name).open('rb') as f:
                    checked=A.CheckedReader(f);tar.addfile(entry,checked);assert checked.sha.hexdigest()==info['sha256']
    writer.finish_part()
    for part in writer.parts:
        name=part['path'].replace('e4_nonweight','e4b_nonweight');(dest/part['path']).rename(dest/name);part['path']=name
    verified=set();joined=io.BufferedReader(A.JoinedReader([dest/p['path'] for p in writer.parts]))
    with tarfile.open(fileobj=joined,mode='r|gz') as tar:
        for entry in tar:
            assert entry.isfile() and entry.name in included and entry.name not in verified
            stream=tar.extractfile(entry);h=hashlib.sha256()
            for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
            assert h.hexdigest()==included[entry.name]['sha256'];verified.add(entry.name)
    assert verified==set(included)
    save(dest/'archive_manifest.json',dict(source_manifest_sha256=sha(H/'run/final_artifact_manifest.json'),
         included_files=len(included),parts=writer.parts,compressed_bytes=writer.total,roundtrip_verified=True,
         excluded_tensor_checkpoints=excluded,exclusion_reason='LoRA tensors retained on server; all non-weight results included.'))
    (dest/'README.txt').write_text('E4b complete non-weight results; all archived members were decompressed and SHA256 verified.\n'
        'Includes all evaluation trajectories, preference pairs, adapter configs, training logs, summaries and final reports.\n'
        'LoRA tensors stay on the server, inventoried with size and hash in archive_manifest.json.\n'
        'Join e4b_nonweight_results.tar.gz.part* in filename order, then extract the gzip tar archive.\n')


def publish():
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.STDOUT,timeout=180).strip()
    assert git('branch','--show-current')=='main'
    assert git('remote','get-url','--push','origin')=='https://github.com/melancholiac99/ForkAudit.git'
    assert not git('diff','--cached','--name-only') and not git('diff','--name-only'),'Preserve unrelated tracked edits'
    files=[H/'e4b.json',H/'E4b_REPORT.txt',RUN/'status.json',RUN/'final_verification.json',RUN/'final_artifact_manifest.json',
           H/'data/pairs/manifest.json',*sorted((H/'results_archive').iterdir())]
    files += sorted((H/'data/adapters').glob('*/train_log.json'))
    files += sorted((H/'data/adapters').glob('*/adapter_config.json'))
    files += sorted((H/'data/rollout').glob('*/*/summary.json'))
    assert all(p.is_file() and p.stat().st_size<40*1024**2 for p in files)
    save(RUN/'publication_content_manifest.json',{str(p.relative_to(H)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    files.append(RUN/'publication_content_manifest.json')
    git('add','-f','--',*[str(p.relative_to(ROOT)) for p in files])
    if git('diff','--cached','--name-only'):git('commit','-m','Publish verified E4b results and complete non-weight archive')
    head=git('rev-parse','HEAD')
    for attempt in range(3):
        try:
            git('push','origin','HEAD:refs/heads/main')
            assert git('ls-remote','--heads','origin','main').split()[0]==head
            status('published_verified',commit=head,files=len(files));return
        except Exception:
            if attempt==2:raise
            time.sleep(20)


def main():
    with (RUN/'publisher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            status('waiting_for_e4b_verified')
            while True:
                state=read(RUN/'status.json');assert state['stage']!='failed','E4b failed; no result publication'
                if state['stage']=='completed_verified':break
                time.sleep(30)
            check=read(RUN/'final_verification.json');assert check['passed'] and check['training_runs']==15 and check['episodes']==13152
            status('archiving_verified_results');archive();status('publishing');publish()
        except BaseException as exc:
            status('publication_failed',error=str(exc));traceback.print_exc();raise


if __name__=='__main__':main()
