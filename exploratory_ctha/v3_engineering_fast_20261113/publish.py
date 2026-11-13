"""Publish verified evidence only. Caller independently reviews before shutdown."""
import argparse,json,subprocess,tarfile,shutil,time
from pathlib import Path
from verify import verify,digest
REPO='https://github.com/melancholiac99/ForkAudit.git'

def cmd(w,*args):return subprocess.check_output(['git','-C',str(w),*args],text=True).strip()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--worktree',type=Path,default=Path('/root/autodl-tmp/ctha_p1_summary_publish_20261113'));a=p.parse_args();d=a.data;w=a.worktree
 verify(d)
 assert cmd(w,'remote','get-url','origin')==REPO
 assert not cmd(w,'status','--porcelain'), 'Publication worktree must be clean'
 cmd(w,'fetch','--no-tags','origin','credit-grok');cmd(w,'merge','--ff-only','origin/credit-grok')
 target=w/'exploratory_ctha/v3_engineering_results_20261113';assert not target.exists(),'Refuse to overwrite prior published results'
 target.mkdir()
 manifest={};excluded={}
 for f in sorted(d.rglob('*')):
  if not f.is_file() or f.is_symlink():continue
  rel=str(f.relative_to(d))
  if rel.startswith('sampling_export_'):continue
  if f.suffix in ('.pt','.safetensors'):
   excluded[rel]={'sha256':digest(f),'bytes':f.stat().st_size,'local_path':str(f)};continue
  manifest[rel]={'sha256':digest(f),'bytes':f.stat().st_size}
 with tarfile.open(target/'engineering_evidence.tar.gz','w:gz') as tar:
  for rel in manifest:tar.add(d/rel,arcname=rel,recursive=False)
 assert (target/'engineering_evidence.tar.gz').stat().st_size<50*1024**2,'Evidence archive too large; retain locally for explicit packaging repair'
 (target/'manifest.json').write_text(json.dumps({'included':manifest,'checkpoints_retained_on_server':excluded,'archive_sha256':digest(target/'engineering_evidence.tar.gz'),'scope':'Engineering evidence only; no learned policy efficacy claims'},indent=2)+'\n')
 for name in ['REPORT.txt','verification.json','registration.json','checkpoint_verification.json']:shutil.copy2(d/name,target/name)
 cmd(w,'add',str(target.relative_to(w)));cmd(w,'commit','-m','Publish CTHA engineering-only acceptance evidence; no learning claim')
 head=cmd(w,'rev-parse','HEAD');cmd(w,'push','origin','HEAD:credit-grok')
 remote=cmd(w,'ls-remote','origin','refs/heads/credit-grok').split()[0];assert head==remote
 record={'verified':True,'branch':'credit-grok','commit':head,'remote_head':remote,'time':time.time(),'path':str(target.relative_to(w)),'archive_sha256':digest(target/'engineering_evidence.tar.gz')}
 (d/'publication.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
