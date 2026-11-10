"""Download a pinned public checkpoint; verify every file against official Hub metadata."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
H=Path(__file__).resolve().parent;R=H/'gigpo_20261110';DEST=Path('/root/autodl-tmp/models/GiGPO-7B-ALFWorld')
REPO='langfeng01/GiGPO-Qwen2.5-7B-Instruct-ALFWorld'

def save(name,x):
 p=R/name;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)

def verify():
 meta=json.loads((R/'hub_metadata.json').read_text());files={}
 for f in meta['siblings']:
  p=DEST/f['rfilename'];assert p.resolve().is_relative_to(DEST.resolve()) and p.is_file()
  assert p.stat().st_size==f['size'],str(p)
  h=hashlib.sha256();g=hashlib.sha1();g.update(b'blob '+str(p.stat().st_size).encode()+b'\0')
  with p.open('rb') as stream:
   for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b);g.update(b)
  if f.get('lfs'):assert h.hexdigest()==f['lfs']['sha256'],str(p)
  else:assert g.hexdigest()==f['blobId'],str(p)
  files[str(p)]=dict(bytes=p.stat().st_size,sha256=h.hexdigest())
 config=json.loads((DEST/'config.json').read_text())
 assert config['model_type']=='qwen2' and config['architectures']==['Qwen2ForCausalLM']
 save('download_verified.json',dict(passed=True,repository=REPO,revision=meta['sha'],files=files,architecture=config['architectures'],time=time.time()))
 return files

if __name__=='__main__':
 meta=json.loads((R/'hub_metadata.json').read_text());errors=[]
 for attempt in range(3):
  try:
   env=dict(os.environ,HF_ENDPOINT='https://huggingface.co',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
   if attempt:env['HF_HUB_DISABLE_XET']='1'
   if attempt==2:env['HF_ENDPOINT']='https://hf-mirror.com'
   args=['/root/autodl-tmp/hf_download_env_20261110/bin/hf','download',REPO,'--revision',meta['sha'],'--local-dir',str(DEST),'--max-workers','4']
   save('download_status.json',dict(stage='downloading',attempt=attempt+1,endpoint=env['HF_ENDPOINT'],revision=meta['sha'],pid=os.getpid(),time=time.time()))
   with (R/'download.log').open('a') as log:subprocess.run(args,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
   save('download_status.json',dict(stage='verifying',pid=os.getpid(),time=time.time()));files=verify()
   save('download_status.json',dict(stage='completed_verified',files=len(files),pid=os.getpid(),time=time.time()));break
  except Exception:
   errors.append(traceback.format_exc());save('download_errors.json',errors)
   if attempt==2:
    save('download_status.json',dict(stage='failed',pid=os.getpid(),time=time.time(),error=errors[-1]));raise
   time.sleep(10)
