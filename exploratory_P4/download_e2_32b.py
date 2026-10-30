"""Download pinned public 32B checkpoint across the two available writable disks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,time,requests
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'exploratory_P4'
LOCAL=Path('/root/autodl-tmp/models/Qwen2.5-32B-Instruct')
SECOND=Path('/autodl-fs/data/forkaudit_models/Qwen2.5-32B-Instruct')
REV='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 meta=json.loads((OUT/'e2_32b_model_source.json').read_text());assert meta['sha']==REV
 LOCAL.mkdir(parents=True,exist_ok=True);SECOND.mkdir(parents=True,exist_ok=True)
 shards=[s for s in meta['siblings'] if s['rfilename'].endswith('.safetensors')]
 other=[s for s in meta['siblings'] if s['rfilename'] in ['config.json','generation_config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt','model.safetensors.index.json','LICENSE']]
 remote=set(s['rfilename'] for s in shards[:8])
 def get(s):
  name=s['rfilename'];path=(SECOND if name in remote else LOCAL)/name
  expected=s.get('lfs',{}).get('sha256')
  if path.exists() and path.stat().st_size==s['size'] and (not expected or digest(path)==expected):pass
  else:
   part=path.with_suffix(path.suffix+'.part')
   for attempt in range(6):
    try:
     offset=part.stat().st_size if part.exists() else 0
     session=requests.Session()
     session.trust_env=not name.endswith('.safetensors')
     url=(f'https://modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct/resolve/master/{name}' if name.endswith('.safetensors') else f'https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/resolve/{REV}/{name}')
     with session.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,120)) as response:
      response.raise_for_status()
      append=offset>0 and response.status_code==206
      if append:assert response.headers['Content-Range'].startswith(f'bytes {offset}-')
      with part.open('ab' if append else 'wb') as f:
       for chunk in response.iter_content(8*1024*1024):
        if chunk:f.write(chunk)
     assert part.stat().st_size==s['size'],(name,part.stat().st_size,s['size'])
     if expected:assert digest(part)==expected,('hash mismatch',name)
     part.replace(path);break
    except Exception as exc:
     print(json.dumps(dict(event='retry',file=name,attempt=attempt,error=str(exc)[:200])),flush=True)
     if attempt==5:raise
     time.sleep(min(5*(attempt+1),20))
  link=LOCAL/name
  if path!=link:
   if link.is_symlink():assert link.resolve()==path.resolve()
   else:link.symlink_to(path)
  print(json.dumps(dict(event='download_verified',file=name,bytes=path.stat().st_size)),flush=True)
  return dict(file=name,path=str(path),sha256=digest(path),bytes=path.stat().st_size)
 with ThreadPoolExecutor(max_workers=17) as pool:files=list(pool.map(get,other+shards))
 (OUT/'e2_32b_download.json').write_text(json.dumps(dict(revision=REV,model=str(LOCAL),files=files,complete=True,weight_transport='ModelScope mirror; exact upstream pinned HF LFS SHA256 enforced'),indent=2)+'\n')
if __name__=='__main__':main()
