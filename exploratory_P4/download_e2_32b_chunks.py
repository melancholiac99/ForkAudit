"""Resume existing weight prefixes using bounded parallel ranges; full pinned HF SHA256 gate."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,json,hashlib,time,threading,requests
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exploratory_P4'
LOCAL=Path('/root/autodl-tmp/models/Qwen2.5-32B-Instruct');SECOND=Path('/autodl-fs/data/forkaudit_models/Qwen2.5-32B-Instruct')
REV='5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd';CHUNK=16*1024*1024
STATE=OUT/'e2_32b_chunks_state.json';lock=threading.Lock();tls=threading.local()
def save(state):
 tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(STATE)
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 meta=json.loads((OUT/'e2_32b_model_source.json').read_text());assert meta['sha']==REV
 shards=[s for s in meta['siblings'] if s['rfilename'].endswith('.safetensors')]
 if STATE.exists():state=json.loads(STATE.read_text());assert state['revision']==REV and state['chunk']==CHUNK
 else:
  state=dict(revision=REV,chunk=CHUNK,files={})
  for i,s in enumerate(shards):
   p=(SECOND if i<8 else LOCAL)/s['rfilename'];part=p.with_suffix(p.suffix+'.part')
   prefix=s['size'] if p.exists() else part.stat().st_size if part.exists() else 0
   assert prefix<=s['size']
   state['files'][s['rfilename']]=dict(path=str(p),size=s['size'],sha256=s['lfs']['sha256'],prefix=prefix,done=[])
  save(state) # durable contiguous-prefix lengths BEFORE any out-of-order writes
 tasks=[];fds={}
 for name,row in state['files'].items():
  p=Path(row['path']);part=p.with_suffix(p.suffix+'.part')
  if p.exists():assert digest(p)==row['sha256'];continue
  fds[name]=os.open(part,os.O_CREAT|os.O_RDWR,0o644)
  for start in range(row['prefix'],row['size'],CHUNK):
   if start not in row['done']:tasks.append((name,start,min(start+CHUNK,row['size'])-1))
 # Spread each batch over files and offsets, never overlap writes.
 tasks.sort(key=lambda t:(t[1],t[0]));started=time.time();received=0;finished=0
 def get(task):
  name,start,end=task;need=end-start+1
  if not hasattr(tls,'sessions'):
   tls.sessions={False:requests.Session(),True:requests.Session()}
   for proxy,sess in tls.sessions.items():sess.trust_env=proxy
  for attempt in range(12):
   proxy=attempt%4==3
   url=(f'https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/resolve/{REV}/{name}' if proxy else f'https://modelscope.cn/models/Qwen/Qwen2.5-32B-Instruct/resolve/master/{name}')
   try:
    session=tls.sessions[proxy]
    with session.get(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},timeout=(10,20),stream=True) as resp:
     resp.raise_for_status();assert resp.status_code==206
     assert resp.headers.get('Content-Range')==f'bytes {start}-{end}/{state["files"][name]["size"]}',resp.headers.get('Content-Range')
     data=bytearray()
     for block in resp.iter_content(1024*1024):
      data.extend(block)
      if len(data)>need:raise ValueError('oversized range')
     assert len(data)==need
    view=memoryview(data);offset=start
    while view:
     n=os.pwrite(fds[name],view,offset);assert n>0;offset+=n;view=view[n:]
    with lock:state['files'][name]['done'].append(start);save(state)
    return need
   except Exception as exc:
    if attempt==11:raise
    if attempt in [2,6]:print(json.dumps(dict(event='range_retry',file=name,start=start,attempt=attempt,error=type(exc).__name__)),flush=True)
    time.sleep(min(1+attempt,5))
 try:
  with ThreadPoolExecutor(max_workers=32) as pool:
   for future in as_completed([pool.submit(get,t) for t in tasks]):
    received+=future.result();finished+=1
    if finished%16==0:
     elapsed=time.time()-started
     print(json.dumps(dict(event='ranges',completed=finished,total=len(tasks),new_GiB=round(received/2**30,2),MiB_s=round(received/elapsed/2**20,2),seconds=round(elapsed,1))),flush=True)
 finally:
  for fd in fds.values():os.fsync(fd);os.close(fd)
 files=[]
 for name,row in state['files'].items():
  p=Path(row['path']);part=p.with_suffix(p.suffix+'.part')
  current=p if p.exists() else part
  assert current.stat().st_size==row['size']
  h=digest(current);assert h==row['sha256'],('full checkpoint hash mismatch',name)
  if current!=p:current.replace(p)
  link=LOCAL/name
  if link!=p:
   if link.is_symlink():assert link.resolve()==p.resolve()
   else:link.symlink_to(p)
  files.append(dict(file=name,path=str(p),sha256=h,bytes=p.stat().st_size))
  print(json.dumps(dict(event='download_verified',file=name)),flush=True)
 for s in meta['siblings']:
  p=LOCAL/s['rfilename']
  if not s['rfilename'].endswith('.safetensors') and p.is_file():files.append(dict(file=p.name,path=str(p),sha256=digest(p),bytes=p.stat().st_size))
 (OUT/'e2_32b_download.json').write_text(json.dumps(dict(revision=REV,model=str(LOCAL),files=files,complete=True,weight_transport='ModelScope bounded ranges with pinned HF fallback; all full-file pinned SHA256 verified',script_sha256=digest(Path(__file__))),indent=2)+'\n')
if __name__=='__main__':main()
