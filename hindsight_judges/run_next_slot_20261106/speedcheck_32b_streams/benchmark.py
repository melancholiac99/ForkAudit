"""Same-shape canonical cache concurrency diagnostic; no outcome selection."""
from pathlib import Path
import sys,time
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parents[2];sys.path.insert(0,str(H))
import exp2_common as C,exp5_token_slot as S,token_slot_backend as K,slot_model_loader as L
import torch
from transformers import AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
OUT=Path(__file__).resolve().parent
DATA=H/'run_next_slot_20261106/speedcheck_7b_cache/previous_aligned_attempt/groups.jsonl.gz'

class Parallel:
 def __init__(self,scorer,tok,prompts,workers):
  self.engines={d:K.Engine(scorer,tok,prompts) for d in ['AF','AM','EF','EM']}
  self.streams={d:torch.cuda.Stream() for d in self.engines}
  self.pool=ThreadPoolExecutor(max_workers=workers)
 def work(self,d,h):
  with torch.inference_mode(),torch.cuda.stream(self.streams[d]):
   out=self.engines[d].lp(d[0],d[1],h)
   self.streams[d].synchronize()
   return out
 def quartet(self,h):
  jobs={d:self.pool.submit(self.work,d,h) for d in self.engines}
  return {d:f.result() for d,f in jobs.items()}
 def close(self):self.pool.shutdown()

with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
 groups,gh=C.decode_rows(DATA)
 tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-32B-Instruct',local_files_only=True)
 scorer,model=L.load_model('32b','/root/autodl-tmp/models/Qwen2.5-32B-Instruct',H.parent)
 scorer.slot_memo_limit=2048
 ranked=sorted(groups,key=lambda g:len(tok.encode(g['prompts']['F']))+len(g['response']['ids']))
 selected=list({g['id']:g for g in [ranked[0],ranked[len(ranked)//2],ranked[-1],next(g for g in groups if g['id']=='s010_t45')]}.values())
 reference={};results={}
 for workers in [0,2,4]:
  mode='serial' if not workers else f'streams{workers}';checks=[];seconds=[]
  for g in selected:
   e=K.Engine(scorer,tok,g['prompts']) if not workers else Parallel(scorer,tok,g['prompts'],workers)
   hs=[g['response']['ids'][:1],g['response']['ids'][:len(g['response']['ids'])//2]]
   h,cands=S.probe_job(tok,g,g['boundary_rules']['boundaries'][0]);hs += [h,h+cands[-1][:-1]]
   torch.cuda.synchronize();start=time.monotonic();vals=[]
   for hi,h in enumerate(hs):
    for d,a in e.quartet(h).items():vals.append(((g['id'],hi,d),a.cpu().clone()))
   torch.cuda.synchronize();seconds.append(time.monotonic()-start)
   for key,a in vals:
    if not workers:reference[key]=a
    tv=float((a.exp()-reference[key].exp()).abs().sum()/2)
    checks.append(dict(id=key[0],position=key[1],distribution=key[2],tv=tv,bitwise_equal=torch.equal(a,reference[key]),passed=tv<.005))
   if workers:e.close()
   del e
   C.save_json(OUT/(mode+'_checks.json'),checks)
   print(mode,g['id'],seconds[-1],flush=True)
  results[mode]=dict(seconds=sum(seconds),checks=len(checks),max_tv=max(x['tv'] for x in checks),all_bitwise_equal=all(x['bitwise_equal'] for x in checks),passed=all(x['passed'] for x in checks),peak_vram_bytes=torch.cuda.max_memory_allocated())
  C.save_json(OUT/'summary.json',dict(results=results,model=model,groups_hash=gh))
  print('RESULT',mode,results[mode],flush=True)
