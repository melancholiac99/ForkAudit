"""Exact-length batching against canonical single-request 1024 reference."""
from pathlib import Path
import sys,time,copy
from collections import defaultdict
H=Path(__file__).resolve().parents[2];sys.path.insert(0,str(H))
import exp2_common as C,exp5_token_slot as S,token_slot_backend as K,slot_model_loader as L
import torch
from transformers import AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
OUT=Path(__file__).resolve().parent
DATA=H/'run_next_slot_20261106/speedcheck_7b_cache/previous_aligned_attempt/groups.jsonl.gz'

def batch(e,slot,evidence,hs):
 inputs=[e.input(slot,evidence,h) for h in hs];stem=inputs[0][1]
 assert all(x[1]==stem and len(x[0])==len(inputs[0][0]) for x in inputs)
 bkey=(slot,evidence)
 if bkey not in e.bases:e.bases[bkey]=e.s._feed(stem,len(stem)-1)
 base,last=e.bases[bkey];kv=copy.deepcopy(base)
 if kv is not None:kv.batch_repeat_interleave(len(hs))
 tails=[x[0][len(stem):] for x in inputs]
 for start in range(0,len(tails[0]),e.s.chunk):
  ids=torch.tensor([t[start:start+e.s.chunk] for t in tails],device=e.s.device)
  z=e.s.model.model(input_ids=ids,past_key_values=kv,use_cache=True)
  kv=z.past_key_values;last=z.last_hidden_state[:,-1].float()
 out=torch.log_softmax(e.s.model.lm_head(last).float(),-1)
 return [v.cpu().clone() for v in out]

with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
 groups,gh=C.decode_rows(DATA)
 tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-32B-Instruct',local_files_only=True)
 scorer,model=L.load_model('32b','/root/autodl-tmp/models/Qwen2.5-32B-Instruct',H.parent);scorer.slot_memo_limit=2048
 ranked=sorted(groups,key=lambda g:len(tok.encode(g['prompts']['F']))+len(g['response']['ids']))
 selected=list({g['id']:g for g in [ranked[0],ranked[len(ranked)//2],ranked[-1],next(g for g in groups if g['id']=='s010_t45')]}.values())
 refs={};results={}
 for bsz in [1,2,4]:
  mode=f'batch{bsz}';checks=[];seconds=[]
  for g in selected:
   e=K.Engine(scorer,tok,g['prompts']);jobs={}
   for b in g['boundary_rules']['boundaries'][:2]:
    h,cands=S.probe_job(tok,g,b)
    for ids in cands:
     for j in range(1,len(ids)):jobs[tuple(h+ids[:j])]=h+ids[:j]
   # Fixed real request groups; keep two largest equal-length buckets per distribution.
   requests=[]
   for slot in ('A','E'):
    for evidence in ('M','F'):
     lengths=defaultdict(list)
     for h in jobs.values():lengths[len(e.input(slot,evidence,h)[0])].append(h)
     for _,hs in sorted(lengths.items(),key=lambda x:(-len(x[1]),x[0]))[:2]:
      requests.append((slot,evidence,hs[:4]))
   torch.cuda.synchronize();start=time.monotonic();vals=[]
   for slot,evidence,hs in requests:
    for i in range(0,len(hs),bsz):
     chunk=hs[i:i+bsz]
     vs=[e.lp(slot,evidence,chunk[0]).cpu().clone()] if bsz==1 else batch(e,slot,evidence,chunk)
     vals.extend(((g['id'],slot+evidence,tuple(h)),v) for h,v in zip(chunk,vs))
   torch.cuda.synchronize();seconds.append(time.monotonic()-start)
   for key,a in vals:
    if bsz==1:refs[key]=a
    tv=float((a.exp()-refs[key].exp()).abs().sum()/2)
    checks.append(dict(id=key[0],distribution=key[1],prefix_tokens=len(key[2]),tv=tv,bitwise_equal=torch.equal(a,refs[key]),passed=tv<.005))
   del e
   C.save_json(OUT/(mode+'_checks.json'),checks);print(mode,g['id'],seconds[-1],flush=True)
  results[mode]=dict(seconds=sum(seconds),checks=len(checks),max_tv=max(x['tv'] for x in checks),all_bitwise_equal=all(x['bitwise_equal'] for x in checks),passed=all(x['passed'] for x in checks),peak_vram_bytes=torch.cuda.max_memory_allocated())
  C.save_json(OUT/'summary.json',dict(results=results,model=model,groups_hash=gh));print('RESULT',mode,results[mode],flush=True)
