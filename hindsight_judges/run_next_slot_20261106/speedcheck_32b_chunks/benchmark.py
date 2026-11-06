"""CPU/GPU diagnostic only: candidate chunk sizes vs fixed 1024 full-vocabulary reference.
No Q/target-effect values are used for selection. Only numerical gates and timing.
"""
from pathlib import Path
import sys,time
H=Path(__file__).resolve().parents[2];sys.path.insert(0,str(H))
import exp2_common as C,exp5_token_slot as S,token_slot_backend as K,slot_model_loader as L
import torch
from transformers import AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
OUT=Path(__file__).resolve().parent
DATA=H/'run_next_slot_20261106/speedcheck_7b_cache/previous_aligned_attempt/groups.jsonl.gz'
with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
 groups,groups_hash=C.decode_rows(DATA);tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-32B-Instruct',local_files_only=True)
 scorer,model=L.load_model('32b','/root/autodl-tmp/models/Qwen2.5-32B-Instruct',H.parent)
 scorer.slot_memo_limit=2048;scorer.slot_cache_policy='canonical_complete_chunks'
 ranked=sorted(groups,key=lambda g:len(tok.encode(g['prompts']['F']))+len(g['response']['ids']))
 selected=list({g['id']:g for g in [ranked[0],ranked[len(ranked)//2],ranked[-1],next(g for g in groups if g['id']=='s010_t45')]}.values())
 reference={};timings=[];results={}
 for chunk in (1024,256,128,64):
  scorer.chunk=chunk;checks=[]
  for g in selected:
   eng=K.Engine(scorer,tok,g['prompts'])
   hs=[g['response']['ids'][:1],g['response']['ids'][:len(g['response']['ids'])//2]]
   h,cands=S.probe_job(tok,g,g['boundary_rules']['boundaries'][0]);hs += [h,h+cands[-1][:-1]]
   torch.cuda.synchronize();start=time.monotonic();outputs=[]
   for hi,h in enumerate(hs):
    for slot in ('A','E'):
     for evidence in ('M','F'):
      a=eng.lp(slot,evidence,h);key=(g['id'],hi,slot,evidence);outputs.append((key,a.cpu().clone()))
   torch.cuda.synchronize();elapsed=time.monotonic()-start;timings.append(dict(chunk=chunk,id=g['id'],requests=len(outputs),seconds=elapsed))
   for key,a in outputs:
    if chunk==1024:reference[key]=a
    b=reference[key];tv=float((a.exp()-b.exp()).abs().sum()/2)
    checks.append(dict(id=key[0],position=key[1],slot=key[2],evidence=key[3],tv=tv,passed=tv<.005))
   del eng
   C.save_json(OUT/'timings.json',timings);C.save_json(OUT/f'chunk_{chunk}_vs1024.json',checks)
   print('TIMING',chunk,g['id'],round(elapsed,3),'maxTV',max(x['tv'] for x in checks),flush=True)
  gates=K.gate(scorer,tok,groups);C.save_json(OUT/f'chunk_{chunk}_gate.json',gates)
  passed=all(x['passed'] for x in checks+gates)
  results[str(chunk)]=dict(passed=passed,max_tv_to1024=max(x['tv'] for x in checks),gate_max_tv=max(x['tv'] for x in gates),seconds=sum(t['seconds'] for t in timings if t['chunk']==chunk),comparisons=len(checks),gates=len(gates))
  C.save_json(OUT/'summary.json',dict(model=model,groups_hash=groups_hash,points=[g['id'] for g in selected],fixed_candidate_order=[1024,256,128,64],selection='fastest candidate passing unchanged TV<0.005 checks AND comparisons with fixed1024 reference',results=results,source={str(p):C.sha_file(p) for p in [Path(__file__),H/'token_slot_backend.py',H/'exp5_token_slot.py']}))
  print('RESULT',chunk,results[str(chunk)],flush=True)
