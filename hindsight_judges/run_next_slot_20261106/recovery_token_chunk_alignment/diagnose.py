"""Diagnostic only: preserve original code and compare aligned caching to unchanged direct gate."""
from pathlib import Path
import sys,time,json
H=Path(__file__).resolve().parents[2];sys.path.insert(0,str(H))
import exp2_common as C,exp5_token_slot as S,token_slot_backend as K,slot_model_loader as L
import torch
from transformers import AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
OUT=Path(__file__).resolve().parent
with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
 groups,meta=S.prepared(S.OUT,32)
 tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-32B-Instruct',local_files_only=True)
 scorer,model=L.load_model('32b','/root/autodl-tmp/models/Qwen2.5-32B-Instruct',H.parent)
 original_contexts=K.contexts
 def aligned(tok,prompt,slot,hids):
  ids,stem,audit=original_contexts(tok,prompt,slot,hids)
  stem=stem[:len(stem)//scorer.chunk*scorer.chunk]
  assert ids[:len(stem)]==stem
  return ids,stem,audit
 baseline=K.gate(scorer,tok,groups);C.save_json(OUT/'baseline_repeat.json',baseline);print('BASELINE',baseline,flush=True)
 K.contexts=aligned
 start=time.monotonic();checks=K.gate(scorer,tok,groups)
 C.save_json(OUT/'aligned_gate.json',dict(checks=checks,seconds=time.monotonic()-start,model=model,source=S.fingerprint(),chunk=scorer.chunk))
 print('ALIGNED',checks,flush=True)
 assert all(x['passed'] for x in checks)
 # Extra deterministic boundary/context checks, no Q values or target outcomes used for this diagnosis.
 extra=[]
 for g in groups[:3]:
  engine=K.Engine(scorer,tok,g['prompts'])
  hs=[g['response']['ids'][:1],g['response']['ids'][:len(g['response']['ids'])//2]]
  h,cands=S.probe_job(tok,g,g['boundary_rules']['boundaries'][0]);hs += [h,h+cands[-1][:-1]]
  for hi,h in enumerate(hs):
   for slot in ('A','E'):
    for evidence in ('M','F'):
     a=engine.lp(slot,evidence,h);ids,stem,_=engine.input(slot,evidence,h)
     kv,last=scorer._feed(ids,len(ids)-1);del kv
     b=torch.log_softmax(scorer.model.lm_head(last[-1].float()).float(),-1)
     tv=float((a.exp()-b.exp()).abs().sum()/2)
     extra.append(dict(id=g['id'],position=hi,slot=slot,evidence=evidence,cached_prefix=len(stem),total_tokens=len(ids),tv=tv,passed=tv<.005))
  del engine
  C.save_json(OUT/'extra_aligned_checks.json',extra);print('EXTRA',g['id'],max(x['tv'] for x in extra),flush=True)
 assert all(x['passed'] for x in extra)
