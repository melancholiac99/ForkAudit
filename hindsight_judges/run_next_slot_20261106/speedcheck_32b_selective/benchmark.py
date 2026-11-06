"""Additional numerical-only performance candidates; no scientific outcomes read."""
from pathlib import Path
import sys,time
H=Path(__file__).resolve().parents[2];sys.path.insert(0,str(H))
import exp2_common as C,exp5_token_slot as S,token_slot_backend as K,slot_model_loader as L
import torch
from transformers import AutoTokenizer
from torch.nn.attention import sdpa_kernel,SDPBackend
OUT=Path(__file__).resolve().parent;DATA=H/'run_next_slot_20261106/speedcheck_7b_cache/previous_aligned_attempt/groups.jsonl.gz'
Base=K.Engine
class Semantic(Base):
 def input(self,slot,evidence,hids):return K.contexts(self.tok,self.prompts[evidence],slot,hids)
with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
 groups,gh=C.decode_rows(DATA);tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-32B-Instruct',local_files_only=True)
 scorer,model=L.load_model('32b','/root/autodl-tmp/models/Qwen2.5-32B-Instruct',H.parent);scorer.slot_memo_limit=2048
 ranked=sorted(groups,key=lambda g:len(tok.encode(g['prompts']['F']))+len(g['response']['ids']))
 selected=list({g['id']:g for g in [ranked[0],ranked[len(ranked)//2],ranked[-1],next(g for g in groups if g['id']=='s010_t45')]}.values())
 reference={};all_results={}
 def check(mode,cls,chunk):
  scorer.chunk=chunk;checks=[];seconds=[]
  for g in selected:
   e=cls(scorer,tok,g['prompts']);hs=[g['response']['ids'][:1],g['response']['ids'][:len(g['response']['ids'])//2]]
   h,cands=S.probe_job(tok,g,g['boundary_rules']['boundaries'][0]);hs += [h,h+cands[-1][:-1]]
   vals=[];torch.cuda.synchronize();start=time.monotonic()
   for hi,h in enumerate(hs):
    for slot in ('A','E'):
     for evidence in ('M','F'):
      vals.append(((g['id'],hi,slot,evidence),e.lp(slot,evidence,h).cpu().clone()))
   torch.cuda.synchronize();seconds.append(time.monotonic()-start)
   for key,a in vals:
    if mode=='reference':reference[key]=a
    tv=float((a.exp()-reference[key].exp()).abs().sum()/2);checks.append(dict(id=key[0],position=key[1],distribution=key[2]+key[3],tv=tv,passed=tv<.005))
   del e
   C.save_json(OUT/(mode+'_checks.json'),checks);print(mode,g['id'],seconds[-1],flush=True)
  previous=K.Engine;K.Engine=cls
  gates=K.gate(scorer,tok,groups);K.Engine=previous
  C.save_json(OUT/(mode+'_gate.json'),gates)
  by={d:max(x['tv'] for x in checks if x['distribution']==d) for d in ['AF','AM','EF','EM']}
  result=dict(passed=all(x['passed'] for x in checks+gates),seconds=sum(seconds),checks=len(checks),max_tv_by_distribution=by,gate_max_tv=max(x['tv'] for x in gates),chunk=chunk)
  all_results[mode]=result;C.save_json(OUT/'summary.json',dict(results=all_results,model=model,groups_hash=gh,selection='Only numerical TV<0.005 and measured throughput; fixed1024 reference; no Q or target-effect selection'))
  print('RESULT',mode,result,flush=True);return by,gates
 check('reference',Base,1024)
 by,gates=check('semantic',Semantic,1024)
 safe={d for d,v in by.items() if v<.005}
 for g in gates:
  if not g['passed'] and 'slot' in g:safe.discard(g['slot']+g['evidence'])
  elif not g['passed']:safe.discard('EM')
 C.save_json(OUT/'numerically_eligible_distributions.json',dict(distributions=sorted(safe),criterion='All explicit comparisons and unchanged gate entries for that distribution TV<0.005'))
 class Hybrid(Base):
  def input(self,slot,evidence,hids):
   return K.contexts(self.tok,self.prompts[evidence],slot,hids) if slot+evidence in safe else super().input(slot,evidence,hids)
 if safe:check('hybrid',Hybrid,1024)
 check('chunk512',Base,512)
