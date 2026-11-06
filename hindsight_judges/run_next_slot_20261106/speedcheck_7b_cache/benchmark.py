"""Read-only 7B speed/accuracy comparison; resume paused production worker even on error."""
from pathlib import Path
import sys,os,signal,time,copy
H=Path(__file__).resolve().parents[2];sys.path.insert(0,str(H))
import exp2_common as C,exp5_token_slot as S,token_slot_backend as K,slot_model_loader as L
OUT=Path(__file__).resolve().parent
pid=C.read_json(H/'run_next_slot_20261106/token_status.json')['child_pid']
assert b'exp5_token_slot.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
def stopped(signum,frame):raise SystemExit('benchmark interrupted')
signal.signal(signal.SIGTERM,stopped)
paused=False
try:
 import torch
 from transformers import AutoTokenizer
 from torch.nn.attention import sdpa_kernel,SDPBackend
 class SemanticEngine(K.Engine):
  def input(self,slot,evidence,hids):return K.contexts(self.tok,self.prompts[evidence],slot,hids)
 with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
  groups,meta=S.prepared(S.OUT,32);tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-7B-Instruct',local_files_only=True)
  # Diagnostic-only load bypasses serialization: 7B+32B CPU parameters fit the verified 184GiB host limit.
  scorer,model=L._load_model('7b','/root/autodl-tmp/models/Qwen2.5-7B-Instruct',H.parent)
  os.kill(pid,signal.SIGSTOP);paused=True;C.save_json(OUT/'paused_worker.json',dict(pid=pid,time=time.time()));time.sleep(.5)
  original_engine=K.Engine;K.Engine=SemanticEngine
  gate=K.gate(scorer,tok,groups);K.Engine=original_engine
  C.save_json(OUT/'semantic_gate.json',gate);print('SEMANTIC GATE max TV',max(x['tv'] for x in gate),flush=True)
  assert all(x['passed'] for x in gate)
  ranked=sorted(groups,key=lambda g:len(tok.encode(g['prompts']['F']))+len(g['response']['ids']))
  selected=[ranked[0],ranked[len(ranked)//2],ranked[-1]];comparison=[];timings=[]
  for g in selected:
   hs=[g['response']['ids'][:1],g['response']['ids'][:len(g['response']['ids'])//2]]
   h,cands=S.probe_job(tok,g,g['boundary_rules']['boundaries'][0]);hs += [h,h+cands[-1][:-1]]
   outputs={}
   for mode,cls in [('semantic',SemanticEngine),('aligned',original_engine)]:
    eng=cls(scorer,tok,g['prompts']);vals=[];torch.cuda.synchronize();start=time.monotonic()
    for hi,h in enumerate(hs):
     for slot in ('A','E'):
      for evidence in ('M','F'):
       lp=eng.lp(slot,evidence,h);vals.append(lp.cpu().clone())
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    timings.append(dict(id=g['id'],mode=mode,requests=len(vals),seconds=elapsed));outputs[mode]=vals;del eng
   for i,(a,b) in enumerate(zip(outputs['semantic'],outputs['aligned'])):
    tv=float((a.exp()-b.exp()).abs().sum()/2);comparison.append(dict(id=g['id'],request=i,tv=tv,passed=tv<.005,max_logp_delta=float((a-b).abs().max())))
   C.save_json(OUT/'comparison.json',comparison);C.save_json(OUT/'timings.json',timings);print('POINT',g['id'],timings[-2:],flush=True)
  assert all(x['passed'] for x in comparison)
  totals={mode:sum(t['seconds'] for t in timings if t['mode']==mode) for mode in ['semantic','aligned']}
  C.save_json(OUT/'summary.json',dict(passed=True,model=model,source=S.fingerprint(),points=len(selected),comparisons=len(comparison),max_tv=max(x['tv'] for x in comparison),timings=totals,aligned_over_semantic=totals['aligned']/totals['semantic'],scope='deterministic short/median/long pilot contexts; no scientific endpoint selection'))
  print('SUMMARY',totals,'speedup',totals['aligned']/totals['semantic'],flush=True)
finally:
 try:
  if paused:os.kill(pid,signal.SIGCONT);C.save_json(OUT/'resumed_worker.json',dict(pid=pid,time=time.time()))
 except ProcessLookupError:pass
