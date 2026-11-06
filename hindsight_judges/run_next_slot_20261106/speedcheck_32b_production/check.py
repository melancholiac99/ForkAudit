"""Production factory versus unchanged serial canonical scoring, before restart."""
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
 groups,gh=C.decode_rows(DATA)
 tok=AutoTokenizer.from_pretrained('/root/autodl-tmp/models/Qwen2.5-32B-Instruct',local_files_only=True)
 scorer,model=L.load_model('32b','/root/autodl-tmp/models/Qwen2.5-32B-Instruct',H.parent)
 scorer.slot_memo_limit=K.MEMO_LIMIT;scorer.slot_cache_policy=K.CACHE_POLICIES['32b']
 ranked=sorted(groups,key=lambda g:len(tok.encode(g['prompts']['F']))+len(g['response']['ids']))
 selected=list({g['id']:g for g in [ranked[0],ranked[len(ranked)//2],ranked[-1],next(g for g in groups if g['id']=='s010_t45')]}.values())
 reference={};results={}
 for workers in (1,K.WORKERS['32b']):
  scorer.slot_workers=workers;mode=f'workers{workers}';checks=[];seconds=[]
  for g in selected:
   e=K.make_engine(scorer,tok,g['prompts'])
   hs=[g['response']['ids'][:1],g['response']['ids'][:len(g['response']['ids'])//2]]
   h,cands=S.probe_job(tok,g,g['boundary_rules']['boundaries'][0]);hs += [h,h+cands[-1][:-1]]
   torch.cuda.synchronize();start=time.monotonic();vals=[]
   for hi,h in enumerate(hs):
    for d,a in e.quartet(h).items():vals.append(((g['id'],hi,d),a.cpu().clone()))
   torch.cuda.synchronize();seconds.append(time.monotonic()-start)
   for key,a in vals:
    if workers==1:reference[key]=a
    tv=float((a.exp()-reference[key].exp()).abs().sum()/2)
    checks.append(dict(id=key[0],position=key[1],distribution=key[2],tv=tv,bitwise_equal=torch.equal(a,reference[key]),passed=tv<.005))
   e.close();del e
   C.save_json(OUT/(mode+'_checks.json'),checks);print(mode,g['id'],seconds[-1],flush=True)
  results[mode]=dict(seconds=sum(seconds),checks=len(checks),max_tv=max(x['tv'] for x in checks),all_bitwise_equal=all(x['bitwise_equal'] for x in checks),passed=all(x['passed'] for x in checks),peak_vram_bytes=torch.cuda.max_memory_allocated())
  C.save_json(OUT/'summary.json',dict(results=results,model=model,groups_hash=gh,source=S.fingerprint()))
  assert results[mode]['passed'] and results[mode]['all_bitwise_equal']
 gates=K.gate(scorer,tok,groups);C.save_json(OUT/'numerical_gate.json',gates)
 assert all(g['passed'] for g in gates)
 C.save_json(OUT/'verification.json',dict(passed=True,checks=64+len(gates),max_tv=max(g['tv'] for g in gates),source=S.fingerprint(),model=model,groups_hash=gh,workers=scorer.slot_workers))
 print('PRODUCTION_VALIDATED',flush=True)
