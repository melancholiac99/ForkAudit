"""Numerical diagnostics only; no formal experiment scores are written."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,importlib.util
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
D=Path(__file__).resolve().parent
H=D.parent.parent;ROOT=H.parent
sys.path.insert(0,str(H));import exp1c as E
from exp1c_backend import ChunkedScorer
def load_module(path):
    spec=importlib.util.spec_from_file_location('frozen',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def prob(lps):
    x=np.array([sum(r) for r in lps]);assert np.isfinite(x).all(), 'nonfinite log probability'
    p=np.exp(x-x.max());return p/p.sum()
rows=[]
def check(scorer,ref,j,kind,precision):
    start=time.time();a=scorer.score(j)
    b=[scorer.direct(j,k) for k in range(5)] if kind=='shared_direct' else ref.score(j)
    tv=float(np.abs(prob(a)-prob(b)).sum()/2)
    r=dict(kind=kind,precision=precision,request_id=j['request_id'],tokens=j['max_tokens'],tv=tv,passed=tv<.005,
           shared_sequence_logp=[sum(r) for r in a],reference_sequence_logp=[sum(r) for r in b],seconds=time.time()-start)
    rows.append(r);E.save(D/'diagnostics.json',rows);print(json.dumps(r),flush=True)

jobs,_,_=E.prepared(H/'results/exp1c_p4')
tok=E.get_tokenizer('/root/autodl-tmp/models/Qwen2.5-32B-Instruct')
jj=[E.tokenize(tok,j) for j in jobs];byid={j['request_id']:j for j in jj}
checks=[max(jj,key=lambda j:j['max_tokens'])]+[next(j for j in jj if j['condition']==c) for c in E.CONDITIONS]
short=[min((j for j in jj if j['condition']==c),key=lambda j:j['max_tokens']) for c in E.CONDITIONS]
short=[j for j in short if j['max_tokens']<=E.UNCHUNKED_CHECK_MAX]
module=load_module(ROOT/'exploratory_P4/p4_e2_backend.py');module.MODEL='/root/autodl-tmp/models/Qwen2.5-32B-Instruct'
with sdpa_kernel(SDPBackend.MATH):
    ref=module.Scorer();scorer=ChunkedScorer(ref)
    print('MODEL LOADED',flush=True)
    for rid in ['s021_t08__L','s001_t07__D1F']:
        check(scorer,ref,byid[rid],'shared_direct','bf16_repeat')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    buffers={n:b.clone() for n,b in ref.model.model.named_buffers()}
    # Module.to(dtype) would also round RoPE frequencies: preserve every buffer.
    for p in ref.model.model.parameters():
        p.data=p.data.to(dtype=torch.float16)
    assert all(torch.equal(b,buffers[n]) and b.dtype==buffers[n].dtype for n,b in ref.model.model.named_buffers())
    assert ref.model.lm_head.weight.dtype==torch.float32
    assert all(torch.isfinite(p).all() for p in ref.model.parameters())
    torch.cuda.empty_cache()
    print('FP16 TRANSFORMER, FP32 HEAD, TF32 OFF, FP16 REDUCED REDUCTION OFF',flush=True)
    for j in checks:check(scorer,ref,j,'shared_direct','fp16')
    for j in short:check(scorer,ref,j,'chunked_onepass','fp16')
    E.save(D/'diagnostic_summary.json',dict(all_fp16_passed=all(r['passed'] for r in rows if r['precision']=='fp16'),
        fp16_checks=sum(r['precision']=='fp16' for r in rows),max_fp16_tv=max(r['tv'] for r in rows if r['precision']=='fp16'),
        peak_gb=torch.cuda.max_memory_allocated()/1e9,complete=True))
