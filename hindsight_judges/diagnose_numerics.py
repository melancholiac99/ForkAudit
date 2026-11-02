"""Diagnostic only: no formal scores, source changes, or threshold adjustments."""
import contextlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers import AutoTokenizer
import exp1 as E
import execution_checks as C

H=Path(__file__).resolve().parent
OUT=H/'diagnostics/numerical_gate_20261102'
OUT.mkdir(parents=True,exist_ok=True)
report=dict(purpose=__doc__,runtime=C.fingerprint(),tests=[])
def save(): E.write_json(OUT/'diagnosis.json',report)
def probability(values):
    a=np.array([sum(v) for v in values]);a=np.exp(a-a.max());return a/a.sum()
def trial(label,job,scorer):
    t=time.time();shared=scorer.score(job);direct=[scorer.direct(job,k) for k in range(5)]
    p,q=probability(shared),probability(direct)
    row=dict(label=label,request_id=job['request_id'],tv=float(abs(p-q).sum()/2),
             shared_sequence_logp=[sum(v) for v in shared],direct_sequence_logp=[sum(v) for v in direct],
             shared_probability=p.tolist(),direct_probability=q.tolist(),seconds=time.time()-t)
    report['tests'].append(row);save();print(json.dumps(row),flush=True)

save();print('Loading frozen original model/backend for diagnostic only',flush=True)
src=Path('/root/autodl-tmp/idea/exploratory_P4/p4_e2_backend.py')
spec=importlib.util.spec_from_file_location('frozen_backend',src);B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
B.MODEL='/root/autodl-tmp/models/Qwen2.5-32B-Instruct'
report['backend_sha256']=E.digest(src)
tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
raw,_,_=E.rows_and_meta(H/'results/exp1_p4')
job=E.tokenize(tok,next(x for x in raw if x['request_id']=='s001_t07__SPE'))
report['prefix_tokens']=len(job['prefix_ids']);report['suffix_tokens']=[len(c['suffix_ids']) for c in job['candidates']]
t=time.time();scorer=B.Scorer();report['load_seconds']=time.time()-t;save()
for i in range(3):trial('original_repeat_'+str(i),job,scorer)
old=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
trial('bf16_reduced_precision_reduction_disabled',job,scorer)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old
from torch.nn.attention import sdpa_kernel,SDPBackend
with sdpa_kernel(SDPBackend.MATH):trial('math_attention_only',job,scorer)
report['complete']=True;save()
print('Diagnostic complete',flush=True)
