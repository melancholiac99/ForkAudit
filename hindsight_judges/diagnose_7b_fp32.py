"""Diagnostic only: test 7B FP32 arithmetic on the fixed 11 numerical gates."""
from pathlib import Path
import importlib.util
import time
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoTokenizer
import exp1 as E
import execution_checks as X

H=Path(__file__).resolve().parent
OUT=H/'diagnostics/numerical_7b_20261102/fp32.json'
spec=importlib.util.spec_from_file_location('original',Path('/root/autodl-tmp/idea/exploratory_P4/p4_e2_backend.py'))
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
B.MODEL='/root/autodl-tmp/models/Qwen2.5-7B-Instruct'
raw,_,_=E.rows_and_meta(H/'results/exp1_p4')
jobs=[next(x for x in raw if x['condition']==c) for c in E.CONDS]+[max(raw,key=lambda r:len(r['text']))]
tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
report=dict(diagnostic_only=True,model='7b',checks=[],complete=False,precision='FP32 transformer and vocabulary projection; original checkpoint values',tf32=False)
E.write_json(OUT,report)
with sdpa_kernel(SDPBackend.MATH):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    scorer=B.Scorer();scorer.model.float()
    report['runtime']=X.fingerprint()
    for job in jobs:
        t=time.time();j=E.tokenize(tok,job)
        shared=scorer.score(j);direct=[scorer.direct(j,k) for k in range(5)]
        logs=[[sum(v) for v in shared],[sum(v) for v in direct]];probs=[]
        for row in logs:
            p=np.asarray(row);p=np.exp(p-p.max());probs.append(p/p.sum())
        tv=float(abs(probs[0]-probs[1]).sum()/2)
        result=dict(request_id=job['request_id'],tv=tv,passed=tv<.005,shared_sequence_logp=logs[0],direct_sequence_logp=logs[1],seconds=time.time()-t)
        report['checks'].append(result);E.write_json(OUT,report);print(result,flush=True)
report['complete']=True;report['passed']=all(r['passed'] for r in report['checks']);report['max_tv']=max(r['tv'] for r in report['checks'])
E.write_json(OUT,report);print('passed',report['passed'],'max TV',report['max_tv'],flush=True)
