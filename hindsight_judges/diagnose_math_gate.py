"""Separate diagnostic of all frozen numerical gate requests; no formal scores."""
import importlib.util
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers import AutoTokenizer
import exp1 as E
import execution_checks as C

H=Path(__file__).resolve().parent
OUT=H/'diagnostics/numerical_gate_20261102/math_gate.json'
source=Path('/root/autodl-tmp/idea/exploratory_P4/p4_e2_backend.py')
spec=importlib.util.spec_from_file_location('original_backend',source)
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)
B.MODEL='/root/autodl-tmp/models/Qwen2.5-32B-Instruct'
report=dict(diagnostic_only=True,backend_sha256=E.digest(source),checks=[],complete=False,
            runtime=C.fingerprint(),override='torch.nn.attention.sdpa_kernel(SDPBackend.MATH)',threshold=0.005)
E.write_json(OUT,report)
tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
raw,_,_=E.rows_and_meta(H/'results/exp1_p4')
jobs=[next(r for r in raw if r['condition']==c) for c in E.CONDS]+[max(raw,key=lambda r:len(r['text']))]
scorer=B.Scorer()
original=E.load_json(H/'diagnostics/numerical_gate_20261102/failed_original_attempt/32b/numerical_check.json')
with sdpa_kernel(SDPBackend.MATH):
    for job,old in zip(jobs,original):
        assert job['request_id']==old['request_id']
        t=time.time();j=E.tokenize(tok,job)
        shared=scorer.score(j);direct=[scorer.direct(j,k) for k in range(5)]
        logs=[[sum(v) for v in shared],[sum(v) for v in direct]]
        probs=[]
        for lp in logs:
            p=np.asarray(lp);p=np.exp(p-p.max());probs.append(p/p.sum())
        tv=float(abs(probs[0]-probs[1]).sum()/2)
        row=dict(request_id=job['request_id'],original_tv=old['tv'],tv=tv,passed=tv<0.005,
                 shared_sequence_logp=logs[0],direct_sequence_logp=logs[1],seconds=time.time()-t)
        report['checks'].append(row);E.write_json(OUT,report);print(row,flush=True)
report['complete']=True;report['passed']=all(c['passed'] for c in report['checks'])
report['max_tv']=max(c['tv'] for c in report['checks']);E.write_json(OUT,report)
print('Complete; passed:',report['passed'],'max TV:',report['max_tv'],flush=True)
