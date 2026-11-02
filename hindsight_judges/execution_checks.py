"""Execution identity and score checks, separate from frozen scientific definitions."""
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import numpy as np


def fingerprint():
    import torch
    p=torch.cuda.get_device_properties(0); uuid=str(getattr(p,'uuid',''))
    assert uuid and uuid!='None', 'GPU UUID required'
    driver=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],text=True)
    driver=[s.strip() for s in driver.splitlines() if uuid.replace('GPU-','') in s]
    assert len(driver)==1, ('GPU UUID mismatch',uuid)
    c=torch.backends.cuda
    return dict(uuid=uuid,name=p.name,capability=[p.major,p.minor],memory=p.total_memory,
                driver=driver[0],python=platform.python_version(),torch=str(torch.__version__),cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),torch_build=torch.__config__.show(),
                versions={k:importlib.metadata.version(k) for k in ('transformers','numpy','safetensors')},
                tf32_matmul=c.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
                bf16_reduction=c.matmul.allow_bf16_reduced_precision_reduction,
                fp16_reduction=c.matmul.allow_fp16_reduced_precision_reduction,
                flash_sdp=c.flash_sdp_enabled(),math_sdp=c.math_sdp_enabled(),
                efficient_sdp=c.mem_efficient_sdp_enabled(),cudnn_sdp=c.cudnn_sdp_enabled(),
                deterministic=torch.are_deterministic_algorithms_enabled(),precision=torch.get_float32_matmul_precision(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                math_environment={k:os.environ.get(k) for k in ('CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE',
                                                               'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')})


def fp_sha(fp):
    return hashlib.sha256(json.dumps(fp,sort_keys=True).encode()).hexdigest()


def require_math_attention(runtime):
    assert runtime['math_sdp'] is True, 'MATH attention must be enabled'
    assert all(runtime[k] is False for k in ('flash_sdp','efficient_sdp','cudnn_sdp')), 'Other attention kernels must be disabled'


def check_attention_run(run,amendment_sha):
    assert run['attention_backend']=='sdpa_math'
    assert run['execution_amendment_sha256']==amendment_sha
    require_math_attention(run['runtime'])


def validate_row(row,job,hardware_sha):
    for k in ('request_id','request_sha256','point_id','condition'):
        assert row[k]==job[k], (job['request_id'],k)
    p=np.asarray(row['candidate_prob'],float);lp=np.asarray(row['sequence_logp'],float)
    assert p.shape==lp.shape==(5,) and np.isfinite(p).all() and np.isfinite(lp).all()
    assert (p>=0).all() and abs(p.sum()-1)<1e-12
    expected=np.exp(lp-lp.max());expected/=expected.sum()
    assert np.max(abs(expected-p))<1e-12 and row['argmax']==int(p.argmax())
    assert row['hardware_sha256']==hardware_sha and row['session']


def load_scores(path,jobs,hardware_sha,complete=False):
    path=Path(path)
    if not path.exists():
        assert not complete, 'Missing scores'
        return {}
    raw=path.read_bytes();assert not raw or raw.endswith(b'\n'), 'Truncated score tail; retain and inspect'
    records=[json.loads(s) for s in raw.splitlines()];by_id={j['request_id']:j for j in jobs}
    done={r['request_id']:r for r in records}
    assert len(done)==len(records) and set(done)<=set(by_id), 'Duplicate or unknown score IDs'
    points={}
    for key,row in done.items():
        validate_row(row,by_id[key],hardware_sha)
        points.setdefault(row['point_id'],[]).append(row)
    for pid,rows in points.items():
        assert len(rows)==10 and len({r['condition'] for r in rows})==10, f'Partial point block: {pid}'
        assert len({r['session'] for r in rows})==1, f'Mixed model loads within point: {pid}'
    if complete:assert set(done)==set(by_id), 'Incomplete score coverage'
    return done


def check_sessions(target,done,jobs,hardware_sha):
    target=Path(target)
    conditions=('S0','SE','SX','SPE','SPX','F0','FE','FX','FPE','FPX')
    expected=[next(r['request_id'] for r in jobs if r['condition']==c) for c in conditions]
    expected.append(max(jobs,key=lambda r:len(r['text']))['request_id'])
    for session in {r['session'] for r in done.values()}:
        info=json.loads((target/'sessions'/f'{session}.json').read_text())
        assert info['hardware_sha256']==hardware_sha
        checks=info['checks'];assert [c['request_id'] for c in checks]==expected
        for c in checks:
            ps=[]
            for key in ('shared_sequence_logp','direct_sequence_logp'):
                lp=np.asarray(c[key],float);assert lp.shape==(5,) and np.isfinite(lp).all()
                p=np.exp(lp-lp.max());p/=p.sum();ps.append(p)
            tv=float(abs(ps[0]-ps[1]).sum()/2)
            assert abs(tv-c['tv'])<1e-12 and tv<.005 and c['passed']
