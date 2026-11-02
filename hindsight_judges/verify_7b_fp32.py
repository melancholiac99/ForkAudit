"""Precision provenance checks plus the unchanged independent statistical verifier."""
import argparse
from pathlib import Path
import exp1 as E
import verify_exp1 as V


def verify(repo,out,arms,name):
    assert name=='7b'
    run=E.load_json(out/name/'run.json')
    assert run['compute_precision']=='float32'
    assert run['runtime']['transformer_compute_dtype']==run['runtime']['vocabulary_projection_dtype']=='float32'
    assert not run['runtime']['tf32_matmul'] and not run['runtime']['tf32_cudnn']
    assert run['precision_amendment_sha256']==E.digest(E.HERE/'EXECUTION_AMENDMENT_7B_FP32_20261102.txt')
    assert run['scorer_script_sha256']==E.digest(E.HERE/'exp1_7b_fp32.py')
    assert run['parent_scorer_sha256']==E.digest(E.HERE/'exp1.py')
    V.verify(repo,out,arms,name)
    path=out/name/'verification.json';result=E.load_json(path)
    result.update(compute_precision='float32',precision_provenance_verified=True,
                  precision_verifier_sha256=E.digest(__file__))
    E.write_json(path,result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for flag in ('repo','out','arms'):p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--name',choices=['7b'],required=True);a=p.parse_args()
    verify(a.repo,a.out,a.arms,a.name)
