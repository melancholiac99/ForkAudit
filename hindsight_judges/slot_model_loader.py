"""Frozen numerical policy, CPU dtype conversion before CUDA transfer on RTX6000D.
Avoids reproducible illegal-address errors in large GPU fill/cast kernels in this
Torch/CUDA/device runtime. Does not alter checkpoint values, attention or scoring.
"""
from pathlib import Path
import exp2_common as C

def load_model(name,path,repo):
 import fcntl
 lock=Path(repo)/'hindsight_judges/run_next_slot_20261106/model_load.lock'
 with lock.open('a') as handle:
  fcntl.flock(handle,fcntl.LOCK_EX)
  return _load_model(name,path,repo)

def _load_model(name,path,repo):
 import torch
 from transformers import AutoModelForCausalLM
 from exp1c_backend import ChunkedScorer
 import exp1c_precision
 path,repo=Path(path),Path(repo)
 original=C.read_json(repo/f'exploratory_P4/e2_{name}/run.json')
 weights={p.name:C.sha_file(p) for p in sorted(path.glob('*.safetensors'))}
 assert weights and weights==original['model_sha256']
 backend=repo/'exploratory_P4/p4_e2_backend.py';assert C.sha_file(backend)==original['scorer_sha256']
 exp1c_precision.set_flags(torch)
 torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
 torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
 model=AutoModelForCausalLM.from_pretrained(str(path),local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').eval()
 assert next(model.parameters()).device.type=='cpu'
 model.lm_head.float()
 if name=='7b':model.float();policy=dict(precision='7b_fp32_math')
 else:
  assert name=='32b';policy=exp1c_precision.configure(type('Frozen',(),{'t':torch,'model':model})())
 # No large device-side conversion or all-parameter GPU elementwise checks.
 model.to('cuda');torch.cuda.synchronize()
 frozen=type('Frozen',(),{'t':torch,'model':model})()
 return ChunkedScorer(frozen),dict(weights=weights,backend_sha256=C.sha_file(backend),precision=policy,chunk=1024,
   loader='cpu_cast_before_cuda_transfer',loader_sha256=C.sha_file(Path(__file__)))
