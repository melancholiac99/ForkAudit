"""Explicit, scoped FP32 launch repair for SM120 shared-memory capacity."""
import inspect,hashlib
from vllm.v1.worker.gpu_worker import Worker
class FP32Worker(Worker):
 def init_device(self):
  import vllm.model_executor.determinism.batch_invariant as module
  source=inspect.getsource(module.matmul_persistent)
  old='"num_stages": _fp32_num_stages if N == 1 else 3,'
  if source.count(old)!=1:raise RuntimeError('Unexpected vLLM kernel wrapper version; refusing patch')
  # Preserve shapes, dot precision and reduction order. Reduce pipelined tile
  # buffering only, to avoid 131072-byte shared allocation on a 101376-byte SM.
  revised=source.replace(old,'"num_stages": 1,')
  print('CTHA FP32 wrapper original SHA256',hashlib.sha256(source.encode()).hexdigest(),flush=True)
  exec(compile(revised,'<ctha_fp32_num_stages_1>','exec'),module.__dict__)
  super().init_device()
