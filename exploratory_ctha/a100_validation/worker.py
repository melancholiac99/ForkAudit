"""Scoped A100 kernel launch candidates; installed vLLM files remain untouched."""
import hashlib
import inspect
import json
import os
from vllm.v1.worker.gpu_worker import Worker


class A100Worker(Worker):
    def init_device(self):
        import vllm.model_executor.determinism.batch_invariant as module
        config = json.loads(os.environ['CTHA_A100_CONFIG_JSON'])
        source = inspect.getsource(module.matmul_persistent)
        old = '"num_stages": _fp32_num_stages if N == 1 else 3,'
        if source.count(old) != 1:
            raise RuntimeError('vLLM internal wrapper changed; candidate requires explicit revision')
        revised = source.replace(old, f'"num_stages": {int(config["stages"])},')
        start = revised.index('        torch.float32: {')
        end = revised.index('    matmul_kernel_persistent[grid]')
        section = revised[start:end]
        if section.count('"BLOCK_SIZE_M": 128') != 1 or section.count('"num_warps": 8') != 1:
            raise RuntimeError('Unexpected FP32 matmul launch dictionary')
        changed = section.replace('"BLOCK_SIZE_M": 128',
                                  f'"BLOCK_SIZE_M": {int(config["small_m"])} if M <= 128 else {int(config["large_m"])}')
        changed = changed.replace('"num_warps": 8', f'"num_warps": {int(config["warps"])}')
        revised = revised.replace(section, changed)
        print('A100_KERNEL', json.dumps(dict(config=config, original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                            revised_sha256=hashlib.sha256(revised.encode()).hexdigest())), flush=True)
        exec(compile(revised, '<ctha_a100_launch_candidate>', 'exec'), module.__dict__)
        super().init_device()
