"""A real two-A100 NCCL collective plus deterministic BF16 matmul comparison."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

dist.init_process_group('nccl')
rank = dist.get_rank()
torch.cuda.set_device(rank)
torch.manual_seed(1234)
a = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
b = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16)
c = a @ b
other = c.clone()
dist.broadcast(other, src=0)
assert torch.equal(c, other), 'Cross-GPU deterministic BF16 matmul differs'
x = torch.tensor(float(rank + 1), device='cuda')
dist.all_reduce(x)
assert x.item() == 3
props = torch.cuda.get_device_properties(rank)
record = dict(rank=rank, gpu=str(props.uuid), name=props.name,
              memory_bytes=props.total_memory, collective=True,
              bf16_matmul_equal=True, torch=torch.__version__, cuda=torch.version.cuda)
Path(os.environ['FORKAUDIT_VERL_OUTPUT'], f'gpu_{rank}.json').write_text(json.dumps(record, indent=2))
dist.destroy_process_group()
