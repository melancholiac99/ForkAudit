"""Exercise the actual CUDA compilation and FlashAttention runtime dependencies."""
import json
import os
from pathlib import Path

import torch
from flash_attn import flash_attn_func


@torch.compile
def compiled(x):
    return torch.sin(x) + x.square()


records = []
for device in range(2):
    with torch.cuda.device(device):
        torch.manual_seed(42)
        x = torch.randn(128, device='cuda')
        torch.testing.assert_close(compiled(x), torch.sin(x) + x.square())
        q, k, v = [torch.randn(2, 32, 2, 64, device='cuda', dtype=torch.bfloat16) for _ in range(3)]
        actual = flash_attn_func(q, k, v, causal=True)
        reference = torch.nn.functional.scaled_dot_product_attention(q.float().transpose(1, 2),
            k.float().transpose(1, 2), v.float().transpose(1, 2), is_causal=True).transpose(1, 2)
        error = (actual.float() - reference).abs().max().item()
        assert error < .025, error
        records.append(dict(gpu=str(torch.cuda.get_device_properties(device).uuid),
                            triton_inductor=True, flash_attention=True, flash_max_abs_error=error))
Path(os.environ['FORKAUDIT_VERL_OUTPUT'], 'kernel_checks.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records))
