"""Read-only size accounting on real saved batches; not a GPU timing benchmark."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'results/throughput_audit/evidence.json'
rows = []
for name in ['verl_agent_bypass_single_gpu_r9', 'verl_agent_grpo_four_rounds_r11']:
    folder = ROOT / '.runtime/runs' / name
    for packet in ['raw_batch_0001', 'ppo_batch_step_1']:
        path = folder / (packet + '.npz')
        with np.load(path) as data:
            mask = data['attention_mask']
            total = int(mask.sum())
            response = int(mask[:, -data['responses'].shape[1]:].sum())
            lengths = mask.sum(-1)
            maximum = max(int(lengths[i:i+16].sum()) for i in range(0, len(lengths), 16))
            rows.append(dict(run=name, packet=packet, rows=len(lengths),
                nonpadding_tokens=total, response_tokens=response,
                prompt_fraction=(total-response)/total,
                all_positions_per_response_position=total/response,
                max_contiguous_micro16_tokens=maximum,
                estimated_bf16_logits_gib=maximum*152064*2/1024**3,
                packet_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
paths = [
 '.runtime/external/verl-agent-20bd331b/verl/workers/sharding_manager/fsdp_vllm.py',
 '.runtime/external/verl-agent-20bd331b/verl/workers/fsdp_workers.py',
 '.runtime/external/verl-agent-20bd331b/verl/workers/actor/dp_actor.py',
 '.runtime/external/verl-agent-20bd331b/agent_system/multi_turn_rollout/rollout_loop.py',
 '.runtime/venvs/verl-agent-20bd331b/lib/python3.12/site-packages/vllm/device_allocator/cumem.py',
 '.runtime/venvs/verl-agent-20bd331b/lib/python3.12/site-packages/vllm/v1/worker/gpu_worker.py',
 '.runtime/venvs/verl-agent-20bd331b/lib/python3.12/site-packages/vllm/lora/worker_manager.py',
 '.runtime/venvs/verl-agent-20bd331b/lib/python3.12/site-packages/vllm/v1/core/kv_cache_utils.py',
]
result = dict(scope='Static code plus real saved-batch accounting; no training or GPU speedup measurement',
    batches=rows,
    frozen_base_parameters=7615616512,
    estimated_base_bf16_gib=7615616512*2/1024**3,
    estimated_base_50_sleep_wake_roundtrips_tib=7615616512*2*2*50/1024**4,
    estimate_limits='Parameter-payload estimate, not measured PCIe traffic; excludes adapter/buffer allocation and transfer overhead. Logits estimate is one BF16 tensor, not actual peak VRAM. Prompt positions still require transformer computation.',
    source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
