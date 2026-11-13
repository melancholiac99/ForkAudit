"""Native checkpoint observations; loaded by external_lib after CUDA assignment.

This changes neither state serialization nor the optimizer. Hash live tensors
before save/after native load; guard FP32 trainable LoRA and AdamW moments.
"""
import functools
import hashlib
import json
import os
from pathlib import Path

import torch


def digest(value):
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            result.update(json.dumps([str(item.dtype), list(item.shape)]).encode())
            raw = item.detach().contiguous().reshape(-1).view(torch.uint8)
            for offset in range(0, raw.numel(), 64 * 1024 * 1024):
                result.update(raw[offset:offset + 64 * 1024 * 1024].cpu().numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            result.update(json.dumps(item, sort_keys=True).encode())

    visit(value)
    return result.hexdigest()


def precision(model):
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.dtype == torch.float32 for p in trainable), 'Trainable LoRA must retain FP32 masters'
    frozen = {}
    for p in model.parameters():
        if not p.requires_grad:
            frozen[str(p.dtype)] = frozen.get(str(p.dtype), 0) + p.numel()
    buffers = {}
    for b in model.buffers():
        buffers[str(b.dtype)] = buffers.get(str(b.dtype), 0) + b.numel()
    return dict(trainable_lora_dtype='torch.float32', trainable_numel=sum(p.numel() for p in trainable),
                frozen_numel_by_dtype=frozen, buffer_numel_by_dtype=buffers)


def fingerprint(manager):
    info = precision(manager.model)
    info.pop('buffer_numel_by_dtype')  # FSDP lazily casts buffers on first forward.
    states = manager.optimizer.state_dict()
    steps = []
    for state in states['state'].values():
        for key in ['exp_avg', 'exp_avg_sq']:
            if key in state:
                assert state[key].dtype == torch.float32, f'{key} is not FP32'
        if 'step' in state:
            steps.append(float(state['step']))
    assert steps and all(x > 0 for x in steps), 'No actual initialized AdamW steps'
    info.update(weights_sha256=digest(dict(manager.model.named_parameters())),
                optimizer_sha256=digest(states), scheduler_sha256=digest(manager.lr_scheduler.state_dict()),
                adamw_steps_unique=sorted(set(steps)), optimizer_state_entries=len(steps))
    return info


def install():
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager as cls
    if getattr(cls, '_forkaudit_evidence', False):
        return
    original_init, original_save, original_load = cls.__init__, cls.save_checkpoint, cls.load_checkpoint

    @functools.wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        record = precision(self.model)
        record['gpu_uuid'] = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        Path(os.environ['FORKAUDIT_VERL_OUTPUT'], f'precision_rank_{self.rank}.json').write_text(json.dumps(record, indent=2))

    @functools.wraps(original_save)
    def save(self, local_path, *args, **kwargs):
        result = original_save(self, local_path, *args, **kwargs)
        if local_path:
            record = fingerprint(self)
            Path(local_path, f'live_fingerprint_rank_{self.rank}.json').write_text(json.dumps(record, indent=2))
        return result

    @functools.wraps(original_load)
    def load(self, local_path, *args, **kwargs):
        result = original_load(self, local_path, *args, **kwargs)
        if local_path:
            expected = json.loads(Path(local_path, f'live_fingerprint_rank_{self.rank}.json').read_text())
            actual = fingerprint(self)
            assert actual == expected, 'Native checkpoint live weight/optimizer/scheduler fingerprint mismatch'
            Path(os.environ['FORKAUDIT_VERL_OUTPUT'], f'restore_rank_{self.rank}.json').write_text(json.dumps(
                dict(passed=True, source=str(local_path), actual=actual,
                     limitation='Inference generator RNG and next online rollout equivalence are not established by state hashes'), indent=2))
        return result

    cls.__init__, cls.save_checkpoint, cls.load_checkpoint = initialize, save, load
    cls._forkaudit_evidence = True


install()
