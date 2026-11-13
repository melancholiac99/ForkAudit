"""Isolated paths, source guards and host-memory coordination for R3."""
import contextlib
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = HERE.parent
VALIDATION = PARENT / 'a100_validation'
FAST = PARENT / 'v3_fp32_fast'
R2 = ROOT / '.runtime/runs/ctha_a100_eng_20261113_r2'
ENGINEERING = Path(os.environ.get('FULL_RESPONSE_ENGINEERING', ROOT / '.runtime/runs/ctha_a100_full_response_r3'))
PILOT = Path(os.environ.get('FULL_RESPONSE_PILOT', ROOT / '.runtime/runs/ctha_a100_baseline_rl_r1'))
MODEL = Path(os.environ['M0_MODEL'])
TB = ROOT / '.runtime/tensorboard/ctha-a100-r2/full_response_r3'
sys.path.insert(0, str(VALIDATION))
from a100_common import fingerprint, hardware, load, save, sha


def sources():
    import a100_common
    paths = {str(p.relative_to(ROOT)): sha(p) for p in [*HERE.glob('*.py'), HERE / 'PLAN.json']}
    paths.update(a100_common.sources())
    paths[str(PARENT.relative_to(ROOT) / 'v3_engineering_fast_20261113/check_checkpoint.py')] = sha(PARENT / 'v3_engineering_fast_20261113/check_checkpoint.py')
    return dict(sorted(paths.items()))


@contextlib.contextmanager
def cpu_model_lock():
    p = ROOT / '.runtime/locks/full_cpu_model.lock'
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def configure(data):
    # util imports historical source names; ROOT/scripts/common.py must not shadow this module.
    import a100_common
    a100_common.DATA = data
    import backend
    backend.DATA = data
    util = backend.configure()
    util.DATA = data
    util.source_hashes = sources
    import merged_policy
    original_export = merged_policy.export_policy

    def locked_export(base, adapter):
        with cpu_model_lock():
            return original_export(base, adapter)

    merged_policy.export_policy = locked_export
    return backend, util


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def emit(folder, run, values, step):
    import time
    from torch.utils.tensorboard import SummaryWriter
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'metrics.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(), run=run, step=step, **values)) + '\n')
    with SummaryWriter(str(TB / run)) as writer:
        for key, value in values.items():
            if isinstance(value, (bool, int, float)):
                writer.add_scalar(key, float(value), step)
        writer.add_text('scope', 'New full-response R3 protocol. R2 remains FAIL. Baseline RL only; no CTHA or generalization evaluation.', 0)

