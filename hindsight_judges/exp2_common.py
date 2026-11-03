"""Non-frozen utilities for Hindsight think localization and OPD experiment 2.

The E1/E1c code and artifacts remain read-only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_bytes(json_bytes(obj))
    tmp.replace(p)


def load_32b(model_path: Path, repo: Path):
    """Frozen P4 checkpoint/scorer and E1c precision policy; caller holds SDPA MATH scope.

    Must never silently replace the judge model with a different checkpoint.
    """
    import torch
    import exp1c_precision
    from exp1c_backend import ChunkedScorer

    model_path, repo = Path(model_path).resolve(), Path(repo).resolve()
    original = read_json(repo / 'exploratory_P4/e2_32b/run.json')
    assert original['complete'], 'frozen P4 scorer not complete'
    weights = {p.name: sha_file(p) for p in sorted(model_path.glob('*.safetensors'))}
    assert weights and weights == original['model_sha256'], '32B weights differ from frozen P4'
    backend = repo / 'exploratory_P4/p4_e2_backend.py'
    assert sha_file(backend) == original['scorer_sha256'], 'frozen P4 backend changed'
    exp1c_precision.set_flags(torch)
    assert torch.backends.cuda.is_available(), 'CUDA is required'
    spec = importlib.util.spec_from_file_location('p4_frozen_exp2', backend)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.MODEL = str(model_path)
    frozen = module.Scorer()
    check = exp1c_precision.configure(frozen)
    assert all(p.dtype == torch.float16 for p in frozen.model.model.parameters())
    assert frozen.model.lm_head.weight.dtype == torch.float32
    return ChunkedScorer(frozen), dict(weights=weights, frozen_scorer_sha256=sha_file(backend),
                                        precision=check, chunk=1024)


def decode_rows(path):
    import gzip
    p = Path(path)
    raw = gzip.decompress(p.read_bytes()) if p.suffix == '.gz' else p.read_bytes()
    return [json.loads(x) for x in raw.decode().splitlines() if x], sha_bytes(raw)


def encode_rows(path, rows):
    import gzip
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = ('\n'.join(json.dumps(r, ensure_ascii=False, allow_nan=False) for r in rows) + '\n').encode()
    if p.suffix == '.gz':
        p.write_bytes(gzip.compress(raw, mtime=0))
    else:
        p.write_bytes(raw)
    return sha_bytes(raw)
