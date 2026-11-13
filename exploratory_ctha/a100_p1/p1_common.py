"""Independent registration and paths for the newly authorized P1 phase."""
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parent
VALIDATION = PARENT / 'a100_validation'
FAST = PARENT / 'v3_fp32_fast'
DATA = Path(os.environ.get('CTHA_P1_ROOT', ROOT / '.runtime/runs/ctha_a100_p1_20261113_r1'))
ENGINEERING = Path(os.environ.get('CTHA_P1_ENGINEERING', ROOT / '.runtime/runs/ctha_a100_eng_20261113_r2'))
MODEL = Path(os.environ['M0_MODEL'])


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value, immutable=False):
    path = Path(path)
    if path.exists() and immutable:
        if load(path) != value:
            raise ValueError(f'Immutable P1 evidence differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sources():
    paths = list(HERE.glob('*.py')) + [HERE / 'PLAN.json']
    paths += list(VALIDATION.glob('*.py')) + [VALIDATION / 'PLAN.json']
    paths += list(FAST.glob('*.py')) + [FAST / 'PLAN.json']
    paths += [PARENT / n for n in ['habits.py', 'habits_v2.py', 'features_v2.py', 'train.py', 'prepare.py', 'run.py']]
    paths += [ROOT / 'scripts' / n for n in ['common.py', 'alfenv.py', 'envpool.py']]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths))}
