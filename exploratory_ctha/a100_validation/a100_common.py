"""Paths, fingerprints and immutable evidence for the separate A100 entry point."""
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parent
FAST = PARENT / 'v3_fp32_fast'
MODEL = Path(os.environ['M0_MODEL'])
DATA = Path(os.environ.get('CTHA_A100_ROOT', ROOT / '.runtime/runs/ctha_a100_eng_20261113_r1'))


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if load(path) != value:
            raise ValueError(f'Immutable evidence already exists: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def hardware():
    text = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version',
        '--format=csv,noheader,nounits',
    ], text=True)
    return [dict(zip(['index', 'uuid', 'name', 'memory_mib', 'driver'],
                     [v.strip() for v in line.split(',')])) for line in text.splitlines()]


def sources():
    paths = list(HERE.glob('*.py')) + [HERE / 'PLAN.json']
    paths += list(FAST.glob('*.py')) + [FAST / 'PLAN.json']
    paths += [PARENT / n for n in ['habits.py', 'habits_v2.py', 'features_v2.py', 'train.py', 'prepare.py', 'run.py']]
    paths += [ROOT / 'scripts' / n for n in ['common.py', 'alfenv.py', 'envpool.py']]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


class PeakMemory:
    """Sample actual device memory, including the separate vLLM engine process."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.peak = {}
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stop_event.is_set():
            try:
                output = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                                                   '--format=csv,noheader,nounits'], text=True, timeout=5)
                for line in output.splitlines():
                    uuid, value = [s.strip() for s in line.split(',')]
                    self.peak[uuid] = max(self.peak.get(uuid, 0), int(value))
            except (subprocess.SubprocessError, ValueError):
                pass
            self.stop_event.wait(.25)

    def start(self):
        self.thread.start()
        return self

    def finish(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        return dict(self.peak)
