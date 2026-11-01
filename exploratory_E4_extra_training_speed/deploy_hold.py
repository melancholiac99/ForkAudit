"""Hold only the scheduler while installing the explicitly authorized optimization."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GUARD = ROOT/'exploratory_E4_extra_scoring_speed/hook.py'
spec = importlib.util.spec_from_file_location('recovery', GUARD)
recovery = importlib.util.module_from_spec(spec); spec.loader.exec_module(recovery)
parent = int(sys.argv[1])
assert os.getpgrp() == os.getpid()
p = recovery.proc(parent); assert p
assert b'e4_pipeline.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
ready = HERE/'deployment_ready.json'
assert not ready.exists()
subprocess.Popen([sys.executable, str(GUARD), '--guard', str(os.getpid()), recovery.proc(os.getpid())['identity'],
                  str(parent), p['identity']], start_new_session=True, stdin=subprocess.DEVNULL,
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    os.kill(parent, signal.SIGSTOP)
    print('Scheduler held; current training continues.', flush=True)
    deadline = time.monotonic()+600
    while not ready.exists():
        assert time.monotonic() < deadline, 'Deployment hold timed out'
        time.sleep(1)
    expected = json.loads(ready.read_text()); assert expected['verified']
    for name, digest in expected['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    print('Verified deployment ready.', flush=True)
finally:
    recovery.resume(parent, p['identity'])
    print('Scheduler resumed.', flush=True)
