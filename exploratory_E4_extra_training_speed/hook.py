"""Run training microbenchmark in the gap between completed learning-rate trials."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GUARD = ROOT/'exploratory_E4_extra_scoring_speed/hook.py'
spec = importlib.util.spec_from_file_location('existing_recovery', GUARD)
recovery = importlib.util.module_from_spec(spec); spec.loader.exec_module(recovery)


def write(stage, **values):
    values.update(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path = HERE/'hook_status.json'; tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(values, indent=2)+'\n'); tmp.replace(path)
    print(json.dumps(values), flush=True)


def main(parent, child):
    assert os.getpgrp() == os.getpid(), 'Use start_new_session=True'
    with (HERE/'hook.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        p, c = recovery.proc(parent), recovery.proc(child)
        assert p and c and c['parent'] == parent
        state = json.loads((ROOT/'exploratory_E4/run/pipeline_status.json').read_text())
        cmd = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
        assert state['stage'] == 'lr_train_5e-06' and b'e4_06_train.py' in cmd
        assert cmd[cmd.index(b'--lr')+1] == b'5e-06'
        assert b'e4_pipeline.py' in Path(f'/proc/{parent}/cmdline').read_bytes()
        guard = subprocess.Popen([sys.executable, str(GUARD), '--guard', str(os.getpid()),
                                  recovery.proc(os.getpid())['identity'], str(parent), p['identity']],
                                 start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            os.kill(parent, signal.SIGSTOP)
            write('waiting_for_first_lr_training', child=child, parent=parent, guard_pid=guard.pid)
            deadline = time.monotonic()+2*3600
            while recovery.alive(child, c['identity']):
                assert time.monotonic() < deadline
                assert recovery.alive(parent, p['identity'])
                time.sleep(5)
            folder = ROOT/'exploratory_E4/data/adapters/Aprime_s0_lr5e-06'
            completed = json.loads((folder/'train_log.json').read_text())
            assert completed['steps'] == 125 and completed['n_examples'] == 1000 and completed['lr'] == 5e-6
            expected = json.loads((ROOT/'exploratory_E4/run/protected_inputs.json').read_text())
            assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == h for name, h in expected.items())
            watched = {str(folder/name):hashlib.sha256((folder/name).read_bytes()).hexdigest()
                       for name in ['train_log.json', 'adapter_model.safetensors']}
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            assert not active, active
            write('benchmark_running', parent=parent)
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='25', OPENBLAS_NUM_THREADS='1')
            with (HERE/'benchmark.log').open('w') as output:
                subprocess.run([sys.executable, '-u', str(HERE/'benchmark.py'), 'benchmark'],
                               cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                               env=env, check=True, timeout=900)
            result = json.loads((HERE/'benchmark.json').read_text())
            assert result['complete']
            assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == h for name, h in watched.items())
            write('benchmark_complete', parent=parent, summary=result['summary'],
                  formal_adapter_hashes_unchanged=watched, formal_code_changed=False)
        except BaseException as exc:
            write('benchmark_failed', parent=parent, error_type=type(exc).__name__, message=str(exc))
            traceback.print_exc(); raise
        finally:
            recovery.resume(parent, p['identity'])
            print('Original formal pipeline resumed.', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
