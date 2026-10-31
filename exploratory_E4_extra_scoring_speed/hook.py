"""Benchmark between F and L without pausing or timing-contaminating the F child."""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def proc(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'state': fields[0], 'parent': int(fields[1]), 'identity': fields[19]}
    except FileNotFoundError:
        return None


def alive(pid, identity):
    p = proc(pid)
    return p is not None and p['identity'] == identity and p['state'] != 'Z'


def resume(pid, identity):
    p = proc(pid)
    if p and p['identity'] == identity and p['state'] in ['T', 't']:
        os.kill(pid, signal.SIGCONT)


def write(stage, **values):
    values.update(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    path = HERE/'hook_status.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(values, indent=2)+'\n'); temporary.replace(path)
    print(json.dumps(values), flush=True)


def guard(owner, owner_identity, parent, parent_identity):
    # Independent process: a killed/failed hook cannot leave the formal queue suspended.
    while alive(owner, owner_identity):
        time.sleep(5)
    # The hook owns its own process group. Remove an orphan benchmark before
    # allowing formal L to use the GPU, including when the hook was SIGKILLed.
    try:
        os.killpg(owner, signal.SIGTERM)
        time.sleep(2)
        os.killpg(owner, signal.SIGKILL)
    except ProcessLookupError:
        pass
    resume(parent, parent_identity)


def main(parent, child):
    assert os.getpgrp() == os.getpid(), 'Launch the hook with start_new_session=True'
    with (HERE/'hook.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent_info, child_info = proc(parent), proc(child)
        assert parent_info and child_info and child_info['parent'] == parent
        parent_identity, child_identity = parent_info['identity'], child_info['identity']
        status = json.loads((ROOT/'exploratory_E4/run/pipeline_status.json').read_text())
        command = Path(f'/proc/{child}/cmdline').read_bytes().split(b'\0')
        assert status['stage'] == 'judge_F' and b'--cond' in command
        assert command[command.index(b'--cond')+1] == b'F'
        parent_command = Path(f'/proc/{parent}/cmdline').read_bytes()
        assert b'e4_pipeline.py' in parent_command
        guard_process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--guard',
                                          str(os.getpid()), proc(os.getpid())['identity'],
                                          str(parent), parent_identity], start_new_session=True,
                                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            # Stop only the orchestration parent, never the timed F worker or its GPU work.
            os.kill(parent, signal.SIGSTOP)
            write('waiting_for_F_completion', parent=parent, child=child, guard_pid=guard_process.pid)
            deadline = time.monotonic()+6*3600
            while alive(child, child_identity):
                assert time.monotonic() < deadline, 'F wait exceeded six hours'
                assert alive(parent, parent_identity), 'Formal parent disappeared'
                time.sleep(5)
            completed = json.loads((ROOT/'exploratory_E4/data/judge/32b_F/run.json').read_text())
            assert completed['complete'] and completed['contexts'] == 6016
            watched = [ROOT/'exploratory_E4/data/judge/32b_F'/name for name in ['run.json', 'scores.jsonl', 'numerical_check.json']]
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
            # Exit releases the F CUDA context even if its stopped parent has not reaped it.
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            assert not active, f'Unexpected GPU processes: {active}'
            write('benchmark_running', parent=parent, F_scoring_seconds=completed['scoring_seconds'])
            with (HERE/'benchmark.log').open('w') as output:
                subprocess.run([sys.executable, '-u', str(HERE/'benchmark.py')], cwd=ROOT,
                               stdout=output, stderr=subprocess.STDOUT, check=True, timeout=1800)
            result = json.loads((HERE/'benchmark.json').read_text())
            assert result['passed']
            assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in before.items())
            # Reviewable automatic analysis, without changing formal scorer or timing data.
            write('benchmark_complete', parent=parent, summary=result['summary'],
                  formal_backend_changed=False, formal_F_hashes_unchanged=before, report=str(HERE/'REPORT.txt'))
        except BaseException as exc:
            write('benchmark_failed', parent=parent, error_type=type(exc).__name__, message=str(exc))
            traceback.print_exc()
            raise
        finally:
            resume(parent, parent_identity)
            print('Formal pipeline resumed with original backend.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=int)
    parser.add_argument('--child', type=int)
    parser.add_argument('--guard', nargs=4)
    args = parser.parse_args()
    if args.guard:
        owner, owner_identity, parent, parent_identity = args.guard
        guard(int(owner), owner_identity, int(parent), parent_identity)
    else:
        assert args.parent and args.child
        main(args.parent, args.child)
