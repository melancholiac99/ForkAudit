"""Finish authorized P8 phase C after E4 verification; preserve every original experiment."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import traceback

H = Path(__file__).resolve().parent
RUN = H / 'run'
E4 = Path('/root/autodl-tmp/idea/exploratory_E4/run')


def read(p):
    return json.loads(Path(p).read_text())


def status(stage, **kw):
    row = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(), **kw)
    path = RUN / 'status.json'; tmp = RUN / 'status.tmp'
    tmp.write_text(json.dumps(row, indent=2)+'\n'); tmp.replace(path)
    print(json.dumps(row), flush=True)


def protect():
    for p, sha in read(RUN / 'protected_inputs.json').items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha, f'Changed frozen input: {p}'
    assert read(H / 'cpu_verification.json')['passed']


def idle():
    return not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()


def wait_e4():
    status('waiting_for_e4_verified', cpu='complete_verified', phase_c='572 reused + 1448 new contexts per model')
    while True:
        assert not (E4 / 'final_hook_error.json').exists(), 'E4 final audit failed'
        state = read(E4 / 'pipeline_status.json')['stage']
        assert 'fail' not in state.lower() and 'error' not in state.lower(), state
        if state == 'completed_verified':
            assert read(E4 / 'final_verification.json')['passed']
            if idle():
                return
        time.sleep(30)


def run(name, script, *args):
    protect()
    assert idle(), 'Another GPU job is running'
    assert shutil.disk_usage(H).free > 8*1024**3, 'Less than 8 GiB free'
    command = [sys.executable, '-u', str(H / script), *args]
    status(name, command=command)
    with (RUN / f'{name}.log').open('a') as log:
        p = subprocess.Popen(command, cwd=H.parent, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                             env=dict(os.environ, PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false'))
        try:
            code = p.wait()
        finally:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                time.sleep(2)
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        assert code == 0, f'{name}: exit {code}; inspect its log'


def main():
    RUN.mkdir(exist_ok=True)
    with (RUN / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            protect()
            wait_e4()
            for name in ('7b', '32b'):
                run('score_' + name, 'p8_judges.py', 'score', '--name', name)
            run('analyze', 'p8_judges.py', 'analyze')
            run('final_verify', 'verify_complete.py')
            assert read(H / 'complete_verification.json')['passed']
            out = read(H / 'p8_complete.json')
            status('completed_verified', reading=out['reading'], P1=out['P1'], P2=out['P2'], S4=out['S4'], S5=out['S5'],
                   report=str(H / 'P8_REPORT.txt'))
        except BaseException as exc:
            status('failed', error=str(exc)); traceback.print_exc(); raise


if __name__ == '__main__':
    main()
