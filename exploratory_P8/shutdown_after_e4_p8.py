"""User-authorized shutdown only after E4 AND P8 independent verification succeeds.

Does not stop experiments, modify protected inputs, or resume E5.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

H = Path(__file__).resolve().parent
R = H / 'run'
E4 = Path('/root/autodl-tmp/idea/exploratory_E4')


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def identity(pid):
    try:
        s = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if s[0] == 'Z' else s[19]
    except FileNotFoundError:
        return None


def write(stage, **kw):
    value = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(), **kw)
    tmp = R / 'shutdown_status.tmp'
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(R / 'shutdown_status.json')
    print(json.dumps(value), flush=True)


def ready(e4_status, p8_status, e4_audit, p8_audit):
    return (e4_status.get('stage') == 'completed_verified' and p8_status.get('stage') == 'completed_verified'
            and e4_audit.get('passed') is True and p8_audit.get('passed') is True)


def validate_results():
    es = read(E4 / 'run/pipeline_status.json'); ps = read(R / 'status.json')
    ev = read(E4 / 'run/final_verification.json'); pv = read(H / 'complete_verification.json')
    assert ready(es, ps, ev, pv)
    e4 = read(E4 / 'e4.json'); p8 = read(H / 'p8_complete.json')
    assert e4['reading'] == ev['reading'] == es['reading']
    assert e4['predictions'] == ev['predictions'] == es['predictions']
    assert p8['reading'] == ps['reading']
    for k in ('P1', 'P2', 'S4', 'S5'):
        assert p8[k] == ps[k]
    assert (E4 / 'run/final_artifact_manifest.json').stat().st_size > 0
    manifest = read(H / 'artifact_manifest.json')
    for name in ('p8_complete.json', 'P8_REPORT.txt', 'complete_verification.json'):
        assert digest(H / name) == manifest[name], name
    for name in ('7b', '32b'):
        run = read(H / 'data/judge' / name / 'run.json')
        assert run['complete'] is True
    return dict(e4_reading=e4['reading'], p8_reading=p8['reading'],
                e4_verification=str(E4 / 'run/final_verification.json'),
                p8_verification=str(H / 'complete_verification.json'), p8_report=str(H / 'P8_REPORT.txt'))


def selftest():
    done, pending, passed, failed = dict(stage='completed_verified'), dict(stage='waiting_for_e4_verified'), dict(passed=True), dict(passed=False)
    assert ready(done, done, passed, passed)
    for args in [(pending, done, passed, passed), (done, pending, passed, passed),
                 (done, done, failed, passed), (done, done, passed, failed), (done, done, {}, passed)]:
        assert not ready(*args)
    result = dict(passed=True, cases=6, shutdown_called=False,
                  checks='both studies completed AND both independent audits passed')
    (R / 'shutdown_selftest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


def main():
    cfg = read(R / 'shutdown_config.json')
    assert cfg['authorized_action'] == 'shutdown_after_e4_and_p8_verified'
    assert digest(__file__) == cfg['hook_sha256']
    assert digest('/usr/bin/shutdown') == cfg['platform_shutdown_sha256']
    with (R / 'shutdown.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            last = None
            while True:
                es = read(E4 / 'run/pipeline_status.json'); ps = read(R / 'status.json')
                assert not (E4 / 'run/final_hook_error.json').exists(), 'E4 independent audit hook failed'
                for name, status in [('E4', es), ('P8', ps)]:
                    assert not any(x in status['stage'].lower() for x in ('fail', 'error', 'paused')), f'{name}: {status}'
                stages = (es['stage'], ps['stage'])
                if stages != last:
                    write('waiting_for_both_verified', E4=stages[0], P8=stages[1])
                    last = stages
                if stages == ('completed_verified', 'completed_verified'):
                    evidence = validate_results()
                    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
                    if not busy:
                        break
                if ps['stage'] != 'completed_verified':
                    p = cfg['p8_queue']
                    assert identity(p['pid']) == p['identity'], 'P8 queue exited before verified completion'
                if es['stage'] != 'completed_verified':
                    assert any(identity(p['pid']) == p['identity'] for p in cfg['e4_processes']), 'E4 queue and audit hook exited before verification'
                time.sleep(30)
            assert digest('/usr/bin/shutdown') == cfg['platform_shutdown_sha256']
            write('verified_shutdown_requested', **evidence)
            os.sync()
            # This instance provides an AutoDL wrapper that stops its supervisor.
            subprocess.run(['/bin/bash', '/usr/bin/shutdown'], check=True, timeout=120)
        except BaseException as exc:
            write('failed_no_shutdown', error=str(exc))
            traceback.print_exc()
            os.sync()
            raise


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    selftest() if args.selftest else main()
