"""Authorized completion hook: wait for E2, report, publish, verify remote, shut down.

Any failure leaves the machine running and records the error. Never force pushes.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exploratory_P4'
STATUS = OUT / 'e2_publish_status.json'
EXPECTED_REMOTE = 'https://github.com/melancholiac99/ForkAudit.git'

def run(args, **kwargs):
    return subprocess.run(args, cwd=ROOT, check=True, text=True, **kwargs)

def git(*args):
    return run(['git', *args], capture_output=True).stdout.strip()

def state(stage, **details):
    data = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **details)
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(STATUS)
    print(json.dumps(data, ensure_ascii=False), flush=True)

def process_identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if raw[0] == 'Z' else raw[19]
    except FileNotFoundError:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver-pid', type=int, required=True)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert git('remote', 'get-url', 'origin') == EXPECTED_REMOTE
    assert git('remote', 'get-url', '--push', 'origin') == EXPECTED_REMOTE
    assert git('branch', '--show-current') == args.branch
    assert Path('/usr/bin/shutdown').is_file()
    identity = process_identity(args.driver_pid)
    assert identity, 'The existing completion driver must be running when hook starts'
    assert b'finish_e2.py' in Path(f'/proc/{args.driver_pid}/cmdline').read_bytes()
    if args.check:
        print('Hook preflight passed; no files committed, no push, no shutdown.')
        return
    with (OUT / 'e2_publish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            state('waiting_for_existing_driver', driver_pid=args.driver_pid, branch=args.branch)
            while process_identity(args.driver_pid) == identity:
                time.sleep(15)
            events = [json.loads(line) for line in (OUT / 'e2_finish.log').read_text().splitlines() if line.startswith('{')]
            assert any(e.get('event') == 'finished' and e.get('command') == ['exploratory_P4/verify_e2.py'] for e in events), 'E2 driver did not finish verification'
            for model in ['7b', '32b']:
                r = json.loads((OUT / f'e2_{model}/run.json').read_text())
                assert r['complete'] and r['contexts'] == 3492
            assert json.loads((OUT / 'e2_verification.json').read_text())['passed']
            result = json.loads((OUT / 'e2.json').read_text())
            assert not result['E3_allowed'], 'E3 gate unexpectedly requires more work; manual review needed'
            state('rendering_verified_results')
            with (OUT / 'e2_render.log').open('w') as log:
                run([sys.executable, 'exploratory_P4/render_e2.py'], stdout=log, stderr=subprocess.STDOUT)
            files = ['e2.json', 'e2_verification.json', 'E2_REPORT.txt', 'e2_summary.png',
                     'e2_summary.svg', 'status.json', 'e2_analysis.log', 'e2_verification.log',
                     'e2_render.log', 'e2_finish.log', 'e2_32b.log', 'e2_32b_download.json',
                     'e2_32b_chunks.log', 'e2_32b_chunks_state.json']
            files += [f'e2_{m}/{f}' for m in ['7b', '32b'] for f in ['run.json', 'numerical_check.json', 'scores.jsonl']]
            files += ['e2_7b.log', 'e2_32b_model_source.json', 'e2_environment.json',
                      'e2_preparation.json', 'e2_requests.jsonl.gz', 'p4_e2_backend.py',
                      'p4_e2_judge_score.py', 'p4_e2_analyze.py', 'verify_e2.py',
                      'render_e2.py', 'publish_e2_when_done.py']
            manifest = {}
            for name in sorted(set(files)):
                path = OUT / name
                assert path.is_file(), name
                assert path.stat().st_size < 95 * 1024**2, ('GitHub size limit', name)
                manifest[str(path.relative_to(ROOT))] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (OUT / 'e2_artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            files.append('e2_artifact_manifest.json')
            assert git('branch', '--show-current') == args.branch
            assert git('remote', 'get-url', '--push', 'origin') == EXPECTED_REMOTE
            assert not git('diff', '--cached', '--name-only'), 'Unrelated staged changes; refusing to commit them'
            run(['git', 'add', '--', *['exploratory_P4/' + f for f in sorted(set(files))]])
            if git('diff', '--cached', '--name-only'):
                run(['git', 'commit', '-m', 'Complete P4 E2: verified 7B/32B results and preregistered analysis'])
            head = git('rev-parse', 'HEAD')
            state('pushing', commit=head, verdict=result['verdict'], predictions=result['predictions'])
            network = ['git', '-c', 'http.version=HTTP/1.1']
            proxy = os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
            if proxy:
                network += ['-c', 'http.proxy=' + proxy]
            for attempt in range(3):
                try:
                    run(network + ['push', 'origin', f'HEAD:refs/heads/{args.branch}'], timeout=300)
                    remote = run(network + ['ls-remote', 'origin', f'refs/heads/{args.branch}'], capture_output=True, timeout=90).stdout.split()
                    assert remote and remote[0] == head, 'Remote commit verification failed'
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired, AssertionError):
                    if attempt == 2:
                        raise
                    time.sleep(30)
            state('github_verified_shutdown_requested', commit=head, branch=args.branch,
                  verdict=result['verdict'], report='exploratory_P4/E2_REPORT.txt')
            os.sync()
            run(['/bin/bash', '/usr/bin/shutdown'], timeout=120)
        except Exception as exc:
            # Do not include command arguments (proxy configuration) in status artifacts.
            state('failed_no_shutdown', error_type=type(exc).__name__)
            traceback.print_exc()
            os.sync()
            raise

if __name__ == '__main__':
    main()
