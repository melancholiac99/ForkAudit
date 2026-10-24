"""User-authorized completion monitor: validate, archive, sync, then stop AutoDL."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from watch_phase1 import identity

ROOT = Path(__file__).resolve().parent.parent


def sha(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def save_archive(root, status):
    dest = root / 'artifacts' / ('phase1c_' + time.strftime('%Y%m%d_%H%M%S'))
    dest.mkdir(parents=True)
    files = []
    for folder in ('scripts', 'data', 'prompts', 'hidden', 'eval', 'logs', 'env'):
        files += [p for p in (root / folder).rglob('*') if p.is_file()
                  and '__pycache__' not in p.parts and not p.name.endswith(('.tmp', '.lock'))]
    files += list(root.glob('*.md'))
    manifest = {}
    with tarfile.open(dest / 'saved_files.tar', 'w') as archive:
        for p in sorted(set(files)):
            name = str(p.relative_to(root))
            with p.open('rb') as f:
                manifest[name] = sha(f)
            archive.add(p, arcname=name, recursive=False)
    # Read back archive bytes and verify every saved file, rather than trusting writes.
    with tarfile.open(dest / 'saved_files.tar', 'r') as archive:
        members = archive.getmembers()
        assert len(members) == len(manifest)
        for member in members:
            with archive.extractfile(member) as f:
                assert sha(f) == manifest[member.name], member.name
    (dest / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (dest / 'status.json').write_text(json.dumps(status, indent=2))
    for p in dest.iterdir():
        with p.open('rb') as f:
            os.fsync(f.fileno())
    os.sync()
    return dest


def validate():
    audit_log = ROOT / 'logs/phase1c_final_audit.log'
    with audit_log.open('w') as log:
        result = subprocess.run(
            ['/root/miniconda3/envs/m0/bin/python', 'scripts/review/r04_live_integrity.py'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Integrity audit failed: exit {result.returncode}; see {audit_log}')
    report = json.loads((ROOT / 'eval/phase1c_live_integrity.json').read_text())
    base = {}
    for p in (ROOT / 'data/replay').glob('*/*.json'):
        r = json.loads(p.read_text())
        if r['cf_defined']:
            base[(r['traj_id'], r['turn'])] = r
    live = set()
    for key, r in base.items():
        b = json.loads((ROOT / f'data/replay_b1/{key[0]}/t{key[1]:02d}.json').read_text())
        outcomes = []
        for record in (r, b):
            outcomes += record['factual_outcomes'] + [x for arm in record['alt_outcomes'] for x in arm]
        if 0 < sum(outcomes) < len(outcomes):
            live.add(key)
    assert report['trajectories'] == 100 and not report['errors']
    for bank in range(2, 6):
        keys = set()
        for p in (ROOT / f'data/replay_b{bank}').glob('*/*.json'):
            r = json.loads(p.read_text())
            keys.add((r['traj_id'], r['turn']))
        assert keys == live, f'Bank {bank + 1} incomplete: {len(keys)}/{len(live)}'
    return report


def shutdown(status):
    status['shutdown_requested_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    p = ROOT / 'logs/phase1c_shutdown_status.json'
    p.write_text(json.dumps(status, indent=2))
    with p.open('rb') as f:
        os.fsync(f.fileno())
    os.sync()
    print('Files verified and synced; executing /usr/bin/shutdown', flush=True)
    result = subprocess.run(['/bin/bash', '/usr/bin/shutdown'])
    print('shutdown command returned', result.returncode, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--thread', required=True)
    args = ap.parse_args()
    with (ROOT / 'logs/phase1c_finish.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print('Armed: waiting for runner', args.pid, 'to finish banks 1–6', flush=True)
        while identity(args.pid) == args.start:
            time.sleep(30)
        status = {'runner_pid': args.pid, 'time': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
        try:
            status['audit'] = validate()
            status['result'] = 'banks_1_to_6_complete_and_validated'
        except Exception as exc:
            status['result'] = 'review_required'
            status['error'] = repr(exc)
        try:
            status['archive'] = str(save_archive(ROOT, status))
        except Exception as exc:
            # Keep the original files; do not delete data to make room for an archive.
            status['archive_error'] = repr(exc)
            status['result'] = 'review_required'
            os.sync()
        if status['result'] == 'review_required':
            (ROOT / 'logs/phase1c_shutdown_status.json').write_text(json.dumps(status, indent=2))
            os.sync()
            message = (
                '【Phase 1C 关机前异常审核】用户已要求任务完成后保存代码、结果和日志，'
                '确认落盘后执行 /usr/bin/shutdown；失败无法恢复时也保存错误并关机。'
                '请读 logs/phase1c_shutdown_status.json 和 logs/phase1c_final_audit.log，'
                '尝试修复并续跑至库6，库7–10已明确取消。若续跑，重新挂载 finish_phase1c.py '
                '监控新runner。若无法恢复，保存错误日志、执行 sync 后调用 /usr/bin/shutdown。'
                '若是归档失败则先解决保存问题。不得删除实验数据或更改科学协议。'
            )
            try:
                result = subprocess.run(['codex', 'queue', '--thread', args.thread,
                                         '--message', message], cwd=ROOT, timeout=45)
                if result.returncode == 0:
                    print('Queued recovery review; reviewer is responsible for final shutdown', flush=True)
                    return
                status['review_error'] = f'queue exit {result.returncode}'
            except Exception as exc:
                status['review_error'] = repr(exc)
            status['result'] = 'recovery_unavailable_original_files_synced'
        shutdown(status)


if __name__ == '__main__':
    main()
