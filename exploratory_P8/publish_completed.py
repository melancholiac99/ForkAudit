"""Publish explicitly selected verified results, preserving branches and raw data.

No checkout, merge, force push, blanket git add, or changes to scientific files.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import time

import shutdown_after_e4_p8 as C

H = Path(__file__).resolve().parent
ROOT = H.parent
E4ROOT = C.E4.parent
REMOTE = 'https://github.com/melancholiac99/ForkAudit.git'
E4_DEST = 'server/e4-results-20261101'
P8_DEST = 'server/p8-preflight-20261101'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n'); temp.replace(path)


def git(root, *args, timeout=120):
    return subprocess.check_output(['git', *args], cwd=root, text=True, stderr=subprocess.STDOUT, timeout=timeout).strip()


def files_e4():
    h = C.E4
    paths = [h/'e4.json', h/'E4_REPORT.txt', h/'run/final_verification.json', h/'run/final_artifact_manifest.json',
             h/'run/learning_rate.json', h/'run/pipeline_status.json', h/'data/pairs/manifest.json',
             h/'data/points_manifest.json', h/'data/replay_manifest.json', h/'data/mechanism.json']
    paths += sorted((h/'data/adapters').glob('*/train_log.json'))
    paths += sorted((h/'data/rollout/valid_unseen').glob('*/summary.json'))
    paths += sorted((h/'data/rollout/valid_seen').glob('*/summary.json'))
    for cond in ('F', 'L'):
        paths += [h/'data/judge'/f'32b_{cond}'/name for name in ('run.json', 'numerical_check.json')]
    return paths


def files_p8():
    paths = [H/name for name in ('P8_REPORT.txt', 'p8_complete.json', 'p8_phase_c.json', 'complete_verification.json',
                                 'artifact_manifest.json', 'data/points.json', 'data/arrays.npz', 'data/judge/requests.jsonl.gz')]
    for model in ('7b', '32b'):
        paths += [H/'data/judge'/model/name for name in ('scores.jsonl', 'run.json', 'numerical_check.json')]
    return paths


def publish(root, paths, dest, expected_branch, message):
    assert git(root, 'remote', 'get-url', '--push', 'origin') == REMOTE
    assert git(root, 'branch', '--show-current') == expected_branch, 'Current branch changed; refusing publication'
    assert not git(root, 'diff', '--cached', '--name-only'), 'Unrelated staged changes must be preserved'
    allowed = set()
    for p in paths:
        assert p.is_file() and not p.is_symlink(), p
        assert p.stat().st_size < 40*1024**2, f'Unexpectedly large publication artifact: {p}'
        allowed.add(str(p.relative_to(root)))
    dirty = set(git(root, 'diff', '--name-only').splitlines())
    assert dirty <= allowed, f'Unexpected tracked changes: {sorted(dirty-allowed)}'
    # Each ignored derived artifact is explicitly named; no raw rollout/model glob.
    git(root, 'add', '-f', '--', *sorted(allowed))
    if git(root, 'diff', '--cached', '--name-only'):
        git(root, 'commit', '-m', message)
    head = git(root, 'rev-parse', 'HEAD')
    errors = []
    for attempt in range(3):
        try:
            git(root, 'push', 'origin', f'HEAD:refs/heads/{dest}', timeout=120)
            remote = git(root, 'ls-remote', '--heads', 'origin', dest, timeout=60).split()
            assert remote and remote[0] == head, 'Remote commit does not match local commit'
            return dict(commit=head, branch=dest, files=len(allowed), verified=True,
                        url=f'https://github.com/melancholiac99/ForkAudit/tree/{dest}')
        except (subprocess.SubprocessError, AssertionError) as exc:
            errors.append(dict(attempt=attempt+1, type=type(exc).__name__))
            save(H/'run/publication_retry_errors.json', dict(branch=dest, commit=head, attempts=errors))
            if attempt < 2:
                time.sleep(20)
    raise RuntimeError(f'Publication failed after three attempts: {dest}; local commit {head} retained')


def success():
    evidence = C.validate_results()
    # Same final hash checks used by the shutdown watcher; no reanalysis or relabeling.
    report = dict(status='publishing_verified_results', evidence=evidence, publications={})
    save(H/'run/publication_status.json', report)
    report['publications']['E4'] = publish(E4ROOT, files_e4(), E4_DEST, 'main', 'Publish independently verified E4 final results')
    save(H/'run/publication_status.json', report)
    report['publications']['P8'] = publish(ROOT, files_p8(), P8_DEST, P8_DEST, 'Publish independently verified P8 complete audit and judge scores')
    report['status'] = 'github_verified'
    save(H/'run/publication_status.json', report)
    print(json.dumps(report, indent=2))
    return report


def redact(text):
    text = re.sub(r'(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?\S+', r'\1[REDACTED]', text)
    text = re.sub(r'(?i)((?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\s*[=:]\s*)\S+', r'\1[REDACTED]', text)
    return re.sub(r'\bsk-[A-Za-z0-9_-]{12,}\b', '[REDACTED_KEY]', text)


def failure(incident):
    incident = Path(incident).resolve()
    assert incident.is_relative_to((H/'run/incidents').resolve()) and incident.is_dir()
    target = H/'failure_reports'/incident.name
    target.mkdir(parents=True, exist_ok=True)
    for name in ('evidence.json', 'log_manifest.json', 'archive_hash.json'):
        if (incident/name).exists():
            (target/name).write_text(redact((incident/name).read_text()))
    lines = ['Incomplete experiment: repair failed. This is diagnostic evidence, not a scientific result.',
             'Full original logs and agent transcripts remain on the server under '+str(H/'run/incidents'), '']
    for base in (C.E4/'run', H/'run'):
        for p in sorted(base.glob('*.log')):
            # Bounded recent experiment log excerpts, excluding credential stores
            # and raw repair-agent transcripts.
            with p.open('rb') as f:
                f.seek(max(0, p.stat().st_size-24000))
                tail = f.read().decode(errors='replace')
            lines += ['FILE: '+str(p), redact(tail), '']
    (target/'DIAGNOSTICS.txt').write_text('\n'.join(lines))
    result = publish(ROOT, list(target.iterdir()), P8_DEST, P8_DEST, 'Archive diagnostic evidence for unsuccessful experiment recovery')
    save(H/'run/failure_publication_status.json', result)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode', choices=['success', 'failure', 'preflight'])
    ap.add_argument('--incident', type=Path)
    args = ap.parse_args()
    if args.mode == 'success': success()
    elif args.mode == 'failure':
        assert args.incident; print(json.dumps(failure(args.incident), indent=2))
    else:
        print(json.dumps(dict(E4_destination=E4_DEST, P8_destination=P8_DEST,
              E4_missing=[str(p) for p in files_e4() if not p.exists()], P8_missing=[str(p) for p in files_p8() if not p.exists()],
              publications_performed=False), indent=2))
