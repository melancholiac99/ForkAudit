"""Power off only after this campaign has fully completed and saved its outputs.

OS poweroff is distinct from provider-side termination and billing cessation.
No provider API, instance release, disk removal, or fixed-time cutoff is used.
"""
import argparse
from collections import Counter
import fcntl
import os
from pathlib import Path
import subprocess

from .common import DEFAULT_RUN, digest_file, read, read_rows, save, utc


def completion_gate(run):
    status_path = run / 'campaign_status.json'
    if not status_path.exists() or read(status_path).get('status') != 'complete':
        return ['campaign_not_complete']
    reasons = []
    required = ['final_checkpoint.json', 'metrics.json', 'REPORT.md', 'audit_base/metrics.json',
                'audit_post/metrics.json', 'splits.json', 'budget_freeze.json']
    missing = [p for p in required if not (run / p).is_file() or (run / p).stat().st_size == 0]
    if missing:
        return ['missing:' + p for p in missing]
    if list((run / 'active_gpu_sessions').glob('*.json')):
        reasons.append('active_gpu_lease')
    final = read(run / 'final_checkpoint.json')
    metrics = read(run / 'metrics.json')
    if not metrics.get('completed_final_evaluation'):
        reasons.append('final_evaluation_not_complete')
    if not (metrics.get('training_updates') == final['update'] == read(status_path).get('training_updates')):
        reasons.append('final_update_disagreement')
    split = read(run / 'splits.json')
    budget = read(run / 'budget_freeze.json')
    expected = {t['task_id'] for t in split['V']}
    for label in ['base', 'final']:
        for decode, repeats in [('greedy', 1), ('sample', budget['final_repeats'])]:
            rows = read_rows(run / 'evaluation' / label / decode / 'episodes.jsonl')
            counts = Counter(r['task_id'] for r in rows)
            if set(counts) != expected or any(n != repeats for n in counts.values()) or len({r['trajectory_id'] for r in rows}) != len(rows):
                reasons.append(f'incomplete_episodes:{label}:{decode}')
            if label == 'final' and any(r['checkpoint_sha'] != final['sha256'] for r in rows):
                reasons.append(f'wrong_checkpoint:{decode}')
    for name, split_name, count_name in [('audit_base', 'E0', 'E0_tasks'), ('audit_post', 'E1', 'E1_tasks')]:
        expected_audit = {t['task_id'] for t in split[split_name][:budget[count_name]]}
        natural = read_rows(run / name / 'natural/episodes.jsonl')
        if {r['task_id'] for r in natural} != expected_audit or len(natural) != len(expected_audit):
            reasons.append(f'incomplete_audit_tasks:{name}')
        arms = read_rows(run / name / 'candidates.jsonl')
        points = read_rows(run / name / 'states.jsonl')
        expected_candidates = {(p['state_id'], i) for p in points for i in range(len(p['actions']))}
        if {(a['state_id'], a['candidate']) for a in arms} != expected_candidates or len(arms) != len(expected_candidates):
            reasons.append(f'incomplete_audit_candidates:{name}')
        if len(read(run / name / 'metrics.json').get('state_records', [])) != len(points):
            reasons.append(f'incomplete_audit_metrics:{name}')
        if name == 'audit_post' and any(p['checkpoint_sha'] != final['sha256'] for p in points):
            reasons.append('wrong_audit_checkpoint')
        for bank in ['a', 'b']:
            rows = read_rows(run / name / f'bank_{bank}/episodes.jsonl')
            counts = Counter(r['trajectory_id'].rsplit(':', 1)[0] for r in rows)
            expected_arms = {f'{a["state_id"]}:arm{a["candidate"]}:{bank.upper()}' for a in arms if not a['terminal']}
            if set(counts) != expected_arms or any(n != 3 for n in counts.values()) or len({r['trajectory_id'] for r in rows}) != len(rows):
                reasons.append(f'incomplete_audit_bank:{name}:{bank}')
    checkpoint = run / 'training/checkpoints' / f'global_step_{final["update"]}'
    for path in [checkpoint / 'data.pt', checkpoint / 'actor/lora_adapter/adapter_model.safetensors',
                 checkpoint / 'actor/replicated_checkpoint.json',
                 *[checkpoint / f'actor/{kind}_world_size_4_rank_{rank}.pt' for kind in ['model', 'optim', 'extra_state'] for rank in range(4)],
                 *[checkpoint / f'actor/live_fingerprint_rank_{rank}.json' for rank in range(4)]]:
        if not path.is_file() or path.stat().st_size == 0:
            reasons.append('missing_checkpoint_component:' + str(path.relative_to(run)))
    if not reasons:
        fingerprints = [read(checkpoint / f'actor/live_fingerprint_rank_{rank}.json') for rank in range(4)]
        if any(row != fingerprints[0] for row in fingerprints):
            reasons.append('checkpoint_rank_disagreement')
        if digest_file(checkpoint / 'actor/lora_adapter/adapter_model.safetensors') != final['sha256']:
            reasons.append('final_adapter_hash_mismatch')
    return reasons


def poll(run, execute=False):
    folder = run / 'completion_shutdown'
    folder.mkdir(exist_ok=True)
    with (folder / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reasons = completion_gate(run)
        request_path = folder / 'request.json'
        request = read(request_path) if request_path.exists() else {}
        if not request.get('researcher_authorized') or request.get('mode') != 'system_poweroff':
            reasons.append('system_poweroff_not_armed')
        if request.get('boot_id') != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
            reasons.append('authorization_not_for_current_boot')
        for service in ['forkaudit-shop-resume.service', 'forkaudit-shop-repair.service']:
            state = subprocess.run(['systemctl', 'is-active', service], capture_output=True, text=True, check=False).stdout.strip()
            if state in ['active', 'activating', 'deactivating']:
                reasons.append('service_still_running:' + service)
        result = {'time': utc(), 'ready': not reasons, 'reasons': reasons, 'executed': False,
                  'billing_note': 'System poweroff does not certify that provider billing has stopped.'}
        save(folder / 'status.json', result)
        if reasons or not execute or (folder / 'poweroff_requested.json').exists():
            return result
        # Preserve a durable receipt before requesting shutdown. Never issue
        # shutdown in tests or mere --check runs, and never power off on failure.
        receipt = {**result, 'command': ['shutdown', '-h', '+2'],
                   'report_sha256': digest_file(run / 'REPORT.md'),
                   'metrics_sha256': digest_file(run / 'metrics.json'),
                   'final_checkpoint': read(run / 'final_checkpoint.json')}
        save(folder / 'poweroff_requested.json', receipt)
        os.sync()
        response = subprocess.run(['shutdown', '-h', '+2'], capture_output=True, text=True, check=False)
        receipt.update(executed=response.returncode == 0, exit_code=response.returncode, stderr=response.stderr)
        save(folder / 'poweroff_requested.json', receipt)
        os.sync()
        return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    poll(args.run.resolve(), args.execute)


if __name__ == '__main__':
    main()
