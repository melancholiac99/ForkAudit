"""Monitor E4/P8, invoke the explicitly requested Codex repair, archive then shut down.

Success still requires both independent final audits. Failed repair is a distinct
shutdown reason and is never reported as a successful experiment.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
import traceback

import shutdown_after_e4_p8 as COMPLETE

H = Path(__file__).resolve().parent
R = H / 'run'
E4 = COMPLETE.E4
CONFIG = R / 'repair_config.json'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n'); temp.replace(path)


def status(stage, **kw):
    value = dict(stage=stage, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), pid=os.getpid(), **kw)
    save(R / 'repair_status.json', value); print(json.dumps(value), flush=True)


def processes():
    found = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            args = (directory/'cmdline').read_bytes().split(b'\0')
            names = {Path(x.decode()).name for x in args if x}
            state = (directory/'stat').read_text().rsplit(')', 1)[1].split()[0]
            if state != 'Z' and names & {'e4_pipeline.py', 'e4_final_hook.py', 'p8_queue.py'}:
                found.append(dict(pid=int(directory.name), names=sorted(names & {'e4_pipeline.py', 'e4_final_hook.py', 'p8_queue.py'}),
                                  state=state, identity=COMPLETE.identity(int(directory.name))))
            # Repair agents may create scoped continuation scripts. Those must
            # record their PID/start identity in this explicit receipt.
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    receipt = R / 'repair_continuations.json'
    if receipt.exists():
        for p in read(receipt):
            if COMPLETE.identity(p['pid']) == p['identity']:
                found.append(p)
    return found


def inspect():
    states = dict(E4=read(E4/'run/pipeline_status.json'), P8=read(R/'status.json'))
    procs = processes(); reasons = []
    if (E4/'run/final_hook_error.json').exists():
        reasons.append('E4 final_hook_error.json exists')
    for name, s in states.items():
        if any(x in s['stage'].lower() for x in ('failed', 'error')):
            reasons.append(name+': '+s['stage'])
    names = {n for p in procs for n in p['names']}
    if states['E4']['stage'] != 'completed_verified' and not names & {'e4_pipeline.py', 'e4_final_hook.py', 'e4_repair_continuation'}:
        reasons.append('E4 queue/auditor absent before verified completion')
    if states['P8']['stage'] != 'completed_verified' and not names & {'p8_queue.py', 'p8_repair_continuation'}:
        reasons.append('P8 queue absent before verified completion')
    hold = (R/'repair_hold').exists() or any('paused' in s['stage'].lower() for s in states.values()) or any(p['state'] == 'T' for p in procs)
    return dict(states=states, processes=procs, reasons=reasons, hold=hold,
                both_done=all(s['stage'] == 'completed_verified' for s in states.values()))


def archive(label, evidence):
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    folder = R/'incidents'/f'{stamp}_{label}'; folder.mkdir(parents=True, exist_ok=False)
    save(folder/'evidence.json', evidence)
    files = []
    for base in (E4/'run', R):
        files += [p for p in base.iterdir() if p.is_file() and p.suffix in ('.log', '.json', '.txt')]
    files += [p for p in (E4/'e4.json', E4/'E4_REPORT.txt', H/'P8_CPU_REPORT.txt', H/'P8_REPORT.txt',
                          H/'p8_cpu.json', H/'p8_complete.json', H/'cpu_verification.json', H/'complete_verification.json') if p.exists()]
    manifest = {}
    with tarfile.open(folder/'experiment_logs.tar.gz', 'w:gz') as tar:
        for p in sorted(set(files)):
            arc = ('E4/' if E4 in p.parents else 'P8/') + str(p.relative_to(E4 if E4 in p.parents else H))
            tar.add(p, arcname=arc, recursive=False)
            manifest[arc] = dict(source=str(p), bytes=p.stat().st_size)
    save(folder/'log_manifest.json', manifest)
    save(folder/'archive_hash.json', dict(file='experiment_logs.tar.gz', sha256=COMPLETE.digest(folder/'experiment_logs.tar.gz')))
    for name, command, cwd in [
        ('gpu', ['nvidia-smi'], H), ('disk', ['df', '-h', str(H)], H),
        ('e4_diff', ['git', 'diff', '--binary'], E4.parent), ('p8_diff', ['git', 'diff', '--binary'], H.parent),
        ('e4_git', ['git', 'status', '--short'], E4.parent), ('p8_git', ['git', 'status', '--short'], H.parent),
    ]:
        with (folder/(name+'.txt')).open('w') as output:
            subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, timeout=60, check=False)
    inventory = {}
    for base in (E4/'data', H/'data'):
        inventory[str(base)] = {str(p.relative_to(base)): p.stat().st_size for p in base.rglob('*') if p.is_file()}
    save(folder/'data_inventory.json', inventory)
    os.sync()
    return folder


def repair(cfg, evidence, attempt):
    folder = archive(f'attempt{attempt}', evidence)
    prompt = (H/'failure_repair_prompt.txt').read_text() + '\nIncident:\n' + json.dumps(dict(evidence=evidence, archive=str(folder), attempt=attempt), indent=2)
    prompt += '\nIf using a scoped continuation rather than existing queue names, register it in '+str(R/'repair_continuations.json')+' as a JSON list of {pid, identity, names:[e4_repair_continuation or p8_repair_continuation], state:S}; identity is /proc/PID/stat starttime.\n'
    (folder/'repair_prompt.txt').write_text(prompt)
    output = folder/'agent_result.json'
    cmd = [cfg['codex'], 'exec', '--ephemeral', '--sandbox', 'danger-full-access', '-c', 'approval_policy="never"',
           '--json', '--output-schema', str(H/'repair_response_schema.json'), '--output-last-message', str(output), '-']
    status('repairing', attempt=attempt, archive=str(folder), timeout_seconds=cfg['attempt_timeout_seconds'])
    with (folder/'agent_events.jsonl').open('w') as out, (folder/'agent_stderr.log').open('w') as err:
        p = subprocess.Popen(cmd, cwd=H.parent, stdin=subprocess.PIPE, stdout=out, stderr=err, text=True, start_new_session=True)
        save(folder/'agent_launch.json', dict(pid=p.pid, identity=COMPLETE.identity(p.pid), command=cmd))
        try:
            p.communicate(prompt, timeout=cfg['attempt_timeout_seconds'])
            code = p.returncode
        except subprocess.TimeoutExpired:
            # Stop the repair agent, not an experiment it explicitly resumed in
            # its own session. Preserve all emitted logs.
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            code = -1
    report = read(output) if output.exists() else dict(outcome='unrecoverable', cause='Agent did not return a result')
    save(folder/'agent_exit.json', dict(returncode=code, report=report))
    # The agent's assertion alone does not count as recovery.
    check = inspect()
    recovered = code == 0 and report.get('outcome') == 'recovered' and not check['reasons'] and not check['hold']
    save(folder/'recovery_check.json', dict(recovered=recovered, evidence=check))
    os.sync()
    return recovered, folder


def shutdown(cfg, reason, **kw):
    assert COMPLETE.digest('/usr/bin/shutdown') == cfg['platform_shutdown_sha256']
    status('shutdown_requested', reason=reason, **kw)
    os.sync()
    subprocess.run(['/bin/bash', '/usr/bin/shutdown'], check=True, timeout=120)


def main():
    cfg = read(CONFIG)
    assert cfg['authorization'] == 'repair_on_failure_then_archive_shutdown_if_unrecoverable'
    assert COMPLETE.digest(__file__) == cfg['guardian_sha256']
    with (R/'repair_guardian.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempts = read(R/'repair_attempts.json')['used'] if (R/'repair_attempts.json').exists() else 0
        last = None
        while True:
            try:
                state = inspect()
            except (OSError, json.JSONDecodeError):
                time.sleep(5); continue  # tolerate an atomic-status handoff
            if state['hold']:
                status('held', reason='Explicit hold, paused status, or stopped queue'); time.sleep(30); continue
            if state['both_done'] and not state['reasons']:
                try:
                    evidence = COMPLETE.validate_results()
                except Exception as exc:
                    state['reasons'].append('Final result validation: '+str(exc))
                else:
                    busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
                    if not busy:
                        shutdown(cfg, 'both_experiments_verified', **evidence)
                        return
            if state['reasons']:
                # Confirm for 30 s to avoid treating normal process/status handoff as failure.
                time.sleep(30)
                again = inspect()
                if again['hold']:
                    continue
                if not again['reasons'] and not state['both_done']:
                    continue
                recovered = False
                while attempts < cfg['max_attempts_total']:
                    attempts += 1; save(R/'repair_attempts.json', dict(used=attempts))
                    try:
                        recovered, _ = repair(cfg, state, attempts)
                    except Exception as exc:
                        status('repair_attempt_error', attempt=attempts, error=str(exc)); traceback.print_exc()
                    if recovered:
                        status('recovered_monitoring', attempt=attempts); break
                    state = inspect()
                if not recovered:
                    folder = archive('unrecoverable_final', state)
                    shutdown(cfg, 'repair_unsuccessful_logs_saved', archive=str(folder), attempts=attempts)
                    return
            key = tuple((k, v['stage']) for k, v in state['states'].items())
            if key != last:
                status('monitoring', states=dict(key), attempts_used=attempts); last = key
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    print(json.dumps(inspect(), indent=2)) if args.inspect else main()
