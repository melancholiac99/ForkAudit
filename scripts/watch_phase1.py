"""Watch an existing Phase 1 runner and queue one review in its Codex thread."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
LOGS = [ROOT / 'logs' / name for name in (
    'phase1_resume.log', '11_replay_pilot_resume.log', '11_replay_full.log')]


def identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if raw[0] == 'Z' else raw[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def classify(alive, complete, free_bytes, age):
    if not alive:
        return 'completed' if complete else 'unexpected_exit'
    if free_bytes < 5 * 1024 ** 3:
        return 'disk_below_5GiB'
    if age > 1800:
        return 'no_log_progress_30min'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pid', type=int, required=True)
    ap.add_argument('--start', required=True)
    ap.add_argument('--thread', required=True)
    args = ap.parse_args()
    # Prevent duplicate monitors for the same experiment.
    with open(ROOT / 'logs/phase1_watch.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            existing = [p for p in LOGS if p.exists()]
            newest = max((p.stat().st_mtime for p in existing), default=time.time())
            status = {
                'time': time.time(), 'pid': args.pid,
                'alive': identity(args.pid) == args.start,
                'free_bytes': shutil.disk_usage(ROOT).free,
                'log_age_seconds': time.time() - newest,
                'complete': 'Full replay completed:' in LOGS[0].read_text(),
            }
            event = classify(status['alive'], status['complete'],
                             status['free_bytes'], status['log_age_seconds'])
            status['event'] = event
            tmp = ROOT / 'logs/phase1_watch_status.json.tmp'
            tmp.write_text(json.dumps(status, indent=2))
            tmp.replace(ROOT / 'logs/phase1_watch_status.json')
            if event:
                message = (
                    f'【Phase 1 自动监控触发】event={event}; '
                    '这是用户授权的实验审核任务。请先阅读 logs/phase1_watch_status.json、'
                    'logs/phase1_resume.log 及最新 replay 日志，核查进程、GPU、磁盘和已保存数据。'
                    '异常时定位原因，必要时修复并验证后从已完成决策点继续；'
                    '若实验仍在运行，不要重复启动或直接杀进程。'
                    '正常完成时核查完整性和 Phase 1 指标。'
                    '遵守 CLAUDE.md 的实验协议，不改模型、温度、特征或判定规则，不删除实验数据。'
                    '修复后如需持续监控，请为新的 runner 重新启动本监控；每次触发只发送一次。'
                )
                try:
                    result = subprocess.run(
                        ['codex', 'queue', '--thread', args.thread, '--message', message],
                        cwd=ROOT, text=True, capture_output=True, timeout=45)
                    print(time.ctime(), event, result.returncode,
                          result.stdout, result.stderr, flush=True)
                    if result.returncode == 0:
                        return
                except (OSError, subprocess.TimeoutExpired) as exc:
                    print(time.ctime(), 'queue failed; retrying', repr(exc), flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
