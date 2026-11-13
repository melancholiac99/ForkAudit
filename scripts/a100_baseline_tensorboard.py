"""Read-only progress and CSV tables for the bounded A100 baseline RL pilot."""
import argparse
import csv
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--logdir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    writers, positions = {}, {}

    def writer(name):
        if name not in writers:
            writers[name] = SummaryWriter(str(args.logdir / name), flush_secs=2)
            writers[name].add_text('scope', 'Baseline RL: exactly four rounds each GRPO/GiGPO-style. Fixed TRAIN tasks; no CTHA or generalization claim. R2 remains FAIL; R3 engineering passed under a new protocol.', 0)
        return writers[name]

    try:
        while True:
            for log in sorted(args.data.glob('*.log')):
                lines = log.read_text(errors='replace').splitlines()
                for line in lines[positions.get(log, 0):]:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict) and row.get('phase') == 'rollout':
                        w = writer('collection/' + log.stem)
                        for key in ['active', 'completed', 'turn']:
                            w.add_scalar('rollout/' + key, row[key], row['turn'])
                        w.flush()
                positions[log] = len(lines)
            for arm in ['grpo', 'gigpo']:
                metrics = args.data / arm / 'metrics.jsonl'
                if not metrics.exists():
                    continue
                rows = [json.loads(line) for line in metrics.read_text().splitlines()]
                # Derived view only; the append-only JSONL remains authoritative.
                path = args.data / arm / 'training_records.csv'
                temporary = path.with_suffix('.tmp')
                columns = sorted(set().union(*(row.keys() for row in rows)))
                with temporary.open('w') as stream:
                    out = csv.DictWriter(stream, fieldnames=columns)
                    out.writeheader()
                    out.writerows(rows)
                temporary.replace(path)
                if rows:
                    last = rows[-1]
                    writer('tables/' + arm).add_text('latest_record', '```json\n' + json.dumps(last, indent=2) + '\n```', int(last['step']))
                    writer('tables/' + arm).flush()
            if args.once:
                break
            time.sleep(3)
    finally:
        for w in writers.values():
            w.close()


if __name__ == '__main__':
    main()
