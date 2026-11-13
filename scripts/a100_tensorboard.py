"""Read-only TensorBoard sidecar for frozen A100 engineering evidence."""
import argparse
import json
import subprocess
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--logdir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.logdir.mkdir(parents=True, exist_ok=True)
    writers = {}
    seen = set()
    positions = {}
    start = read(args.data / 'registration.json')['time']

    def writer(name):
        if name not in writers:
            writers[name] = SummaryWriter(str(args.logdir / name), flush_secs=1)
            writers[name].add_text('scope', 'CTHA A100 engineering acceptance only. Learning: NOT_RUN; Generalization: NOT_RUN; CTHA online efficacy: NOT_TESTED.', 0)
        return writers[name]

    try:
        while True:
            roots = [args.data.parent / 'ctha_a100_eng_20261113_r1', args.data]
            for root in roots:
                for folder in sorted((root / 'candidates').glob('*')):
                    key = str(folder)
                    if key in seen or not (folder / 'parity.json').exists() or not (folder / 'performance.json').exists():
                        continue
                    registration = read(folder / 'registration.json')
                    config = registration['config']
                    parity = read(folder / 'parity.json')
                    perf = read(folder / 'performance.json')
                    w = writer('candidates/' + folder.name)
                    values = {**{'parity/' + k: parity[k] for k in ['mean_abs', 'p99_abs', 'max_abs', 'passed']},
                              'performance/tokens_per_second': perf['tokens_per_second'],
                              'performance/generation_seconds': perf['generation_seconds']}
                    if 'selected_gpu_peak_used_mib' in perf:
                        values['performance/peak_memory_gib'] = perf['selected_gpu_peak_used_mib'] / 1024
                    for tag, value in values.items():
                        w.add_scalar(tag, float(value), 0)
                    w.add_text('config', '```json\n' + json.dumps(config, indent=2) + '\n```', 0)
                    parameters = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
                    parameters['gpu'] = registration['visible_devices']
                    w.add_hparams(parameters, {k: float(v) for k, v in values.items()}, run_name='hparams')
                    w.flush()
                    seen.add(key)
            for name in ['repeat_a', 'repeat_b', 'initial_train']:
                path = args.data / (name + '.log')
                if not path.exists():
                    continue
                lines = path.read_text(errors='replace').splitlines()
                for line in lines[positions.get(str(path), 0):]:
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if row.get('phase') == 'rollout':
                        w = writer('rollouts/' + name)
                        step = row['turn']
                        w.add_scalar('rollout/turn', step, step)
                        w.add_scalar('rollout/active_episodes', row['active'], step)
                        w.add_scalar('rollout/terminated_episodes', row['completed'], step)
                        w.add_scalar('rollout/horizon_progress', step / 50, step)
                        w.flush()
                positions[str(path)] = len(lines)
            metrics = args.data / 'training_telemetry/metrics.jsonl'
            if metrics.exists():
                lines = metrics.read_text().splitlines()
                for line in lines[positions.get(str(metrics), 0):]:
                    row = json.loads(line)
                    step = int(row.get('optimizer_step', 0))
                    w = writer('engineering_direction_probe')
                    for key, value in row.items():
                        if key != 'time' and isinstance(value, (int, float, bool)):
                            w.add_scalar(key, float(value), step)
                    w.flush()
                positions[str(metrics)] = len(lines)
            status_path = args.data / 'status.json'
            if status_path.exists():
                status = read(status_path)
                tag = ('status', status['time'])
                if tag not in seen:
                    writer('status').add_text('current_stage', json.dumps(status, indent=2), int(time.time() - start))
                    seen.add(tag)
            try:
                values = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                                   '--format=csv,noheader,nounits'], text=True, timeout=5)
                for line in values.splitlines():
                    index, memory, utilization = [s.strip() for s in line.split(',')]
                    w = writer('hardware/gpu' + index)
                    step = max(0, int(time.time() - start))
                    w.add_scalar('gpu/memory_gib', float(memory) / 1024, step)
                    w.add_scalar('gpu/utilization_percent', float(utilization), step)
                    w.flush()
            except (subprocess.SubprocessError, ValueError):
                pass
            if args.once:
                break
            time.sleep(2)
    finally:
        for w in writers.values():
            w.close()


if __name__ == '__main__':
    main()
