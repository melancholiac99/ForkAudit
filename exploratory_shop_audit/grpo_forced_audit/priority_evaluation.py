"""Run a frozen interim evaluation at a clean, GPU-leased stage boundary.

The enclosing campaign stage owns and accounts for all GPUs. This module must
not acquire a second budget lease or add its timings to cost_ledger.jsonl.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time
import traceback

from .common import (DEFAULT_RUN, HERE, INTERFACE, SEED, append, digest_file,
                     read, read_rows, save, utc, write_once)


def queue(run, checkpoint_update=30, boundary_update=40):
    folder = run / 'priority_evaluation'
    request_path = folder / 'request.json'
    if request_path.exists():
        request = read(request_path)
        assert (request['checkpoint_update'], request['boundary_update']) == (checkpoint_update, boundary_update)
        return request
    adapter = run / f'training/checkpoints/global_step_{checkpoint_update}/actor/lora_adapter'
    frozen = [run / 'splits.json', run / 'PROTOCOL.md',
              run / 'snapshots/product_prices_frozen.json', run / 'snapshots/goals_frozen.json',
              adapter / 'adapter_model.safetensors', adapter / 'adapter_config.json']
    request = {
        'time': utc(), 'reason': 'Researcher requested full evaluation before continuing training.',
        'checkpoint_update': checkpoint_update, 'boundary_update': boundary_update,
        'checkpoint': str(adapter.resolve()), 'label': f'interim_checkpoint_{checkpoint_update:04d}',
        'tasks': [t['task_id'] for t in read(run / 'splits.json')['V']],
        'sample_repeats': 4, 'greedy_repeats': 1, 'gpu_ids': [0, 1, 2, 3],
        'interface': INTERFACE, 'master_seed': SEED,
        'seed_rule': 'Existing jobs/Runner trajectory IDs; base and interim have independent sampling streams.',
        'frozen_sha256': {str(p.resolve()): digest_file(p) for p in frozen},
        'code_sha256': {p.name: digest_file(p) for p in HERE.glob('*.py')},
        'scope': 'Interim validation, not final held-out testing or the final campaign verdict.',
        'resume': 'Preserve the boundary checkpoint including optimizer/RNG/dataloader; then follow the existing campaign.',
        'accounting': 'Substage GPU occupation is included in the enclosing campaign stage; do not sum twice.',
    }
    folder.mkdir(parents=True, exist_ok=True)
    write_once(request_path, request)
    save(folder / 'status.json', {'status': 'queued', 'time': utc(), 'request': request})
    return request


def validate_request(run, request):
    for path, expected in request['frozen_sha256'].items():
        assert digest_file(Path(path)) == expected, f'Frozen input changed: {path}'
    assert [t['task_id'] for t in read(run / 'splits.json')['V']] == request['tasks']
    stage = f'train_{request["boundary_update"]:04d}'
    assert any(r['event'] == 'end' and r['stage'] == stage and r['exit_code'] == 0
               for r in read_rows(run / 'cost_ledger.jsonl')), 'Wait for the complete training stage, not just an adapter file'
    root = run / f'training/checkpoints/global_step_{request["boundary_update"]}'
    for path in [root / 'data.pt', root / 'actor/lora_adapter/adapter_model.safetensors']:
        assert path.exists(), f'Missing resume state: {path}'
    for kind in ['model', 'optim', 'extra_state']:
        assert list((root / 'actor').glob(f'{kind}_world_size_*_rank_0.pt')), kind
    assert [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x] == request['gpu_ids']
    leases = [read(p) for p in (run / 'active_gpu_sessions').glob('*.json')]
    assert len(leases) == 1 and leases[0]['gpu_ids'] == request['gpu_ids'], 'Requires the enclosing four-GPU lease'
    return leases[0]


def validate_outputs(run, request, label, checkpoint_hash):
    """Reject missing/duplicate/wrong-checkpoint episodes, never select retries."""
    from .rollout import jobs
    tasks = read(run / 'splits.json')['V']
    result = {}
    for decode, repeats in [('greedy', 1), ('sample', request['sample_repeats'])]:
        path = run / 'evaluation' / label / decode / 'episodes.jsonl'
        rows = read_rows(path)
        step_rows = read_rows(path.parent / 'steps.jsonl')
        by_id = defaultdict(list)
        for step in step_rows:
            assert step['checkpoint_sha'] == checkpoint_hash
            assert step['interface_version'] == request['interface']
            by_id[step['trajectory_id']].append(step)
        expected = {j['trajectory_id']: j for j in jobs(tasks, 'V', label, repeats, decode)}
        counts = Counter(r['trajectory_id'] for r in rows)
        assert set(counts) == set(expected) and all(n == 1 for n in counts.values()), (label, decode, 'episode IDs/counts')
        assert set(by_id) == set(expected), (label, decode, 'step IDs')
        for row in rows:
            job = expected[row['trajectory_id']]
            assert row['task_id'] == job['task_id'] and row['goal_index'] == job['goal_index']
            assert row['checkpoint_sha'] == checkpoint_hash and row['decode'] == decode
            ss = by_id[row['trajectory_id']]
            assert [s['step'] for s in ss] == list(range(row['length']))
            assert ss[-1]['done'] and all(not s['done'] for s in ss[:-1])
            assert ss[-1]['episode_result'] == row
        result[decode] = {'episodes': len(rows), 'sha256': digest_file(path)}
    return result


def analyze(run, request):
    """Only saved trajectories and frozen environment scoring enter the report."""
    from .analysis import evaluate_saved, interval
    from .checks import environment
    summaries = {}
    engine = environment(run, 'priority_evaluation_analysis')
    try:
        for label in ['base', request['label']]:
            for decode in ['greedy', 'sample']:
                folder = run / 'evaluation' / label / decode
                summaries[label + '_' + decode] = evaluate_saved(
                    run, read_rows(folder / 'episodes.jsonl'), read_rows(folder / 'steps.jsonl'),
                    lambda i, a: engine.call(cmd='maximum', goal_index=i, asin=a), label + decode)
    finally:
        engine.close()
    differences = {}
    for decode in ['greedy', 'sample']:
        differences[decode] = {}
        for stratum in ['all_unweighted', 'easy', 'hard']:
            differences[decode][stratum] = {}
            for field in ['reward', 'strict_success', 'purchase', 'length']:
                means = []
                for label in ['base', request['label']]:
                    grouped = defaultdict(list)
                    for row in summaries[label + '_' + decode][stratum]['records']:
                        grouped[row['task_id']].append(float(row[field]))
                    means.append({k: sum(v) / len(v) for k, v in grouped.items()})
                a, b = means
                assert a.keys() == b.keys()
                differences[decode][stratum][field] = interval(
                    [b[k] - a[k] for k in sorted(a)], f'interim:{decode}:{stratum}:{field}')
    result = {'request': request, 'evaluation': summaries, 'paired_task_changes': differences,
              'final_campaign_evaluation': False, 'time': utc()}
    folder = run / 'priority_evaluation'
    save(folder / 'metrics.json', result)
    lines = [f'# 第{request["checkpoint_update"]}步中途完整评测', '',
             f'固定验证集{len(request["tasks"])}题；每模型每题{request["sample_repeats"]}条采样轨迹及1条贪心轨迹。不是最终留出测试或终点判定。',
             f'训练在完整保存第{request["boundary_update"]}步后让出四张卡；评测后沿原平台期规则继续。', '',
             '| 模型 | 解码 | 加权奖励 | 加权严格成功率 |', '| - | - | - | - |']
    for key, summary in summaries.items():
        m = summary['weighted_all']
        lines.append(f'| {key} | {key.rsplit("_", 1)[-1]} | {m["reward"]} | {m["strict_success"]} |')
    lines += ['', '按任务配对的变化区间、有效率、终止原因及分层漏斗见 metrics.json。',
              '存在已记录的BF16训推数值路径差异；本次评测不证明该问题已修复，也不隔离它的因果影响。',
              '成本明细见 events.jsonl；其占卡时间已包含于主 cost_ledger 的外层阶段，不能重复相加。']
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return result


@contextmanager
def inference_devices(ids):
    keys = ['SHOP_GPU_IDS', 'SHOP_GPU_ID']
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(SHOP_GPU_IDS=','.join(map(str, ids)), SHOP_GPU_ID=str(ids[0]))
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_pending(run):
    if run is None:
        return False
    run = Path(run).resolve()
    folder = run / 'priority_evaluation'
    if not (folder / 'request.json').exists():
        return False
    with (folder / 'execution.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if read(folder / 'status.json')['status'] == 'complete':
            return False
        request = read(folder / 'request.json')
        lease = validate_request(run, request)
        state = {'status': 'running', 'time': utc(), 'pid': os.getpid(),
                 'parent_stage': lease['stage'], 'session_id': lease['session_id'],
                 'gpu_ids': request['gpu_ids'], 'checkpoint_update': request['checkpoint_update']}
        save(folder / 'status.json', state)
        from .rollout import evaluate
        try:
            with inference_devices(request['gpu_ids']):
                for label, checkpoint in [('base', 'base'), (request['label'], request['checkpoint'])]:
                    start = time.time()
                    save(folder / 'status.json', {**state, 'substage': label, 'time': utc()})
                    append(folder / 'events.jsonl', {**state, 'event': 'start', 'label': label, 'time': utc()})
                    evaluate(run, checkpoint, label, request['sample_repeats'])
                    sha = digest_file(Path(checkpoint) / 'adapter_model.safetensors') if checkpoint != 'base' else 'Qwen2.5-7B-Instruct:a09a35458c702b33eeacc393d103063234e8bc28'
                    counts = validate_outputs(run, request, label, sha)
                    append(folder / 'events.jsonl', {**state, 'event': 'end', 'label': label, 'time': utc(),
                        'wall_seconds': time.time() - start, 'gpu_hours_included_in_parent': (time.time() - start) * len(request['gpu_ids']) / 3600,
                        'outputs': counts})
            save(folder / 'status.json', {**state, 'substage': 'analysis', 'time': utc()})
            analyze(run, request)
            save(folder / 'status.json', {**state, 'status': 'complete', 'substage': 'complete', 'time': utc()})
        except BaseException:
            error = traceback.format_exc()
            save(folder / 'status.json', {**state, 'status': 'failed', 'time': utc(), 'error': error})
            append(folder / 'events.jsonl', {'event': 'error', 'time': utc(), 'error': error})
            raise
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    queue(args.run.resolve())


if __name__ == '__main__':
    main()
