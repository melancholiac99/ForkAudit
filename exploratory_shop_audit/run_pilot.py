"""Run the frozen 20 x 2 policy pilot; fail closed on every acceptance gate."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import digest_file, digest_value, sample_tasks, stable_seed, write_once
from exploratory_shop_audit.runtime import (HERE, MODEL, SOURCE, SYSTEM, Model, append,
    environment, make_input, parse_generation, read_rows, utc)


def public_state(state):
    return {key: value for key, value in state.items() if key not in {'backend', 'stop_score', 'tables'}}


def freeze(run):
    config = json.loads((HERE / 'config.json').read_text())
    goals = json.loads((run / 'snapshots/goals_frozen.json').read_text())
    products = json.loads((SOURCE / 'community_products.json').read_text())
    tasks = sample_tasks(products, goals, config)
    write_once(run / 'tasks.json', tasks)
    write_once(run / 'config.json', config)
    write_once(run / 'freeze.json', {'time': utc(), 'system_prompt': SYSTEM,
        'files': {p.name: digest_file(p) for p in sorted(HERE.iterdir()) if p.suffix in {'.py', '.md', '.json'}},
        'products_sha256': digest_file(SOURCE / 'community_products.json'),
        'index_sha256': digest_file(SOURCE / 'products.sqlite3'),
        'price_snapshot_sha256': digest_file(run / 'snapshots/product_prices_frozen.json'),
        'goal_snapshot_sha256': digest_file(run / 'snapshots/goals_frozen.json'),
        'model_revision': config['model']['expected_revision']})
    return tasks


def assert_same(left, right, label):
    # Key/session URLs are not policy observations; native text and backend state must match too.
    if left != right:
        keys = [k for k in left if left.get(k) != right.get(k)]
        raise RuntimeError(f'{label} mismatch: {keys}')


def cpu_acceptance(run, a, b, tasks):
    assert_same(a.ready['tables'], b.ready['tables'], 'independent frozen tables')
    parser = a.call(cmd='parser_audit')
    rows = []
    for i, task in enumerate(tasks['pilot']):
        key = f'cpu:{i}'
        first = a.call(cmd='open', key=key, goal_index=task['goal_index'])
        same = b.call(cmd='open', key=key, goal_index=task['goal_index'])
        assert_same(first, same, 'reset')
        for action in ['invalid[action]', 'finish[no_suitable_product]']:
            ra = a.call(cmd='step', key=key, action=action)
            rb = b.call(cmd='step', key=key, action=action)
            assert_same(ra, rb, 'invalid/finish transition')
        reset = a.call(cmd='open', key=key, goal_index=task['goal_index'])
        assert_same(first, reset, 'reset after terminal')
        rows.append({'task_id': task['task_id'], 'reset_and_invalid_and_finish_equal': True})
        a.call(cmd='close', key=key)
        b.call(cmd='close', key=key)
    result = {'tables': a.ready['tables'], 'parser': parser, 'checks': rows, 'passed': True}
    write_once(run / 'cpu_acceptance.json', result)


def collect(run, a, b, model, tokenizer, tasks, phase='pilot', repeats=2):
    existing = {r['trajectory_id']: r for r in read_rows(run / 'episodes.jsonl')}
    saved_steps = read_rows(run / 'steps.jsonl')
    saved_by_id = {}
    for row in saved_steps:
        saved_by_id.setdefault(row['trajectory_id'], []).append(row)
    audit = read_rows(run / 'acceptance_checks.jsonl')
    restored = sum(r['check'] == 'policy_prefix' for r in audit)
    buys = sum(r['check'] == 'isolated_buy' for r in audit)
    todo = [(task, repeat) for task in tasks for repeat in range(repeats)]
    for offset in range(0, len(todo), 8):
        active = []
        for task, repeat in todo[offset:offset + 8]:
            trajectory = f'{phase}:{task["task_id"]}:{repeat}'
            if trajectory in existing:
                continue
            prior = saved_by_id.get(trajectory, [])
            if prior and prior[-1]['done']:
                raise RuntimeError('Terminal step without episode: reconcile saved evidence before resume')
            prefix = [s['action'] for s in prior]
            state = a.call(cmd='open', key=trajectory, goal_index=task['goal_index'], prefix=prefix)
            history = [(s['before']['observation'], s['generation']['text']) for s in prior]
            active.append({'task': task, 'repeat': repeat, 'id': trajectory, 'state': state,
                           'prefix': prefix, 'history': history,
                           'seed': stable_seed(20261115, task['task_id'], repeat, phase)})
        while active:
            requests = []
            inputs = []
            for episode in active:
                state = episode['state']
                try:
                    actual = make_input(tokenizer, state['task'], episode['history'], state['observation'])
                except ValueError:
                    append(run / 'overflow_evidence.jsonl', {'trajectory_id': episode['id'],
                           'step': len(episode['prefix']), 'state': public_state(state),
                           'history': episode['history']})
                    raise
                inputs.append(actual)
                requests.append({'token_ids': actual['token_ids'],
                    'seed': stable_seed(episode['seed'], 'decision', len(episode['prefix']))})
            outputs = model.generate(requests)
            survivors = []
            for episode, actual, request, generated in zip(active, inputs, requests, outputs):
                key, before = episode['id'], episode['state']
                step = len(episode['prefix'])
                action = parse_generation(generated['text'])
                # Acceptance uses the same task, prefix, input and action, never outcome selection.
                restore = phase == 'pilot' and restored < 20
                if restore:
                    restored_state = b.call(cmd='open', key=key, goal_index=episode['task']['goal_index'], prefix=episode['prefix'])
                    assert_same(before, restored_state, 'policy prefix restore')
                if phase == 'pilot' and before['stop_available'] and buys < 20:
                    stop_key = key + ':buy_audit'
                    stopped = b.call(cmd='open', key=stop_key, goal_index=episode['task']['goal_index'], prefix=episode['prefix'])
                    assert_same(before, stopped, 'isolated buy restore')
                    purchase = b.call(cmd='step', key=stop_key, action='click[buy now]')
                    if not purchase['status'].get('purchase') or purchase['status']['reward'] != before['stop_score']['reward']:
                        raise RuntimeError('Stop score differs from actual isolated purchase')
                    if purchase['no_budget_gate'] != before['stop_score']['no_budget_gate']:
                        raise RuntimeError('Diagnostic stop score differs')
                    append(run / 'acceptance_checks.jsonl', {'check': 'isolated_buy', 'trajectory_id': key,
                           'step': step, 'actual': purchase['status'], 'offline': before['stop_score']})
                    buys += 1
                    b.call(cmd='close', key=stop_key)
                result = a.call(cmd='step', key=key, action=action)
                if restore:
                    replayed = b.call(cmd='step', key=key, action=action)
                    assert_same(result, replayed, 'restored next step')
                    append(run / 'acceptance_checks.jsonl', {'check': 'policy_prefix', 'trajectory_id': key,
                           'step': step, 'state_sha256': digest_value(before), 'next_sha256': digest_value(result)})
                    restored += 1
                    b.call(cmd='close', key=key)
                row = {'task_id': episode['task']['task_id'], 'trajectory_id': key, 'phase': phase,
                       'repeat': episode['repeat'], 'trajectory_seed': episode['seed'], 'seed': request['seed'],
                       'step': step, 'remaining': before['remaining'], 'input': actual, 'generation': generated,
                       'action': action, 'legal': result['legal'], 'before': public_state(before),
                       'after': public_state(result['state']), 'reward': result['status']['reward'],
                       'done': result['status']['done']}
                append(run / 'steps.jsonl', row)
                append(run / 'backend_steps.jsonl', {'trajectory_id': key, 'step': step,
                       'before': before['backend'], 'after': result['state']['backend'],
                       'stop_score': before['stop_score'], 'status': result['status'],
                       'no_budget_gate': result['no_budget_gate']})
                episode['prefix'].append(action)
                episode['history'].append((before['observation'], generated['text']))
                episode['state'] = result['state']
                if result['status']['done']:
                    status = result['status']
                    record = {'task_id': episode['task']['task_id'], 'trajectory_id': key, 'phase': phase,
                              'repeat': episode['repeat'], 'category': episode['task']['category'],
                              'seed': episode['seed'], 'length': len(episode['prefix']),
                              'reward': status['reward'], 'no_budget_gate': result['no_budget_gate'],
                              'success': status.get('reward_detail', {}).get('reward_type') == 'gold_purchase' and status.get('reward_valid', False),
                              'purchase': bool(status.get('purchase')), 'reward_valid': status.get('reward_valid'),
                              'termination_reason': status.get('termination_reason'), 'status': status}
                    append(run / 'episodes.jsonl', record)
                    existing[key] = record
                    a.call(cmd='close', key=key)
                else:
                    if len(episode['prefix']) >= 35:
                        raise RuntimeError('Native environment did not terminate at 35 actions')
                    survivors.append(episode)
            active = survivors
        progress = {'time': utc(), 'phase': phase, 'complete_episodes': len(existing),
                    'restores': restored, 'buys': buys, 'gpu_hours': model.hours()}
        append(run / 'progress.jsonl', progress)
        print(json.dumps(progress), flush=True)
    return {'restores': restored, 'isolated_buys': buys, 'episodes': len(existing)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    a = b = model = None
    try:
        a = environment(run, 'engine_a')
        b = environment(run, 'engine_b')
        if (run / 'freeze.json').exists():
            raise RuntimeError('Run already frozen; resume requires explicit saved-step reconciliation')
        tasks = freeze(run)
        cpu_acceptance(run, a, b, tasks)
        if args.cpu_only:
            return
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        model = Model(run)
        accepted = collect(run, a, b, model, tokenizer, tasks['pilot'])
        accepted['passed'] = accepted['restores'] >= 20 and accepted['isolated_buys'] >= 20 and accepted['episodes'] == 40
        accepted['throughput_fork_pending'] = True
        write_once(run / 'pilot_acceptance.json', accepted)
        print(json.dumps(accepted), flush=True)
    except Exception:
        evidence = {'time': utc(), 'error': traceback.format_exc(), 'formal_collection': False}
        append(run / 'errors.jsonl', evidence)
        print(json.dumps(evidence), flush=True)
        raise
    finally:
        if model: model.close()
        if a: a.close()
        if b: b.close()
        hours = sum(row['gpu_hours'] for row in read_rows(run / 'gpu_sessions.jsonl'))
        (run / 'costs.json').write_text(json.dumps({'gpu_hours': hours, 'gpu_count': 1 if hours else 0,
            'cap_gpu_hours': 16, 'formal_collection': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
