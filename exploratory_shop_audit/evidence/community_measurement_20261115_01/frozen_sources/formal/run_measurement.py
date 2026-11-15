"""Continue only a passed policy pilot; freeze throughput tier before formal data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import digest_file, write_once
from exploratory_shop_audit.runtime import HERE, MODEL, Model, append, environment, read_rows, utc
from exploratory_shop_audit.run_pilot import collect
from exploratory_shop_audit.forks import audit_state, judge_state, select_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--throughput-only', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    accepted = json.loads((run / 'pilot_acceptance.json').read_text())
    if not accepted['passed']:
        raise RuntimeError('Pilot gate failed; refuse formal collection')
    tasks = json.loads((run / 'tasks.json').read_text())
    a = b = model = None
    stage = 'throughput'
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
        a, b = environment(run, 'engine_a'), environment(run, 'engine_b')
        if a.ready['tables'] != b.ready['tables']:
            raise RuntimeError('Cross-process tables changed')
        model = Model(run)
        pilot_steps = [r for r in read_rows(run / 'steps.jsonl') if r['phase'] == 'pilot']
        pilot_points = select_points(pilot_steps, [t['task_id'] for t in tasks['pilot']], 3, 'pilot')
        write_once(run / 'pilot_fork_manifest.json', pilot_points)
        if pilot_points['selected'] != 3:
            raise RuntimeError('Insufficient three-page pilot throughput states')
        results = [audit_state(run, a, model, tokenizer, p, 'pilot_forks') for p in pilot_points['points']]
        # Cost stratification uses episode lengths and measured times, never rewards.
        pilot_episodes = [r for r in read_rows(run / 'episodes.jsonl') if r['phase'] == 'pilot']
        length_counts = dict(sorted(Counter(r['length'] for r in pilot_episodes).items()))
        inference = read_rows(run / 'inference_costs.jsonl')
        max_seconds_per_request = max(r['seconds_including_retries'] / r['requests'] for r in inference)
        # Upper-envelope 35-action cost, plus observed restore overhead, all four
        # arms treated as nonterminal. This deliberately includes unused slots.
        restore_per_candidate = max(r['restore_seconds'] / max(1, r['candidate_count']) for r in results)
        cost_per_state = max(max(r['total_seconds'] * 4 / r['candidate_count'] for r in results),
                             16 * max_seconds_per_request + 4 * 12 * 34 * max_seconds_per_request + 4 * restore_per_candidate)
        # Pilot actual environment and auditing CPU overhead is included in an
        # additional per-step envelope derived from the occupied pilot session.
        prior_sessions = read_rows(run / 'gpu_sessions.jsonl')
        pilot_seconds_per_step = sum(r['gpu_hours'] for r in prior_sessions) * 3600 / max(1, len(pilot_steps))
        episode_cost = max(pilot_seconds_per_step, max_seconds_per_request) * 35
        options = []
        for n_tasks, states in [(100, 30), (200, 30), (200, 60), (200, 120)]:
            future_seconds = n_tasks * 4 * episode_cost + states * cost_per_state + states * 2 * max_seconds_per_request
            options.append({'tasks': n_tasks, 'states': states, 'remaining_gpu_hours_with_25pct_margin': future_seconds * 1.25 / 3600,
                            'fits': model.hours() + future_seconds * 1.25 / 3600 < 16})
        feasible = [r for r in options if r['fits']]
        if not feasible:
            raise RuntimeError('No preregistered tier fits measured throughput plus margin')
        selection = feasible[-1]
        tier = {'selected': selection, 'options': options, 'gpu_hours_at_freeze': model.hours(),
                'pilot_length_counts': length_counts, 'seconds_per_episode_envelope': episode_cost,
                'seconds_per_state_envelope': cost_per_state, 'safety_multiplier': 1.25,
                'method': 'measured upper envelope; 35 steps and four nonterminal arms; no outcome-dependent sizing',
                'scripts': {p.name: digest_file(p) for p in HERE.glob('*.py')}}
        if (run / 'tier_freeze.json').exists():
            tier = json.loads((run / 'tier_freeze.json').read_text())
            selection = tier['selected']
        else:
            write_once(run / 'tier_freeze.json', tier)
        append(run / 'progress.jsonl', {'time': utc(), 'phase': 'tier_frozen', **selection, 'gpu_hours': model.hours()})
        print(json.dumps(tier), flush=True)
        if args.throughput_only:
            return
        stage = 'formal'
        formal_tasks = tasks['formal'][:selection['tasks']]
        collect(run, a, b, model, tokenizer, formal_tasks, phase='formal', repeats=4)
        stage = 'formal_forks'
        steps = read_rows(run / 'steps.jsonl')
        manifest = select_points(steps, [t['task_id'] for t in formal_tasks], selection['states'], 'formal')
        # Save any shortfall; no retrospective outcome-driven substitute task selection.
        write_once(run / 'state_manifest.json', manifest)
        for point in manifest['points']:
            audit_state(run, a, model, tokenizer, point, 'formal_forks')
        stage = 'information_control'
        candidates = read_rows(run / 'formal_forks/candidates.jsonl')
        for point in manifest['points']:
            judge_state(run, model, tokenizer, point, [r for r in candidates if r['state_id'] == point['state_id']])
        write_once(run / 'collection_complete.json', {'time': utc(), 'formal_tasks': len(formal_tasks),
                   'requested_states': selection['states'], 'selected_states': manifest['selected'],
                   'gpu_hours': model.hours(), 'analysis_pending': True})
    except Exception:
        error = {'time': utc(), 'stage': stage, 'error': traceback.format_exc(), 'formal_collection_stopped': True}
        append(run / 'errors.jsonl', error)
        print(json.dumps(error), flush=True)
        raise
    finally:
        if model: model.close()
        if a: a.close()
        if b: b.close()
        hours = sum(r['gpu_hours'] for r in read_rows(run / 'gpu_sessions.jsonl'))
        (run / 'costs.json').write_text(json.dumps({'gpu_hours': hours, 'gpu_count': 1, 'cap_gpu_hours': 16,
            'collection_complete': (run / 'collection_complete.json').exists()}, indent=2) + '\n')


if __name__ == '__main__':
    main()
