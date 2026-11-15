"""Validate provenance, independence, counts, exact prompts and no hidden-input path."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import digest_file, stable_seed
from exploratory_shop_audit.runtime import MODEL, SYSTEM, make_input, read_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True)
    tasks = json.loads((run / 'tasks.json').read_text())
    assert all(t['tag'] == 'train' for t in tasks['pilot'] + tasks['formal'])
    assert not {t['target_asin'] for t in tasks['pilot']} & {t['target_asin'] for t in tasks['formal']}
    episodes = read_rows(run / 'episodes.jsonl')
    steps = read_rows(run / 'steps.jsonl')
    by_episode = defaultdict(list)
    for step in steps:
        by_episode[step['trajectory_id']].append(step)
    assert len({r['trajectory_id'] for r in episodes}) == len(episodes)
    counts = Counter((r['phase'], r['task_id']) for r in episodes)
    assert all(n == (2 if phase == 'pilot' else 4) for (phase, tid), n in counts.items())
    for ep in episodes:
        rows = sorted(by_episode[ep['trajectory_id']], key=lambda r: r['step'])
        assert [r['step'] for r in rows] == list(range(ep['length']))
        assert rows[-1]['done'] and not any(r['done'] for r in rows[:-1])
        assert rows[-1]['reward'] == ep['reward']
        assert 1 <= ep['length'] <= 35
        history = []
        for row in rows:
            assert row['remaining'] == 35 - row['step']
            assert row['seed'] == stable_seed(ep['seed'], 'decision', row['step'])
            reconstructed = make_input(tokenizer, row['before']['task'], history, row['before']['observation'])
            assert json.loads(json.dumps(reconstructed)) == row['input']
            assert row['input']['token_count'] <= 8192
            public = row['before']['public']
            assert not {'goal', 'reward', 'reward_detail', 'goal_options', 'target_asin', 'answer'} & public.keys()
            assert all(field not in row['input']['text'] for field in ['reward_detail', 'goal_options', 'price_upper', 'target_asin'])
            history.append((row['before']['observation'], row['generation']['text']))
    manifest = json.loads((run / 'state_manifest.json').read_text())
    points = {p['state_id']: p for p in manifest['points']}
    assert len({p['task_id'] for p in points.values()}) == len(points)
    assert Counter(p['page_type'] for p in points.values()) == {'search_home': 20, 'search_results': 20, 'product_detail': 20}
    generated_total = len(steps)
    bank_rows_total = 0
    reconstructed_replay_inputs = 0
    for phase in ['pilot_forks', 'formal_forks']:
        root = run / phase
        drafts = read_rows(root / 'drafts.jsonl')
        draft_counts = Counter(r['state_id'] for r in drafts)
        assert all(n == 16 for n in draft_counts.values())
        assert len({(r['state_id'], r['draft']) for r in drafts}) == len(drafts)
        candidates = read_rows(root / 'candidates.jsonl')
        state_results = {r['state_id']: r for r in read_rows(root / 'state_results.jsonl')}
        banks = {b: read_rows(root / f'replay_bank_{b.lower()}.jsonl') for b in ['A', 'B']}
        keys = []
        seeds = []
        for bank, rows in banks.items():
            for r in rows:
                assert r['bank'] == bank
                assert r['seed'] == stable_seed(20261115, r['state_id'], r['candidate'], bank, r['repeat'])
                assert r['length'] <= r['initial_remaining']
                keys.append(r['replay_id']); seeds.append(r['seed'])
        assert len(set(keys)) == len(keys) and len(set(seeds)) == len(seeds)
        bank_rows_total += len(keys)
        for c in candidates:
            for bank in ['A', 'B']:
                samples = [r for r in banks[bank] if r['state_id'] == c['state_id'] and r['candidate'] == c['candidate']]
                expected = 0 if c['terminal'] or (phase == 'formal_forks' and not state_results[c['state_id']]['defined']) else 6
                assert len(samples) == expected
            if c['terminal']:
                assert c['clean_feedback'] == '该动作终止回合。'
            else:
                assert c['clean_feedback'] == c['one_step']['state']['observation']
        replay_steps = read_rows(root / 'replay_steps.jsonl')
        assert len({(r['replay_id'], r['step']) for r in replay_steps}) == len(replay_steps)
        phase_manifest = json.loads((run / ('pilot_fork_manifest.json' if phase == 'pilot_forks' else 'state_manifest.json')).read_text())
        phase_points = {r['state_id']: r for r in phase_manifest['points']}
        arm_index = {(r['state_id'], r['candidate']): r for r in candidates}
        replay_index = {r['replay_id']: r for rows in banks.values() for r in rows}
        replay_histories = {}
        for row in replay_steps:
            episode = replay_index[row['replay_id']]
            point = phase_points[row['state_id']]
            candidate = arm_index[(row['state_id'], episode['candidate'])]
            if row['replay_id'] not in replay_histories:
                replay_histories[row['replay_id']] = list(point['history']) + [(point['before']['observation'], 'Action: ' + candidate['action'])]
            history = replay_histories[row['replay_id']]
            actual = make_input(tokenizer, row['before']['task'], history, row['before']['observation'])
            assert json.loads(json.dumps(actual)) == row['input']
            assert row['seed'] == stable_seed(episode['seed'], 'decision', row['step'])
            assert row['remaining'] == episode['initial_remaining'] - row['step']
            history.append((row['before']['observation'], row['generation']['text']))
            reconstructed_replay_inputs += 1
        generated_total += len(drafts) + len(replay_steps)
    outputs = read_rows(run / 'judge_outputs.jsonl')
    generated_judges = [r for r in outputs if 'generation' in r]
    assert len({(r['state_id'], r['condition']) for r in generated_judges}) == len(generated_judges)
    requests = {(r['state_id'], r['condition']): r for r in read_rows(run / 'judge_requests.jsonl')}
    for sid in {key[0] for key in requests}:
        s, f = requests[(sid, 'S')], requests[(sid, 'F')]
        assert s['order'] == f['order']
        assert s['messages'][:-1] == f['messages'][:-1] == points[sid]['input']['messages']
        assert s['temperature'] == f['temperature'] == 0
        for req in [s, f]:
            assert tokenizer.apply_chat_template(req['messages'], tokenize=True, add_generation_prompt=True) == req['token_ids']
        assert not any(field in str(s['messages']) + str(f['messages']) for field in ['reward_detail', 'goal_options', 'price_upper', 'target_asin'])
    generated_total += len(generated_judges)
    accounted_requests = sum(r['requests'] for r in read_rows(run / 'inference_costs.jsonl'))
    assert generated_total == accounted_requests, (generated_total, accounted_requests)
    cost = sum(r['gpu_hours'] for r in read_rows(run / 'gpu_sessions.jsonl'))
    assert cost <= 16
    maximum_checks = read_rows(run / 'r_star_native_checks.jsonl')
    assert len({r['task_id'] for r in maximum_checks}) >= 20
    assert all(r['expected'] == r['actual'] and r['passed'] for r in maximum_checks)
    for stage, filename, key in [('pilot', 'freeze.json', 'files'), ('formal', 'tier_freeze.json', 'scripts')]:
        freeze = json.loads((run / filename).read_text())
        for name, sha in freeze[key].items():
            assert digest_file(run / 'frozen_sources' / stage / name) == sha
    result = {'passed': True, 'episodes': len(episodes), 'formal_episodes': sum(r['phase'] == 'formal' for r in episodes),
              'exact_policy_inputs_reconstructed': len(steps), 'formal_points': len(points),
              'exact_replay_inputs_reconstructed': reconstructed_replay_inputs,
              'independent_replay_outcomes': bank_rows_total, 'unique_judge_generations': len(generated_judges),
              'accounted_model_requests': accounted_requests, 'gpu_hours': cost,
              'native_optimum_checks': len(maximum_checks)}
    (run / 'saved_data_validation.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
