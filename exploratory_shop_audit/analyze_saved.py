"""Recompute exploratory metrics solely from saved JSON/JSONL artifacts."""
import argparse
from collections import Counter, defaultdict
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.audit_statistics import (bootstrap_mean, bootstrap_reliability,
    funnel, reliability_components)
from exploratory_shop_audit.data import stable_seed
from exploratory_shop_audit.runtime import read_rows

METRICS = ['reward', 'no_budget_gate']


def estimate(values, name):
    return bootstrap_mean(values, stable_seed(20261115, 'bootstrap', name))


def task_equal(rows, value, name):
    """Rows represent one episode or one unique-task state."""
    grouped = defaultdict(list)
    for row in rows:
        x = value(row)
        if x is not None and np.isfinite(x):
            grouped[row['task_id']].append(x)
    return estimate([np.mean(v) for v in grouped.values()], name)


def quantiles(x):
    return dict(zip(['min', 'q25', 'median', 'q75', 'max'], np.quantile(x, [0, .25, .5, .75, 1]).tolist())) if len(x) else None


def weighted_correlation(rows):
    """Each task equal, then each episode equal, then eligible observations equal."""
    grouped = defaultdict(lambda: defaultdict(list))
    for tid, eid, x, y in rows:
        grouped[tid][eid].append((x, y))
    points, weights = [], []
    for episodes in grouped.values():
        for values in episodes.values():
            points.extend(values)
            weights.extend([1 / len(grouped) / len(episodes) / len(values)] * len(values))
    if not points:
        return None
    x, y = np.asarray(points).T
    w = np.asarray(weights)
    x, y = x - w @ x, y - w @ y
    denominator = np.sqrt((w @ (x * x)) * (w @ (y * y)))
    return float(w @ (x * y) / denominator) if denominator > 1e-12 else None


def reward_groups(episodes, metric):
    groups = defaultdict(list)
    for row in episodes:
        groups[row['task_id']].append(row)
    records = []
    for tid, rows in groups.items():
        if len(rows) != 4:
            continue
        values = [r[metric] for r in rows]
        gap = max(values) - min(values)
        equal = gap <= 1e-8
        records.append({'task_id': tid, 'range': gap, 'equal': equal,
            'all_failed': all(not r['success'] for r in rows) if metric == 'reward' else all(v < 1 - 1e-8 for v in values),
            'zero_equal': equal and abs(values[0]) <= 1e-8,
            'full_equal': equal and abs(values[0] - 1) <= 1e-8,
            'negative_equal': equal and values[0] < -1e-8,
            'partial_nonzero_equal': equal and 1e-8 < values[0] < 1 - 1e-8,
            'strict_pairs': sum(abs(a - b) > 1e-8 for a, b in itertools.combinations(values, 2)),
            'values': values})
    return {'groups': len(records), 'mixed_groups': sum(not r['equal'] for r in records),
            'strict_pairs': sum(r['strict_pairs'] for r in records),
            'range_quantiles': quantiles([r['range'] for r in records]),
            **{key: task_equal(records, lambda r, k=key: r[k], metric + key) for key in
               ['equal', 'all_failed', 'zero_equal', 'full_equal', 'negative_equal', 'partial_nonzero_equal']},
            'range_le_005': task_equal(records, lambda r: r['range'] <= .05, metric + 'range005'),
            'records': records}


def stop_metrics(episodes, steps, backend, metric):
    by_episode = defaultdict(list)
    for step in steps:
        by_episode[step['trajectory_id']].append(step)
    rows, correlations, samples = [], [], []
    for ep in episodes:
        relevant = by_episode[ep['trajectory_id']]
        stops, continuation, purchasing = [], [], []
        for step in relevant:
            score = backend[(ep['trajectory_id'], step['step'])]['stop_score']
            if not step['before']['stop_available']:
                if score is not None:
                    raise ValueError('Non-buyable state has stop score')
                continue
            c = score[metric]
            delta = ep[metric] - c
            buy = step['action'].lower() == 'click[buy now]'
            stops.append((c, delta))
            (purchasing if buy else continuation).append((c, delta))
            correlations.append((ep['task_id'], ep['trajectory_id'], c, ep[metric]))
            samples.append({'task_id': ep['task_id'], 'trajectory_id': ep['trajectory_id'], 'step': step['step'],
                            'remaining': step['remaining'], 'C': c, 'R': ep[metric], 'delta': delta, 'buy': buy})
        rows.append({'task_id': ep['task_id'], 'trajectory_id': ep['trajectory_id'],
                     'coverage': len(stops) / len(relevant) if relevant else None, 'episode_coverage': bool(stops),
                     'C_mean': np.mean([c for c, d in stops]) if stops else None,
                     'delta_mean': np.mean([d for c, d in stops]) if stops else None,
                     'opportunity': np.mean([abs(d) > .05 for c, d in continuation]) if continuation else None,
                     'improvement': np.mean([d > .05 for c, d in continuation]) if continuation else None,
                     'regression': np.mean([d < -.05 for c, d in continuation]) if continuation else None,
                     'buy_delta': np.mean([d for c, d in purchasing]) if purchasing else None})
    result = {key: task_equal(rows, lambda r, k=key: r[k], metric + 'stop' + key) for key in
              ['coverage', 'episode_coverage', 'C_mean', 'delta_mean', 'opportunity', 'improvement', 'regression', 'buy_delta']}
    result.update({'correlation': weighted_correlation(correlations), 'eligible_steps': len(samples),
                   'continuation_steps': sum(not r['buy'] for r in samples),
                   'C_quantiles_unweighted_descriptive': quantiles([r['C'] for r in samples]),
                   'delta_quantiles_unweighted_descriptive': quantiles([r['delta'] for r in samples]),
                   'by_remaining': {}})
    for low, high in [(1, 5), (6, 15), (16, 25), (26, 35)]:
        filtered = [r for r in samples if low <= r['remaining'] <= high]
        grouped = defaultdict(list)
        for row in filtered:
            grouped[row['trajectory_id']].append(row)
        per_ep = [{'task_id': v[0]['task_id'], 'delta': np.mean([r['delta'] for r in v]),
                   'C': np.mean([r['C'] for r in v])} for v in grouped.values()]
        result['by_remaining'][f'{low}-{high}'] = {'steps': len(filtered),
            'C': task_equal(per_ep, lambda r: r['C'], metric + str(low) + 'C'),
            'delta': task_equal(per_ep, lambda r: r['delta'], metric + str(low) + 'D')}
    return result, samples


def shown(state):
    public = state['public']
    ids = {str(p['asin']) for p in public.get('products', [])}
    if public.get('product'):
        ids.add(str(public['product']['asin']))
    return ids


def funnels(episodes, steps, maxima, metric):
    surfaces, opened = defaultdict(set), defaultdict(set)
    for step in steps:
        eid = step['trajectory_id']
        for state in [step['before'], step['after']]:
            surfaces[eid] |= shown(state)
            if state['public'].get('product'):
                opened[eid].add(str(state['public']['product']['asin']))
    rows, excluded = [], []
    for ep in episodes:
        eid, tid = ep['trajectory_id'], ep['task_id']
        if not ep['purchase']:
            excluded.append({'trajectory_id': eid, 'reason': 'no_purchase', 'reward': ep[metric]})
            continue
        item = str(ep['status']['purchase']['asin'])
        if item not in opened[eid] or not opened[eid] <= surfaces[eid]:
            raise ValueError('Actual purchase not in nested surfaced/opened sets')
        try:
            surface = max(maxima[(tid, a)]['r_star'][metric] for a in surfaces[eid])
            op = max(maxima[(tid, a)]['r_star'][metric] for a in opened[eid])
            chosen = maxima[(tid, item)]['r_star'][metric]
            values = funnel(surface, op, chosen, ep[metric])
        except (KeyError, ValueError) as exc:
            excluded.append({'trajectory_id': eid, 'reason': str(exc), 'reward': ep[metric]})
            continue
        rows.append({'task_id': tid, 'trajectory_id': eid, 'Phi_surf': surface, 'Phi_open': op,
                     'R_item': chosen, 'R': ep[metric], 'gaps': values})
    return {'eligible_episodes': len(rows), 'excluded': excluded, 'records': rows,
            'gaps': [task_equal(rows, lambda r, i=i: r['gaps'][i], metric + 'funnel' + str(i)) for i in range(4)]}


def replay_metrics(run, metric):
    root = run / 'formal_forks'
    candidates = read_rows(root / 'candidates.jsonl')
    states = {r['state_id']: r for r in read_rows(root / 'state_results.jsonl')}
    banks = {bank: read_rows(root / f'replay_bank_{bank.lower()}.jsonl') for bank in ['A', 'B']}
    index = defaultdict(list)
    seen_seeds = set()
    for bank, rows in banks.items():
        for row in rows:
            if row['seed'] in seen_seeds:
                raise ValueError('Replay streams duplicate seed')
            seen_seeds.add(row['seed'])
            index[(row['state_id'], row['candidate'], bank)].append(row)
    by_state = defaultdict(list)
    values = {}
    raw = {}
    table = []
    for candidate in candidates:
        sid, cid = candidate['state_id'], candidate['candidate']
        by_state[sid].append(candidate)
        row = {'state_id': sid, 'candidate': cid, 'terminal': candidate['terminal'], 'banks': {}}
        combined = []
        complete = True
        for bank in ['A', 'B']:
            samples = sorted(index[(sid, cid, bank)], key=lambda r: r['repeat'])
            if candidate['terminal']:
                status = candidate['one_step']['status']
                value = status['reward'] if metric == 'reward' else candidate['one_step']['no_budget_gate']
                row['banks'][bank] = {'mean': value, 'variance': 0.0, 'independent_model_continuations': 0,
                    'success_rate': float(status.get('reward_detail', {}).get('reward_type') == 'gold_purchase'), 'mean_length': 0.0}
                values[(sid, cid, bank)] = value
            elif len(samples) == 6:
                y = [r[metric] for r in samples]
                row['banks'][bank] = {'mean': float(np.mean(y)), 'variance': float(np.var(y, ddof=1)),
                    'independent_model_continuations': 6, 'success_rate': float(np.mean([r['success'] for r in samples])),
                    'mean_length': float(np.mean([r['length'] for r in samples]))}
                values[(sid, cid, bank)] = float(np.mean(y))
                combined.extend(y)
            else:
                complete = False
                row['banks'][bank] = {'incomplete': len(samples)}
        if complete:
            raw[(sid, cid)] = [values[(sid, cid, 'A')]] * 12 if candidate['terminal'] else combined
        table.append(row)
    components, all_components, weighted_components = [], [], []
    improvements, condition_noise, state_signal = [], [], []
    matrices = []
    components_by_page = defaultdict(list)
    stop_rows = []
    for sid, arms in by_state.items():
        if not states.get(sid, {}).get('defined') or any((sid, a['candidate']) not in raw for a in arms):
            continue
        arms.sort(key=lambda a: a['candidate'])
        nonterminal = [a for a in arms if not a['terminal']]
        if len(nonterminal) >= 2:
            y = np.asarray([raw[(sid, a['candidate'])] for a in nonterminal])
            components.append(reliability_components(y))
            components_by_page[states[sid]['page_type']].append(components[-1])
            matrices.append(y)
            condition_noise.append(float(np.var(y, axis=1, ddof=1).mean()))
            state_signal.append(float(np.var(y.mean(axis=1))))
            weights = [a['draft_frequency'] for a in nonterminal]
            if np.count_nonzero(weights) >= 2:
                weighted_components.append(reliability_components(y, weights))
        all_components.append(reliability_components([raw[(sid, a['candidate'])] for a in arms]))
        chosen_a = max(arms, key=lambda a: (values[(sid, a['candidate'], 'A')], -a['candidate']))['candidate']
        chosen_b = max(arms, key=lambda a: (values[(sid, a['candidate'], 'B')], -a['candidate']))['candidate']
        improvements.append({'task_id': arms[0]['task_id'], 'state_id': sid,
            'A_select_B_evaluate': values[(sid, chosen_a, 'B')] - values[(sid, 0, 'B')],
            'B_select_A_evaluate': values[(sid, chosen_b, 'A')] - values[(sid, 0, 'A')],
            'chosen_a': chosen_a, 'chosen_b': chosen_b, 'page_type': states[sid]['page_type']})
        buy = next((a for a in arms if a['parsed_action'] == ['click', 'buy now']), None)
        if buy:
            c = values[(sid, buy['candidate'], 'A')]
            for arm in arms:
                if arm['candidate'] != buy['candidate']:
                    stop_rows.append({'task_id': arms[0]['task_id'], 'state_id': sid, 'candidate': arm['candidate'],
                        'original': arm['original'], 'delta_A': values[(sid, arm['candidate'], 'A')] - c,
                        'delta_B': values[(sid, arm['candidate'], 'B')] - c})
    checks = {}
    rng = np.random.default_rng(stable_seed(20261115, 'random-split-check'))
    for n in [1, 3, 6]:
        correlations, rankings = [], []
        for repeat in range(100):
            x_all, y_all, agree = [], [], []
            for y in matrices:
                perm = rng.permutation(12)
                x, z = y[:, perm[:n]].mean(axis=1), y[:, perm[n:2*n]].mean(axis=1)
                x_all.extend(x - x.mean()); y_all.extend(z - z.mean())
                pairs = [(np.sign(x[i] - x[j]), np.sign(z[i] - z[j])) for i, j in itertools.combinations(range(len(x)), 2)]
                agree.append(np.mean([1.0 if a == b else (.5 if a == 0 or b == 0 else 0.0) for a, b in pairs]))
            if x_all and np.std(x_all) > 0 and np.std(y_all) > 0:
                correlations.append(float(np.corrcoef(x_all, y_all)[0, 1]))
            if agree: rankings.append(float(np.mean(agree)))
        checks[str(n)] = {'correlation_split_distribution': quantiles(correlations),
                         'ranking_split_distribution': quantiles(rankings), 'splits': 100,
                         'warning': 'Splits are dependent diagnostics, not 100 independent samples'}
    result = {'defined_states': len(improvements), 'candidate_estimates': table,
              'reliability': bootstrap_reliability(components, stable_seed(metric, 'rho')),
              'including_exact_terminal': bootstrap_reliability(all_components, stable_seed(metric, 'rho_terminal')),
              'draft_frequency_weighted': bootstrap_reliability(weighted_components, stable_seed(metric, 'rho_weighted')),
              'conditional_return_noise': estimate(condition_noise, metric + 'conditional_noise'),
              'observed_action_mean_variance': estimate(state_signal, metric + 'observed_action_variance'),
              'crossbank_records': improvements,
              'crossbank_gain': task_equal(improvements, lambda r: (r['A_select_B_evaluate'] + r['B_select_A_evaluate']) / 2, metric + 'crossbank'),
              'A_select_B_gain': task_equal(improvements, lambda r: r['A_select_B_evaluate'], metric + 'ab'),
              'B_select_A_gain': task_equal(improvements, lambda r: r['B_select_A_evaluate'], metric + 'ba'),
              'split_checks': checks, 'stop_comparisons': stop_rows,
              'stop_mean_delta_B': task_equal(stop_rows, lambda r: r['delta_B'], metric + 'stopB'),
              'by_page_descriptive': {page: {'states': sum(r['page_type'] == page for r in improvements),
                  'reliability': bootstrap_reliability(components_by_page[page], stable_seed(metric, page, 'rho')),
                  'gain': task_equal([r for r in improvements if r['page_type'] == page],
                                    lambda r: (r['A_select_B_evaluate'] + r['B_select_A_evaluate']) / 2, metric + page)}
                  for page in ['search_home', 'search_results', 'product_detail']}}
    fixed_x, fixed_y, fixed_rank, fixed_ties = [], [], [], []
    for y in matrices:
        x, z = y[:, :6].mean(axis=1), y[:, 6:].mean(axis=1)
        fixed_x.extend(x - x.mean()); fixed_y.extend(z - z.mean())
        pairs = [(np.sign(x[i] - x[j]), np.sign(z[i] - z[j])) for i, j in itertools.combinations(range(len(x)), 2)]
        fixed_rank.append(np.mean([1.0 if a == b else (.5 if a == 0 or b == 0 else 0.0) for a, b in pairs]))
        fixed_ties.append(np.mean([a == 0 or b == 0 for a, b in pairs]))
    result['fixed_6_plus_6_check'] = {
        'centered_correlation': float(np.corrcoef(fixed_x, fixed_y)[0, 1]) if fixed_x and np.std(fixed_x) > 0 and np.std(fixed_y) > 0 else None,
        'state_mean_ranking_agreement': float(np.mean(fixed_rank)) if fixed_rank else None,
        'state_mean_pair_tie_fraction': float(np.mean(fixed_ties)) if fixed_ties else None}
    judges = {(r['state_id'], r['condition']): r for r in read_rows(run / 'judge_outputs.jsonl')}
    judge_rows = []
    for sid, arms in by_state.items():
        s, f = judges.get((sid, 'S')), judges.get((sid, 'F'))
        if not s or not f or s['selected'] is None or f['selected'] is None:
            continue
        if any((sid, r['selected'], bank) not in values for r in [s, f] for bank in ['A', 'B']):
            continue
        judge_rows.append({'task_id': arms[0]['task_id'], 'state_id': sid,
            **{bank: values[(sid, f['selected'], bank)] - values[(sid, s['selected'], bank)] for bank in ['A', 'B']}})
    result['information_control'] = {'paired_valid_states': len(judge_rows),
        'invalid_requests': sum(r['selected'] is None for r in judges.values()),
        **{bank: task_equal(judge_rows, lambda r, b=bank: r[b], metric + 'judge' + bank) for bank in ['A', 'B']}, 'records': judge_rows}
    return result, values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    run = args.run.resolve()
    episodes = [r for r in read_rows(run / 'episodes.jsonl') if r['phase'] == 'formal']
    completed = {r['trajectory_id'] for r in episodes}
    steps = [r for r in read_rows(run / 'steps.jsonl') if r['trajectory_id'] in completed]
    backend = {(r['trajectory_id'], r['step']): r for r in read_rows(run / 'backend_steps.jsonl') if r['trajectory_id'] in completed}
    maxima = {(r['task_id'], r['asin']): r for r in read_rows(run / 'r_star.jsonl')}
    results = {'formal_episodes': len(episodes), 'formal_tasks': len({r['task_id'] for r in episodes}),
               'collection_complete': (run / 'collection_complete.json').exists(),
               'costs': json.loads((run / 'costs.json').read_text()), 'reward_metrics': {}}
    for metric in METRICS:
        stop, stop_samples = stop_metrics(episodes, steps, backend, metric)
        replay, replay_values = replay_metrics(run, metric)
        results['reward_metrics'][metric] = {'groups': reward_groups(episodes, metric), 'stop': stop,
            'funnel': funnels(episodes, steps, maxima, metric), 'replay': replay}
        (run / f'stop_samples_{metric}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in stop_samples))
    native = results['reward_metrics']['reward']
    equal, coverage, opportunity = native['groups']['equal']['estimate'], native['stop']['coverage']['estimate'], native['stop']['opportunity']['estimate']
    reliability = native['replay']['reliability']
    rho = reliability['rho'].get('3')
    results['stop_rules'] = {
        'ctha': {'triggered': equal < .1 if equal is not None else None, 'estimate': equal, 'threshold': '<0.1'},
        'stop_baseline': {'triggered': (coverage < .2 or opportunity < .15) if coverage is not None and opportunity is not None else None,
                          'coverage': coverage, 'opportunity': opportunity, 'threshold': 'coverage<0.2 OR opportunity<0.15'},
        'replay_noise': {'triggered': rho > .7 if reliability['identifiable'] else None,
                         'rho3': rho, 'identifiable': reliability['identifiable'], 'threshold': '>0.7'}}
    def crosses(ci, threshold):
        return bool(ci[0] <= threshold <= ci[1]) if ci is not None else None
    results['stop_rules']['ctha']['boundary_uncertain'] = crosses(native['groups']['equal']['ci95'], .1)
    results['stop_rules']['stop_baseline']['coverage_boundary_uncertain'] = crosses(native['stop']['coverage']['ci95'], .2)
    results['stop_rules']['stop_baseline']['opportunity_boundary_uncertain'] = crosses(native['stop']['opportunity']['ci95'], .15)
    results['stop_rules']['replay_noise']['boundary_uncertain'] = crosses(reliability.get('rho_ci95', {}).get('3'), .7)
    truncated = [r for r in steps if r['input']['removed_pairs']]
    truncation = {'decisions': len(steps), 'truncated_decisions': len(truncated),
        'truncated_decision_fraction': len(truncated) / len(steps) if steps else None,
        'episodes': len(episodes), 'truncated_episodes': len({r['trajectory_id'] for r in truncated}),
        'removed_tokens': sum(r['input']['removed_tokens'] for r in steps),
        'max_input_tokens': max([r['input']['token_count'] for r in steps], default=0),
        'generation_truncations': sum(r['generation']['finish_reason'] == 'length' for r in steps),
        'removed_pairs_with_product_evidence': sum(any(marker in obs for marker in ['title:', 'key_attributes:', 'available_options:', 'rank|asin|'])
            for r in truncated for obs, reply in r['input']['removed_pairs'])}
    (run / 'truncation_audit.json').write_text(json.dumps(truncation, ensure_ascii=False, indent=2) + '\n')
    results['truncation'] = truncation
    (run / 'metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    with (run / 'summary.csv').open('w') as f:
        writer = csv.writer(f); writer.writerow(['reward', 'metric', 'estimate', 'ci_low', 'ci_high', 'tasks'])
        for metric in METRICS:
            r = results['reward_metrics'][metric]
            for name, obj in [('equal_groups', r['groups']['equal']), ('all_failed', r['groups']['all_failed']),
                              ('stop_coverage', r['stop']['coverage']), ('stop_opportunity', r['stop']['opportunity']),
                              ('crossbank_gain', r['replay']['crossbank_gain'])]:
                writer.writerow([metric, name, obj['estimate'], *(obj['ci95'] or [None, None]), obj['n_tasks']])
    print(json.dumps({'formal_tasks': results['formal_tasks'], 'formal_episodes': len(episodes), 'stop_rules': results['stop_rules']}))


if __name__ == '__main__':
    main()
