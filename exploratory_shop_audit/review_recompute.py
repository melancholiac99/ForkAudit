"""Post-review offline diagnostics. Original trajectories and rewards are read-only."""
import argparse
from collections import Counter, defaultdict
import contextlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import digest_file, digest_value, stable_seed, write_once
from exploratory_shop_audit.runtime import COMMUNITY, SOURCE, append, read_rows
from exploratory_shop_audit.env_worker import Engine, parse_action_fixed
from exploratory_shop_audit.derive_environment import configurations, shown
from exploratory_shop_audit.audit_statistics import bootstrap_mean, bootstrap_reliability, reliability_components, funnel


def parent_path(revision):
    return Path(json.loads((revision / 'parent_manifest.json').read_text())['parent'])


def verify_parent(revision):
    manifest = json.loads((revision / 'parent_manifest.json').read_text())
    parent = Path(manifest['parent'])
    for name, checksum in manifest['files'].items():
        if digest_file(parent / name) != checksum:
            raise RuntimeError('Parent artifact changed: ' + name)
    return parent


def validity(scored, metric):
    return scored['valid'] if metric == 'reward' else scored['diagnostic_detail']['reward_valid']


def summarize_configurations(scored_rows):
    result = {}
    for metric in ['reward', 'no_budget_gate']:
        best = max(scored_rows, key=lambda r: r['score'][metric])
        good = [r for r in scored_rows if validity(r['score'], metric)]
        verified = max(good, key=lambda r: r['score'][metric]) if good else None
        value = best['score'][metric]
        result[metric] = {'numeric_max': value, 'numeric_argmax_valid': validity(best['score'], metric),
            'numeric_max_has_valid_tie': any(abs(r['score'][metric] - value) <= 1e-8 for r in good),
            'valid_max': verified['score'][metric] if verified else None,
            'valid_argmax': verified['options'] if verified else None,
            'configurations': len(scored_rows), 'valid_configurations': len(good),
            'invalid_configurations': len(scored_rows) - len(good),
            'invalid_numeric_max_strictly_exceeds_valid': bool(verified and value > verified['score'][metric] + 1e-8)}
    return result


def derive_maxima(revision):
    parent = verify_parent(revision)
    old = read_rows(parent / 'r_star.jsonl')
    existing = {(r['task_id'], r['asin']) for r in read_rows(revision / 'validity_maxima.jsonl')}
    with contextlib.redirect_stdout(sys.stderr):
        engine = Engine(COMMUNITY, SOURCE / 'community_products.json', SOURCE / 'products.sqlite3', parent / 'snapshots')
    for i, original in enumerate(old):
        tid, asin = original['task_id'], original['asin']
        if (tid, asin) in existing:
            continue
        product = engine.server.product_item_dict[asin]
        goal = engine.server.goals[int(tid.split(':')[1])]
        before = digest_value(product)
        rows = [{'options': options, 'score': engine.score(product, goal, options, None)} for options in configurations(product)]
        result = summarize_configurations(rows)
        assert digest_value(product) == before
        for metric in result:
            assert abs(result[metric]['numeric_max'] - original['r_star'][metric]) <= 1e-8
            assert result[metric]['numeric_argmax_valid'] == original['argmax_reward_valid'][metric]
        append(revision / 'validity_maxima.jsonl', {'task_id': tid, 'asin': asin, 'metrics': result})
        if i % 500 == 0:
            print(json.dumps({'phase': 'validity_maxima', 'complete': i + 1, 'total': len(old)}), flush=True)


def summarize(values):
    values = list(values)
    return {'n': len(values), 'mean': float(np.mean(values)) if values else None,
            'quantiles': np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else None,
            'counts': dict(sorted(Counter(values).items()))}


def nested(rows, field, seed):
    episodes = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            episodes[(row['task_id'], row['trajectory_id'])].append(float(value))
    tasks = defaultdict(list)
    for (tid, eid), values in episodes.items():
        tasks[tid].append(float(np.mean(values)))
    result = bootstrap_mean([np.mean(v) for v in tasks.values()], stable_seed(20261115, 'revision', seed))
    result.update({'n_episodes': len(episodes), 'n_steps': sum(len(v) for v in episodes.values())})
    return result


def action_class(step):
    if not step['legal']:
        return 'invalid'
    name, arg = parse_action_fixed(step['action'])
    arg = arg.lower() if isinstance(arg, str) else arg
    if name == 'search': return 'search'
    if name == 'finish': return 'abstain'
    if name != 'click': return 'other'
    if arg == 'buy now': return 'purchase'
    if arg in ['description', 'features', 'reviews']: return 'inspect'
    if arg in ['back to search', '< prev', 'next >']: return 'navigate'
    options = step['before']['public'].get('available_options', {})
    if any(arg in values for values in options.values()): return 'specification'
    products = step['before']['public'].get('products', [])
    if arg in {str(p['asin']).lower() for p in products}: return 'open_product'
    return 'other'


def option_completeness(product, selected):
    options = product.get('options', {})
    if not options: return 'no_options'
    return 'complete' if all(selected.get(axis.lower()) in values for axis, values in options.items()) else 'incomplete'


def episode_diagnostics(episodes, by_episode):
    records = []
    for ep in episodes:
        rows = by_episode[ep['trajectory_id']]
        opened, ranked = set(), defaultdict(list)
        searches = []
        for row in rows:
            if action_class(row) == 'search': searches.append(row['action'])
            for state in [row['before'], row['after']]:
                public = state['public']
                if public.get('product'): opened.add(public['product']['asin'])
                for product in public.get('products', []):
                    if product.get('rank') is not None:
                        ranked[product['asin']].append(product['rank'])
        asin = ep['status'].get('purchase', {}).get('asin')
        ranks = ranked.get(asin, [])
        records.append({'task_id': ep['task_id'], 'trajectory_id': ep['trajectory_id'], 'length': ep['length'],
            'success': ep['success'], 'reward_valid': ep['reward_valid'], 'purchase': ep['purchase'],
            'termination_reason': ep['termination_reason'], 'searches': len(searches), 'distinct_queries': len(set(searches)),
            'opened_products': len(opened), 'purchase_first_rank': ranks[0] if ranks else None,
            'purchase_last_rank': ranks[-1] if ranks else None})
    return {'termination_counts': dict(Counter(r['termination_reason'] for r in records)),
        'length_all': summarize(r['length'] for r in records),
        'length_by_success': {str(v): summarize(r['length'] for r in records if r['success'] == v) for v in [True, False]},
        'length_by_validity': {str(v): summarize(r['length'] for r in records if r['reward_valid'] is v) for v in [True, False, None]},
        **{field: summarize(r[field] for r in records if r[field] is not None) for field in ['searches', 'distinct_queries', 'opened_products', 'purchase_first_rank', 'purchase_last_rank']},
        'records': records}


def stop_diagnostics(episodes, by_episode, backend, metric='reward'):
    all_rows = []
    for ep in episodes:
        for step in by_episode[ep['trajectory_id']]:
            back = backend[(ep['trajectory_id'], step['step'])]
            if not step['before']['stop_available']: continue
            score = back['stop_score']
            c_valid = validity(score, metric)
            r_valid = ep['reward_valid'] if metric == 'reward' else ep['_diagnostic_valid']
            requirements = score['detail']['evidence']['preference_scoring']['dimensions']['key_options']
            req_status = ('not_required' if not requirements['active'] else
                          'satisfied' if requirements['score'] == 1 and requirements['coverage'] == 1 else 'not_satisfied_or_unverifiable')
            row = {'task_id': ep['task_id'], 'trajectory_id': ep['trajectory_id'], 'step': step['step'],
                'action_class': action_class(step), 'specification_state': option_completeness(back['before']['current_product'], back['before']['session']['options']),
                'requirement_status': req_status, 'C_valid': c_valid, 'R_valid': r_valid,
                'both_valid': c_valid is True and r_valid is True,
                'C': score[metric], 'R': ep[metric], 'delta': ep[metric] - score[metric]}
            row.update({'opportunity': abs(row['delta']) > .05, 'improvement': row['delta'] > .05,
                        'regression': row['delta'] < -.05, 'invalid_C': c_valid is False,
                        'invalid_R': r_valid is False})
            all_rows.append(row)
    continuation = [r for r in all_rows if r['action_class'] != 'purchase']
    strata = {'all_continuation': continuation}
    fixed_values = {'action_class': ['specification', 'inspect', 'search', 'navigate', 'open_product', 'abstain', 'invalid', 'other'],
                    'specification_state': ['complete', 'incomplete', 'no_options']}
    for field in ['action_class', 'specification_state', 'requirement_status']:
        for value in fixed_values.get(field, sorted({r[field] for r in continuation})):
            strata[f'{field}:{value}'] = [r for r in continuation if r[field] == value]
    for action in sorted({r['action_class'] for r in continuation}):
        for completeness in ['complete', 'incomplete', 'no_options']:
            strata[f'{action}/{completeness}'] = [r for r in continuation if r['action_class'] == action and r['specification_state'] == completeness]
    results = {}
    for name, rows in strata.items():
        good = [r for r in rows if r['both_valid']]
        results[name] = {'native_numeric': {f: nested(rows, f, name + f) for f in ['opportunity', 'improvement', 'regression', 'delta', 'invalid_C', 'invalid_R', 'both_valid']},
            'both_valid_conditional': {f: nested(good, f, name + 'valid' + f) for f in ['opportunity', 'improvement', 'regression', 'delta']},
            'excluded_steps': len(rows) - len(good), 'warning': 'Validity-conditioned subset, not an unbiased replacement of the native estimand.'}
    return {'buyable_steps': len(all_rows), 'continuation_steps': len(continuation), 'strata': results, 'records': all_rows}


def funnel_diagnostics(episodes, by_episode, maxima, metric):
    records, excluded = [], []
    for ep in episodes:
        tid, eid = ep['task_id'], ep['trajectory_id']
        if not ep['purchase']:
            excluded.append({'trajectory_id': eid, 'reason': 'no_purchase'}); continue
        validity_flag = ep['reward_valid']
        if metric == 'no_budget_gate':
            # Saved no-budget validity at the final pre-purchase state is exact
            # for the same product/options and unmodified deterministic scorer.
            validity_flag = ep['_diagnostic_valid']
        if validity_flag is not True:
            excluded.append({'trajectory_id': eid, 'reason': 'actual_purchase_unverifiable'}); continue
        surfaced, opened = set(), set()
        for step in by_episode[eid]:
            for state in [step['before'], step['after']]:
                surfaced |= shown(state)
                if state['public'].get('product'): opened.add(state['public']['product']['asin'])
        item = ep['status']['purchase']['asin']
        valid_surfaced = {a for a in surfaced if maxima[(tid, a)]['metrics'][metric]['valid_max'] is not None}
        valid_opened = opened & valid_surfaced
        if item not in valid_opened:
            excluded.append({'trajectory_id': eid, 'reason': 'no_valid_item_configuration'}); continue
        s = max(maxima[(tid, a)]['metrics'][metric]['valid_max'] for a in valid_surfaced)
        o = max(maxima[(tid, a)]['metrics'][metric]['valid_max'] for a in valid_opened)
        chosen = maxima[(tid, item)]['metrics'][metric]['valid_max']
        gaps = funnel(s, o, chosen, ep[metric])
        records.append({'task_id': tid, 'trajectory_id': eid, 'Phi_surf': s, 'Phi_open': o, 'R_item': chosen, 'R': ep[metric],
            'gaps': gaps, 'surfaced_no_valid_configuration': len(surfaced - valid_surfaced),
            'opened_no_valid_configuration': len(opened - valid_opened)})
    return {'warning': 'Conditional on an evaluable actual purchase; unevaluable catalog items removed from diagnostic sets only.',
            'episodes': len(records), 'exclusions': dict(Counter(r['reason'] for r in excluded)), 'excluded_records': excluded,
            'gaps': [nested([{**r, 'value': r['gaps'][i]} for r in records], 'value', metric + 'funnel' + str(i)) for i in range(4)], 'records': records}


def replay_diagnostics(parent):
    candidates = read_rows(parent / 'formal_forks/candidates.jsonl')
    rows = read_rows(parent / 'formal_forks/replay_bank_a.jsonl') + read_rows(parent / 'formal_forks/replay_bank_b.jsonl')
    outcomes = defaultdict(list)
    for row in rows: outcomes[(row['state_id'], row['candidate'])].append(row)
    manifest = {r['state_id']: r for r in json.loads((parent / 'state_manifest.json').read_text())['points']}
    by_state = defaultdict(list)
    candidate_records = []
    for c in candidates:
        samples = sorted(outcomes[(c['state_id'], c['candidate'])], key=lambda r: (r['bank'], r['repeat']))
        validity_flags = [r['status'].get('reward_valid') for r in samples]
        candidate_records.append({'state_id': c['state_id'], 'candidate': c['candidate'], 'terminal': c['terminal'],
            'samples': len(samples), 'valid_samples': sum(v is True for v in validity_flags),
            'invalid_samples': sum(v is False for v in validity_flags), 'unknown_validity_samples': sum(v is None for v in validity_flags),
            'invalid_fraction': sum(v is False for v in validity_flags) / len(samples) if samples else None,
            'terminal_reward_valid': c['one_step']['status'].get('reward_valid') if c['terminal'] else None,
            'length': summarize(r['length'] for r in samples)})
        if not c['terminal'] and len(samples) == 12: by_state[c['state_id']].append((c, samples))
    state_records, strata = [], defaultdict(list)
    for sid, arms in by_state.items():
        if len(arms) < 2: continue
        samples = [r for c, rows in arms for r in rows]
        mean_length = float(np.mean([r['length'] for r in samples]))
        length_bin = '<=2' if mean_length <= 2 else '(2,4]' if mean_length <= 4 else '>4'
        validity_bin = 'all_valid' if all(r['status'].get('reward_valid') is True for r in samples) else 'contains_invalid_or_unknown'
        component = reliability_components([[r['reward'] for r in rows] for c, rows in arms])
        record = {'state_id': sid, 'task_id': manifest[sid]['task_id'], 'page': manifest[sid]['page_type'],
            'mean_length': mean_length, 'length_bin': length_bin, 'validity_bin': validity_bin,
            'S': component[0], 'N': component[1], 'candidate_count': len(arms),
            'invalid_samples': sum(r['status'].get('reward_valid') is False for r in samples), 'samples': len(samples)}
        state_records.append(record)
        for key in ['all', 'page:' + record['page'], 'length:' + length_bin, 'validity:' + validity_bin,
                    'page:' + record['page'] + '/validity:' + validity_bin]:
            strata[key].append(component)
    return {'primary_scope': 'Frozen natural candidates, current policy and observed short continuations; not the entire environment.',
        'strata': {k: bootstrap_reliability(v, stable_seed(20261115, 'review-reliability', k)) for k, v in strata.items()},
        'selection_warning': 'Length and validity strata are post-outcome descriptive strata. All 12 samples are retained in every included arm.',
        'state_records': state_records, 'candidate_records': candidate_records}


def analyze(revision):
    parent = verify_parent(revision)
    episodes = [r for r in read_rows(parent / 'episodes.jsonl') if r['phase'] == 'formal']
    ids = {r['trajectory_id'] for r in episodes}
    by_episode = defaultdict(list)
    for r in read_rows(parent / 'steps.jsonl'):
        if r['trajectory_id'] in ids: by_episode[r['trajectory_id']].append(r)
    for rows in by_episode.values(): rows.sort(key=lambda r: r['step'])
    backend = {(r['trajectory_id'], r['step']): r for r in read_rows(parent / 'backend_steps.jsonl') if r['trajectory_id'] in ids}
    maxima = {(r['task_id'], r['asin']): r for r in read_rows(revision / 'validity_maxima.jsonl')}
    for ep in episodes:
        ep['_diagnostic_valid'] = backend[(ep['trajectory_id'], ep['length'] - 1)]['stop_score']['diagnostic_detail']['reward_valid'] if ep['purchase'] else ep['reward_valid']
    result = {'episode_diagnostics': episode_diagnostics(episodes, by_episode),
              'stop': stop_diagnostics(episodes, by_episode, backend), 'validity_maxima': {},
              'stop_no_budget_gate': stop_diagnostics(episodes, by_episode, backend, 'no_budget_gate'),
              'valid_funnel': {m: funnel_diagnostics(episodes, by_episode, maxima, m) for m in ['reward', 'no_budget_gate']},
              'replay': replay_diagnostics(parent)}
    for metric in ['reward', 'no_budget_gate']:
        values = [r['metrics'][metric] for r in maxima.values()]
        result['validity_maxima'][metric] = {'task_product_pairs': len(values),
            'numeric_argmax_invalid_pairs': sum(not r['numeric_argmax_valid'] for r in values),
            'numeric_max_without_valid_tie_pairs': sum(not r['numeric_max_has_valid_tie'] for r in values),
            'pairs_with_no_valid_configuration': sum(r['valid_max'] is None for r in values),
            'invalid_max_strictly_above_valid_pairs': sum(r['invalid_numeric_max_strictly_exceeds_valid'] for r in values),
            'all_configurations': sum(r['configurations'] for r in values),
            'invalid_configurations': sum(r['invalid_configurations'] for r in values)}
    previous = json.loads((parent / 'metrics.json').read_text())
    for metric in ['reward', 'no_budget_gate']:
        original = {r['trajectory_id']: r for r in previous['reward_metrics'][metric]['funnel']['records']}
        cohort = result['valid_funnel'][metric]['records']
        assert all(r['trajectory_id'] in original for r in cohort)
        result['valid_funnel'][metric]['numeric_same_cohort_gaps'] = [nested(
            [{**r, 'value': original[r['trajectory_id']]['gaps'][i]} for r in cohort], 'value', metric + 'same_cohort' + str(i)) for i in range(4)]
        result['valid_funnel'][metric]['paired_valid_minus_numeric_gaps'] = [nested(
            [{**r, 'value': r['gaps'][i] - original[r['trajectory_id']]['gaps'][i]} for r in cohort], 'value', metric + 'paired_funnel' + str(i)) for i in range(4)]
    assert abs(result['stop']['strata']['all_continuation']['native_numeric']['opportunity']['estimate'] - previous['reward_metrics']['reward']['stop']['opportunity']['estimate']) < 1e-12
    for key in ['S', 'N']:
        assert abs(result['replay']['strata']['all'][key] - previous['reward_metrics']['reward']['replay']['reliability'][key]) < 1e-12
    (revision / 'offline_metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'phase': 'offline_complete', 'episodes': len(episodes), 'maxima': len(maxima), 'native_metrics_preserved': True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--derive', action='store_true')
    args = parser.parse_args()
    if args.derive: derive_maxima(args.revision)
    analyze(args.revision)


if __name__ == '__main__': main()
