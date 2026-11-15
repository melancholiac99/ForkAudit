"""CPU-only exact product maxima and evidence records from the frozen corpus.

This produces score primitives. The final analysis consumes those saved
primitives and does not import the shopping environment.
"""
import argparse
from collections import defaultdict
import contextlib
import itertools
import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.data import digest_value, stable_seed
from exploratory_shop_audit.env_worker import Engine
from exploratory_shop_audit.runtime import COMMUNITY, SOURCE, append, read_rows


def shown(state):
    public = state['public']
    ids = {str(p['asin']) for p in public.get('products', [])}
    if public.get('product'):
        ids.add(str(public['product']['asin']))
    return ids


def item_text(observation, asin):
    """Only this product's renderer block; never task/query/model reasoning."""
    if f'\nasin: {asin}\n' in observation:
        return observation.split('\nasin: ', 1)[1].split('\n\n', 1)[0]
    return '\n'.join(line for line in observation.splitlines()
                     if re.match(r'^\d+\|' + re.escape(str(asin)) + r'\|', line))


def evidence_label(text, requirement, normalized=False):
    if normalized:
        text = re.sub(r'\s+', '', text).casefold()
        requirement = re.sub(r'\s+', '', requirement).casefold()
    if any(prefix + requirement in text for prefix in ['不含', '不支持', '非', '无']):
        return 'conflict_proxy'
    return 'support_proxy' if requirement and requirement in text else 'unknown'


def configurations(product):
    """Enumerate native radio-click reachable selections, including omission.

    Values collide across axes in native text_to_clickable: the last axis wins.
    Native special navigation names cannot select a radio value.
    """
    reserved = {'buy now', 'back to search', '< prev', 'next >', 'description', 'features', 'reviews', 'search'}
    mapping = {value: axis.lower() for axis, values in product['options'].items() for value in values if value not in reserved}
    axes = defaultdict(list)
    for value, axis in mapping.items():
        axes[axis].append(value)
    names = list(axes)
    for combination in itertools.product(*[[None] + axes[axis] for axis in names]):
        yield {axis: value for axis, value in zip(names, combination) if value is not None}


def maximum(engine, goal_index, asin):
    product = engine.server.product_item_dict[asin]
    goal = engine.server.goals[goal_index]
    before = digest_value(product)
    best = {'reward': -float('inf'), 'no_budget_gate': -float('inf')}
    best_options, best_valid = {}, {}
    count = 0
    all_values = []
    for options in configurations(product):
        scored = engine.score(product, goal, options, None)
        count += 1
        for metric in best:
            if scored[metric] > best[metric]:
                best[metric] = scored[metric]
                best_options[metric] = options
                best_valid[metric] = scored['valid'] if metric == 'reward' else scored['diagnostic_detail']['reward_valid']
        all_values.append((options, scored['reward'], scored['no_budget_gate']))
    if before != digest_value(product):
        raise RuntimeError('Exact maximization mutated frozen product')
    return {'task_id': f'goal:{goal_index}', 'asin': asin, 'r_star': best,
            'argmax': best_options, 'argmax_reward_valid': best_valid, 'configurations': count,
            'method': 'exhaustive native radio mapping plus unselected axes; no combination cutoff',
            'enumeration_sha256': digest_value(all_values), 'error': None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    with contextlib.redirect_stdout(sys.stderr):
        engine = Engine(COMMUNITY, SOURCE / 'community_products.json', SOURCE / 'products.sqlite3', run / 'snapshots')
    tasks = json.loads((run / 'tasks.json').read_text())
    formal = {t['task_id']: t for t in tasks['formal']}
    steps = [r for r in read_rows(run / 'steps.jsonl') if r['phase'] == 'formal']
    requested = set()
    for row in steps:
        for state in (row['before'], row['after']):
            requested.update((row['task_id'], asin) for asin in shown(state))
    for row in read_rows(run / 'formal_forks/candidates.jsonl'):
        requested.update((row['task_id'], asin) for asin in shown(row['one_step']['state']))
    for task_id in {r['task_id'] for r in steps}:
        requested.add((task_id, formal[task_id]['target_asin']))
    existing = {(r['task_id'], r['asin']) for r in read_rows(run / 'r_star.jsonl')}
    started = time.monotonic()
    for n, (task_id, asin) in enumerate(sorted(requested)):
        if (task_id, asin) in existing:
            continue
        result = maximum(engine, formal[task_id]['goal_index'], asin)
        append(run / 'r_star.jsonl', result)
        if n % 100 == 0:
            print(json.dumps({'pairs': n + 1, 'requested': len(requested), 'seconds': time.monotonic() - started}), flush=True)
    maxima = {(r['task_id'], r['asin']): r for r in read_rows(run / 'r_star.jsonl')}
    # Verify exact maximizing configurations through actual native clicks/buys,
    # using the first fresh opened item in 20 distinct formal tasks.
    checked = {r['task_id'] for r in read_rows(run / 'r_star_native_checks.jsonl')}
    trajectory_prefix = defaultdict(list)
    for row in steps:
        tid, eid = row['task_id'], row['trajectory_id']
        public = row['before']['public']
        if len(checked) < 20 and tid not in checked and row['before']['stop_available'] and not public.get('selected_options'):
            asin = public['product']['asin']
            exact = maxima[(tid, asin)]
            for metric in ['reward', 'no_budget_gate']:
                key = 'rstar:' + tid + ':' + metric
                with contextlib.redirect_stdout(sys.stderr):
                    state = engine.open(key, formal[tid]['goal_index'], trajectory_prefix[eid])
                    for axis, value in exact['argmax'][metric].items():
                        result = engine.step(key, f'click[{value}]')
                        if result['status']['done']:
                            raise RuntimeError('Maximizing specification path terminates before purchase')
                    bought = engine.step(key, 'click[buy now]')
                actual = bought['status']['reward'] if metric == 'reward' else bought['no_budget_gate']
                if actual != exact['r_star'][metric]:
                    raise RuntimeError('Exact specification optimum disagrees with native action path')
                append(run / 'r_star_native_checks.jsonl', {'task_id': tid, 'asin': asin, 'metric': metric,
                       'configurations': exact['configurations'], 'argmax': exact['argmax'][metric],
                       'expected': exact['r_star'][metric], 'actual': actual, 'passed': True})
                engine.close(key)
            checked.add(tid)
        trajectory_prefix[eid].append(row['action'])
    # An evidence record is specific to a displayed item and decision context;
    # prompts are read verbatim, backend preferences remain analysis-only.
    pairs = []
    history = defaultdict(list)
    for row in steps:
        tid, eid = row['task_id'], row['trajectory_id']
        goal = engine.server.goals[formal[tid]['goal_index']]
        requirements = [('attribute', x) for x in goal.get('expected_core_functions', [])]
        requirements += [('option', x.get('value', '')) for x in goal.get('required_options_by_key', {}).values()]
        product = row['before']['public'].get('product')
        if product:
            asin = product['asin']
            backend = engine.server.product_item_dict[asin]
            historical = '\n'.join(item_text(obs, asin) for obs in history[eid] + [row['before']['observation']])
            current = '\n'.join(item_text(m['content'], asin) for m in row['input']['messages'] if m['role'] == 'user')
            visible = item_text(row['before']['observation'], asin)
            score = engine.score(backend, goal, row['before']['public'].get('selected_options', {}), None)
            for kind, required in requirements:
                required = str(required)
                # These are text-presence proxies, not semantic truth labels.
                def norm(s): return re.sub(r'\s+', '', s).casefold()
                norm_requirement = norm(required)
                literal_h, literal_c = required in historical, required in current
                fuzzy_h, fuzzy_c = norm_requirement in norm(historical), norm_requirement in norm(current)
                public_text = visible
                pairs.append({'task_id': tid, 'trajectory_id': eid, 'step': row['step'], 'asin': asin,
                    'target': asin == goal['asin'], 'kind': kind, 'requirement': required,
                    'historical_literal_presence': literal_h, 'current_literal_presence': literal_c,
                    'historical_normalized_presence': fuzzy_h, 'current_normalized_presence': fuzzy_c,
                    'current_product_literal_support': required in public_text,
                    'current_product_normalized_support': norm_requirement in norm(public_text),
                    'historical_label': evidence_label(historical, required),
                    'current_input_label': evidence_label(current, required),
                    'historical_normalized_label': evidence_label(historical, required, True),
                    'current_input_normalized_label': evidence_label(current, required, True),
                    'history_lost_from_current': literal_h and not literal_c,
                    'truncation_present': bool(row['input']['removed_pairs']),
                    'visible_text': public_text, 'backend_attributes': backend.get('Attributes', []),
                    'backend_options': backend.get('options'),
                    'backend_scoring_detail': score['detail'],
                    'review_reason': 'Product-specific renderer blocks only; absence is unknown; lexical support/negation needs human semantic review.'})
        history[eid].append(row['before']['observation'])
    (run / 'visibility_pairs.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in pairs))
    strata = defaultdict(list)
    for pair in pairs:
        strata[(pair['target'], pair['current_product_literal_support'], pair['kind'])].append(pair)
    for rows in strata.values():
        rows.sort(key=lambda p: stable_seed(20261115, 'manual-review', p['task_id'], p['asin'], p['requirement'], p['step']))
    selected, seen = [], set()
    while len(selected) < 50 and any(strata.values()):
        for key in sorted(strata):
            while strata[key]:
                pair = strata[key].pop(0)
                identity = (pair['task_id'], pair['asin'], pair['kind'], pair['requirement'])
                if identity not in seen:
                    seen.add(identity)
                    selected.append(pair)
                    break
            if len(selected) == 50:
                break
    (run / 'visibility_manual_review.jsonl').write_text(''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in selected))
    print(json.dumps({'completed_pairs': len(requested), 'visibility_pairs': len(pairs), 'manual_review': len(selected)}), flush=True)


if __name__ == '__main__':
    main()
