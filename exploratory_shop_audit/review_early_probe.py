"""Landmark probes from exact predecision inputs, without a future action."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exploratory_shop_audit.runtime import read_rows, parse_generation
from exploratory_shop_audit.data import digest_value, stable_seed
from exploratory_shop_audit.review_recompute import verify_parent, action_class
from exploratory_shop_audit.predictive_probe import center_by_task, task_pair_accuracy
from exploratory_shop_audit.audit_statistics import bootstrap_mean


def landmark_features(rows, landmark):
    selected = None
    for i, previous in enumerate(rows[:-1]):
        kind = action_class(previous)
        qualifies = kind == ('search' if landmark == 'after_first_search' else 'open_product')
        if qualifies and not previous['done']:
            selected = rows[i + 1]
            break
    if selected is None: return None
    messages = selected['input']['messages']
    actions = [parse_generation(m['content']) for m in messages if m['role'] == 'assistant']
    counts = Counter(a.split('[', 1)[0] for a in actions)
    simple = {f'past_action_{k}': counts[k] for k in ['search', 'click', 'finish']}
    simple['retained_past_actions'] = len(actions)
    simple['visible_input_tokens'] = selected['input']['token_count']
    full = dict(simple)
    for kind in ['search', 'click']:
        subset = [a for a in actions if a.startswith(kind + '[')]
        full[kind + '_repetitions'] = len(subset) - len(set(subset))
    visible = '\n'.join(m['content'] for m in messages if m['role'] != 'system')
    for field in ['title:', 'price:', 'key_attributes:', 'available_options:', 'content:', 'rank|asin|']:
        full['seen_' + field] = visible.count(field)
    return {'simple': simple, 'full': full, 'text': visible,
            'landmark_step': selected['step'], 'input_sha256': digest_value(selected['input']),
            'largest_feature_action_step': selected['step'] - 1,
            'current_action_used': False, 'future_observation_used': False}


def fit_fold(job):
    category, categories, task_ids, labels, features, landmark = job
    categories, task_ids = np.asarray(categories), np.asarray(task_ids)
    y = np.asarray(labels)
    test, train = np.flatnonzero(categories == category), np.flatnonzero(categories != category)
    outputs = {}
    for name in ['simple', 'full']:
        vectorizer = DictVectorizer(sparse=False)
        x_train = vectorizer.fit_transform([features[i][name] for i in train])
        x_test = vectorizer.transform([features[i][name] for i in test])
        if name == 'full':
            text = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=512, min_df=2)
            train_text = text.fit_transform([features[i]['text'] for i in train]).toarray()
            test_text = text.transform([features[i]['text'] for i in test]).toarray()
            x_train = np.concatenate([x_train, train_text], axis=1)
            x_test = np.concatenate([x_test, test_text], axis=1)
        x_train = center_by_task(x_train, task_ids[train])
        x_test = center_by_task(x_test, task_ids[test])
        scaler = StandardScaler().fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
        model = Ridge(alpha=10, fit_intercept=False)
        outputs[name] = model.fit(x_train, y[train]).predict(x_test).tolist()
        if name == 'full':
            rng = np.random.default_rng(stable_seed(20261115, 'early-shuffle', landmark, category))
            shuffled = np.repeat(y[train, None], 100, axis=1)
            for tid in sorted(set(task_ids[train])):
                indices = np.flatnonzero(task_ids[train] == tid)
                for repeat in range(100): shuffled[indices, repeat] = rng.permutation(y[train][indices])
            outputs['shuffle'] = model.fit(x_train, shuffled).predict(x_test).T.tolist()
    return {'category': category, 'test_indices': test.tolist(), 'train_tasks': len(set(task_ids[train])),
            'test_tasks': len(set(task_ids[test])), 'outputs': outputs}


def evaluate(episodes, predictions, seed):
    values, pairs = task_pair_accuracy(episodes, predictions)
    both, both_pairs = task_pair_accuracy(episodes, predictions, both_bought=True)
    good_indices = [i for i, r in enumerate(episodes) if r['reward_valid'] is True]
    good, good_pairs = task_pair_accuracy([episodes[i] for i in good_indices], np.asarray(predictions)[good_indices])
    return {'accuracy': bootstrap_mean(list(values.values()), stable_seed(seed, 'all')), 'pairs': pairs,
            'both_purchased': bootstrap_mean(list(both.values()), stable_seed(seed, 'bought')), 'both_purchased_pairs': both_pairs,
            'valid_outcomes_conditional': bootstrap_mean(list(good.values()), stable_seed(seed, 'valid')), 'valid_pairs': good_pairs}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--revision', type=Path, required=True); p.add_argument('--workers', type=int, default=4)
    args = p.parse_args(); parent = verify_parent(args.revision)
    episodes = [r for r in read_rows(parent / 'episodes.jsonl') if r['phase'] == 'formal']
    by_episode = defaultdict(list)
    for row in read_rows(parent / 'steps.jsonl'):
        if row['phase'] == 'formal': by_episode[row['trajectory_id']].append(row)
    for rows in by_episode.values(): rows.sort(key=lambda r: r['step'])
    results = {}
    manifests = []
    for landmark in ['after_first_search', 'after_first_open']:
        selected, features = [], []
        for ep in episodes:
            feature = landmark_features(by_episode[ep['trajectory_id']], landmark)
            if feature is not None:
                selected.append(ep); features.append(feature)
                manifests.append({'landmark': landmark, 'trajectory_id': ep['trajectory_id'], **{k: v for k, v in feature.items() if k != 'text'}})
        task_ids = np.asarray([e['task_id'] for e in selected])
        labels = center_by_task(np.asarray([e['reward'] for e in selected], dtype=float), task_ids)
        categories = [e['category'] for e in selected]
        strict = defaultdict(int)
        for tid in sorted(set(task_ids)):
            values = [e['reward'] for e in selected if e['task_id'] == tid]
            strict[tid] = sum(abs(a - b) > 1e-8 for a, b in itertools.combinations(values, 2))
        result = {'eligible_episodes': len(selected), 'missing_episodes': len(episodes) - len(selected),
            'tasks': len(set(task_ids)), 'mixed_tasks': sum(n > 0 for n in strict.values()), 'strict_pairs': sum(strict.values()),
            'landmark_steps': dict(Counter(f['landmark_step'] for f in features)),
            'status': 'INSUFFICIENT', 'native_numeric_labels': True}
        if result['mixed_tasks'] >= 20 and result['strict_pairs'] >= 200 and len(set(categories)) >= 3:
            predictions = {k: np.full(len(selected), np.nan) for k in ['simple', 'full']}
            shuffled = np.full((100, len(selected)), np.nan)
            jobs = [(c, categories, task_ids.tolist(), labels.tolist(), features, landmark) for c in sorted(set(categories))]
            folds = []
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for index, fold in enumerate(pool.map(fit_fold, jobs)):
                    test = fold['test_indices']
                    for name in predictions: predictions[name][test] = fold['outputs'][name]
                    shuffled[:, test] = fold['outputs']['shuffle']
                    folds.append({k: v for k, v in fold.items() if k != 'outputs'})
                    if index % 10 == 0: print(json.dumps({'landmark': landmark, 'folds': index + 1, 'total': len(jobs)}), flush=True)
            predictions['final_valid_purchase_diagnostic'] = np.array([float(e['purchase'] and e['reward_valid'] is True) for e in selected])
            result.update({'status': 'COMPLETE_PREDICTIVE_ONLY', 'folds': folds,
                'comparisons': {k: evaluate(selected, v, landmark + k) for k, v in predictions.items()}})
            actual, _ = task_pair_accuracy(selected, predictions['full'])
            null = defaultdict(list)
            for v in shuffled:
                scores, _ = task_pair_accuracy(selected, v)
                for tid, score in scores.items(): null[tid].append(score)
            result['shuffled_control'] = {'repeats': 100, 'mean_task_accuracy': float(np.mean([np.mean(v) for v in null.values()])),
                'paired_difference': bootstrap_mean([actual[t] - np.mean(v) for t, v in null.items()], stable_seed(landmark, 'null-diff'))}
            result['predictions'] = [{'trajectory_id': ep['trajectory_id'], **{k: float(v[i]) for k, v in predictions.items()}} for i, ep in enumerate(selected)]
        results[landmark] = result
        (args.revision / 'early_probe_metrics.json').write_text(json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    (args.revision / 'early_feature_manifest.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in manifests))
    print(json.dumps({k: v['status'] for k, v in results.items()}), flush=True)


if __name__ == '__main__': main()
