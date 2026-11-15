"""Task-centered Ridge probe, category holdout; prediction only, no policy training."""
from collections import Counter, defaultdict
import itertools
import re

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy import sparse

from exploratory_shop_audit.data import stable_seed
from exploratory_shop_audit.audit_statistics import bootstrap_mean


def feature_rows(episodes, steps):
    by_episode = defaultdict(list)
    for step in steps:
        by_episode[step['trajectory_id']].append(step)
    simple, full, texts, early = [], [], [], []
    for ep in episodes:
        rows = sorted(by_episode[ep['trajectory_id']], key=lambda r: r['step'])[:8]
        actions = [r['action'] for r in rows]
        types = Counter(a.split('[', 1)[0] for a in actions)
        base = {f'action_{a}': types[a] for a in ['search', 'click', 'finish']}
        base['visible_decisions'] = len(rows)
        base['mean_remaining'] = float(np.mean([r['remaining'] for r in rows]))
        f = dict(base)
        for kind in ['search', 'click']:
            selected = [a for a in actions if a.startswith(kind + '[')]
            f[kind + '_repetitions'] = len(selected) - len(set(selected))
        visible_texts = [r['before']['observation'] for r in rows]
        for field in ['title:', 'price:', 'key_attributes:', 'available_options:', 'content:', 'rank|asin|']:
            f['seen_' + field] = sum(field in t for t in visible_texts)
        f['specification_actions'] = sum(r['action'].startswith('click[') and any(
            r['action'][6:-1].lower() == str(v).lower() for vals in r['before']['public'].get('available_options', {}).values() for v in vals) for r in rows)
        simple.append(base); full.append(f)
        # Character 2/3-grams retain Chinese text; training fold vocabulary only.
        texts.append('\n'.join(visible_texts))
        early.append(any(a.lower() == 'click[buy now]' for a in actions))
    return simple, full, texts, early


def center_by_task(matrix, task_ids):
    result = np.asarray(matrix).copy()
    for task in set(task_ids):
        index = np.flatnonzero(task_ids == task)
        result[index] -= result[index].mean(axis=0)
    return result


def task_pair_accuracy(episodes, predictions, both_bought=False, early=None, early_value=None):
    grouped = defaultdict(list)
    for index, row in enumerate(episodes):
        grouped[row['task_id']].append(index)
    values, pair_count = {}, 0
    for tid, indices in grouped.items():
        correct = []
        for i, j in itertools.combinations(indices, 2):
            if both_bought and not (episodes[i]['purchase'] and episodes[j]['purchase']):
                continue
            if early is not None and not (early[i] == early_value and early[j] == early_value):
                continue
            delta = episodes[i]['reward'] - episodes[j]['reward']
            if abs(delta) <= 1e-8 or not np.isfinite(predictions[i]) or not np.isfinite(predictions[j]):
                continue
            difference = predictions[i] - predictions[j]
            correct.append(.5 if abs(difference) <= 1e-12 else float(difference * delta > 0))
        if correct:
            values[tid] = float(np.mean(correct))
            pair_count += len(correct)
    return values, pair_count


def run_probe(episodes, steps, group_metrics):
    if group_metrics['mixed_groups'] < 20 or group_metrics['strict_pairs'] < 200:
        return {'status': 'INSUFFICIENT_REWARD_DIFFERENCES', 'mixed_tasks': group_metrics['mixed_groups'],
                'strict_pairs': group_metrics['strict_pairs'], 'required': [20, 200]}
    categories = np.array([r['category'] for r in episodes])
    unique = sorted(set(categories))
    if len(unique) < 3:
        return {'status': 'INSUFFICIENT_CATEGORIES', 'categories': len(unique)}
    task_ids = np.array([r['task_id'] for r in episodes])
    y = center_by_task(np.array([r['reward'] for r in episodes], dtype=float), task_ids)
    simple, full, texts, early = feature_rows(episodes, steps)
    predictions = {name: np.full(len(episodes), np.nan) for name in ['full', 'action_progress']}
    shuffled = np.full((100, len(episodes)), np.nan)
    folds = []
    for category in unique:
        test = np.flatnonzero(categories == category)
        train = np.flatnonzero(categories != category)
        for name, dictionaries in [('full', full), ('action_progress', simple)]:
            vectorizer = DictVectorizer(sparse=False)
            x_train = vectorizer.fit_transform([dictionaries[i] for i in train])
            x_test = vectorizer.transform([dictionaries[i] for i in test])
            if name == 'full':
                text_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=512, min_df=2)
                t_train = text_vectorizer.fit_transform([texts[i] for i in train]).toarray()
                t_test = text_vectorizer.transform([texts[i] for i in test]).toarray()
                x_train = np.concatenate([x_train, t_train], axis=1)
                x_test = np.concatenate([x_test, t_test], axis=1)
            x_train = center_by_task(x_train, task_ids[train])
            x_test = center_by_task(x_test, task_ids[test])
            scaler = StandardScaler().fit(x_train)
            x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
            ridge = Ridge(alpha=10, fit_intercept=False)
            predictions[name][test] = ridge.fit(x_train, y[train]).predict(x_test)
            if name == 'full':
                rng = np.random.default_rng(stable_seed(20261115, 'probe-shuffle', category))
                labels = np.repeat(y[train, None], 100, axis=1)
                for task in set(task_ids[train]):
                    index = np.flatnonzero(task_ids[train] == task)
                    for repeat in range(100):
                        labels[index, repeat] = rng.permutation(y[train][index])
                # Multi-output Ridge is exactly the same fixed-alpha fit per shuffle.
                shuffled[:, test] = ridge.fit(x_train, labels).predict(x_test).T
        folds.append({'held_out_category': category, 'train_tasks': len(set(task_ids[train])), 'test_tasks': len(set(task_ids[test]))})
    predictions['final_valid_purchase_diagnostic'] = np.array([float(r['purchase'] and bool(r['reward_valid'])) for r in episodes])
    result = {'status': 'COMPLETED_PREDICTIVE_ONLY', 'folds': folds, 'window': 8, 'ridge_alpha': 10,
              'text_features': 'train-fold Chinese-safe character 2/3-gram TFIDF, max512, min_df2', 'comparisons': {}}
    for name, p in predictions.items():
        values, count = task_pair_accuracy(episodes, p)
        both, both_count = task_pair_accuracy(episodes, p, both_bought=True)
        result['comparisons'][name] = {'accuracy': bootstrap_mean(list(values.values()), stable_seed(name, 'probe')),
            'pairs': count, 'both_purchased_accuracy': bootstrap_mean(list(both.values()), stable_seed(name, 'both')),
            'both_purchased_pairs': both_count,
            'early_purchase': {}}
        for value in [False, True]:
            stratified, pairs = task_pair_accuracy(episodes, p, early=early, early_value=value)
            result['comparisons'][name]['early_purchase'][str(value)] = {
                'episodes': sum(x == value for x in early), 'pairs': pairs,
                'accuracy': bootstrap_mean(list(stratified.values()), stable_seed(name, value, 'early'))}
    actual, _ = task_pair_accuracy(episodes, predictions['full'])
    random_per_task = defaultdict(list)
    shuffle_means = []
    for prediction in shuffled:
        values, _ = task_pair_accuracy(episodes, prediction)
        shuffle_means.append(float(np.mean(list(values.values()))))
        for tid, value in values.items():
            random_per_task[tid].append(value)
    result['shuffled_control'] = {'repeats': 100, 'mean_accuracy': float(np.mean(shuffle_means)),
        'shuffle_accuracy_distribution': np.quantile(shuffle_means, [0, .25, .5, .75, 1]).tolist(),
        'paired_task_accuracy_difference': bootstrap_mean([actual[t] - np.mean(v) for t, v in random_per_task.items()], stable_seed('probe-difference'))}
    result['predictions'] = [{'trajectory_id': r['trajectory_id'], **{name: float(p[i]) for name, p in predictions.items()}} for i, r in enumerate(episodes)]
    return result
