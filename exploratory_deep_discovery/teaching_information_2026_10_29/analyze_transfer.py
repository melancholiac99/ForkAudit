"""Exploratory preference transfer across native / information-completed worlds.

This is NOT a measurement of a hindsight LLM teacher or a distillation run.
Measured outcome vectors are converted to simple decision rules, then evaluated
on disjoint continuation repeats. Repeated splits are averaged within each task;
they are not treated as additional independent observations.
"""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = ROOT / 'exploratory_deep_discovery/regime_mechanism_2026_10_28'
SEED = 2026102907


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(q, tau):
    if tau is None:
        p = np.isclose(q, np.max(q)).astype(float)
    else:
        p = np.exp((q - np.max(q)) / tau)
    return p / p.sum()


def ucov(x, y):
    """Centered action-vector cross product excluding matched repeat indices."""
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    n, arms = x.shape
    return float((x.sum(0) @ y.sum(0) - (x*y).sum()) / (n*(n-1)*arms))


def interval(values, rng):
    values = np.asarray(values, float)
    means = values[rng.integers(len(values), size=(5000, len(values)))].mean(1)
    return dict(n_tasks=len(values), mean=float(values.mean()),
                ci95=np.quantile(means, [.025, .975]).tolist())


def main():
    manifest_path = SRC / 'results/run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    points = [p for p in manifest['design']['points']
              if p['id'] in manifest['config']['point_ids']]
    assert len(points) == 24
    assert len({p['task_id'] for p in points}) == 24
    assert manifest['config']['reps'] == 16 and not manifest['config']['dry']
    hashes = {str(manifest_path.relative_to(ROOT)):
              hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    arrays = {}
    raw_blocks = {}
    for p in points:
        y = np.zeros((2, 16, 5))
        for ci, condition in enumerate(['native', 'completed']):
            for rep in range(16):
                for ai in range(5):
                    key = f"{p['id']}__a{ai}__{condition}__r{rep:03d}"
                    path = SRC / 'runs/main_24x16/results' / (key + '.json')
                    content = path.read_bytes()
                    r = json.loads(content)
                    assert r['key'] == key and r['status'] == 'complete'
                    assert not r['dry'] and r['condition'] == condition
                    assert r['run_fingerprint'] == manifest['run_fingerprint']
                    assert r['task_id'] == p['task_id']
                    y[ci, rep, ai] = int(r['won'])
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(content).hexdigest()
        arrays[p['id']] = y
        for rep in range(16):
            raw_blocks[p['id'], rep] = y[:, rep, :]
    # Independently check the compact outcome export against all raw records.
    export = SRC / 'results/summary_blocks.csv'
    with export.open() as f:
        exported = list(csv.DictReader(f))
    assert len(exported) == 24*16
    for r in exported:
        block = raw_blocks[r['point_id'], int(r['rep'])]
        for ci, condition in enumerate(['native', 'completed']):
            for ai in range(5):
                assert int(r[f'a{ai}_{condition}']) == block[ci, ai]
    hashes[str(export.relative_to(ROOT))] = hashlib.sha256(export.read_bytes()).hexdigest()
    dump('input_manifest.json', hashes)

    rng = np.random.default_rng(SEED)
    # Predeclared here before observing this script's output: tau=.2 is primary;
    # .1, .5 and greedy are sensitivity summaries, not selected for performance.
    settings = [('softmax_0.2', .2), ('softmax_0.1', .1),
                ('softmax_0.5', .5), ('greedy', None)]
    transfer = []
    profiles = []
    for p in points:
        y = arrays[p['id']]
        q0, q1 = y.mean(1)
        visible = np.array([p['arms'][f'a{i}']['clue_object_in_feedback']
                            for i in range(5)], float)
        c0, c1, c01 = ucov(y[0], y[0]), ucov(y[1], y[1]), ucov(y[0], y[1])
        profiles.append(dict(point_id=p['id'], old_regime=p['old_regime'],
            native_mu=float(q0.mean()), completed_mu=float(q1.mean()),
            native_C=c0, completed_C=c1, cross_condition_covariance=c01,
            centered_preference_change=c0+c1-2*c01,
            native_q=q0.tolist(), completed_q=q1.tolist(),
            visible_arms=visible.astype(int).tolist(),
            actions=[p['arms'][f'a{i}']['action'] for i in range(5)]))
        # Paired conditions and all five arms share repeat blocks. Fit/evaluation
        # blocks never overlap, including across information conditions.
        splits = [rng.permutation(16) for _ in range(256)]
        for name, tau in settings:
            accum = []
            for order in splits:
                for train, test in [(order[:8], order[8:]), (order[8:], order[:8])]:
                    assert not set(train).intersection(test)
                    fit0, fit1 = y[:, train, :].mean(1)
                    test0, test1 = y[:, test, :].mean(1)
                    p0, p1 = distribution(fit0, tau), distribution(fit1, tau)
                    accum.append([
                        p0@test0, p1@test0, p0@test1, p1@test1,
                        p0@visible, p1@visible,
                        test0.mean(), test1.mean(),
                    ])
            a = np.mean(accum, axis=0)
            transfer.append(dict(point_id=p['id'], task_id=p['task_id'],
                old_regime=p['old_regime'], rule=name,
                native_rule_native_outcome=float(a[0]),
                completed_rule_native_outcome=float(a[1]),
                native_rule_completed_outcome=float(a[2]),
                completed_rule_completed_outcome=float(a[3]),
                native_rule_visible_mass=float(a[4]),
                completed_rule_visible_mass=float(a[5]),
                uniform_native_outcome=float(a[6]), uniform_completed_outcome=float(a[7]),
                transfer_penalty_native=float(a[0]-a[1]),
                matched_gain_completed=float(a[3]-a[2]),
                completed_minus_native_visible_mass=float(a[5]-a[4]),
                native_rule_gain_over_uniform=float(a[0]-a[6]),
                completed_rule_gain_over_uniform_native=float(a[1]-a[6])))
    write_csv('transfer_points.csv', transfer)
    dump('action_profiles.json', profiles)
    summaries = {}
    metrics = list(transfer[0])[4:]
    for group in ['all', 'Fork', 'Indifferent', 'Dead']:
        summaries[group] = {}
        for name, _ in settings:
            rows = [r for r in transfer if r['rule'] == name and
                    (group == 'all' or r['old_regime'] == group)]
            summaries[group][name] = {key: interval([r[key] for r in rows], rng)
                                     for key in metrics}
    dump('summary.json', dict(status='exploratory; no teacher inference or training',
         raw_records=3840, independent_tasks=24, n_per_old_regime=8,
         primary_rule='softmax_0.2', split_design='256 random 8/8 partitions, both directions',
         uncertainty='5000 task bootstrap draws; split repeats averaged inside task',
         seed=SEED, groups=summaries))
    dump('verification.json', dict(raw_records_match_export=True, n_records=3840,
         unique_tasks=24, n_repeats=16, train_test_disjoint=True,
         added_gpu_rollouts=0, teacher_probabilities_measured=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, pid in zip(axes[:2], ['s044_t18', 's010_t29']):
        p = next(x for x in profiles if x['point_id'] == pid)
        ax.plot(range(5), p['native_q'], 'o-', label='Native continuation')
        ax.plot(range(5), p['completed_q'], 's-', label='Same policy + location clue')
        ax.set(xticks=range(5), xticklabels=['chosen', 'alt1', 'alt2', 'alt3', 'alt4'],
               ylim=(-.03, 1.04), title=pid, ylabel='Measured task success')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    for i, group in enumerate(['Dead', 'Indifferent', 'Fork']):
        s = summaries[group]['softmax_0.2']['transfer_penalty_native']
        m = 100*s['mean']; lo, hi = np.array(s['ci95'])*100
        axes[2].errorbar(i, m, yerr=[[m-lo], [hi-m]], fmt='o', capsize=4)
    axes[2].axhline(0, color='gray', lw=.8)
    axes[2].set(xticks=range(3), xticklabels=['Dead', 'Indifferent', 'Fork'],
        ylabel='Native-condition transfer penalty (pp)',
        title='Outcome-derived rules, disjoint evaluation')
    fig.suptitle('Information changes the action preference being taught (exploratory proxy)')
    fig.tight_layout()
    fig.savefig(OUT/'preference_transfer.png', dpi=180)
    fig.savefig(OUT/'preference_transfer.svg')
    for group in summaries:
        s = summaries[group]['softmax_0.2']
        print(group, json.dumps({k:s[k] for k in ['native_rule_native_outcome',
            'completed_rule_native_outcome','transfer_penalty_native',
            'completed_minus_native_visible_mass']}))


if __name__ == '__main__':
    main()
