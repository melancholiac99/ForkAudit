"""Summarize all four cells, paired rescue, and gap reduction; keep null/negative results.

Only complete four-cell replicate blocks enter comparisons. A default full analysis
requires the entire run; --allow-partial writes separately named partial artifacts.
Bootstrap whole tasks, preserving the paired conditions and selected points.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from protocol import ARMS, CONDITIONS, read, write
from run import result_key, validate_record


def bootstrap(rows, field, reps=4000):
    if not rows:
        return dict(n_points=0, n_tasks=0, mean=None, ci95=None)
    tasks = sorted({r['task_id'] for r in rows})
    sums = np.array([sum(r[field] for r in rows if r['task_id'] == t) for t in tasks])
    counts = np.array([sum(r['task_id'] == t for r in rows) for t in tasks])
    rng = np.random.default_rng(20261028)
    weights = rng.multinomial(len(tasks), np.ones(len(tasks)) / len(tasks), size=reps)
    values = (weights @ sums) / (weights @ counts)
    return dict(n_points=len(rows), n_tasks=len(tasks), mean=float(sums.sum() / counts.sum()),
                ci95=np.quantile(values, [.025, .975]).tolist())


def collect(run_dir, allow_partial=False):
    manifest = read(run_dir / 'run_manifest.json')
    config, design = manifest['config'], manifest['design']
    selected = set(config['point_ids'])
    points = [p for p in design['points'] if p['id'] in selected]
    rows = []
    blocks = []
    present = 0
    total_steps = total_tokens = 0
    for point in points:
        cells = {(arm, condition): [] for arm in ARMS for condition in CONDITIONS}
        use = {(arm, condition): [] for arm in ARMS for condition in CONDITIONS}
        complete_reps = []
        for rep in range(config['reps']):
            records = {}
            for arm in ARMS:
                for condition in CONDITIONS:
                    key = result_key(point, arm, condition, rep)
                    path = run_dir / 'results' / f'{key}.json'
                    if path.exists():
                        record = read(path)
                        validate_record(record, manifest['run_fingerprint'], key)
                        if record['dry'] != config['dry']:
                            raise ValueError('Mixed dry and scientific outputs')
                        if (record['point_id'], record['arm'], record['condition'], record['rep']) != (
                                point['id'], arm, condition, rep):
                            raise ValueError(f'Wrong record identity: {key}')
                        records[arm, condition] = record
                        present += 1
                        total_steps += record['generated_steps']
                        total_tokens += record['generated_tokens']
            if len(records) == 4:
                complete_reps.append(rep)
                block = dict(point_id=point['id'], task_id=point['task_id'], rep=rep)
                for (arm, condition), record in records.items():
                    cells[arm, condition].append(int(record['won']))
                    generated = [t for t in record['transitions'] if not t['forced']]
                    took = any(t['action'].startswith('take ' + point['clue']['object'] + ' from ')
                               for t in generated[:3])
                    use[arm, condition].append(int(took))
                    block[f'{arm}_{condition}'] = int(record['won'])
                blocks.append(block)
        if not complete_reps:
            continue
        row = dict(point_id=point['id'], task_id=point['task_id'], task_type=point['task_type'],
                   old_fork=point['old_fork'], old_masked=point['old_masked'],
                   paired_reps=len(complete_reps), clue=point['clue']['text'],
                   revealing_action=point['arms']['revealing']['action'],
                   other_action=point['arms']['other']['action'])
        for arm in ARMS:
            for condition in CONDITIONS:
                row[f'q_{arm}_{condition}'] = float(np.mean(cells[arm, condition]))
                row[f'take3_{arm}_{condition}'] = float(np.mean(use[arm, condition]))
        row['native_gap'] = row['q_revealing_native'] - row['q_other_native']
        row['completed_gap'] = row['q_revealing_completed'] - row['q_other_completed']
        row['gap_reduction'] = row['native_gap'] - row['completed_gap']
        row['other_lift'] = row['q_other_completed'] - row['q_other_native']
        row['revealing_lift'] = row['q_revealing_completed'] - row['q_revealing_native']
        assert np.isclose(row['gap_reduction'], row['other_lift'] - row['revealing_lift'])
        row['take3_other_lift'] = row['take3_other_completed'] - row['take3_other_native']
        rows.append(row)
    expected = len(points) * 4 * config['reps']
    if present != expected and not allow_partial:
        raise ValueError(f'Run incomplete: {present}/{expected}. Finish/resume, or use --allow-partial.')
    return manifest, rows, blocks, dict(complete=present, expected=expected,
                                       generated_steps=total_steps, generated_tokens=total_tokens)


def export_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, summary, output, dry):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for arm, color in [('revealing', '#2678b2'), ('other', '#e58b28')]:
        metrics = [summary['groups']['all'][f'q_{arm}_{condition}'] for condition in CONDITIONS]
        means = np.array([m['mean'] for m in metrics]) * 100
        errors = np.array([[m['mean'] - m['ci95'][0] for m in metrics],
                           [m['ci95'][1] - m['mean'] for m in metrics]]) * 100
        axes[0].errorbar([0, 1], means, yerr=errors, marker='o', capsize=4, color=color, label=arm)
    axes[0].set_xticks([0, 1], ['Native feedback', 'Shared location clue'])
    axes[0].set_ylabel('Success rate (%)')
    axes[0].set_ylim(-5, 105)
    axes[0].legend()
    for fork, marker in [(True, 'o'), (False, '^')]:
        subset = [r for r in rows if r['old_fork'] == fork]
        axes[1].scatter([100 * r['native_gap'] for r in subset],
                        [100 * r['completed_gap'] for r in subset], marker=marker,
                        label='Original Fork' if fork else 'Original non-Fork', alpha=.8)
    axes[1].plot([-100, 100], [-100, 100], '--', color='gray', linewidth=1)
    axes[1].axhline(0, color='gray', linewidth=.5)
    axes[1].axvline(0, color='gray', linewidth=.5)
    axes[1].set_xlabel('Native arm gap (percentage points)')
    axes[1].set_ylabel('Completed arm gap (percentage points)')
    axes[1].legend()
    fig.suptitle('DRY RUN — NOT SCIENTIFIC RESULTS' if dry else 'Exploratory location completion')
    fig.tight_layout()
    fig.savefig(output.with_suffix('.png'), dpi=180)
    fig.savefig(output.with_suffix('.svg'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    manifest, rows, blocks, cost = collect(args.run_dir, args.allow_partial)
    if not rows:
        parser.error('No complete four-cell replicate blocks yet')
    fields = ['q_revealing_native', 'q_other_native', 'q_revealing_completed', 'q_other_completed',
              'native_gap', 'completed_gap', 'gap_reduction', 'other_lift', 'revealing_lift',
              'take3_other_lift']
    groups = dict(all=rows, original_fork=[r for r in rows if r['old_fork']],
                  original_non_fork=[r for r in rows if not r['old_fork']])
    summary = dict(status='CPU DRY RUN — NOT SCIENTIFIC RESULTS' if manifest['config']['dry'] else 'exploratory',
                   complete=cost['complete'] == cost['expected'],
                   run_fingerprint=manifest['run_fingerprint'], cost=cost,
                   estimand='Point-average signed arm gap reduction; revealing arm fixed by feedback, not outcomes.',
                   primary_group='original_fork',
                   groups={name: {field: bootstrap(subset, field) for field in fields}
                           for name, subset in groups.items()})
    stem = 'summary' if summary['complete'] else 'partial_summary'
    write(args.run_dir / f'{stem}.json', summary)
    export_csv(args.run_dir / f'{stem}_points.csv', rows)
    export_csv(args.run_dir / f'{stem}_paired_outcomes.csv', blocks)
    lines = [f"# {summary['status']}", '', f"Completed continuations: {cost['complete']}/{cost['expected']}.", '',
             'Signed gap = success(revealing) − success(other). Gap reduction = native gap − completed gap.',
             'A smaller gap is evidence of rescue only alongside an improvement of the other arm; '
             'a decline of the revealing arm alone is not rescue.', '',
             '| Group | Points | Native gap | Completed gap | Gap reduction | Other-arm lift | Revealing-arm lift |',
             '|---|---:|---:|---:|---:|---:|---:|']
    def cell(metric):
        if metric['mean'] is None:
            return '—'
        lo, hi = metric['ci95']
        return f"{100 * metric['mean']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    for name, values in summary['groups'].items():
        cells = [cell(values[f]) for f in ['native_gap', 'completed_gap', 'gap_reduction',
                                          'other_lift', 'revealing_lift']]
        lines.append(f"| {name} | {values['native_gap']['n_points']} | " + ' | '.join(cells) + ' |')
    lines += ['', 'All numbers are percentage points; 95% intervals resample whole tasks.',
              'The 24/8 enriched sample is not a population prevalence estimate. '
              'The cue is given once and then ages out with the original memory. '
              'Positions, affordances and original action history may still differ between arms.', '',
              '## Every selected point', '',
              '| Point | Fork | Paired reps | Native gap | Completed gap | Other lift | Revealing lift |',
              '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['point_id']} | {row['old_fork']} | {row['paired_reps']} | " +
                     ' | '.join(f"{100 * row[f]:+.1f}" for f in
                                ['native_gap', 'completed_gap', 'other_lift', 'revealing_lift']) + ' |')
    (args.run_dir / f'{stem}.md').write_text('\n'.join(lines) + '\n')
    if not args.no_plot:
        plot(rows, summary, args.run_dir / f'{stem}_figure', manifest['config']['dry'])
    print('\n'.join(lines[:14]))
    print(f'Analysis written to {args.run_dir}/{stem}.*')


if __name__ == '__main__':
    main()
