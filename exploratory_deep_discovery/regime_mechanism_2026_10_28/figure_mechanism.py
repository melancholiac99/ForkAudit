"""Illustrative post-hoc cases; full sample is in summary_figure_all_profiles."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', type=Path)
    args = ap.parse_args()
    with (args.run_dir / 'summary_arms.csv').open() as f:
        arms = list(csv.DictReader(f))
    with (args.run_dir / 'summary_points.csv').open() as f:
        points = {r['point_id']: r for r in csv.DictReader(f)}
    examples = [
        ('s005_t31', 'Shared execution obstacle',
         'Clue supplied, all five branches still fail',
         ['examine\ndesk', 'go to\nsidetable', 'go to\ndrawer', 'go to\ncabinet', 'inventory']),
        ('s044_t18', 'Information gap reduced',
         'Weaker branches gain more; action variance falls',
         ['go to\ncountertop', 'go to\ncabinet 4', 'go to\nfridge', 'examine\ncabinet 2', 'go to\ncabinet 1']),
        ('s010_t29', 'Execution difference exposed',
         'Four branches improve or stay high; one stays at zero',
         ['go to\ndrawer', 'examine\ndresser', 'look', 'take\npencil', 'go to\nshelf'])]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.8), sharey=True)
    for ax, (pid, title, note, labels) in zip(axes, examples):
        rows = [r for r in arms if r['point_id'] == pid]
        q0 = np.array([float(r['native_q']) for r in rows])
        q1 = np.array([float(r['completed_q']) for r in rows])
        x = np.arange(5)
        ax.bar(x - .18, q0, width=.34, color='#4477aa', label='Native')
        ax.bar(x + .18, q1, width=.34, color='#ee7733', label='Shared location clue')
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(-.01, 1.15)
        ax.set_yticks([0, .25, .5, .75, 1])
        ax.grid(axis='y', alpha=.2)
        r = points[pid]
        c0, c1 = [0. if abs(float(r[k])) < 1e-12 else float(r[k])
                  for k in ['native_C', 'completed_C']]
        ax.set_title(f'{title}\n{pid} · old {r["old_regime"]}\n'
                     f'μ: {q0.mean():.2f} → {q1.mean():.2f};  '
                     f'C: {c0:.3f} → {c1:.3f}', fontsize=10)
        ax.text(.5, -.23, note, transform=ax.transAxes, ha='center', fontsize=8.5)
    axes[0].set_ylabel('Measured success probability (16 per arm / condition)')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[2].annotate('Held pencil blocks target pickup', xy=(3.15, .015), xytext=(2.15, .32),
                     fontsize=8, ha='center', arrowprops=dict(arrowstyle='->', lw=1))
    fig.suptitle('Sharing a truthful location clue can preserve, reduce, or amplify action differences', y=1.02)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(args.run_dir / ('mechanism_cases.' + ext), dpi=190, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
