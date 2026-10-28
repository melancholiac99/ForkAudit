"""Post-hoc mechanism diagnostics with two interpretable metadata contrasts.

Written after inspecting the first five complete points, including s010_t29.
No outcome-based exclusion. Report all 24 points. These are not additional
randomized interventions and do not identify mediation effects.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from protocol import HERE, ARMS, read, write
from metrics import bootstrap
from analyze import csv_write


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run_dir', type=Path)
    args = ap.parse_args()
    summary = read(args.run_dir / 'summary.json')
    assert summary['complete']
    manifest = read(args.run_dir / 'run_manifest.json')
    design = manifest['design']['points']
    with (args.run_dir / 'summary_points.csv').open() as f:
        values = {r['point_id']: r for r in csv.DictReader(f)}
    with (args.run_dir / 'summary_arms.csv').open() as f:
        arms = list(csv.DictReader(f))
    metadata = read(HERE / 'affordance_checks.json')['all_arm_metadata']
    rows = []
    for p in design:
        pid = p['id']
        if pid not in values:
            continue
        held = {r['arm']: bool(r['held_objects']) and p['clue']['object'] not in r['held_objects']
                for r in metadata if r['point_id'] == pid}
        count = sum(held.values())
        group = 'shared-held-other' if count == 5 else 'branch-held-other' if count else 'no-held-other'
        ar = [a for a in arms if a['point_id'] == pid]
        visible = {a: p['arms'][a]['clue_object_in_feedback'] for a in ARMS}
        lifts_visible = [float(a['lift']) for a in ar if visible[a['arm']]]
        lifts_hidden = [float(a['lift']) for a in ar if not visible[a['arm']]]
        row = dict(point_id=pid, old_regime=p['old_regime'], held_group=group,
            held_other_arms=count, clue_visible_arms=sum(visible.values()),
            **{k: float(values[pid][k]) for k in ['native_C', 'completed_C', 'C_change', 'mean_lift']},
            visible_arm_lift=float(np.mean(lifts_visible)) if lifts_visible else None,
            hidden_arm_lift=float(np.mean(lifts_hidden)) if lifts_hidden else None)
        row['hidden_minus_visible_lift'] = (row['hidden_arm_lift'] - row['visible_arm_lift']
            if lifts_hidden and lifts_visible else None)
        rows.append(row)
    contrasts = {}
    for name in ['all', 'Dead', 'Indifferent', 'Fork']:
        rs = [r for r in rows if name == 'all' or r['old_regime'] == name]
        contrasts[name] = {k: bootstrap([r[k] for r in rs if r[k] is not None]) for k in
            ['visible_arm_lift', 'hidden_arm_lift', 'hidden_minus_visible_lift']}
    held_groups = {}
    for name in ['no-held-other', 'shared-held-other', 'branch-held-other']:
        rs = [r for r in rows if r['held_group'] == name]
        held_groups[name] = dict(n=len(rs), point_ids=[r['point_id'] for r in rs],
            **{k: float(np.mean([r[k] for r in rs])) for k in
               ['native_C', 'completed_C', 'C_change', 'mean_lift']})
    write(args.run_dir / 'mechanism_summary.json', dict(
        status='post-hoc explanatory analysis', selection=__doc__,
        visible_definition='The exact clue object instance appears in the native immediate feedback.',
        held_definition='Post-action world facts contain a held object other than the clue object. '
                        'This is a metadata grouping; seven representative local constraints were CPU-checked. '
                        'It is not a task-blockage label: s025_t31 already holds another valid potato instance.',
        contrasts=contrasts, held_groups=held_groups,
        limitations=['Visible/hidden branches are not randomized arm types.',
            'Held-object grouping was motivated by observed complete cases; tiny groups are descriptive.',
            'Information salience and new semantic content are not separated by a placebo condition.',
            'No mediated fraction is identified from a change in action variance.']))
    csv_write(args.run_dir / 'mechanism_points.csv', rows)
    print({'contrasts': contrasts, 'held_groups': held_groups})


if __name__ == '__main__':
    main()
