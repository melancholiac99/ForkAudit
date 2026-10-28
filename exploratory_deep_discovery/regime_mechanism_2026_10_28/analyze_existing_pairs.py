"""Re-use the completed two-arm intervention; do not claim five-arm transitions."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from metrics import bootstrap, completion_effect, outcome_components

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE.parent / 'opportunity_completion/runs/main_32x16'


def main():
    manifest = json.loads((SOURCE/'run_manifest.json').read_text())
    rows = []
    hashes = {}
    for p in manifest['design']['points']:
        arrays = {}
        for cond in ['native', 'completed']:
            arr = []
            for rep in range(manifest['config']['reps']):
                one = []
                for arm in ['revealing', 'other']:
                    path = SOURCE/'results'/f"{p['id']}__{arm}__{cond}__r{rep:03d}.json"
                    record = json.loads(path.read_text())
                    assert record['status'] == 'complete' and not record['dry']
                    assert record['run_fingerprint'] == manifest['run_fingerprint']
                    one.append(int(record['won']))
                    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                arr.append(one)
            arrays[cond] = np.asarray(arr)
        row = dict(point_id=p['id'], task_id=p['task_id'], old_fork=p['old_fork'])
        row.update(completion_effect(arrays['native'], arrays['completed']))
        for cond, arr in arrays.items():
            row.update({f'{cond}_{k}': v for k, v in outcome_components(arr).items()})
        rows.append(row)
    assert len({r['task_id'] for r in rows}) == len(rows), 'Task bootstrap assumes unique tasks'
    fields = [k for k in rows[0] if k not in ['point_id', 'task_id', 'old_fork']]
    groups = {'all': rows, 'old_fork': [r for r in rows if r['old_fork']],
              'old_nonfork': [r for r in rows if not r['old_fork']]}
    out = dict(status='exploratory reuse; two arms only',
               definition='C=Var_a(q_a), uniform over the two tested arms. '
                          'Quadratic estimates exclude same-repeat products. '
                          'Not a test of frozen five-arm regime transitions.',
               identity='C_completed-C_native = 2 Cov_a(q_native, lift) + Var_a(lift)',
               groups={g: {f: bootstrap([r[f] for r in rs]) for f in fields}
                       for g, rs in groups.items()})
    (HERE/'existing_pair_analysis.json').write_text(json.dumps(out, indent=2)+'\n')
    with (HERE/'existing_pair_points.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (HERE/'existing_pair_inputs.json').write_text(json.dumps(hashes, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
