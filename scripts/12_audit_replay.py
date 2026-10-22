"""Validate raw Phase 1 records and build decision_points without feature correlations."""
from collections import Counter
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import common


def main():
    root = Path(common.ROOT)
    rows, manifest, errors = [], {}, []
    continuations = divergences = 0
    commits = Counter()
    expected_paths = set()
    trajectories = sorted((root / 'data/rollout').glob('*.json'))
    assert len(trajectories) == 50
    for path in trajectories:
        tr = json.loads(path.read_text())
        manifest[str(path.relative_to(root))] = common.file_sha256(path)
        for st in tr['steps']:
            rp = root / 'data/replay' / tr['traj_id'] / f"t{st['turn']:02d}.json"
            expected_paths.add(rp)
            try:
                r = json.loads(rp.read_text())
                manifest[str(rp.relative_to(root))] = common.file_sha256(rp)
                assert (r['traj_id'], r['turn'], r['seed'], r['task_id'], r['task_type']) == (
                    tr['traj_id'], st['turn'], tr['seed'], tr['task_id'], tr['task_type'])
                assert r['factual_action'] == st['action']
                assert (root / st['prompt_path']).is_file()
                assert r['provenance']['inputs'][str(path.relative_to(root))] == manifest[str(path.relative_to(root))]
                commits[r['provenance']['commit']] += 1
                alts = r['alternatives']
                assert len(alts) == len(set(alts)) <= 4
                assert all(a in st['admissible'] and a != st['action'] for a in alts)
                assert r['cf_defined'] == (len(alts) == 4)
                assert r['exclude_reason'] == (None if r['cf_defined'] else 'lt4_alternatives')
                assert len(r['drafts']) == r['n_draws'] <= 300
                assert r['n_draws'] >= 32
                if not r['cf_defined']:
                    assert r['n_draws'] == 300
                supported = {d['action'] for d in r['drafts'] if d['valid'] and d['admissible']}
                assert set(alts) <= supported
                fac, alt = r['factual_outcomes'], r['alt_outcomes']
                assert len(fac) == 3 and all(v in (0, 1) for v in fac)
                roles = [1] + (list(range(2, 6)) if r['cf_defined'] else [])
                assert len(r['continuations']) == 3 * len(roles)
                assert {(c['role'], c['rep']) for c in r['continuations']} == {
                    (role, rep) for role in roles for rep in range(3)}
                assert len(alt) == len(alts)
                assert all(len(a) == (3 if r['cf_defined'] else 0) for a in alt)
                for c in r['continuations']:
                    assert 1 <= len(c['actions']) == len(c['valids']) <= 51 - st['turn']
                    assert c['len'] == st['turn'] - 1 + len(c['actions'])
                    action = st['action'] if c['role'] == 1 else alts[c['role'] - 2]
                    assert c['first_action'] == c['actions'][0] == action
                    outcomes = fac if c['role'] == 1 else alt[c['role'] - 2]
                    assert int(c['won']) == outcomes[c['rep']]
                    divergences += c['refeed_div']
                continuations += len(r['continuations'])
                # Integer contrast numerator avoids classifying roundoff as pivotal.
                numerator = 4 * sum(fac) - sum(map(sum, alt)) if r['cf_defined'] else None
                outcomes = fac + [v for a in alt for v in a]
                rows.append(dict(
                    traj_id=tr['traj_id'], turn=st['turn'], task_id=tr['task_id'],
                    task_type=tr['task_type'], seed=tr['seed'], prompt_path=st['prompt_path'],
                    chosen_action=st['action'], admissible_actions=st['admissible'],
                    A_replay=numerator / 12 if numerator is not None else None,
                    n_factual=3, n_alt=4 if r['cf_defined'] else 0,
                    factual_outcomes=fac, alt_outcomes=alt,
                    sigma_floor=float(np.std(fac, ddof=0)), crit_sd=float(np.std(outcomes, ddof=0)),
                    cf_defined=r['cf_defined'], exclude_reason=r['exclude_reason'], source='own_replay'))
            except (AssertionError, OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({'path': str(rp), 'error': repr(exc)})
    extra = set((root / 'data/replay').glob('*/*.json')) - expected_paths
    assert not extra, extra
    assert not errors, errors[:20]
    assert divergences == 0, divergences
    df = pd.DataFrame(rows)
    assert not df.duplicated(['traj_id', 'turn']).any()
    df.to_parquet(root / 'data/decision_points.parquet', index=False)
    defined = df[df.cf_defined]
    report = dict(
        provenance=common.provenance(__file__),
        input_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        trajectories=len(trajectories), decision_points=len(df), defined=len(defined),
        defined_fraction=len(defined) / len(df), pivotal=int(defined.A_replay.ne(0).sum()),
        pivotal_fraction=float(defined.A_replay.ne(0).mean()),
        sigma_floor_zero_fraction_all=float(df.sigma_floor.eq(0).mean()),
        sigma_floor_zero_fraction_defined=float(defined.sigma_floor.eq(0).mean()),
        excluded=int((~df.cf_defined).sum()), continuations=continuations,
        refeed_divergences=divergences, replay_commits=dict(commits),
        schema_notes={'n_alt': 'number of complete alternative arms; each has 3 replicas',
                      'std_ddof': 0},
        pending=['chosen-action logprob exclusion shift requires action-only teacher-forced scoring',
                 'HF-vLLM chosen-action scoring alignment not yet performed'],
        minimum_defined_1200_pass=len(defined) >= 1200,
    )
    common.write_json(str(root / 'eval/phase1_input_manifest.json'), manifest)
    common.write_json(str(root / 'eval/phase1_audit.json'), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
