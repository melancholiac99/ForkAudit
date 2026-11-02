"""Independent raw-bank reconstruction and direct trajectory bootstrap for P8 CPU results."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

H = Path(__file__).resolve().parent
S = Path('/root/autodl-tmp/idea')


def read(p):
    return json.loads(Path(p).read_text())


def main():
    out = read(H / 'p8_cpu.json'); rows = read(H / 'data/points.json')
    cached = np.load(H / 'data/arrays.npz')['q']
    source_hashes = read(H / 'science_inputs.json')
    for path, sha in source_hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
    lab = pd.read_parquet(S / 'data/labels_final.parquet').set_index(['traj_id', 'turn'])
    strata = pd.read_parquet(S / 'exploratory_P4/strata.parquet').set_index(['traj_id', 'turn'])
    gu, diff, tids, pop, nonfork, wf, live = [], [], [], [], [], [], []
    for i, r in enumerate(rows):
        tid, turn = r['traj_id'], r['turn']
        a = read(S / f'data/replay/{tid}/t{turn:02d}.json')
        b = read(S / f'data/replay_b1/{tid}/t{turn:02d}.json')
        oa = [a['factual_outcomes']] + a['alt_outcomes']
        ob = [b['factual_outcomes']] + b['alt_outcomes']
        q = np.array([(sum(x)+sum(y))/6 for x, y in zip(oa, ob)])
        assert np.array_equal(q, cached[i])
        counts = [sum(d['batch'] == 0 and d['valid'] and d['admissible'] and d['action'] == act for d in a['drafts']) for act in r['actions']]
        total = sum(counts)+2.5
        p = np.array([(c+.5)/total for c in counts])
        assert np.array_equal(p, r['policy'])
        value = sum((p-.2)*q)
        gu.append(value); diff.append(sum((np.array(r['R1_window'])-p)*q))
        tids.append(tid); pop.append(not strata.loc[(tid, turn), 'prompt_target'])
        ll = bool(lab.loc[(tid, turn), 'live']); ff = bool(lab.loc[(tid, turn), 'het'])
        assert ll == r['live'] and ff == r['fork']
        live.append(ll); wf.append(ff); nonfork.append(ll and not ff)
    gu, diff = np.array(gu), np.array(diff)
    pop, nonfork, wf, live = map(lambda x: np.array(x, bool), (pop, nonfork, wf, live))
    assert np.array_equal(pop, [r['H_prompt'] for r in rows])
    all_t = sorted(set(tids)); ti = {t: i for i, t in enumerate(all_t)}
    inv = np.array([ti[t] for t in tids])
    draws = np.random.default_rng(20261101).multinomial(len(all_t), np.full(len(all_t), 1/len(all_t)), size=10000)
    verified = {}
    for name, vals, mask, expected in [
        ('P1', gu, pop, out['table']['X2']['H_prompt']['G_u']),
        ('P2', gu, pop & nonfork, out['table']['X2']['H_nonFork_Live']['G_u']),
        ('S4', diff, pop, out['contrasts']['R1_window-policy']['H_prompt']),
    ]:
        values, counts = [], []
        for t in all_t:
            ix = mask & (np.array(tids) == t)
            values.append(float(sum(vals[ix]))); counts.append(int(ix.sum()))
        # Deliberately direct loop, independent of the production matrix aggregation.
        dist = []
        for weights in draws:
            den = sum(int(w)*c for w, c in zip(weights, counts))
            if den:
                dist.append(sum(int(w)*v for w, v in zip(weights, values))/den)
        estimate = float(vals[mask].mean()); ci = np.percentile(dist, [2.5, 97.5])
        error = max(abs(estimate-expected['estimate']), float(max(abs(ci-expected['ci95']))))
        assert error < 1e-12, (name, error)
        verified[name] = dict(estimate=estimate, ci95=ci.tolist(), max_error=error)
    sample = read(H / 'blind_sample.json'); expected_ids = set()
    for tid in sorted(set(np.array(tids)[pop])):
        items = [r for r in rows if r['traj_id'] == tid and r['H_prompt']]
        items.sort(key=lambda r: hashlib.sha256(f"P8-v1|{tid}|{r['turn']}".encode()).hexdigest())
        expected_ids.update(r['point_id'] for r in items[:8])
    assert {r['point_id'] for r in sample['points']} == expected_ids
    result = dict(passed=True, raw_bank_values_exact=True, raw_draft_probabilities_exact=True,
                  input_hashes_verified=len(source_hashes), direct_cluster_bootstrap=verified,
                  history_only_sample_verified=True, formal_gpu_results_pending=True)
    (H / 'cpu_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
