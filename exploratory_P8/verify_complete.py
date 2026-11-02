"""Independent validation of P8 phase C coverage, provenance, and IPW S5 intervals."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

H = Path(__file__).resolve().parent
S = Path('/root/autodl-tmp/idea')


def read(p):
    return json.loads(Path(p).read_text())


def main():
    assert read(H / 'cpu_verification.json')['passed']
    result = read(H / 'p8_complete.json'); phase = result['phase_c']
    prep = read(H / 'phase_c_preparation.json')
    raw = gzip.decompress((H / 'data/judge/requests.jsonl.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == prep['requests_sha256']
    jobs = {r['request_id']: r for r in map(json.loads, raw.splitlines())}
    sample = read(H / 'blind_sample.json'); sample_map = {r['point_id']: r for r in sample['points']}
    assert len(jobs) == 4*len(sample_map)
    rows = read(H / 'data/points.json'); qi = np.load(H / 'data/arrays.npz')['q']
    index = {r['point_id']: i for i, r in enumerate(rows)}
    gains, cost = {}, {}
    for name in ('7b', '32b'):
        target = H / 'data/judge' / name
        run = read(target / 'run.json'); original = read(S / f'exploratory_P4/e2_{name}/run.json')
        assert run['complete'] and run['contexts'] == len(jobs)
        assert run['revision'] == original['revision'] and run['model_sha256'] == original['model_sha256']
        assert run['backend_sha256'] == original['scorer_sha256'] and run['requests_sha256'] == prep['requests_sha256']
        checks = read(target / 'numerical_check.json')
        assert len(checks) == 9 and all(c['passed'] and c['tv'] < .005 for c in checks)
        saved = [json.loads(l) for l in (target / 'scores.jsonl').read_text().splitlines()]
        assert len(saved) == len(jobs) and len({r['request_id'] for r in saved}) == len(jobs)
        old = {}
        for prefix in ('e2', 'e2m'):
            for r in map(json.loads, (S / f'exploratory_P4/{prefix}_{name}/scores.jsonl').read_text().splitlines()):
                old[r['request_id']] = r
        reused = 0
        for r in saved:
            j = jobs[r['request_id']]
            assert r['request_sha256'] == j['request_sha256']
            copy = dict(j); copy.pop('request_sha256')
            assert hashlib.sha256(json.dumps(copy, sort_keys=True, ensure_ascii=False).encode()).hexdigest() == j['request_sha256']
            i = index[r['point_id']]
            assert j['actions'] == rows[i]['actions'] == sample_map[r['point_id']]['actions']
            lp = np.array(r['sequence_logp']); prob = np.exp(lp-lp.max()); prob /= prob.sum()
            assert len(prob) == 5 and np.isfinite(prob).all() and np.max(abs(prob-r['candidate_prob'])) < 1e-12
            if r['p8_reused']:
                reused += 1
                assert r['candidate_prob'] == old[r['request_id']]['candidate_prob']
                assert r['request_sha256'] == old[r['request_id']]['request_sha256']
            gains[name, r['condition'], r['point_id']] = float(sum((prob-.2)*qi[i]))
        assert reused == prep['reuse'][name]['contexts']
        cost[name] = dict(reused=reused, computed=len(jobs)-reused, max_tv=max(c['tv'] for c in checks))
    tids = sorted({r['traj_id'] for r in rows}); ti = {t: i for i, t in enumerate(tids)}
    W = np.random.default_rng(20261101).multinomial(len(tids), np.full(len(tids), 1/len(tids)), size=10000)
    checked = {}
    for name in ('32b/F - 32b/M', '32b/F - R1_full'):
        values = np.zeros(len(tids)); den = np.zeros(len(tids))
        for s in sample['points']:
            pid = s['point_id']; i = index[pid]; t = ti[s['traj_id']]
            baseline = gains['32b', 'M', pid] if name.endswith('32b/M') else sum((np.array(rows[i]['R1_full'])-.2)*qi[i])
            w = s['inverse_probability']
            assert w == 1/s['inclusion_probability'] or abs(w-1/s['inclusion_probability']) < 1e-12
            values[t] += w*(gains['32b', 'F', pid]-baseline); den[t] += w
        dist = np.array([sum(w*values)/sum(w*den) for w in W])
        estimate = float(values.sum()/den.sum()); ci = np.percentile(dist, [2.5, 97.5])
        expected = phase['contrasts'][name]['H_prompt']
        error = max(abs(estimate-expected['estimate']), float(max(abs(ci-expected['ci95']))))
        assert error < 1e-12
        checked[name] = dict(estimate=estimate, ci95=ci.tolist(), max_error=error)
    assert phase['S5'] == all(c['ci95'][0] > 0 for c in checked.values()) == result['S5']
    for path, sha in read(H / 'run/protected_inputs.json').items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha, path
    verification = dict(passed=True, models=cost, independent_ipw=checked, all_contexts_complete=True,
                        frozen_inputs_unchanged=True, original_labels_and_reports_unchanged=True)
    (H / 'complete_verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    manifest = {str(p.relative_to(H)): hashlib.sha256(p.read_bytes()).hexdigest() for p in H.rglob('*')
                if p.is_file() and p.suffix in ('.json', '.npz', '.gz', '.txt', '.py', '.md') and 'run' not in p.parts and '__pycache__' not in p.parts}
    (H / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
