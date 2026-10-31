"""E2-M analysis (exploratory_P4/PREREG_E2M.md). Reads complete M/MF scores and the existing E2 S/F scores.

Requires both models' e2m runs complete with passing numerical checks. Action classes are the frozen definition in
post_e2/post_e2_mechanism.py (arm_classes). Bootstrap = E2's: evalcore.Boot over all Live trajectories, B=10,000,
seed 30102026. Writes exploratory_P4/e2m.json.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
import argparse, gzip, hashlib, json, subprocess, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(OUT / 'post_e2'))
from post_e2_mechanism import arm_classes, load_parquet  # noqa: E402
MODELS, LAYERS = ['7b', '32b'], ['I', 'U']


def main(scores_dir, arms_path):
    from evalcore import Boot
    inputs = {}

    def track(p):
        b = Path(p).read_bytes(); inputs[str(p)] = hashlib.sha256(b).hexdigest(); return b

    e2 = json.loads(track(OUT / 'e2.json'))
    track(arms_path)
    assert inputs[str(arms_path)] == e2['inputs']['exploratory_deep_discovery/exploratory_arms.parquet']
    prep = json.loads(track(OUT / 'e2m_preparation.json'))
    body = gzip.decompress(track(OUT / 'e2m_requests.jsonl.gz'))
    assert hashlib.sha256(body).hexdigest() == prep['requests_sha256']
    requests = {r['request_id']: r for r in map(json.loads, body.decode().splitlines())}
    for m in MODELS:
        run = json.loads(track(scores_dir / f'e2m_{m}/run.json'))
        if not run.get('complete'):
            raise SystemExit(f'E2-M incomplete: {m}; no reading issued.')
        assert run['requests_sha256'] == prep['requests_sha256']
        assert all(c['passed'] for c in json.loads(track(scores_dir / f'e2m_{m}/numerical_check.json')))

    track(OUT / 'strata.parquet')
    s = load_parquet(OUT / 'strata.parquet')
    a = load_parquet(arms_path)
    key = lambda t, u: f'{t}_t{int(u):02d}'
    live = [i for i in range(len(s['traj_id'])) if s['live'][i]]
    boot = Boot([s['traj_id'][i] for i in live], B=10000, seed=30102026)
    pts = sorted((s['traj_id'][i], int(s['turn'][i]), s['stratum'][i], bool(s['het'][i]))
                 for i in live if s['stratum'][i] in LAYERS)
    assert len(pts) == 873
    pid = [key(t, u) for t, u, _, _ in pts]
    idx = {p: n for n, p in enumerate(pid)}
    boot.inv = np.searchsorted(boot.t, [t for t, _, _, _ in pts])
    layer = np.array([z for _, _, z, _ in pts]); het = np.array([h for _, _, _, h in pts])

    Q = {src: np.full((873, 5), np.nan) for src in ['12', 'gold']}
    actions = [[None] * 5 for _ in range(873)]
    for i in range(len(a['traj_id'])):
        k = key(a['traj_id'][i], a['turn'][i])
        if k in idx:
            n, j = idx[k], int(a['j'][i])
            Q['12'][n, j] = a['q_12'][i]
            Q['gold'][n, j] = np.nan if a['q_gold'][i] is None else a['q_gold'][i]
            actions[n][j] = a['action'][i]

    e2raw = gzip.decompress(track(OUT / 'e2_requests.jsonl.gz'))
    episodes, dec = {}, json.JSONDecoder()
    for line in e2raw.decode().splitlines():
        r = json.loads(line)
        if r['condition'] == 'L' and r['traj_id'] not in episodes:
            episodes[r['traj_id']] = dec.raw_decode(r['text'][r['text'].find('{"task":'):])[0]
    cls = []
    for n, (t, u, _, _) in enumerate(pts):
        steps = episodes[t]['steps']
        assert steps[u - 1]['executed_action'] == actions[n][0]
        cls.append(arm_classes(actions[n], [(st['executed_action'], st['observation_after']) for st in steps[:u - 1]]))
    cls = np.array(cls)

    P = {}
    for m in MODELS:
        for name, path in [('e2', OUT / f'e2_{m}/scores.jsonl'), ('e2m', scores_dir / f'e2m_{m}/scores.jsonl')]:
            for line in track(path).decode().splitlines():
                r = json.loads(line)
                if r['condition'] in ('S', 'F', 'M', 'MF'):
                    if name == 'e2m':
                        assert r['request_sha256'] == requests[r['request_id']]['request_sha256']
                        assert requests[r['request_id']]['actions'] == actions[idx[r['point_id']]]
                    P.setdefault((m, r['condition']), np.full((873, 5), np.nan))[idx[r['point_id']]] = r['candidate_prob']
    for v in P.values():
        assert np.isfinite(v).all() and np.allclose(v.sum(1), 1)

    def stat(v, mask):
        mask = np.asarray(mask) & np.isfinite(v)
        if not mask.any():
            return dict(n=0, estimate=None, ci95=None), None
        sums = np.bincount(boot.inv[mask], weights=v[mask], minlength=len(boot.t))
        cnt = np.bincount(boot.inv[mask], minlength=len(boot.t))
        with np.errstate(divide='ignore', invalid='ignore'):
            dist = (boot.W @ sums) / (boot.W @ cnt)
        return dict(n=int(mask.sum()), estimate=float(v[mask].mean()), ci95=Boot.ci(dist)), dist

    def gu(m, c, src):
        q = Q[src]; return (P[m, c] * q).sum(1) - q.mean(1)

    def mass(m, c, k):
        return (P[m, c] * (cls == k)).sum(1)

    table, contrasts = {}, {}
    for src in ['12', 'gold']:
        for L in LAYERS:
            mk = (layer == L) & het
            for m in MODELS:
                for c in ['S', 'F', 'M', 'MF']:
                    table[f'{src}/{m}/{c}/G_u/{L}'] = stat(gu(m, c, src), mk)[0]
                    if src == '12':
                        for k in ['E_new', 'R_seen']:
                            table[f'{m}/{c}/mass_{k}/{L}'] = stat(mass(m, c, k), mk)[0]
                for lhs, rhs in [('M', 'S'), ('F', 'M'), ('MF', 'F'), ('MF', 'M')]:
                    contrasts[f'{src}/{m}/{lhs}-{rhs}/G_u/{L}'] = stat(gu(m, lhs, src) - gu(m, rhs, src), mk)[0]
                if src == '12':
                    for k in ['E_new', 'R_seen']:
                        contrasts[f'{m}/M-S/mass_{k}/{L}'] = stat(mass(m, 'M', k) - mass(m, 'S', k), mk)[0]
            inter = (gu('32b', 'M', src) - gu('7b', 'M', src)) - (gu('32b', 'S', src) - gu('7b', 'S', src))
            contrasts[f'{src}/interaction_(32b-7b)_M_minus_S/G_u/{L}'] = stat(inter, mk)[0]
            contrasts[f'{src}/32b-7b/M/G_u/{L}'] = stat(gu('32b', 'M', src) - gu('7b', 'M', src), mk)[0]

    # consistency gate with E2
    for m in MODELS:
        for c in ['S', 'F']:
            for L in LAYERS:
                d = abs(table[f'12/{m}/{c}/G_u/{L}']['estimate'] - e2['table']['12'][f'{m}/{c}/soft/G_u/{L}']['estimate'])
                assert d < 1e-9, (m, c, L, d)

    lo = lambda x: x['ci95'] is not None and x['ci95'][0] > 0
    hi = lambda x: x['ci95'] is not None and x['ci95'][1] < 0
    C = contrasts
    m1 = hi(C['32b/M-S/mass_R_seen/I']) and all(lo(C[f'{m}/M-S/mass_E_new/I']) for m in MODELS)
    m2 = {m: lo(C[f'12/{m}/M-S/G_u/I']) for m in MODELS}
    m3 = {m: lo(C[f'12/{m}/F-M/G_u/I']) for m in MODELS}
    m4 = lo(C['12/interaction_(32b-7b)_M_minus_S/G_u/I'])
    if not any(m2.values()):
        reading = 'R-M-NULL'
    elif m1 and all(m2.values()) and all(m3.values()):
        reading = 'R-M-TWO-GAPS'
    else:
        reading = 'R-M-PARTIAL'
    for p in [Path(__file__), OUT / 'post_e2/post_e2_mechanism.py', ROOT / 'scripts/evalcore.py']:
        track(p)
    out = dict(table=table, contrasts=contrasts,
               predictions={'M-1': m1, 'M-2': m2, 'M-3': m3, 'M-4': m4}, reading=reading,
               prereg='exploratory_P4/PREREG_E2M.md', inputs=inputs,
               commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    (OUT / 'e2m.json').write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(predictions=out['predictions'], reading=reading), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores-dir', type=Path, default=OUT)
    ap.add_argument('--arms', type=Path, default=ROOT / 'exploratory_deep_discovery/exploratory_arms.parquet')
    args = ap.parse_args()
    main(args.scores_dir, args.arms)
