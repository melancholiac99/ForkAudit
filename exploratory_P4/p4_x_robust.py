"""P4-X robustness checks for the I-layer policy-preference result (exploratory_P4/PREREG_X.md).

Run `--selftest` first (synthetic data, no repository inputs). The main run reads the same inputs as E1
plus bank-1 drafts and factual rollouts, recomputes E1's policy G_u as a consistency gate, and writes x.json.
No fits, no new continuations, no change to any P4 prediction or verdict.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import argparse, hashlib, json, re, subprocess, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
KEY = ['traj_id', 'turn']
SEED = 30102026
NBOOT = 10000
LAYERS = ['I', 'U', 'N']
SOURCES = ['12', 'gold']

_GOTO = re.compile(r'^go to (.+)$')
_OPEN = re.compile(r'^open (.+)$')


# ----------------------------------------------------------------------------- pure helpers (self-tested)

def history_sets(actions_before):
    """Receptacles reached by `go to` and opened by `open` in the factual actions strictly before t."""
    visited, opened = set(), set()
    for a in actions_before:
        a = a.strip().lower()
        m = _GOTO.match(a)
        if m:
            visited.add(m.group(1).strip())
        m = _OPEN.match(a)
        if m:
            opened.add(m.group(1).strip())
    return visited, opened


def act_class(action, visited, opened):
    """'E' (exploratory: go to an unvisited receptacle, or open an unopened one) or 'N' (everything else)."""
    a = action.strip().lower()
    m = _GOTO.match(a)
    if m:
        return 'E' if m.group(1).strip() not in visited else 'N'
    m = _OPEN.match(a)
    if m:
        return 'E' if m.group(1).strip() not in opened else 'N'
    return 'N'


def smoothed(counts, alpha=0.5):
    c = np.asarray(counts, float) + alpha
    return c / c.sum()


def gains(p, q):
    """Per-point G_u = J - mean(q) and G_c = J - q[chosen] for selection distributions p over the five arms."""
    j = (p * q).sum(1)
    return j - q.mean(1), j - q[:, 0]


def class_stats(q, cls, p_e):
    """-> (d_EX, regret, v_E, v_N) for one point; NaNs unless both classes are present among the arms."""
    e = [i for i, c in enumerate(cls) if c == 'E']
    n = [i for i, c in enumerate(cls) if c == 'N']
    if not e or not n or not np.isfinite(q).all():
        return (np.nan,) * 4
    v_e, v_n = float(q[e].mean()), float(q[n].mean())
    regret = max(v_e, v_n) - (p_e * v_e + (1 - p_e) * v_n)
    return v_e - v_n, regret, v_e, v_n


def within_gain(q, cls, counts, alpha=0.5):
    """Within-E selection gain with smoothed draft counts renormalised over E arms; NaN if fewer than two E arms."""
    e = [i for i, c in enumerate(cls) if c == 'E']
    if len(e) < 2 or not np.isfinite(q).all():
        return np.nan
    p = smoothed([counts[i] for i in e], alpha)
    return float((p * q[e]).sum() - q[e].mean())


def stat(x, mask, boot, traj_ids):
    """Point mean with trajectory-cluster bootstrap (same estimator as E1's `stat` with den=None)."""
    x = np.asarray(x, float)
    mask = np.asarray(mask, bool) & np.isfinite(x)
    if not mask.any():
        return dict(n=0, n_traj=0, estimate=None, ci95=None), np.full(boot.W.shape[0], np.nan)
    v = np.bincount(boot.inv[mask], weights=x[mask], minlength=len(boot.t))
    c = np.bincount(boot.inv[mask], minlength=len(boot.t)).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dist = (boot.W @ v) / (boot.W @ c)
    ok = dist[np.isfinite(dist)]
    return dict(n=int(mask.sum()), n_traj=int(len(set(np.asarray(traj_ids)[mask]))), estimate=float(x[mask].mean()),
                ci95=[float(np.percentile(ok, 2.5)), float(np.percentile(ok, 97.5))],
                invalid_bootstrap=int((~np.isfinite(dist)).sum())), dist


def loto(x, mask, traj_ids):
    """Min and max of the point-mean estimate leaving one trajectory out."""
    x = np.asarray(x, float)
    mask = np.asarray(mask, bool) & np.isfinite(x)
    t = np.asarray(traj_ids)
    vals = [float(x[mask & (t != g)].mean()) for g in sorted(set(t[mask])) if (mask & (t != g)).any()]
    return dict(min=min(vals), max=max(vals), n_traj=len(vals)) if vals else None


def reading(px1, px2, px3, px4, within_hi):
    if not px1 and not px2:
        return 'R-X-ARTIFACT'
    if within_hi is not None and within_hi < 0:
        return 'R-X-MISDIRECT'
    if px3 and px4:
        return 'R-X-UNDEREXPLORE'
    if px4 and not px3:
        return 'R-X-DILUTION'
    return 'R-X-OTHER'


# ----------------------------------------------------------------------------- self-test

class _Boot:
    def __init__(self, groups, B, seed):
        self.t, self.inv = np.unique(np.asarray(groups), return_inverse=True)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(len(self.t), np.full(len(self.t), 1 / len(self.t)), size=B).astype(np.float32)


def selftest():
    # classification
    v, o = history_sets(['go to cabinet 1', 'open cabinet 1', 'take ladle 1 from cabinet 1', 'go to sinkbasin 1'])
    assert v == {'cabinet 1', 'sinkbasin 1'} and o == {'cabinet 1'}
    assert act_class('go to cabinet 2', v, o) == 'E' and act_class('go to cabinet 1', v, o) == 'N'
    assert act_class('open cabinet 1', v, o) == 'N' and act_class('open drawer 3', v, o) == 'E'
    assert act_class('examine cabinet 2', v, o) == 'N' and act_class('inventory', v, o) == 'N'
    # gains: one-hot chosen gives q0 - mean; uniform gives zero
    q = np.array([[.2, .8, .2, .2, .2], [1., 0, 0, 0, 0]])
    gu, gc = gains(np.eye(5)[[0, 0]], q)
    assert np.allclose(gu, q[:, 0] - q.mean(1)) and np.allclose(gc, 0)
    gu, _ = gains(np.full((2, 5), .2), q)
    assert np.allclose(gu, 0)
    # class stats and regret
    d, r, ve, vn = class_stats(np.array([.2, .8, .6, .2, .2]), ['N', 'E', 'E', 'N', 'N'], 0.25)
    assert np.isclose(d, .7 - .2) and np.isclose(r, .7 - (.25 * .7 + .75 * .2))
    assert all(np.isnan(class_stats(np.ones(5), ['N'] * 5, .5)))
    # within-E gain: equal counts -> zero; counts on the better E arm -> positive
    assert np.isclose(within_gain(np.array([.2, .8, .4, .2, .2]), ['N', 'E', 'E', 'N', 'N'], [5, 3, 3, 0, 0]), 0)
    assert within_gain(np.array([.2, .8, .4, .2, .2]), ['N', 'E', 'E', 'N', 'N'], [5, 30, 0, 0, 0]) > 0
    assert np.isnan(within_gain(np.ones(5), ['N', 'E', 'N', 'N', 'N'], [1] * 5))
    # dilution scenario: blind within E, policy puts half its mass on 'explore' spread over many options.
    # Expect negative G_u (set over-represents rare E arms) but within-E gain ~ 0.
    rng = np.random.default_rng(0)
    n, traj = 4000, np.repeat(np.arange(80), 50)
    P, Q, W = np.zeros((n, 5)), np.zeros((n, 5)), np.zeros(n)
    cls = ['N', 'E', 'E', 'E', 'E']
    for i in range(n):
        p_e_each = 0.5 / 10                                   # ten unvisited options, policy blind among them
        pol = np.array([0.5, p_e_each, p_e_each, p_e_each, p_e_each]); P[i] = pol / pol.sum()
        good = rng.integers(1, 5)                              # one of the four E arms reveals the target
        Q[i] = np.where(np.arange(5) == good, .8, .2)
        cnt = rng.multinomial(400, np.r_[0.5, [p_e_each] * 4, 0.5 - 4 * p_e_each])[:5]
        W[i] = within_gain(Q[i], cls, cnt)
    gu, _ = gains(P, Q)
    boot = _Boot(traj, 2000, 1)
    res_gu, _ = stat(gu, np.ones(n, bool), boot, traj)
    res_w, _ = stat(W, np.ones(n, bool), boot, traj)
    assert res_gu['ci95'][1] < 0, res_gu                       # dilution alone produces negative G_u
    assert res_w['ci95'][0] <= 0 <= res_w['ci95'][1], res_w    # ...but no within-E misdirection
    assert reading(True, True, False, True, res_w['ci95'][1]) == 'R-X-DILUTION'
    assert reading(False, False, True, True, None) == 'R-X-ARTIFACT'
    assert reading(True, False, True, True, 0.01) == 'R-X-UNDEREXPLORE'
    assert reading(True, False, True, False, -0.01) == 'R-X-MISDIRECT'
    lo = loto(np.r_[np.ones(10), -np.ones(10)], np.ones(20, bool), np.repeat([0, 1], 10))
    assert lo == dict(min=-1.0, max=1.0, n_traj=2)
    print('selftest passed')


# ----------------------------------------------------------------------------- main run

def main():
    import pandas as pd
    from scipy.special import softmax
    sys.path.insert(0, str(ROOT / 'scripts'))
    from evalcore import Boot
    start = time.time()
    inputs = {}

    def read(path):
        b = (ROOT / path).read_bytes()
        inputs[path] = hashlib.sha256(b).hexdigest()
        return b

    def pq(path):
        read(path)
        return pd.read_parquet(ROOT / path)

    assert json.loads(read('exploratory_P4/strata_audit.json'))['passed']
    e1 = json.loads(read('exploratory_P4/e1.json'))
    read('exploratory_P4/PREREG_X.md')
    strata = pq('exploratory_P4/strata.parquet')
    labels = pq('data/labels_final.parquet')
    labels = labels[labels.live]
    d = labels.merge(strata[KEY + ['stratum']], on=KEY, validate='one_to_one').reset_index(drop=True)
    assert len(d) == 1478
    boot = Boot(d.traj_id, B=NBOOT, seed=SEED)
    scored = pq('data/scores_fp32.parquet').set_index(KEY + ['action'])
    w1 = pq('data/features_W1.parquet').set_index(KEY)

    n = len(d)
    Q = {k: np.full((n, 5), np.nan) for k in SOURCES}
    pol_sum, pol_mean, pol_d0, pol_dall = (np.zeros((n, 5)) for _ in range(4))
    p_e = np.full(n, np.nan)
    within = np.full(n, np.nan)
    cstats = np.full((n, 4), np.nan)
    loop = np.zeros(n)
    task_type = np.empty(n, dtype=object)
    reveal_e = np.full(n, np.nan)
    rollouts = {}
    for i, z in enumerate(d.to_dict('records')):
        tid, t = z['traj_id'], int(z['turn'])
        acts, banks, b1 = None, {}, None
        for b in range(1, 8):
            folder = ('replay' if b == 1 else f'replay_b{b-1}') if b < 7 else 'replay_gold'
            path = f'data/{folder}/{tid}/t{t:02d}.json'
            if b == 7 and not (ROOT / path).exists():
                continue
            r = json.loads(read(path))
            aa = [r['factual_action']] + r['alternatives']
            if acts is None:
                acts = aa
            assert aa == acts and len(set(aa)) == 5
            oo = [r['factual_outcomes']] + r['alt_outcomes']
            assert all(len(v) == (20 if b == 7 else 3) for v in oo)
            banks[b] = np.asarray(oo, float)
            if b == 1:
                b1 = r
        Q['12'][i] = np.concatenate([banks[1], banks[2]], axis=1).mean(1)
        if 7 in banks:
            Q['gold'][i] = banks[7].mean(1)
        rows = [scored.loc[(tid, t, a)] for a in acts]
        pol_sum[i] = softmax([float(r_['logp_sum']) for r_ in rows])
        pol_mean[i] = softmax([float(r_['logp_mean']) for r_ in rows])
        drafts = [x for x in b1['drafts'] if x['valid'] and x['admissible']]
        first = [x for x in drafts if x['batch'] == 0]
        c0 = [sum(x['action'] == a for x in first) for a in acts]
        call = [sum(x['action'] == a for x in drafts) for a in acts]
        pol_d0[i], pol_dall[i] = smoothed(c0), smoothed(call)
        if tid not in rollouts:
            rollouts[tid] = json.loads(read(f'data/rollout/{tid}.json'))
        tr = rollouts[tid]
        st = tr['steps'][t - 1]
        assert int(st['turn']) == t and st['action'] == acts[0]
        before = [s['action'] for s in tr['steps'][:t - 1]]
        visited, opened = history_sets(before)
        cls = [act_class(a, visited, opened) for a in acts]
        if first:
            p_e[i] = float(np.mean([act_class(x['action'], visited, opened) == 'E' for x in first]))
        cstats[i] = class_stats(Q['12'][i], cls, p_e[i])          # regret is NaN when P_E is undefined
        within[i] = within_gain(Q['12'][i], cls, call)
        loop[i] = float(acts[0] in before[-5:])
        task_type[i] = tr['task_type']
        pk = json.loads(read(f'data/peek_alts/{tid}/t{t:02d}.json'))
        peeks = sorted(pk['peeks'], key=lambda p: (p['role'] != 'factual', p['alt_index'] or 0))
        assert [p['action'] for p in peeks] == acts
        rx = re.compile(rf"\b{re.escape(w1.loc[(tid, t), 'target_word'])} \d+\b")
        hits = [bool(rx.search(p['obs'].lower())) for p in peeks]
        if any(hits):
            reveal_e[i] = float(np.mean([c == 'E' for c, h in zip(cls, hits) if h]))
        if i % 300 == 0:
            print('points', i, flush=True)

    tids = d.traj_id.to_numpy()
    het = d.het.to_numpy(bool)
    layer_mask = {L: d.stratum.eq(L).to_numpy() for L in LAYERS}

    # consistency gate: E1 policy G_u recomputed from logp_sum
    gu_sum, _ = gains(pol_sum, Q['12'])
    chk = stat(gu_sum, layer_mask['I'] & het, boot, tids)[0]['estimate']
    ref = e1['action']['I']['12']['policy']['G_u']['estimate']
    if abs(chk - ref) > 1e-9:
        raise SystemExit(f'consistency gate failed: {chk} vs e1 {ref}')

    out, dists = {}, {}
    for L in LAYERS:
        fork = layer_mask[L] & het
        out[L] = {}
        for src in SOURCES:
            q = Q[src]
            sel = {'X1_logp_mean': pol_mean, 'X2_drafts_batch0': pol_d0, 'X2b_drafts_all': pol_dall,
                   'E1_logp_sum': pol_sum, 'X3_chosen': np.eye(5)[np.zeros(n, int)]}
            out[L][src] = {}
            for name, p in sel.items():
                gu, gc = gains(p, q)
                r_u, dist_u = stat(gu, fork, boot, tids)
                r_c, _ = stat(gc, fork, boot, tids)
                out[L][src][name] = {'G_u': r_u, 'G_c': r_c}
                dists[L, src, name] = dist_u
        out[L]['X4'] = {
            'd_EX': stat(cstats[:, 0], fork, boot, tids)[0],
            'regret': stat(cstats[:, 1], fork, boot, tids)[0],
            'v_E': stat(cstats[:, 2], fork, boot, tids)[0],
            'v_N': stat(cstats[:, 3], fork, boot, tids)[0],
            'P_E': stat(p_e, fork, boot, tids)[0],
            'within_E_gain': stat(within, fork, boot, tids)[0],
            'revealing_arms_E_share': stat(reveal_e, fork, boot, tids)[0]}
        out[L]['X5'] = {'loop_rate': stat(loop, fork, boot, tids)[0]}
        for k, m in [('loop', loop == 1), ('no_loop', loop == 0)]:
            out[L]['X5']['X3_' + k] = stat(Q['12'][:, 0] - Q['12'].mean(1), fork & m, boot, tids)[0]

    # X6: first Live I point per trajectory, then its Fork points
    first_i = np.zeros(n, bool)
    di = d[layer_mask['I']].sort_values(KEY)
    first_i[di.groupby('traj_id').head(1).index.to_numpy()] = True
    x3 = Q['12'][:, 0] - Q['12'].mean(1)
    gu_d0, _ = gains(pol_d0, Q['12'])
    out['X6'] = {'n_first_I_points': int(first_i.sum()),
                 'X2_drafts_batch0': stat(gu_d0, first_i & het, boot, tids)[0],
                 'X3_chosen': stat(x3, first_i & het, boot, tids)[0]}
    # X7: leave-one-trajectory-out and task types on I Fork
    fI = layer_mask['I'] & het
    out['X7'] = {'loto_X2': loto(gu_d0, fI, tids), 'loto_X3': loto(x3, fI, tids), 'by_task_type': {}}
    for tt in sorted(set(task_type[fI])):
        m = fI & (task_type == tt)
        out['X7']['by_task_type'][tt] = {'n': int(m.sum()), 'X2': float(np.nanmean(gu_d0[m])), 'X3': float(np.nanmean(x3[m]))}

    I12 = out['I']['12']
    px1 = I12['X2_drafts_batch0']['G_u']['ci95'][1] < 0
    px2 = I12['X3_chosen']['G_u']['ci95'][1] < 0
    x4 = out['I']['X4']
    px3 = x4['d_EX']['ci95'] is not None and x4['d_EX']['ci95'][0] > 0 and x4['P_E']['estimate'] < 0.5
    wci = x4['within_E_gain']['ci95']
    px4 = wci is not None and wci[0] <= 0 <= wci[1]
    px5 = out['X6']['X3_chosen']['estimate'] is not None and out['X6']['X3_chosen']['estimate'] < 0
    pred = {'PX-1': bool(px1), 'PX-2': bool(px2), 'PX-3': bool(px3), 'PX-4': bool(px4), 'PX-5_direction': bool(px5)}
    result = dict(script='exploratory_P4/p4_x_robust.py',
                  commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  prereg='exploratory_P4/PREREG_X.md', seed=SEED, bootstrap=NBOOT,
                  consistency_gate=dict(recomputed=chk, e1=ref, passed=True),
                  results=out, predictions=pred,
                  reading=reading(px1, px2, px3, px4, wci[1] if wci else None),
                  seconds=time.time() - start, inputs=inputs)
    for path in ['exploratory_P4/p4_x_robust.py', 'scripts/evalcore.py', 'scripts/common.py']:
        read(path)
    (OUT / 'x.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(predictions=pred, reading=result['reading']), indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    selftest() if args.selftest else main()
