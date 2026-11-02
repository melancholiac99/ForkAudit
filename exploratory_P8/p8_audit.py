"""Frozen P8 CPU audit and blind sample. Reads source data; writes only exploratory_P8."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import time
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = Path('/root/autodl-tmp/idea')
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'exploratory_P4')]
import common
import p4_x_robust as X

SEED, B = 20261101, 10000
KEY = ['traj_id', 'turn']


def sha(b):
    return hashlib.sha256(b).hexdigest()


def commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


class Reader:
    def __init__(self):
        self.hashes = {}

    def raw(self, p):
        p = Path(p)
        full = p if p.is_absolute() else SOURCE / p
        b = full.read_bytes()
        self.hashes[str(full)] = sha(b)
        return b

    def js(self, p):
        return json.loads(self.raw(p))

    def pq(self, p, **kwargs):
        import io
        return pd.read_parquet(io.BytesIO(self.raw(p)), **kwargs)


def rule(actions, before):
    visited, opened = X.history_sets(before)
    new = np.array([X.act_class(a, visited, opened) == 'E' for a in actions])
    return (new / new.sum() if new.any() else np.full(5, .2)), bool(new.any()), new


def descriptors(reader):
    """Population selection and rules never read Fork/Live flags or outcome values."""
    keys = reader.pq('data/labels_final.parquet', columns=KEY).sort_values(KEY)
    target = reader.pq('data/features_W1.parquet', columns=KEY + ['target_word']).set_index(KEY)
    assert not keys.duplicated(KEY).any()
    rows, trs = [], {}
    for i, (tid, t) in enumerate(keys.itertuples(index=False, name=None)):
        t = int(t)
        if tid not in trs:
            trs[tid] = reader.js(f'data/rollout/{tid}.json')
        tr = trs[tid]
        st = tr['steps'][t - 1]
        assert st['turn'] == t
        prompt = reader.raw(st['prompt_path']).decode()
        before = tr['steps'][:t - 1]
        memory = [dict(text_obs=s['obs_before'], action=s['action']) for s in before]
        assert common.build_prompt(tr['task'], memory, st['obs_before'], st['admissible']) == prompt, (tid, t, 'prompt mismatch')
        b1 = reader.js(f'data/replay/{tid}/t{t:02d}.json')
        b2 = reader.js(f'data/replay_b1/{tid}/t{t:02d}.json')
        actions = [b1['factual_action']] + b1['alternatives']
        assert b1['cf_defined'] and len(actions) == len(set(actions)) == 5
        assert actions == [b2['factual_action']] + b2['alternatives'] and actions[0] == st['action']
        for bank in (b1, b2):
            assert len(bank['factual_outcomes']) == 3 and len(bank['alt_outcomes']) == 4
            assert all(len(v) == 3 for v in bank['alt_outcomes'])
        target_word = target.loc[(tid, t), 'target_word']
        assert isinstance(target_word, str) and target_word
        rx = re.compile(rf'\b{re.escape(target_word)} \d+\b')
        prompt_target = bool(rx.search(prompt.lower()))
        history_text = '\n'.join(s['obs_before'] + '\n' + '\n'.join(s['admissible']) for s in tr['steps'][:t])
        history_target = bool(rx.search(history_text.lower()))
        pwin, win_cover, win_new = rule(actions, [s['action'] for s in before[-2:]])
        pfull, full_cover, full_new = rule(actions, [s['action'] for s in before])
        first = [d for d in b1['drafts'] if d['batch'] == 0]
        assert len(first) == 32, (tid, t, len(first))
        valid = [d for d in first if d['valid'] and d['admissible']]
        all_valid = [d for d in b1['drafts'] if d['valid'] and d['admissible']]
        c0 = [sum(d['action'] == a for d in valid) for a in actions]
        ca = [sum(d['action'] == a for d in all_valid) for a in actions]
        rows.append(dict(traj_id=tid, turn=t, point_id=f'{tid}_t{t:02d}', task_type=tr['task_type'],
                         actions=actions, H_prompt=not prompt_target, H_full=not prompt_target and not history_target,
                         prompt_target=prompt_target, history_target=history_target,
                         R1_window=pwin.tolist(), R1_full=pfull.tolist(), window_coverage=win_cover,
                         full_coverage=full_cover, window_new=win_new.tolist(), full_new=full_new.tolist(),
                         policy=X.smoothed(c0).tolist(), policy_all=X.smoothed(ca).tolist(),
                         first_drafts=len(first), valid_first=len(valid), total_drafts=len(b1['drafts']),
                         valid_all=len(all_valid), prompt_sha256=sha(prompt.encode())))
        if i % 500 == 0:
            print('history descriptors', i, flush=True)
    return rows, trs


def freeze_sample():
    assert not (HERE / 'blind_sample.json').exists(), 'Sample already frozen; do not overwrite'
    rd = Reader()
    rows, _ = descriptors(rd)
    chosen, populations = [], []
    for tid in sorted({r['traj_id'] for r in rows}):
        eligible = [r for r in rows if r['traj_id'] == tid and r['H_prompt']]
        if not eligible:
            continue
        order = sorted(eligible, key=lambda r: sha(f"P8-v1|{tid}|{r['turn']}".encode()))
        n = len(eligible)
        populations.append(dict(traj_id=tid, N_H=n, sampled=min(n, 8)))
        for r in order[:8]:
            chosen.append(dict(traj_id=tid, turn=r['turn'], point_id=r['point_id'], inclusion_probability=min(n, 8)/n,
                               inverse_probability=n/min(n, 8), hash=sha(f"P8-v1|{tid}|{r['turn']}".encode()),
                               prompt_sha256=r['prompt_sha256'], actions=r['actions']))
    save(HERE / 'blind_sample.json', dict(algorithm='SHA256(P8-v1|traj_id|turn), max 8 per H_prompt trajectory',
         selection_reads_no_outcomes_or_labels=True, source=str(SOURCE), commit=commit(),
         population=sum(p['N_H'] for p in populations), n=len(chosen), trajectories=populations, points=chosen))
    save(HERE / 'blind_sample_inputs.json', rd.hashes)
    print('Blind sample frozen:', len(chosen), 'of', sum(p['N_H'] for p in populations), flush=True)


class Bootstrap:
    def __init__(self, tids, B=B, seed=SEED):
        self.t, self.inv = np.unique(tids, return_inverse=True)
        self.W = np.random.default_rng(seed).multinomial(len(self.t), np.full(len(self.t), 1/len(self.t)), size=B).astype(float)

    def stat(self, values, mask, weights=None, equal_traj=False):
        values, mask = np.asarray(values, float), np.asarray(mask, bool)
        finite = np.isfinite(values)
        use = mask & finite
        weights = np.ones(len(values)) if weights is None else np.asarray(weights, float)
        assert np.isfinite(weights).all() and (weights >= 0).all()
        info = dict(n=int(use.sum()), n_traj=int(len(np.unique(self.inv[use]))), eligible=int(mask.sum()),
                    missing=int((mask & ~finite).sum()), missing_rate=float((mask & ~finite).sum()/max(mask.sum(), 1)))
        if not use.any():
            return dict(**info, estimate=None, ci95=None, invalid_bootstrap=len(self.W)), np.full(len(self.W), np.nan)
        c = np.bincount(self.inv[use], weights=weights[use], minlength=len(self.t))
        v = np.bincount(self.inv[use], weights=weights[use]*values[use], minlength=len(self.t))
        if equal_traj:
            v = np.divide(v, c, out=np.zeros_like(v), where=c > 0)
            c = (c > 0).astype(float)
        with np.errstate(invalid='ignore', divide='ignore'):
            dist = (self.W @ v)/(self.W @ c)
        good = dist[np.isfinite(dist)]
        ci = np.percentile(good, [2.5, 97.5]).tolist() if info['n_traj'] >= 2 and len(good) else None
        return dict(**info, estimate=float(v.sum()/c.sum()), ci95=ci, invalid_bootstrap=int((~np.isfinite(dist)).sum())), dist


def gu(p, q):
    return (p*q).sum(1)-q.mean(1)


def reading(p1, p2):
    if p1['ci95'] is None:
        return 'UNAVAILABLE', 'R-P8-UNDERPOWERED/UNAVAILABLE'
    lo, hi = p1['ci95']
    if hi < 0:
        return 'NEG', 'R-P8-MISDIRECT-HISTORY'
    if lo >= 0:
        return 'NONNEG', 'R-P8-SELECTION' if p2 else 'R-P8-PARTIAL'
    return 'INCONCLUSIVE', 'R-P8-INCONCLUSIVE'


def load_science(rd, rows):
    labels = rd.pq('data/labels_final.parquet').set_index(KEY)
    strata = rd.pq('exploratory_P4/strata.parquet').set_index(KEY)
    scores = rd.pq('data/scores_fp32.parquet').set_index(KEY + ['action'])
    assert rd.js('exploratory_P4/strata_audit.json')['passed']
    Q, GOLD, MEAN, SUM = [], [], [], []
    bankcounts, peek_count = {}, 0
    for i, r in enumerate(rows):
        tid, t = r['traj_id'], r['turn']
        lab, st = labels.loc[(tid, t)], strata.loc[(tid, t)]
        assert bool(st['prompt_target']) == r['prompt_target']
        r.update(live=bool(lab['live']), fork=bool(lab['het']), old_stratum=str(st['stratum']), gold=False)
        banks = {}
        for b in range(1, 8):
            folder = ('replay' if b == 1 else f'replay_b{b-1}') if b < 7 else 'replay_gold'
            p = f'data/{folder}/{tid}/t{t:02d}.json'
            if b == 7 and not (SOURCE / p).exists():
                continue
            if b in (3, 4, 5, 6) and not r['live']:
                continue
            data = rd.js(p)
            assert [data['factual_action']] + data['alternatives'] == r['actions']
            values = np.asarray([data['factual_outcomes']] + data['alt_outcomes'], float)
            assert values.shape == (5, 20 if b == 7 else 3) and np.isin(values, [0, 1]).all()
            assert all(c['refeed_div'] == 0 for c in data['continuations'])
            banks[b] = values
            bankcounts[str(b)] = bankcounts.get(str(b), 0) + 1
        q = np.concatenate([banks[1], banks[2]], axis=1).mean(1)
        live = 0 < sum(v.sum() for b, v in banks.items() if b in (1, 2)) < 30
        assert live == r['live'] and (not r['fork'] or live)
        Q.append(q)
        GOLD.append(banks[7].mean(1) if 7 in banks else np.full(5, np.nan))
        r['gold'] = 7 in banks
        for field, arr in [('logp_mean', MEAN), ('logp_sum', SUM)]:
            s = np.array([scores.loc[(tid, t, a), field] for a in r['actions']], float)
            assert np.isfinite(s).all()
            p = np.exp(s-s.max()); arr.append(p/p.sum())
        p = f'data/peek_alts/{tid}/t{t:02d}.json'
        r['peek_available'] = (SOURCE / p).exists()
        if r['peek_available']:
            pk = rd.js(p)
            checks = pk['checks']
            assert checks['factual_obs_match'] and checks['refeed_div_total'] == 0 and checks['repeat_all_match']
            peeks = sorted(pk['peeks'], key=lambda p: (p['role'] != 'factual', p['alt_index'] or 0))
            assert [p['action'] for p in peeks] == r['actions']
            peek_count += 1
        if i % 500 == 0:
            print('integrity/science inputs', i, flush=True)
    return np.array(Q), np.array(GOLD), np.array(MEAN), np.array(SUM), dict(bank_point_counts=bankcounts, peek_points=peek_count)


def audit():
    rd = Reader()
    sample = json.loads((HERE / 'blind_sample.json').read_text())
    assert subprocess.check_output(['git', 'show', 'HEAD:exploratory_P8/blind_sample.json'], cwd=ROOT) == (HERE / 'blind_sample.json').read_bytes()
    rows, _ = descriptors(rd)
    q, gold, mean, psum, integrity = load_science(rd, rows)
    n = len(rows)
    tids = np.array([r['traj_id'] for r in rows]); boot = Bootstrap(tids)
    field = lambda k: np.array([r[k] for r in rows])
    pol = np.array([r['policy'] for r in rows]); g = gu(pol, q)
    old = (field('old_stratum') == 'I') & field('fork')
    expected = rd.js('exploratory_P4/x.json')['results']['I']['12']['X2_drafts_batch0']['G_u']
    ref_e1 = rd.js('exploratory_P4/e1.json')['action']['I']['12']['policy']['G_u']['estimate']
    gate = dict(n=int(old.sum()), expected_n=184, X2=float(g[old].mean()), expected_X2=expected['estimate'],
                E1=float(gu(psum, q)[old].mean()), expected_E1=ref_e1)
    gate['passed'] = gate['n'] == 184 and abs(gate['X2']-gate['expected_X2']) < 1e-9 and abs(gate['E1']-ref_e1) < 1e-9
    save(HERE / 'consistency_gate.json', gate)
    assert gate['passed'], gate
    print('Old-result consistency gate passed', flush=True)
    h, live, fork = field('H_prompt'), field('live'), field('fork')
    assert np.max(np.abs(g[~live])) < 1e-12
    groups = {'H_prompt': h, 'H_full': field('H_full'), 'H_Live': h & live,
              'H_Fork': h & fork, 'H_nonFork_Live': h & live & ~fork, 'H_Dead': h & ~live,
              'old_I_Fork': old}
    first = np.zeros(n, bool)
    for tid in np.unique(tids[h]):
        ids = np.where(h & (tids == tid))[0]
        first[ids[np.argmin(field('turn')[ids])]] = True
    selectors = dict(X2=pol, drafts_all=np.array([r['policy_all'] for r in rows]),
                     chosen=np.eye(5)[np.zeros(n, int)], logp_mean=mean,
                     R0=np.full((n, 5), .2), R1_window=np.array([r['R1_window'] for r in rows]),
                     R1_full=np.array([r['R1_full'] for r in rows]))
    table = {}
    for name, p in selectors.items():
        gains = gu(p, q)
        table[name] = {group: dict(G_u=boot.stat(gains, mask)[0], G_c=boot.stat((p*q).sum(1)-q[:, 0], mask)[0],
                                  trajectory_equal=boot.stat(gains, mask, equal_traj=True)[0],
                                  gold=boot.stat(gu(p, gold), mask)[0]) for group, mask in groups.items()}
    p1 = table['X2']['H_prompt']['G_u']
    p2stat = table['X2']['H_nonFork_Live']['G_u']
    p2 = bool(p2stat['ci95'] is not None and p2stat['ci95'][0] > 0)
    p1class, read = reading(p1, p2)
    contrasts = {}
    for name in ('R1_window', 'R1_full'):
        contrasts[name + '-policy'] = {group: boot.stat(gu(selectors[name], q)-g, mask)[0] for group, mask in groups.items()}
        contrasts[name + '-policy_information_budget'] = 'same prompt' if name == 'R1_window' else 'extra full history; not same-information'
    s4stat = contrasts['R1_window-policy']['H_prompt']
    s4 = bool(s4stat['ci95'] is not None and s4stat['ci95'][0] > 0)
    decomposition = {}
    for norm, mask in [('Live', h & live), ('H_prompt', h)]:
        total, dt = boot.stat(g, mask)
        fc, df = boot.stat(g*fork, mask)
        nc, dn = boot.stat(g*(live & ~fork), mask)
        wf, dw = boot.stat(fork.astype(float), mask)
        lhs_f, dgf = boot.stat(g, mask & fork)
        lhs_n, dgn = boot.stat(g, mask & live & ~fork)
        ok = np.isfinite(dt) & np.isfinite(df) & np.isfinite(dn)
        error = float(np.max(np.abs(dt[ok]-df[ok]-dn[ok])))
        assert error < 1e-12
        if norm == 'Live':
            ok2 = np.isfinite(dt) & np.isfinite(dgf) & np.isfinite(dgn)
            identity = float(np.max(np.abs(dt[ok2] - dw[ok2]*dgf[ok2]-(1-dw[ok2])*dgn[ok2]))) if ok2.any() else None
            assert identity is None or identity < 1e-12
        else:
            identity = None
        decomposition[norm] = dict(total=total, fork_weight=wf, no_compensation=fc, nonfork_compensation=nc,
                                   bootstrap_sum_error=error, bootstrap_reweighted_identity_error=identity)
    decomposition['Live_share_in_H'] = boot.stat(live.astype(float), h)[0]
    subgroups = {}
    rng = np.ptp(q, axis=1)
    bins = np.where(rng <= 1e-12, 0, np.where(rng <= 1/3+1e-12, 1, np.where(rng <= 2/3+1e-12, 2, 3)))
    quartiles = np.quantile(field('turn')[h], [.25, .5, .75], method='linear')
    turns = np.searchsorted(quartiles, field('turn'), side='left')
    submasks = {f'range/{name}': h & (bins == i) for i, name in enumerate(['0', '(0,1/3]', '(1/3,2/3]', '(2/3,1]'])}
    submasks.update({f'turn/Q{i+1}': h & (turns == i) for i in range(4)})
    submasks.update({f'task/{tt}': h & (field('task_type') == tt) for tt in np.unique(field('task_type'))})
    submasks['first_H_per_trajectory'] = first
    for name, mask in submasks.items():
        subgroups[name] = {sel: boot.stat(gu(selectors[sel], q), mask)[0] for sel in ('X2', 'chosen', 'R1_window', 'R1_full')}
    best = np.isclose(q, q.max(1)[:, None], rtol=0, atol=1e-12)
    unique = best.sum(1) == 1
    pbest = (pol*best).sum(1)
    excess = pbest-best.mean(1)
    a2 = dict(descriptive_only=True, unique_best=int((h & live & unique).sum()),
              tie_rate=boot.stat((~unique).astype(float), h & live)[0],
              best_set_probability_excess={name: boot.stat(excess, mask)[0] for name, mask in groups.items()}, bins={})
    pbins = np.searchsorted([.2, .4, .6, .8], pbest, side='left')
    for k in range(5):
        mask = h & live & unique & (pbins == k)
        a2['bins'][str(k)] = dict(interval=[k/5, (k+1)/5], fork_rate=boot.stat(fork.astype(float), mask)[0],
                                  policy_best_mass=boot.stat(pbest, mask)[0])
    cross = pd.DataFrame({k: field(k) for k in ['prompt_target', 'history_target', 'live', 'fork', 'gold']})
    cross['old_I'] = field('old_stratum') == 'I'; cross['traj_id'] = tids
    ct = cross.groupby(['prompt_target', 'history_target', 'old_I', 'live', 'fork', 'gold'], dropna=False).agg(
        n=('traj_id', 'size'), n_traj=('traj_id', 'nunique')).reset_index().to_dict('records')
    sampleids = {r['point_id'] for r in sample['points']}; sampled = np.array([r['point_id'] in sampleids for r in rows])
    assert sampled.sum() == sample['n'] and (sampled <= h).all() and int(h.sum()) == sample['population']
    coverage = {name: {group: dict(coverage=boot.stat(field(k).astype(float), mask)[0],
                                  fallback=int((mask & ~field(k)).sum())) for group, mask in groups.items()}
                for name, k in [('R1_window', 'window_coverage'), ('R1_full', 'full_coverage')]}
    out = dict(commit=commit(), source=str(SOURCE), prereg_sha256=sha((HERE / 'PREREG_P8.md').read_bytes()),
               units='fraction; multiply by 100 for percentage points', seed=SEED, bootstrap=B,
               counts=dict(points=n, trajectories=len(set(tids)), H_prompt=int(h.sum()), H_full=int(field('H_full').sum())),
               integrity=integrity, old_consistency=gate, P1=p1class, P2=p2, S4=s4, S5='PENDING_PHASE_C', reading=read,
               selection_interpretation_permitted=p1class == 'NONNEG' and p2 and table['X2']['H_Fork']['G_u']['estimate'] < 0,
               table=table, contrasts=contrasts, decomposition=decomposition, subgroups=subgroups, turn_quartile_edges=quartiles.tolist(),
               A2=a2, cross_tab=ct, coverage=coverage, R2='UNAVAILABLE: no validated strict negative-evidence/path parser',
               R3='UNAVAILABLE: no audited independent training-prior table',
               first_batch_drafts=dict(total=int(field('first_drafts').sum()), valid_admissible=int(field('valid_first').sum()),
                                       parse_rate=float(field('valid_first').sum()/field('first_drafts').sum())),
               phase_c_composition={name: dict(population=int(mask.sum()), sampled=int((sampled & mask).sum())) for name, mask in groups.items()})
    save(HERE / 'p8_cpu.json', out)
    (HERE / 'data').mkdir(exist_ok=True)
    save(HERE / 'data/points.json', rows)
    np.savez_compressed(HERE / 'data/arrays.npz', q=q, gold=gold, logp_mean=mean)
    for p in [Path(__file__), HERE / 'PREREG_P8.md', ROOT / 'scripts/common.py', ROOT / 'exploratory_P4/p4_x_robust.py']:
        rd.raw(p)
    save(HERE / 'science_inputs.json', rd.hashes)
    render_cpu(out)
    print(json.dumps(dict(P1=p1class, P2=p2, S4=s4, reading=read)), flush=True)


def fmt(s):
    if s['estimate'] is None:
        return 'UNAVAILABLE'
    ci = '' if s['ci95'] is None else f" [{100*s['ci95'][0]:+.2f}, {100*s['ci95'][1]:+.2f}]"
    return f"{100*s['estimate']:+.2f}{ci} pp (n={s['n']}, trajectories={s['n_traj']}, missing={s['missing']})"


def render_cpu(out):
    lines = ['P8 CPU audit (post-hoc ALFWorld audit; not independent confirmation)',
             'P1: ' + out['P1'], 'P2: ' + str(out['P2']), 'S4: ' + str(out['S4']), 'Reading: ' + out['reading'],
             'S5 and same-population model comparisons: pending phase C', '', 'X2 policy population results:']
    for group, value in out['table']['X2'].items():
        lines.append(group + ': ' + fmt(value['G_u']))
    for name, row in out['decomposition'].items():
        if name == 'Live_share_in_H':
            continue
        lines.extend(['', name + ' weighted decomposition:'] + [k + ': ' + fmt(row[k]) for k in ('total', 'fork_weight', 'no_compensation', 'nonfork_compensation')])
    lines.extend(['', 'R1-window minus policy: ' + fmt(out['contrasts']['R1_window-policy']['H_prompt']),
                  'R1-full minus policy (additional history): ' + fmt(out['contrasts']['R1_full-policy']['H_prompt']),
                  'R2: ' + out['R2'], 'R3: ' + out['R3'],
                  'Old I/Fork n=184 and both X2/E1 gains reproduced within 1e-9.',
                  'A2 is descriptive only and does not establish hidden-world surprise.',
                  'All strata, coverage, Gold, task/turn/range and equal-trajectory sensitivity results: p8_cpu.json.'])
    (HERE / 'P8_CPU_REPORT.txt').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['freeze', 'analyze'])
    args = ap.parse_args()
    freeze_sample() if args.stage == 'freeze' else audit()
