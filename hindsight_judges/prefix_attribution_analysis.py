"""Paired point-level estimates with frozen trajectory-cluster resampling."""
import numpy as np
import exp1c as E

CONDITIONS = ('AS', 'AO', 'QS', 'QO', 'QN', 'AP', 'KEEP', 'DROP')
SEED = 20261105
NBOOT = 4000
LIMITATIONS = (
    'Restricted five-action continuation diagnostic, not observed policy success or an OPD gradient. '
    'QO−QS identifies a stated-authorship wording intervention, not internal self-recognition. '
    'QS−AP matches restart and final request but still changes message/history/quotation structure. '
    'AP−AS combines turn completion, restart and a new user request. '
    'QS−AS and QO−AO combine placement, generation restart and prompt/task format. '
    'CIs are exploratory without multiplicity correction; crossing zero is unresolved, not equivalence. '
    'Null quote effects do not establish a content-only mechanism. '
    'A value gain without a corresponding Q-aligned evidence-log-odds effect is not recovered evidence assimilation. '
    'No Student training, trajectory collection, HLN or OPD was performed by this diagnostic.'
)


def evidence_metrics(m, f, q):
    """Unclipped alternative-versus-factual change; independent replay Q is fixed."""
    m, f, q = (np.asarray(x, dtype=float) for x in (m, f, q))
    assert m.shape == f.shape == q.shape == (5,)
    assert np.isfinite([m, f, q]).all()
    shift = (f[1:] - f[0]) - (m[1:] - m[0])
    dq = q[1:] - q[0]
    non_tie = dq != 0
    if non_tie.any():
        x, y = dq[non_tie], shift[non_tie]
        concordance = float(np.mean(np.where(y == 0, .5, (np.sign(y) == np.sign(x)).astype(float))))
        aligned = float(np.mean(y * np.sign(x)))
        slope = float(np.sum(x * y) / np.sum(x * x))
    else:
        concordance = aligned = slope = None
    centered_q = q - q.mean()
    centered_delta = (f - m) - (f - m).mean()
    return dict(centered_q=centered_q.tolist(), centered_delta=centered_delta.tolist(),
                centered_xy=float(centered_q @ centered_delta), centered_xx=float(centered_q @ centered_q), shift_nats=shift.tolist(), qgap=dq.tolist(), non_tie_pairs=int(non_tie.sum()),
                aligned_shift_nat=aligned, concordance=concordance, slope_nat_per_q=slope)


def cell_metrics(ll, q):
    ll, q = np.asarray(ll, float), np.asarray(q, float)
    p = np.exp(ll - ll.max()); p /= p.sum()
    top = p == p.max()
    bad = bool(q[0] < q.max())
    return dict(value_pp=float(100 * p @ q), factual_argmax_rate=float(np.argmax(p) == 0),
                factual_probability=float(p[0]),
                bad_factual_corrected_rate=float(np.mean(q[top] > q[0])) if bad else None)


def ci(values, clusters, unit):
    good = [i for i, v in enumerate(values) if v is not None]
    if not good:
        return dict(n=0, n_trajectories=0, estimate=None, ci95=None, unit=unit, undefined=len(values))
    result = E.mean_ci([values[i] for i in good], [clusters[i] for i in good], seed=SEED, nboot=NBOOT)
    result.update(unit=unit, undefined=len(values) - len(good))
    return result


def subtract(a, b):
    return None if a is None or b is None else a - b


def verdict(stat):
    if stat['ci95'] is None: return 'unresolved'
    lo, hi = stat['ci95']
    if lo > 0: return 'supported: positive contrast'
    if hi < 0: return 'not supported: contrast is negative'
    return 'unresolved: CI crosses zero; not evidence of equivalence'


def analyze_rows(rows, scores):
    indexed = {}
    for r in rows:
        key = (r['point_id'], r['evidence'], r['condition'])
        assert key not in indexed
        indexed[key] = r
    pids = sorted({r['point_id'] for r in rows})
    points = []
    for pid in pids:
        base = indexed[pid, 'M', 'AS']
        q = base['q3456']
        point = dict(point_id=pid, traj=base['traj'], q3456=q, cells={}, evidence={})
        for c in CONDITIONS:
            for e in ('M', 'F'):
                r = indexed[pid, e, c]
                assert r['q3456'] == q and r['traj'] == base['traj']
                point['cells'][e + '_' + c] = cell_metrics(scores[r['id']]['sequence_logp'], q)
            m, f = [scores[indexed[pid, e, c]['id']]['sequence_logp'] for e in ('M', 'F')]
            point['evidence'][c] = evidence_metrics(m, f, q)
        points.append(point)
    clusters = [p['traj'] for p in points]
    cell_names = ('value_pp', 'factual_argmax_rate', 'factual_probability', 'bad_factual_corrected_rate')
    evidence_names = ('aligned_shift_nat', 'concordance', 'slope_nat_per_q')
    units = dict(value_pp='percentage points', factual_argmax_rate='0–1', factual_probability='0–1',
                 bad_factual_corrected_rate='0–1', aligned_shift_nat='nat', concordance='0–1',
                 slope_nat_per_q='nat per unit replay Q')
    cells = {f'{e}_{c}': {k: ci([p['cells'][f'{e}_{c}'][k] for p in points], clusters, units[k])
                         for k in cell_names} for e in ('M', 'F') for c in CONDITIONS}
    evidence = {c: {k: ci([p['evidence'][c][k] for p in points], clusters, units[k])
                    for k in evidence_names} for c in CONDITIONS}
    pairs = {'matched_restart_slot_QS_minus_AP': ('QS', 'AP'),
             'restart_AP_minus_AS': ('AP', 'AS'),
             'neutral_QN_minus_QS': ('QN', 'QS'),
             'neutral_QN_minus_QO': ('QN', 'QO'),
             'quote_provenance_QO_minus_QS': ('QO', 'QS'),
             'assistant_provenance_AO_minus_AS': ('AO', 'AS'),
             'mixed_placement_QS_minus_AS': ('QS', 'AS'),
             'mixed_placement_QO_minus_AO': ('QO', 'AO'),
             'explicit_self_AS_minus_KEEP': ('AS', 'KEEP'),
             'anchor_DROP_minus_KEEP': ('DROP', 'KEEP')}
    contrasts = {}
    for label, (a, b) in pairs.items():
        for e in ('M', 'F'):
            contrasts[f'{e}_{label}'] = {k: ci([subtract(p['cells'][f'{e}_{a}'][k], p['cells'][f'{e}_{b}'][k])
                                                for p in points], clusters, units[k]) for k in cell_names}
        contrasts[f'evidence_{label}'] = {k: ci([subtract(p['evidence'][a][k], p['evidence'][b][k])
                                               for p in points], clusters, units[k]) for k in evidence_names}
        contrasts[f'F_minus_M_{label}'] = {k: ci([
            subtract(subtract(p['cells'][f'F_{a}'][k], p['cells'][f'M_{a}'][k]),
                     subtract(p['cells'][f'F_{b}'][k], p['cells'][f'M_{b}'][k]))
            for p in points], clusters, units[k]) for k in cell_names}
    for c in CONDITIONS:
        contrasts[f'F_minus_M_{c}'] = {k: ci([subtract(p['cells'][f'F_{c}'][k], p['cells'][f'M_{c}'][k])
                                             for p in points], clusters, units[k]) for k in cell_names}
    for e in ('M', 'F'):
        contrasts[f'{e}_role_by_provenance'] = {k: ci([
            subtract(subtract(p['cells'][f'{e}_QO'][k], p['cells'][f'{e}_QS'][k]),
                     subtract(p['cells'][f'{e}_AO'][k], p['cells'][f'{e}_AS'][k]))
            for p in points], clusters, units[k]) for k in cell_names}
    primary = dict(SLOT_F=contrasts['F_matched_restart_slot_QS_minus_AP']['value_pp'],
                   SLOT_M=contrasts['M_matched_restart_slot_QS_minus_AP']['value_pp'],
                   I_SLOT=contrasts['F_minus_M_matched_restart_slot_QS_minus_AP']['value_pp'], P_Q_F=contrasts['F_quote_provenance_QO_minus_QS']['value_pp'],
                   P_Q_M=contrasts['M_quote_provenance_QO_minus_QS']['value_pp'],
                   I_Q=contrasts['F_minus_M_quote_provenance_QO_minus_QS']['value_pp'])
    result = dict(n_points=len(points), n_clusters=len(set(clusters)), contexts=len(rows),
                  bootstrap=dict(draws=NBOOT, seed=SEED, unit='original trajectory',
                                 estimator='pointwise paired contrast, then point-weighted mean'),
                  definitions=dict(bad_factual='q3456[f] < max(q3456)',
                                   correction='argmax action has Q > factual Q; score ties averaged',
                                   factual_argmax='numpy first-index argmax (factual first), consistent with old anchor',
                                   evidence_concordance='mean over non-tie Q pairs within point; zero shift = 0.5',
                                   aligned_shift='mean sign(Q_j-Q_f)*E_j over non-tie Q pairs within point',
                                   slope='within-point through-origin sum(Qgap*E)/sum(Qgap^2)',
                                   undefined='all-Q-tied points retained for V; E-alignment undefined and counts reported'),
                  centered=centered_statistics(points, pairs), cells=cells, evidence=evidence, contrasts=contrasts, primary=primary,
                  interpretation={k: verdict(v) for k, v in primary.items()}, limitations=LIMITATIONS)
    return result, points


def report(result):
    lines = ['PREFIX PROVENANCE × MESSAGE ROLE',
             f"{result.get('model', '')} / {result.get('cohort', '')}: {result['n_points']} points, "
             f"{result['n_clusters']} trajectories, {result['contexts']} contexts.",
             '95% cluster bootstrap, 4000 draws, seed 20261105. No multiplicity correction.', '',
             'Primary paired contrasts (percentage points):']
    for k, s in result['primary'].items():
        lines.append(f"{k}: {s['estimate']:.6f}, CI {s['ci95']}; {result['interpretation'][k]}")
    lines += ['', 'Absolute M/F values and factual anchoring (all conditions):',
              'Cell | value pp [CI] | factual argmax rate | p_f | bad factual correction rate']
    for k, m in result['cells'].items():
        lines.append(f"{k} | {m['value_pp']['estimate']:.6f} {m['value_pp']['ci95']} | "
                     f"{m['factual_argmax_rate']['estimate']} | {m['factual_probability']['estimate']} | "
                     f"{m['bad_factual_corrected_rate']['estimate']}")
    lines += ['', 'Paired contrasts: value pp, F−M interaction, and Q-aligned evidence support:']
    for label, metrics in result['contrasts'].items():
        lines.append(label)
        for k, s in metrics.items():
            lines.append(f"  {k}: {s['estimate']} [{s['ci95']}], {s['unit']}; n={s['n']}")
    lines += ['', 'Absolute evidence-induced log-odds alignment (factual reference; no clipping):']
    for c, metrics in result['evidence'].items():
        for k, s in metrics.items():
            lines.append(f"{c} {k}: {s['estimate']}, CI {s['ci95']}; n={s['n']}, undefined={s['undefined']}")
    lines += ['', 'Pooled five-arm centered evidence slopes and paired DROP ratios:']
    for c, stat in result['centered']['conditions'].items():
        lines.append(f"{c}: slope {stat['slope']}; slope/DROP {stat['ratio_to_DROP']}")
    for label, stat in result['centered']['paired_slope_contrasts'].items():
        lines.append(f'{label}: {stat}')
    lines += ['', 'Historical KEEP/DROP: ' + str(result.get('historical_anchors', {})),
              'Original action markers outside Think preserved: ' + str(result.get('known_preexisting_action_marker_points', []))]
    lines += ['', 'All raw four-alternative E values and Q gaps are in point_metrics.jsonl.gz.',
              'Exact user/assistant prompts and role token IDs: ../../exact_prompt_examples.json.',
              'All numerical gates, model/source hashes and runtime/VRAM: analysis.json and run.json.', '', LIMITATIONS]
    return '\n'.join(lines) + '\n'


def centered_statistics(points, pairs):
    """Pooled within-point-centered slope; paired cluster draws shared by ALL conditions."""
    labels = sorted({p['traj'] for p in points})
    groups = np.array([labels.index(p['traj']) for p in points])
    draws = np.random.default_rng(SEED).multinomial(len(labels), [1 / len(labels)] * len(labels), size=NBOOT)
    xx = np.array([p['evidence']['DROP']['centered_xx'] for p in points])
    den = float(xx.sum())
    bd = draws @ np.bincount(groups, weights=xx, minlength=len(labels))
    samples, estimates = {}, {}
    for c in CONDITIONS:
        xy = np.array([p['evidence'][c]['centered_xy'] for p in points])
        bn = draws @ np.bincount(groups, weights=xy, minlength=len(labels))
        estimates[c] = float(xy.sum() / den) if den != 0 else None
        samples[c] = np.divide(bn, bd, out=np.full(NBOOT, np.nan), where=bd != 0)
    def summary(value, draws_, unit):
        valid = np.isfinite(draws_)
        return dict(estimate=value, ci95=np.percentile(draws_[valid], [2.5, 97.5]).tolist() if valid.all() else None,
                    undefined_draws=int((~valid).sum()), unit=unit)
    reference = summary(estimates['DROP'], samples['DROP'], 'nat per replay Q')
    unstable = reference['ci95'] is None or reference['ci95'][0] <= 0 <= reference['ci95'][1]
    conditions = {}
    for c in CONDITIONS:
        ratio = estimates[c] / estimates['DROP'] if estimates['DROP'] not in (0, None) else None
        dist = np.divide(samples[c], samples['DROP'], out=np.full(NBOOT, np.nan), where=samples['DROP'] != 0)
        ratio_stat = summary(ratio, dist, 'relative to DROP')
        ratio_stat['denominator_ci_contains_zero_or_undefined'] = unstable
        if unstable:
            ratio_stat['ci95'] = None
            ratio_stat['interpretation'] = 'Unstable denominator; do not interpret a bounded ratio interval.'
        conditions[c] = dict(slope=summary(estimates[c], samples[c], 'nat per replay Q'), ratio_to_DROP=ratio_stat)
    contrasts = {label: summary(subtract(estimates[a], estimates[b]), samples[a] - samples[b], 'nat per replay Q')
                 for label, (a, b) in pairs.items()}
    return dict(definition='sum_i,j (Q_ij-mean_j Q_i)*(delta_ij-mean_j delta_i) / sum_i,j (Q_ij-mean_j Q_i)^2; delta=ell_F-ell_M',
                conditions=conditions, paired_slope_contrasts=contrasts, bootstrap_draws=NBOOT, seed=SEED,
                comparison_to_prior_ratios='Only compare to prior 0.55/0.36 if their estimator is confirmed identical; historical anchors are recomputed using this formula.')
