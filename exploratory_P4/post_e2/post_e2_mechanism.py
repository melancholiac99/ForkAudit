"""POST-HOC (not pre-registered) mechanism analysis of P4 E2 and P4-X.

Question: at information-limited (I) Forks, where does each action selector put its probability mass, and what are
those actions worth? Action classes are computed from the agent's own factual history before t (taken from the full
factual episode embedded in the committed E2 `L` requests); they never read outcomes.

Classes for the five arms at decision t (history = executed actions/observations at turns < t):
  E_new     `go to X` with X never reached before t, or `open X` with X never opened before t
  R_seen    `go to X` with X reached before t, outside the 2-turn prompt window, and X's contents already observed
  R_recent  `go to X` with X reached within the 2-turn prompt window
  R_unseen  `go to X` reached before t, outside the window, contents never observed
  Other     everything else (examine, take, put, inventory, look, ...)

Inputs (SHA256 recorded; the arms file must equal the one used by p4_e2_analyze.py):
  exploratory_P4/strata.parquet, exploratory_deep_discovery/exploratory_arms.parquet (deep branch c6d03ee),
  exploratory_P4/e2_requests.jsonl.gz, exploratory_P4/e2_{7b,32b}/scores.jsonl, exploratory_P4/e2.json.
Gate: recomputed G_u for every judge condition must equal e2.json to 1e-9, else stop.
Output: exploratory_P4/post_e2/post_e2_mechanism.json
"""
from pathlib import Path
import argparse, gzip, hashlib, json, re, subprocess, sys
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SEED, NBOOT = 31102026, 4000
GOTO, OPEN = re.compile(r'^go to (.+)$'), re.compile(r'^open (.+)$')
CLASSES = ['E_new', 'R_seen', 'R_recent', 'R_unseen', 'Other']
SELECTORS = ['uniform', 'policy_drafts'] + [f'{m}_{c}' for m in ['7b', '32b'] for c in ['S', 'L', 'F', 'Fx']]


def load_parquet(path):
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return {c: df[c].tolist() for c in df.columns}
    except ImportError:
        sys.path.insert(0, str(OUT))
        import minipq
        return minipq.read(str(path))


def arm_classes(actions, history):
    """actions: five candidate commands; history: list of (executed_action, observation_after) for turns < t."""
    reached, opened, seen = set(), set(), set()
    for act, obs in history:
        a, o = act.strip().lower(), obs.lower()
        m = GOTO.match(a)
        if m:
            reached.add(m.group(1))
            if 'you see' in o or 'is open' in o:
                seen.add(m.group(1))
        m = OPEN.match(a)
        if m:
            opened.add(m.group(1))
            if 'you see' in o or 'you open' in o:
                seen.add(m.group(1))
    window = {GOTO.match(a.strip().lower()).group(1) for a, _ in history[-2:] if GOTO.match(a.strip().lower())}
    out = []
    for x in actions:
        a = x.strip().lower()
        m, o = GOTO.match(a), OPEN.match(a)
        if m:
            r = m.group(1)
            out.append('E_new' if r not in reached else 'R_recent' if r in window else 'R_seen' if r in seen else 'R_unseen')
        elif o:
            out.append('E_new' if o.group(1) not in opened else 'Other')
        else:
            out.append('Other')
    return out


def selftest():
    h = [('go to cabinet 1', 'You arrive at cabinet 1. The cabinet 1 is closed.'),
         ('open cabinet 1', 'You open the cabinet 1. The cabinet 1 is open. In it, you see a cup 1.'),
         ('go to drawer 1', 'You arrive at drawer 1. On the drawer 1, you see nothing.'),
         ('go to countertop 1', 'You arrive at countertop 1. On the countertop 1, you see a apple 1.'),
         ('examine countertop 1', 'On the countertop 1, you see a apple 1.')]
    c = arm_classes(['go to cabinet 1', 'go to drawer 1', 'go to countertop 1', 'go to shelf 2', 'open drawer 1'], h)
    assert c == ['R_seen', 'R_seen', 'R_recent', 'E_new', 'E_new'], c
    h2 = [('go to fridge 1', 'You arrive at fridge 1. The fridge 1 is closed.'), ('look', 'x'), ('inventory', 'y')]
    assert arm_classes(['go to fridge 1', 'open fridge 1', 'take a from b'], h2) == ['R_unseen', 'E_new', 'Other']
    print('selftest passed')


def main(arms_path):
    inputs = {}

    def track(p):
        b = Path(p).read_bytes()
        inputs[str(p)] = hashlib.sha256(b).hexdigest()
        return b

    e2 = json.loads(track(ROOT / 'exploratory_P4/e2.json'))
    track(arms_path)
    want = e2['inputs']['exploratory_deep_discovery/exploratory_arms.parquet']
    assert inputs[str(arms_path)] == want, 'arms file differs from the one used by p4_e2_analyze.py'
    track(ROOT / 'exploratory_P4/strata.parquet')
    s = load_parquet(ROOT / 'exploratory_P4/strata.parquet')
    a = load_parquet(arms_path)
    key = lambda t, u: f'{t}_t{int(u):02d}'
    info = {key(s['traj_id'][i], s['turn'][i]): dict(traj=s['traj_id'][i], turn=int(s['turn'][i]), layer=s['stratum'][i],
                                                    het=bool(s['het'][i]))
            for i in range(len(s['traj_id'])) if s['live'][i]}
    arms = defaultdict(dict)
    for i in range(len(a['traj_id'])):
        arms[key(a['traj_id'][i], a['turn'][i])][int(a['j'][i])] = dict(
            action=a['action'][i], q12=float(a['q_12'][i]),
            qg=float('nan') if a['q_gold'][i] is None else float(a['q_gold'][i]), drafts=float(a['draft_count'][i]))
    raw = track(ROOT / 'exploratory_P4/e2_requests.jsonl.gz')
    assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == e2['inputs']['exploratory_P4/e2_requests.jsonl']
    acts, episodes, dec = {}, {}, json.JSONDecoder()
    for line in gzip.decompress(raw).decode().splitlines():
        r = json.loads(line)
        if r['condition'] == 'S':
            acts[r['point_id']] = r['actions']
        if r['condition'] == 'L' and r['traj_id'] not in episodes:
            t = r['text']
            episodes[r['traj_id']] = dec.raw_decode(t[t.find('{"task":'):])[0]
    P = {}
    for m in ['7b', '32b']:
        for line in track(ROOT / f'exploratory_P4/e2_{m}/scores.jsonl').decode().splitlines():
            r = json.loads(line)
            P[m, r['condition'], r['point_id']] = np.array(r['candidate_prob'])

    traj = sorted({z['traj'] for z in info.values()})
    tix = {t: i for i, t in enumerate(traj)}
    W = np.random.default_rng(SEED).multinomial(len(traj), np.full(len(traj), 1 / len(traj)), size=NBOOT).astype(float)

    def mean_ci(vals, trajs):
        v = np.asarray(vals, float); g = np.asarray([tix[t] for t in trajs])
        ok = np.isfinite(v); v, g = v[ok], g[ok]
        if not len(v):
            return dict(n=0, estimate=None, ci95=None)
        sv, sc = np.bincount(g, v, len(traj)), np.bincount(g, None, len(traj))
        with np.errstate(invalid='ignore', divide='ignore'):
            d = (W @ sv) / (W @ sc)
        d = d[np.isfinite(d)]
        return dict(n=int(len(v)), n_traj=int(len(set(g))), estimate=float(v.mean()),
                    ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])

    def loto(vals, trajs):
        v, t = np.asarray(vals, float), np.asarray(trajs)
        e = [v[t != x].mean() for x in sorted(set(trajs))]
        return dict(min=float(min(e)), max=float(max(e)))

    def point(k):
        z = info[k]
        steps = episodes[z['traj']]['steps']
        assert all(int(st['turn']) == n + 1 for n, st in enumerate(steps))
        assert steps[z['turn'] - 1]['executed_action'] == arms[k][0]['action']
        hist = [(st['executed_action'], st['observation_after']) for st in steps[:z['turn'] - 1]]
        cand = [arms[k][j]['action'] for j in range(5)]
        assert acts[k] == cand
        q = np.array([arms[k][j]['q12'] for j in range(5)])
        qg = np.array([arms[k][j]['qg'] for j in range(5)])
        dr = np.array([arms[k][j]['drafts'] for j in range(5)]) + .5
        sel = {'uniform': np.full(5, .2), 'policy_drafts': dr / dr.sum()}
        for m in ['7b', '32b']:
            for c in ['S', 'L', 'F', 'Fx']:
                sel[f'{m}_{c}'] = P[m, c, k]
        return q, qg, sel, arm_classes(cand, hist), int(episodes[z['traj']]['final_reward'])

    res = {'gate': {}, 'layers': {}}
    for layer in ['I', 'U']:
        ks = sorted(k for k, z in info.items() if z['layer'] == layer and z['het'])
        tr = [info[k]['traj'] for k in ks]
        acc = defaultdict(list); vq = defaultdict(list); vt = defaultdict(list); vqg = defaultdict(list); vtg = defaultdict(list)
        chosen = Counter()
        for k in ks:
            q, qg, sel, cls, _ = point(k)
            chosen[cls[0]] += 1
            for c in set(cls):
                idx = [j for j in range(5) if cls[j] == c]
                vq[c].append(q[idx].mean()); vt[c].append(info[k]['traj'])
                if np.isfinite(qg).all():
                    vqg[c].append(qg[idx].mean()); vtg[c].append(info[k]['traj'])
            E = [j for j in range(5) if cls[j] == 'E_new']
            for n, p in sel.items():
                acc[n, 'G_u'].append(float(p @ q - q.mean()))
                for c in CLASSES:
                    m_ = sum(p[j] for j in range(5) if cls[j] == c)
                    acc[n, 'mass_' + c].append(float(m_))
                    acc[n, 'excess_' + c].append(float(m_ - sum(.2 for j in range(5) if cls[j] == c)))
                acc[n, 'within_E'].append(float((p[E] / p[E].sum()) @ q[E] - q[E].mean()) if len(E) >= 2 else np.nan)
        out = {'n_fork_points': len(ks), 'chosen_class_counts': dict(chosen),
               'class_value_q12': {c: mean_ci(vq[c], vt[c]) for c in CLASSES if vq[c]},
               'class_value_gold': {c: mean_ci(vqg[c], vtg[c]) for c in CLASSES if vqg[c]}, 'selectors': {}}
        for n in SELECTORS:
            out['selectors'][n] = {m: mean_ci(acc[n, m], tr) for m in ['G_u', 'within_E'] + [f'{p}_{c}' for p in ['mass', 'excess'] for c in CLASSES]}
            out['selectors'][n]['excess_R_seen_loto'] = loto(acc[n, 'excess_R_seen'], tr)
            if n not in ('uniform', 'policy_drafts'):
                m, c = n.split('_')
                ref = e2['table']['12'][f'{m}/{c}/soft/G_u/{layer}']['estimate']
                diff = abs(out['selectors'][n]['G_u']['estimate'] - ref)
                res['gate'][f'{n}/{layer}'] = diff
                if diff > 1e-9:
                    raise SystemExit(f'gate failed {n}/{layer}: {diff}')
        res['layers'][layer] = out

    # chosen-class distribution by layer x Fork over all classifiable Live points
    dist = {}
    for k, z in info.items():
        if z['traj'] not in episodes or k not in arms:
            continue
        steps = episodes[z['traj']]['steps']
        hist = [(st['executed_action'], st['observation_after']) for st in steps[:z['turn'] - 1]]
        c0 = arm_classes([arms[k][0]['action']], hist)[0]
        d = dist.setdefault(f"{z['layer']}/fork{int(z['het'])}", Counter())
        d[c0] += 1; d['_n'] += 1
    res['chosen_class_by_layer_fork'] = {k: dict(v) for k, v in sorted(dist.items())}

    # anchoring: L minus S change in probability on the executed (factual) action, by episode outcome
    anch = {}
    for m in ['7b', '32b']:
        for layer in ['I', 'U']:
            for won in [1, 0]:
                ks = [k for k, z in info.items() if z['layer'] == layer and int(episodes[z['traj']]['final_reward']) == won]
                anch[f'{m}/{layer}/won{won}'] = mean_ci([P[m, 'L', k][0] - P[m, 'S', k][0] for k in ks], [info[k]['traj'] for k in ks])
    res['anchoring_dP_executed_L_minus_S'] = anch
    for p in [Path(__file__), OUT / 'minipq.py']:
        track(p)
    res.update(script='exploratory_P4/post_e2/post_e2_mechanism.py', status='POST-HOC exploratory; not pre-registered',
               seed=SEED, bootstrap=NBOOT, inputs=inputs,
               commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    (OUT / 'post_e2_mechanism.json').write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    print('gate max diff', max(res['gate'].values()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--arms', type=Path, default=ROOT / 'exploratory_deep_discovery/exploratory_arms.parquet')
    args = ap.parse_args()
    selftest() if args.selftest else main(args.arms)
