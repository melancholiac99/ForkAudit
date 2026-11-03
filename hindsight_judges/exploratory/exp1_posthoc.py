"""Post-hoc analysis of hindsight experiment 1 (EXPLORATORY; not part of the frozen protocol v2).

Usage: python hindsight_judges/exploratory/exp1_posthoc.py <repo root> <exploratory_arms.parquet> <out.txt>
  - repo root: research/hindsight-judges checkout containing hindsight_judges/results/exp1_p4/{32b,7b}
  - arms: deep-discovery exploratory_arms.parquet (SHA must match exploratory_P4/e2.json inputs)

Notation per point: f = executed arm (index 0); b = q12-best alternative (pointmeta swap_arm);
l_c(a) = sequence log-prob of candidate a under condition c; lo_c = l_c(f) - l_c(b) (nats).
Conditions: 0 none; E 'executed = f'; X 'executed = b'; PE 'arbitrary marker on f'; PX 'marker on b'.
Values use held-out q3456 only. CIs: 95% trajectory-cluster bootstrap (2000 draws).
Section 7 compares with ORIGINAL P4 E2/E2-M scores from another GPU run: order of magnitude only.
"""
import sys, os, json, hashlib
from collections import defaultdict
import numpy as np

REPO, ARMS, OUT = sys.argv[1:4]
R = os.path.join(REPO, 'hindsight_judges/results/exp1_p4')
P4 = os.path.join(REPO, 'exploratory_P4')
sys.path.insert(0, os.path.join(P4, 'post_e2'))
import minipq

want = json.load(open(os.path.join(P4, 'e2.json')))['inputs']['exploratory_deep_discovery/exploratory_arms.parquet']
assert hashlib.sha256(open(ARMS, 'rb').read()).hexdigest() == want, 'arms SHA mismatch'

lines = []
def pr(*x):
    s = ' '.join(str(y) for y in x); print(s); lines.append(s)

meta = {m['point_id']: m for m in json.load(open(os.path.join(R, 'pointmeta.json')))}
display = {r['point_id']: r['display_order'] for r in json.load(open(os.path.join(P4, 'e2_point_audit.json')))}
keys = sorted(meta)
num = lambda x: float('nan') if x is None else float(x)
a = minipq.read(ARMS)
arm = defaultdict(dict)
for i in range(len(a['traj_id'])):
    k = f"{a['traj_id'][i]}_t{int(a['turn'][i]):02d}"
    arm[k][int(a['j'][i])] = dict(q=num(a['q_3456'][i]), tm=bool(a['target_mention'][i]))
trajs = sorted({meta[k]['traj'] for k in keys}); tix = {t: i for i, t in enumerate(trajs)}
W = np.random.default_rng(31102026).multinomial(len(trajs), np.full(len(trajs), 1 / len(trajs)), size=2000).astype(float)

def ci(vals, ks, scale=1.0):
    v = np.asarray(vals, float) * scale; g = np.array([tix[meta[k]['traj']] for k in ks]); ok = np.isfinite(v); v, g = v[ok], g[ok]
    if len(v) < 3: return f'n={len(v)}'
    sv, sc = np.bincount(g, v, len(trajs)), np.bincount(g, None, len(trajs))
    with np.errstate(all='ignore'): d = (W @ sv) / (W @ sc)
    d = d[np.isfinite(d)]
    return f'{v.mean():+6.2f} [{np.percentile(d,2.5):+6.2f}, {np.percentile(d,97.5):+6.2f}] n={len(v)}'

def slope(x, y, ks, seed=5, nb=1000):
    g = np.array([tix[meta[k]['traj']] for k in ks]); rng = np.random.default_rng(seed); bs = []
    for _ in range(nb):
        w = rng.integers(0, len(trajs), len(trajs)); idx = np.concatenate([np.where(g == t)[0] for t in w])
        bs.append(np.polyfit(x[idx], y[idx], 1)[0])
    return f'{np.polyfit(x, y, 1)[0]:+.2f} [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]'

q = lambda k: np.array([arm[k][j]['q'] for j in range(5)])
B = lambda k: meta[k]['swap_arm']
groups_ev = [('F shows f: no target; b: target', [k for k in keys if not arm[k][0]['tm'] and arm[k][B(k)]['tm']]),
             ('F shows f finds target', [k for k in keys if arm[k][0]['tm']]),
             ('neither shows target', [k for k in keys if not arm[k][0]['tm'] and not arm[k][B(k)]['tm']])]

for model in ['32b', '7b']:
    Lg, P = {}, {}
    for line in open(os.path.join(R, model, 'scores.jsonl')):
        r = json.loads(line); Lg[r['condition'], r['point_id']] = np.array(r['sequence_logp']); P[r['condition'], r['point_id']] = np.array(r['candidate_prob'])
    pr(f'\n################ judge {model} ################')
    for ctx in ['S', 'F']:
        L = lambda c, k: Lg[ctx + c, k]
        lo = lambda c, k: L(c, k)[0] - L(c, k)[B(k)]
        pr(f'\n== {ctx} context ==')
        pr('  1. [check vs REPORT] identity switch', ci([(lo('E', k) - lo('X', k)) / 2 for k in keys], keys),
           '| placebo switch', ci([(lo('PE', k) - lo('PX', k)) / 2 for k in keys], keys))
        pr('  2. mean p(f) / p(b) by condition: ' + '  '.join(
            f"{c}: {np.mean([P[ctx+c,k][0] for k in keys]):.3f}/{np.mean([P[ctx+c,k][B(k)] for k in keys]):.3f}" for c in ['0', 'E', 'X', 'PE', 'PX']))
        cf = np.mean([L('0', k)[0] > -0.1 for k in keys]); cb = np.mean([L('0', k)[B(k)] > -0.1 for k in keys])
        pr(f'     ceiling share (no-label seq-logp > -0.1): f {cf:.2f}, b {cb:.2f}')
        def pull(c, k, j):
            d = L(c, k) - L('0', k); o = [i for i in range(5) if i not in (0, B(k))]; return d[j] - d[o].mean()
        hf = [k for k in keys if L('0', k)[0] < -1]; hb = [k for k in keys if L('0', k)[B(k)] < -1]
        pr('  3. pull on pointed arm vs 3 untouched arms, only where pointed arm has headroom (no-label logp < -1), nat:')
        pr('     E->f', ci([pull('E', k, 0) for k in hf], hf), '| X->b', ci([pull('X', k, B(k)) for k in hb], hb))
        pr('     PE->f', ci([pull('PE', k, 0) for k in hf], hf), '| PX->b', ci([pull('PX', k, B(k)) for k in hb], hb))
        both = [k for k in keys if L('0', k)[0] < -1 and L('0', k)[B(k)] < -1]
        idn = np.array([(lo('E', k) - lo('X', k)) / 2 for k in both]); pln = np.array([(lo('PE', k) - lo('PX', k)) / 2 for k in both])
        pr(f'     both arms with headroom: identity {ci(idn, both)} | placebo {ci(pln, both)} | ratio {pln.mean()/idn.mean():.1f}x')
        pr('  4. where the effect lands, by display offset from labeled position N (mean logp change, centred, nat):')
        for c, sk in [('E', 'real_display_number'), ('X', 'swap_display_number'), ('PE', 'real_display_number'), ('PX', 'swap_display_number')]:
            acc = {o: [] for o in (-1, 0, 1)}
            for k in keys:
                d = L(c, k) - L('0', k); d = d - d.mean(); N = meta[k][sk]
                for o in acc:
                    if 1 <= N + o <= 5: acc[o].append(d[display[k][N + o - 1]])
            pr(f'     {c:3s} ' + '  '.join(f'N{o:+d}: {np.mean(v):+.2f}' for o, v in acc.items()))
        adv = np.array([q(k)[0] - q(k)[B(k)] for k in keys])
        pr('  5. quality dependence, slope on q3456(f)-q3456(b) [nat per unit]: identity',
           slope(adv, np.array([(lo('E', k) - lo('X', k)) / 2 for k in keys]), keys),
           '| placebo', slope(adv, np.array([(lo('PE', k) - lo('PX', k)) / 2 for k in keys]), keys))
        pr('  6. evidence dependence (meaningful in F, where every arm\'s one-step observation is shown):')
        for name, kk in groups_ev:
            pr(f'     {name:33s} identity {ci([(lo("E",k)-lo("X",k))/2 for k in kk], kk)} | placebo {ci([(lo("PE",k)-lo("PX",k))/2 for k in kk], kk)}'
               f" | argmax=f: none {np.mean([P[ctx+'0',k].argmax()==0 for k in kk]):.2f}, E {np.mean([P[ctx+'E',k].argmax()==0 for k in kk]):.2f}, PE {np.mean([P[ctx+'PE',k].argmax()==0 for k in kk]):.2f}")
        V = lambda c, k: P[ctx + c, k] @ q(k)
        pr('  7. decision value vs no label (pp, q3456): ' + ' | '.join(f'{c} {ci([V(c,k)-V("0",k) for k in keys], keys, 100)}' for c in ['E', 'X', 'PE', 'PX']))
        ke = [(k, 0, 'E') for k in keys if P[ctx+'0', k][0] > .9] + [(k, B(k), 'X') for k in keys if P[ctx+'0', k][B(k)] > .9]
        km = [(k, j, 'P' + c) for k, j, c in ke]
        pr(f'  8. pointed arm already chosen (p>0.9) and label drops it below 0.5: executed-label {sum(P[ctx+c,k][j]<.5 for k,j,c in ke)}/{len(ke)}, '
           f'marker {sum(P[ctx+c,k][j]<.5 for k,j,c in km)}/{len(km)}')

pr('\n################ scale reference: original P4 E2/E2-M full-log anchoring (other GPU run; order of magnitude only) ################')
for model in ['32b', '7b']:
    old = {}
    for f in [f'e2_{model}', f'e2m_{model}']:
        for line in open(os.path.join(P4, f, 'scores.jsonl')):
            r = json.loads(line); old[r['condition'], r['point_id']] = np.array(r['sequence_logp'])
    lo_old = lambda c, k: old[c, k][0] - old[c, k][B(k)]
    pr(f'  {model}, same 386 Forks, f:b log-odds: L - M', ci([lo_old('L', k) - lo_old('M', k) for k in keys], keys),
       '| L - S', ci([lo_old('L', k) - lo_old('S', k) for k in keys], keys))
    rel = lambda c, k: old[c, k][0] - old[c, k][1:].mean()
    y = np.array([rel('L', k) - rel('M', k) for k in keys]); x = np.array([q(k)[0] - q(k)[1:].mean() for k in keys])
    gi = np.array([tix[meta[k]['traj']] for k in keys]); tm = np.array([y[gi == i].mean() for i in gi])
    pr(f'    executed minus mean alternative, L - M: mean {y.mean():+.2f} sd {y.std():.2f}; slope on held-out advantage {slope(x, y, keys)}; '
       f'within-trajectory share of variance {1-np.var(tm)/np.var(y):.2f}')
open(OUT, 'w').write('\n'.join(lines) + '\n')
