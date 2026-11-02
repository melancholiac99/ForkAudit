"""Post-hoc reanalysis (exploratory, NOT pre-registered): does the full-log judge (L) raise the executed
action regardless of its quality, and what does that cost?  Zero new GPU work; reads existing scores.

Usage:
  python anchoring_reanalysis.py <ForkAudit repo root> <exploratory_arms.parquet> <E4 archive extract root> <out.txt>
  - exploratory_arms.parquet: from branch research/deep-discovery-2026-10-27 (same SHA256 as e2.json inputs)
  - E4 extract root: directory containing data/replay/ from exploratory_E4/results_archive

Value is always evaluated on replay libraries independent of the ones used for stratification:
stratify on q_12 (libs 1-2), evaluate on q_3456 (libs 3-6); Gold subset evaluates on q_gold.
CIs: 95%, trajectory-cluster bootstrap.
"""
import sys, json, gzip, glob, os
from collections import defaultdict
import numpy as np

REPO, ARMS, E4X, OUT = sys.argv[1:5]
sys.path.insert(0, os.path.join(REPO, 'exploratory_P4/post_e2'))
import minipq  # bundled fallback parquet reader

lines = []
def pr(*x):
    s = ' '.join(str(y) for y in x); print(s); lines.append(s)

def make_ci(trajs_of, seed=2112026, nboot=4000):
    trajs = sorted(set(trajs_of.values())); tix = {t: i for i, t in enumerate(trajs)}
    W = np.random.default_rng(seed).multinomial(len(trajs), np.full(len(trajs), 1 / len(trajs)), size=nboot).astype(float)
    def ci(vals, ks):
        v = np.asarray(vals, float); g = np.asarray([tix[trajs_of[k]] for k in ks]); ok = np.isfinite(v); v, g = v[ok], g[ok]
        if len(v) < 3: return f'n={len(v)}'
        sv, sc = np.bincount(g, v, len(trajs)), np.bincount(g, None, len(trajs))
        with np.errstate(all='ignore'): d = (W @ sv) / (W @ sc)
        d = d[np.isfinite(d)]
        return f'{100*v.mean():+6.1f} [{100*np.percentile(d,2.5):+6.1f}, {100*np.percentile(d,97.5):+6.1f}]  n={len(v)}, traj={len(set(g.tolist()))}'
    return ci

# ---------------- P4 E2 / E2-M ----------------
s = minipq.read(os.path.join(REPO, 'exploratory_P4/strata.parquet')); a = minipq.read(ARMS)
key = lambda t, u: f'{t}_t{int(u):02d}'
num = lambda x: float('nan') if x is None else float(x)
info = {key(s['traj_id'][i], s['turn'][i]): dict(traj=s['traj_id'][i], turn=int(s['turn'][i]), layer=s['stratum'][i], fork=bool(s['het'][i]))
        for i in range(len(s['traj_id'])) if s['live'][i] and s['stratum'][i] in ('I', 'U')}
arms = defaultdict(dict)
for i in range(len(a['traj_id'])):
    arms[key(a['traj_id'][i], a['turn'][i])][int(a['j'][i])] = dict(
        action=a['action'][i], obs=a['obs'][i], tm=bool(a['target_mention'][i]),
        q12=num(a['q_12'][i]), q3456=num(a['q_3456'][i]), qg=num(a['q_gold'][i]))
episodes, dec = {}, json.JSONDecoder()
for line in gzip.open(os.path.join(REPO, 'exploratory_P4/e2_requests.jsonl.gz'), 'rt'):
    r = json.loads(line)
    if r['condition'] == 'L' and r['traj_id'] not in episodes:
        t = r['text']; episodes[r['traj_id']] = dec.raw_decode(t[t.find('{"task":'):])[0]
P = {}
for m in ['7b', '32b']:
    for f in [f'e2_{m}', f'e2m_{m}']:
        for line in open(os.path.join(REPO, f'exploratory_P4/{f}/scores.jsonl')):
            r = json.loads(line); P[m, r['condition'], r['point_id']] = np.array(r['candidate_prob'])
keys = sorted(k for k in info if len(arms[k]) == 5)
ci = make_ci({k: info[k]['traj'] for k in keys})
q = lambda k, c: np.array([arms[k][j][c] for j in range(5)])
gap12 = lambda k: np.nanmax(q(k, 'q12')[1:]) - q(k, 'q12')[0]
strata = [('fact_best (gap<=0)', lambda k: gap12(k) <= 0), ('fact_worse (0<gap<1/3)', lambda k: 0 < gap12(k) < 1/3),
          ('fact_much_worse (gap>=1/3)', lambda k: gap12(k) >= 1/3)]

pr('#### P4: candidate index 0 = executed action. M = own full history before t (no future, no CoT, no reward); L = M + CoT + step t + future + reward.')
for m in ['32b', '7b']:
    for pop in ['FORK', 'ALL']:
        pr(f'\n== judge {m} | I/U Live {pop} ==')
        base = [k for k in keys if pop == 'ALL' or info[k]['fork']]
        for name, cond in strata:
            kk = [k for k in base if cond(k)]
            pr(f'  {name:28s} dP(exec) L-M {ci([P[m,"L",k][0]-P[m,"M",k][0] for k in kk], kk)}')
            pr(f'  {"":28s} dValue  L-M {ci([(P[m,"L",k]-P[m,"M",k]) @ q(k,"q3456") for k in kk], kk)}  (q3456)')
        kg = [k for k in base if np.isfinite(q(k, 'qg')).all() and gap12(k) > 0]
        pr(f'  Gold, fact worse on q12       dValue  L-M {ci([(P[m,"L",k]-P[m,"M",k]) @ q(k,"qg") for k in kg], kg)}  (q_gold)')
pr('\n== 32B, Fork points, by layer ==')
for L in ['I', 'U']:
    for name, cond in [strata[0], strata[2]]:
        kk = [k for k in keys if info[k]['fork'] and info[k]['layer'] == L and cond(k)]
        pr(f'  {L} {name:28s} dP(exec) L-M {ci([P["32b","L",k][0]-P["32b","M",k][0] for k in kk], kk)} | dValue L-M {ci([(P["32b","L",k]-P["32b","M",k]) @ q(k,"q3456") for k in kk], kk)}')

pr('\n== Does the boost depend on what the log shows about the executed action? (I/U Live, all) ==')
def disc(k, strict):
    e = arms[k][0]
    return (not e['tm']) and any(arms[k][j]['tm'] for j in range(1, 5)) and (not strict or ('you see' in e['obs'].lower() and gap12(k) >= 1/3))
for m in ['32b', '7b']:
    for name, kk in [('executed obs reveals target', [k for k in keys if arms[k][0]['tm']]),
                     ('executed obs no target, an alt would reveal it', [k for k in keys if disc(k, False)]),
                     ('  strict: obs lists contents w/o target & gap>=1/3', [k for k in keys if disc(k, True)])]:
        pr(f'  {m} {name:50s} pM={np.mean([P[m,"M",k][0] for k in kk]):.2f} pL={np.mean([P[m,"L",k][0] for k in kk]):.2f} '
           f'pF={np.mean([P[m,"F",k][0] for k in kk]):.2f} | argmax=exec M {np.mean([P[m,"M",k].argmax()==0 for k in kk]):.2f} '
           f'L {np.mean([P[m,"L",k].argmax()==0 for k in kk]):.2f} F {np.mean([P[m,"F",k].argmax()==0 for k in kk]):.2f} | n={len(kk)}')

pr('\n== Fork points where executed is much worse (q12 gap>=1/3): share where judge argmax = executed ==')
for m in ['32b', '7b']:
    kk = [k for k in keys if info[k]['fork'] and gap12(k) >= 1/3]
    pr('  ' + m + '  ' + '  |  '.join(f'{c}: {ci([float(P[m,c,k].argmax()==0) for k in kk], kk)}' for c in ['S', 'M', 'L', 'F']))

pr('\n== Copy channel: dP(alt) L-M for alternatives executed LATER in the logged future vs not (I/U Live, all) ==')
for m in ['32b', '7b']:
    fut, fk, no, nk = [], [], [], []
    for k in keys:
        later = {st['executed_action'].strip().lower() for st in episodes[info[k]['traj']]['steps'][info[k]['turn']:]}
        for j in range(1, 5):
            d = P[m, 'L', k][j] - P[m, 'M', k][j]
            if arms[k][j]['action'].strip().lower() in later: fut.append(d); fk.append(k)
            else: no.append(d); nk.append(k)
    pr(f'  {m} alt appears later: {ci(fut, fk)} | alt not later: {ci(no, nk)}')

# ---------------- E4 (training set, 6,016 points, 3 replays/arm) ----------------
pr('\n#### E4: same 32B judge, L vs F, 3-run replay per arm (noisy). H trains on "L-argmax > executed", A on "F-argmax > executed".')
J = {}
for c in ['L', 'F']:
    for line in open(os.path.join(REPO, f'exploratory_E4/data/judge/32b_{c}/scores.jsonl')):
        r = json.loads(line); J[c, r['point_id']] = np.array(r['candidate_prob'])
rows = {}
for f in glob.glob(os.path.join(E4X, 'data/replay/*/*.json')):
    x = json.load(open(f))
    if not x.get('cf_defined') or x['alt_outcomes'] is None: continue
    pid = f"{x['traj_id']}_t{x['turn']:02d}"
    if ('L', pid) in J:
        rows[pid] = dict(traj=x['traj_id'], q=np.array([np.mean(x['factual_outcomes'])] + [np.mean(o) for o in x['alt_outcomes']]))
ci4 = make_ci({k: v['traj'] for k, v in rows.items()}, seed=3, nboot=2000)
g4 = lambda k: rows[k]['q'][1:].max() - rows[k]['q'][0]
for name, kk in [('all', list(rows)), ('flat (all arms 0/3)', [k for k in rows if rows[k]['q'].max() == 0]),
                 ('executed best, non-flat', [k for k in rows if rows[k]['q'].max() > 0 and g4(k) <= 0]),
                 ('executed worse by >=1/3', [k for k in rows if g4(k) >= 1/3]), ('executed worse by >=2/3', [k for k in rows if g4(k) >= 2/3])]:
    pr(f'  argmax=executed  {name:26s} L: {ci4([J["L",k].argmax()==0 for k in kk], kk)} | F: {ci4([J["F",k].argmax()==0 for k in kk], kk)}')
kk = [k for k in rows if g4(k) >= 1/3]
ok = lambda c, k: J[c, k].argmax() != 0 and rows[k]['q'][J[c, k].argmax()] > rows[k]['q'][0]
pr(f'  executed worse by >=1/3: judge yields a pair whose preferred arm beats executed on replay  L: {ci4([ok("L",k) for k in kk], kk)} | F: {ci4([ok("F",k) for k in kk], kk)}')
open(OUT, 'w').write('\n'.join(lines) + '\n')
