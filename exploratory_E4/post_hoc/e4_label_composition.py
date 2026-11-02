"""E4 post-hoc (exploratory, NOT preregistered): composition of the preference labels of each arm.

Questions: which action classes (P4-X arm_classes: E_new / R_seen / R_recent / R_unseen / Other) the preferred and
rejected arms belong to; whether the rejected arm is the factual action; how often each arm's preference agrees with
the 3-run replay estimate at the same point; and where the pairs sit (replay-flat points vs points with a 3-run gap).
Written after E4's registered reading (R-L-HEURISTIC) was known; no E4 result or threshold depends on it.

Inputs: the extracted E4 archive (see results_archive/README.txt): <root>/points, <root>/replay, <root>/pairs.
Usage (from the repository root): python exploratory_E4/post_hoc/e4_label_composition.py /tmp/e4_results/data
"""
import glob, json, os, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, "exploratory_P4/post_e2")
from post_e2_mechanism import arm_classes
D = sys.argv[1]
pts = {}
for f in glob.glob(os.path.join(D, "points", "*.json")):
    r = json.load(open(f))
    if not r["cf_defined"]:
        continue
    acts = [r["factual"]] + r["alternatives"]
    hist = [(h["executed_action"], h["observation_after"]) for h in r["history"]]
    rp = os.path.join(D, "replay", f"s{r['seed']}", f"t{r['turn']:02d}.json")
    q = None
    if os.path.exists(rp):
        x = json.load(open(rp))
        q = [np.mean(x["factual_outcomes"])] + [np.mean(o) for o in x["alt_outcomes"]]
    pts[r["point_id"]] = dict(cls=arm_classes(acts, hist), q=q, seed=r["seed"])
print("cf-defined points", len(pts), "with replay", sum(v["q"] is not None for v in pts.values()))
fc = Counter(v["cls"][0] for v in pts.values())
print("factual class:", {k: f"{v/len(pts):.2f}" for k, v in fc.items()})
anyE = np.mean([("E_new" in v["cls"][1:]) for v in pts.values()])
print(f"points with >=1 E_new alternative: {anyE:.2f}")
# mean replay value by class (all points with replay, arms pooled)
byc = defaultdict(list)
for v in pts.values():
    if v["q"] is None: continue
    for c, qq in zip(v["cls"], v["q"]):
        byc[c].append(qq)
print("mean 3-run q by class:", {c: f"{np.mean(x):.3f} (n={len(x)})" for c, x in byc.items()})
rows = []
for arm in ["A", "Aprime", "H", "D", "Rnd", "Bfull", "Bmatch", "Cmatch"]:
    P = [json.loads(l) for l in open(os.path.join(D, "pairs", f"{arm}.jsonl"))]
    n = len(P)
    rejf = np.mean([p["rejected_arm"] == 0 for p in P])
    pc = [pts[p["point_id"]]["cls"][p["preferred"]] for p in P]
    rc = [pts[p["point_id"]]["cls"][p["rejected_arm"]] for p in P]
    pE = np.mean([c == "E_new" for c in pc])
    rE = np.mean([c == "E_new" for c in rc])
    eVsNon = np.mean([(a == "E_new") and (b != "E_new") for a, b in zip(pc, rc)])
    nonVsE = np.mean([(a != "E_new") and (b == "E_new") for a, b in zip(pc, rc)])
    agree = []
    for p in P:
        q = pts[p["point_id"]]["q"]
        if q is None: continue
        d = q[p["preferred"]] - q[p["rejected_arm"]]
        agree.append(np.sign(d))
    agree = np.array(agree)
    print(f"{arm:7s} n={n:4d} rej=factual {rejf:.2f} | pref E_new {pE:.2f} rej E_new {rE:.2f} | E>nonE {eVsNon:.2f} nonE>E {nonVsE:.2f} "
          f"| replay(3-run): pref>rej {np.mean(agree>0):.2f} tie {np.mean(agree==0):.2f} pref<rej {np.mean(agree<0):.2f} (n={len(agree)})")
    print("         preferred classes", dict(Counter(pc)), " rejected classes", dict(Counter(rc)))

print("\n--- where the pairs sit (3-run replay at the point) ---")
def kind(q):
    if q is None: return "noreplay"
    if max(q) == min(q):
        return "flat0" if max(q) == 0 else ("flat1" if min(q) == 1 else "flat_mid")
    return "diff"
allk = Counter(kind(v["q"]) for v in pts.values())
print("all points:", {k: f"{v/len(pts):.2f}" for k, v in allk.items()})
for arm in ["A", "Aprime", "H", "D", "Rnd", "Bfull"]:
    P = [json.loads(l) for l in open(os.path.join(D, "pairs", f"{arm}.jsonl"))]
    k = Counter(kind(pts[p["point_id"]]["q"]) for p in P)
    print(f"{arm:7s}", {kk: f"{v/len(P):.2f}" for kk, v in sorted(k.items())})
# turn position of pairs
for arm in ["A", "D", "Bfull"]:
    P = [json.loads(l) for l in open(os.path.join(D, "pairs", f"{arm}.jsonl"))]
    t = [int(p["point_id"].split("_t")[1]) for p in P]
    print(arm, "turn quartiles", np.percentile(t, [25, 50, 75]).tolist())

print("\n--- within-point value of E_new vs other arms (3-run q), cluster bootstrap by trajectory ---")
d, cl = [], []
for pid, v in pts.items():
    q, c = v["q"], v["cls"]
    e = [qq for qq, cc in zip(q, c) if cc == "E_new"]
    o = [qq for qq, cc in zip(q, c) if cc != "E_new"]
    if e and o:
        d.append(np.mean(e) - np.mean(o)); cl.append(v["seed"])
d, cl = np.array(d), np.array(cl)
u, inv = np.unique(cl, return_inverse=True)
rng = np.random.default_rng(1)
W = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=5000).astype(float)
bs = (W @ np.bincount(inv, d, len(u))) / (W @ np.bincount(inv, None, len(u)))
print(f"E_new - other: {100*d.mean():+.2f} pp [{100*np.percentile(bs,2.5):+.2f}, {100*np.percentile(bs,97.5):+.2f}] n={len(d)} traj={len(u)}")
nz = np.array([max(pts[p]['q']) > min(pts[p]['q']) for p in pts if pts[p]['q'] and ('E_new' in pts[p]['cls']) and any(c != 'E_new' for c in pts[p]['cls'])])
print("share of those points with non-flat q:", nz.mean())
