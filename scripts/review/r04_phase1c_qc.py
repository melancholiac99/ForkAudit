"""QC of Phase 1C outputs (read-only, no features): invariants + cross-bank consistency + rollout health."""
import collections, glob, json, os
import numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def load(root):
    return {(os.path.basename(os.path.dirname(p)), int(os.path.basename(p)[1:3] if False else os.path.basename(p)[1:-5])): json.load(open(p))
            for p in glob.glob(f"{R}/data/{root}/s*/t*.json")}
B1, B2, B3 = load("replay"), load("replay_b1"), load("replay_b2")
print("points: bank1", len(B1), "bank2", len(B2), "bank3(partial)", len(B3))
# 1 rollouts
T = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{R}/data/rollout/s*.json")}
new = {k: v for k, v in T.items() if int(k[1:]) >= 50}; old = {k: v for k, v in T.items() if int(k[1:]) < 50}
for nm, g in [("old s000-049", old), ("new s050-099", new)]:
    st = [s for t in g.values() for s in t["steps"]]
    print(f"rollout {nm}: traj {len(g)} success {np.mean([t['won'] for t in g.values()]):.2f} steps {len(st)} "
          f"valid {np.mean([s['valid'] for s in st]):.4f} admissible_hit {np.mean([s['admissible_hit'] for s in st]):.3f} "
          f"fmt {dict(collections.Counter(s['fmt'] for s in st))}")
print("  new by type won:", {k: sum(t['won'] for t in new.values() if t['task_type'] == k) for k in sorted({t['task_type'] for t in new.values()})})
# 2 structural invariants + seed uniqueness + alternatives identical
for nm, B, bank in [("bank1", B1, 0), ("bank2", B2, 1), ("bank3", B3, 2)]:
    bad = collections.Counter(); seeds = collections.Counter(); div = 0
    for (tj, turn), r in B.items():
        if bank >= 1:
            b1 = B1[(tj, turn)]
            bad["alts_differ"] += r["alternatives"] != b1["alternatives"]
            bad["factual_differs"] += r["factual_action"] != b1["factual_action"]
            bad["not_cf"] += not b1["cf_defined"]
        else:
            bad["n_alts"] += r["cf_defined"] != (len(r["alternatives"]) == 4)
        if r["cf_defined"]:
            bad["n_out"] += len(r["factual_outcomes"]) != 3 or any(len(a) != 3 for a in r["alt_outcomes"])
        for c in r["continuations"]:
            div += c["refeed_div"]
            bad["len_vs_actions"] += c["len"] != turn - 1 + len(c["actions"])
            bad["won_after_cap"] += c["won"] and c["len"] > 50
            bad["lost_short"] += (not c["won"]) and c["len"] < 50
            if bank >= 1:
                for s in c["seeds"]: seeds[s] += 1
                bad["offset_range"] += any(not (bank * 10**11 <= s < (bank + 1) * 10**11) for s in c["seeds"])
                bad["first_action_wrong"] += c["first_action"] != (r["factual_action"] if c["role"] == 1 else r["alternatives"][c["role"] - 2])
    dup = sum(v > 1 for v in seeds.values())
    print(f"{nm}: violations {dict(bad)} refeed_div {div} duplicate_seeds {dup}")
# 3 cross-bank consistency on points present in all banks
common = [k for k in B3 if k in B2 and B1[k]["cf_defined"]]
def A(r): 
    f = np.mean(r["factual_outcomes"]); a = np.mean([o for x in r["alt_outcomes"] for o in x]); return f - a
def pool(r): return r["factual_outcomes"] + [o for x in r["alt_outcomes"] for o in x]
a1 = np.array([A(B1[k]) for k in common]); a2 = np.array([A(B2[k]) for k in common])
live1 = np.array([0 < sum(pool(B1[k])) < 15 for k in common]); live2 = np.array([0 < sum(pool(B2[k])) < 15 for k in common])
print(f"\ncross-bank on {len(common)} points present in banks 1-3:")
print(f"  live(bank1)={live1.mean():.3f} live(bank2)={live2.mean():.3f} both={np.mean(live1&live2):.3f}; "
      f"agreement of live flag {np.mean(live1==live2):.3f}")
nz = (a1 != 0) & (a2 != 0)
print(f"  A bank1 vs bank2: pearson {np.corrcoef(a1,a2)[0,1]:.3f}, sign agree (both nonzero, n={nz.sum()}) {np.mean(np.sign(a1[nz])==np.sign(a2[nz])):.3f}")
# 4 coverage of live set for later banks
allD = [k for k in B1 if B1[k]["cf_defined"] and k in B2]
L12 = [k for k in allD if 0 < sum(pool(B1[k]) + pool(B2[k])) < 30]
print(f"\nD (banks 1&2 present) {len(allD)}; live(b1+b2) = {len(L12)} ({len(L12)/max(1,len(allD)):.1%}); banks>=3 coverage so far {sum(k in B3 for k in L12)}")
print("  live points by trajectory (n traj with >=5):", sum(v >= 5 for v in collections.Counter(k[0] for k in L12).values()), "of", len({k[0] for k in L12}))
print("  no-cf rate bank1 s050-099:", np.mean([not B1[k]["cf_defined"] for k in B1 if int(k[0][1:]) >= 50]).round(3), " s000-049:", np.mean([not B1[k]["cf_defined"] for k in B1 if int(k[0][1:]) < 50]).round(3))
