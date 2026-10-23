"""Independent review of Phase 1 replay (target-structure only; no feature-vs-A_replay associations, §9.2)."""
import collections, glob, json, os
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rep = {}
for p in glob.glob(f"{R}/data/replay/s*/t*.json"):
    r = json.load(open(p)); rep[(r["traj_id"], r["turn"])] = r
df = pd.read_parquet(f"{R}/data/decision_points.parquet")
roll = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(f"{R}/data/rollout/s*.json")}

# 1. independent A_replay / sigma / crit recomputation
bad = 0
for _, x in df.iterrows():
    r = rep[(x.traj_id, x.turn)]
    f = np.array(r["factual_outcomes"])
    if r["cf_defined"]:
        a = np.array(r["alt_outcomes"]).ravel()
        A = f.mean() - a.mean()
        bad += not np.isclose(A, x.A_replay)
        bad += not np.isclose(np.concatenate([f, a]).std(), x.crit_sd)
    else:
        bad += not np.isnan(x.A_replay)
    bad += not np.isclose(f.std(), x.sigma_floor)
print("1. recompute mismatches:", bad, "of", len(df))

D = df[df.cf_defined]
piv = D.A_replay.ne(0)
print(f"2. total {len(df)} |D| {len(D)} ({len(D)/len(df):.1%}) pivotal {piv.sum()} ({piv.mean():.1%}) "
      f"sigma0(D) {D.sigma_floor.eq(0).mean():.1%}")

# 3. divisibility classes (CWGT App. E): constant pool / forced nonzero / genuinely stochastic
cls = collections.Counter()
for _, x in D.iterrows():
    f = np.array(x.factual_outcomes); a = np.concatenate(list(x.alt_outcomes))
    pool = np.concatenate([f, a])
    if pool.min() == pool.max(): cls["constant_pool"] += 1
    elif 5 * f.sum() != pool.sum(): cls["forced_nonzero"] += 1
    else: cls["stochastic_zero"] += 1
print("3. divisibility classes:", dict(cls), " CWGT: 1204/509/55 of 1768 (68.1%/28.8%/3.1%)")
print("   constant-pool split:", collections.Counter(int(np.concatenate([x.factual_outcomes, *x.alt_outcomes])[0])
      for _, x in D.iterrows() if np.ptp(np.concatenate([x.factual_outcomes, *x.alt_outcomes])) == 0))

# 4. outcome base rates
fo = np.concatenate(D.factual_outcomes.values); ao = np.concatenate([np.concatenate(list(v)) for v in D.alt_outcomes])
print(f"4. continuation success: factual {fo.mean():.3f} alt {ao.mean():.3f}; rollout success "
      f"{np.mean([t['won'] for t in roll.values()]):.2f}")

# 5. per-trajectory heterogeneity (traj-level, by rollout outcome)
g = D.assign(piv=piv).groupby("traj_id").agg(n=("piv", "size"), piv=("piv", "mean"))
g["won"] = [roll[t]["won"] for t in g.index]
print("5. pivotal frac by rollout outcome:", g.groupby("won").apply(lambda s: f"{(s.piv*s.n).sum()/s.n.sum():.1%} n={s.n.sum()} traj={len(s)}").to_dict())
print("   per-traj pivotal frac quantiles:", np.round(np.percentile(g.piv, [0, 25, 50, 75, 100]), 2))
print("   pilot s000-s004 pivotal:", f"{piv[D.traj_id.isin([f's00{i}' for i in range(5)])].mean():.1%}")

# 6. exclusions
E = df[~df.cf_defined]
nalt = [len(rep[(x.traj_id, x.turn)]["alternatives"]) for _, x in E.iterrows()]
print("6. excluded", len(E), "by #alts found:", dict(collections.Counter(nalt)))
print("   excluded at turn 1:", int((E.turn == 1).sum()), "of", df.turn.eq(1).sum(), "trajectories")
dr = [r for r in rep.values()]
print("   draft fmt:", dict(collections.Counter(d["fmt"] for r in dr for d in r["drafts"])))
print("   drafts valid&admissible:", np.mean([d["valid"] and d["admissible"] for r in dr for d in r["drafts"]]).round(4))
print("   'help' used as factual/alt:", sum(r["factual_action"] == "help" for r in dr), sum("help" in r["alternatives"] for r in dr))
print("   n_draws defined pts pctl:", np.percentile([r["n_draws"] for r in dr if r["cf_defined"]], [50, 90, 99, 100]))

# 7. continuation sanity
L = [c for r in dr for c in r["continuations"]]
print("7. continuations", len(L), "won", np.mean([c["won"] for c in L]).round(3), "hit cap(50) & lost",
      np.mean([(c["len"] >= 50) and not c["won"] for c in L]).round(3), "refeed div", sum(c["refeed_div"] for c in L))
bad_len = sum(1 for r in dr for c in r["continuations"] if c["len"] != r["turn"] - 1 + len(c["actions"]))
print("   len == (turn-1)+#actions violations:", bad_len)
won_early = sum(1 for r in dr for c in r["continuations"] if c["won"] and c["len"] > 50)
print("   won beyond cap:", won_early)
# factual action at t completes task?
fin = [(r["traj_id"], r["turn"]) for r in dr if roll[r["traj_id"]]["steps"][r["turn"]-1].get("won_after")]
print("   points where factual action wins immediately:", len(fin))
