"""Gold set (FEATURES_FROZEN §1; CLAUDE.md §6.3a; FREEZE ADDENDUM §1 Flag 2): finish audit + replication rates -> D-045.

Audit (D-028 style): 300 points present; 5 arms x 20 continuations each; alternatives and factual equal bank 1;
every seed inside bank index 6's range and no duplicate seed; re-feed divergence 0; outcomes consistent with won.
Replication (independent 20 per arm only): among G-candidate points flagged by the 12-run rule, the fraction flagged
again by the same rule on the 20 new runs per arm:
  Credit-pivotal: factual 20 vs pool 80, Newcombe 90% excludes 0
  het (Fork)    : five-arm G, 10,000 permutations, alpha 0.10
  cap           : both
plus the null replication rate (the same rules after permuting each point's 100 new outcomes across arms) and the
merged-32 version (selection-biased, reported). Switch: cap replication < 0.50 -> B3 report-only.
G-unbiased: regime fractions on 32 runs per arm and the 12-run rules' recall against the independent-20 flags.
Output eval/gold.json.
"""
import importlib
import json
import os

import numpy as np
import pandas as pd

import common

R = importlib.import_module("15_regimes")
H = importlib.import_module("18_het_fork")
N_PERM = 10_000
SEED_OFF = 6 * 10 ** 11


def flags(f, a, n_perm, rng):
    """f (n_f,), a (4, n_a) 0/1 outcomes -> (credit, het, p_het)."""
    nf, na = len(f), a.shape[1]
    lo, hi = R.newcombe(f.sum(), nf, a.sum(), 4 * na)
    credit = bool(lo > 0 or hi < 0)
    if nf != na:
        return credit, None, None
    c = np.concatenate([[f.sum()], a.sum(1)]).astype(float)
    G0 = H.g_stat(c[None, :], n=nf)[0]
    o = np.concatenate([f, a.ravel()])
    perm = rng.permuted(np.tile(o, (n_perm, 1)), axis=1).reshape(n_perm, 5, nf).sum(2).astype(float)
    p = (1 + np.sum(H.g_stat(perm, n=nf) >= G0 - 1e-9)) / (n_perm + 1)
    return credit, bool(p <= 0.10), float(p)


def main():
    gp = json.load(open(os.path.join(common.ROOT, "eval", "gold_points.json")))
    lab = pd.read_parquet(os.path.join(common.ROOT, "data", "labels_final.parquet")).set_index(["traj_id", "turn"])
    base = {(r["traj_id"], r["turn"]): r for r in R.load().to_dict("records")}
    audit = {"missing": [], "bad_counts": 0, "alt_mismatch": 0, "seed_out_of_range": 0, "dup_seeds": 0,
             "refeed_div": 0, "outcome_won_mismatch": 0}
    seen, rows = set(), []
    for t, u in gp["points"]:
        p = os.path.join(common.ROOT, "data", "replay_gold", t, f"t{u:02d}.json")
        if not os.path.exists(p):
            audit["missing"].append([t, u])
            continue
        d = json.load(open(p))
        b1 = json.load(open(os.path.join(common.ROOT, "data", "replay", t, f"t{u:02d}.json")))
        f = np.array(d["factual_outcomes"])
        a = np.array(d["alt_outcomes"])
        audit["bad_counts"] += int(len(f) != 20 or a.shape != (4, 20))
        audit["alt_mismatch"] += int(d["alternatives"] != b1["alternatives"] or d["factual_action"] != b1["factual_action"])
        for c in d["continuations"]:
            audit["refeed_div"] += int(c.get("refeed_div", 0))
            for s in c["seeds"]:
                audit["seed_out_of_range"] += int(not (SEED_OFF <= s < SEED_OFF + 10 ** 11))
                audit["dup_seeds"] += int(s in seen)
                seen.add(s)
        won = {(c["role"], c["rep"]): int(c["won"]) for c in d["continuations"]}
        audit["outcome_won_mismatch"] += int(sorted(won.get((1, r), -1) for r in range(20)) != sorted(f.tolist()))
        b = base[(t, u)]
        f12 = np.concatenate([b[f"f{k}"] for k in (3, 4, 5, 6)])
        a12 = np.concatenate([b[f"a{k}"] for k in (3, 4, 5, 6)], axis=1)
        rows.append({"traj_id": t, "turn": u, "f20": f, "a20": a, "f32": np.concatenate([f12, f]),
                     "a32": np.concatenate([a12, a], axis=1),
                     "layer": "candidate" if [t, u] in gp["layers"]["candidate"] else "unbiased"})
    rng = np.random.default_rng(45)
    for r in rows:
        r["cp20"], r["het20"], r["p20"] = flags(r["f20"], r["a20"], N_PERM, rng)
        r["cp32"], r["het32"], _ = flags(r["f32"], r["a32"], N_PERM, rng)
        # null replication: permute the 100 new outcomes across arms (one permutation set; a permuted data set's het
        # p-value is its rank in the same permutation distribution, as in 18_het_fork)
        o = np.concatenate([r["f20"], r["a20"].ravel()])
        cp_ = rng.permuted(np.tile(o, (N_PERM, 1)), axis=1).reshape(N_PERM, 5, 20).sum(2).astype(float)
        Gp = H.g_stat(cp_, n=20)
        srt = np.sort(Gp)
        pp = (1 + (N_PERM - np.searchsorted(srt, Gp - 1e-9, side="left"))) / (N_PERM + 1)
        lo, hi = R.newcombe(cp_[:, 0], 20, cp_[:, 1:].sum(1), 80)
        ck, hk = (lo > 0) | (hi < 0), pp <= 0.10
        r["null_cp"], r["null_het"], r["null_cap"] = float(ck.mean()), float(hk.mean()), float((ck & hk).mean())
        L = lab.loc[(r["traj_id"], r["turn"])]
        r["cp12"], r["het12"] = bool(L.credit_pivotal), bool(L.het)
    D = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in rows])
    D["cap12"], D["cap20"], D["cap32"] = D.cp12 & D.het12, D.cp20 & D.het20, D.cp32 & D.het32
    C = D[D.layer == "candidate"]
    rep = {}
    for rule, null in (("cp", "null_cp"), ("het", "null_het"), ("cap", "null_cap")):
        sel = C[C[f"{rule}12"]]
        rep[rule] = {"n_flagged_12": int(len(sel)),
                     "replication_indep20": H.frac_ci(sel.traj_id.values, sel[f"{rule}20"].values) if len(sel) else None,
                     "null_replication_indep20": float(sel[null].mean()) if len(sel) else None,
                     "replication_merged32_selection_biased": float(sel[f"{rule}32"].mean()) if len(sel) else None}
    switch = rep["cap"]["replication_indep20"]["frac"] < 0.50 if rep["cap"]["replication_indep20"] else None
    U = D[D.layer == "unbiased"]
    unb = {"n": int(len(U)), "fork32": H.frac_ci(U.traj_id.values, U.het32.values),
           "credit32": H.frac_ci(U.traj_id.values, U.cp32.values), "cap32": H.frac_ci(U.traj_id.values, U.cap32.values),
           "recall_het12_vs_het20": H.frac_ci(U[U.het20].traj_id.values, U[U.het20].het12.values) if U.het20.any() else None,
           "recall_cp12_vs_cp20": H.frac_ci(U[U.cp20].traj_id.values, U[U.cp20].cp12.values) if U.cp20.any() else None}
    out = {"audit": audit | {"n_points": len(rows), "n_continuations": int(sum(len(json.load(open(os.path.join(
               common.ROOT, "data", "replay_gold", r["traj_id"], f"t{r['turn']:02d}.json")))["continuations"]) for r in rows))},
           "replication_candidate_layer": rep, "B3_report_only_switch_fires": bool(switch) if switch is not None else None,
           "unbiased_layer": unb,
           "provenance": common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}}
    D.to_parquet(os.path.join(common.ROOT, "eval", "gold_points_flags.parquet"))
    common.write_json(os.path.join(common.ROOT, "eval", "gold.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=1, default=float))


if __name__ == "__main__":
    main()
