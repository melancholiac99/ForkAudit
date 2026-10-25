"""Phase 2 preconditions (CLAUDE.md §7 2.0 and §3), run BEFORE FEATURES_FROZEN.md -- uses cf_defined only,
never A_replay or any replay outcome (§9.2 safe).

  select : 200 points = 140 random from D + 60 random cf-undefined (numpy seed 0); the first 20 of the D points form the
           HF-vs-vLLM alignment subset.  -> eval/prereq/points.json
  (then run 20_score_candidates.py --backend hf on all 200 and --backend vllm on the 20, out_dir eval/prereq)
  analyze:
    (a) D-003 pipeline check: chosen conf (bare) included-minus-excluded mean, trajectory cluster bootstrap CI.
        Expected negative, near CWGT -0.70. Same for emptythink / realized (sensitivity rows).
    (b) HF vs vLLM on the 20 points: per-point Spearman of candidate logp_sum (bare and notask), median and min;
        max |Δ logp_mean|. Gate: median Spearman > 0.95.
Usage: python 13_align_check.py select | analyze
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import common

OUT = os.path.join(common.ROOT, "eval", "prereq")


def select():
    dp = pd.read_parquet(os.path.join(common.ROOT, "data", "decision_points.parquet"))
    rng = np.random.default_rng(0)
    inc = dp[dp.cf_defined][["traj_id", "turn"]].values.tolist()
    exc = dp[~dp.cf_defined][["traj_id", "turn"]].values.tolist()
    pi = [inc[i] for i in rng.choice(len(inc), 140, replace=False)]
    pe = [exc[i] for i in rng.choice(len(exc), 60, replace=False)]
    sel = {"included": [[t, int(u)] for t, u in pi], "excluded": [[t, int(u)] for t, u in pe],
           "align20": [[t, int(u)] for t, u in pi[:20]], "provenance": common.provenance(__file__)}
    common.write_json(os.path.join(OUT, "points.json"), sel)
    for k in ("included", "excluded", "align20"):
        print(k, len(sel[k]))
    allp = sel["included"] + sel["excluded"]
    print("ONLY_ALL=" + ",".join(f"{t}:{u}" for t, u in allp))
    print("ONLY_20=" + ",".join(f"{t}:{u}" for t, u in sel["align20"]))


def cluster_diff(df, col, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    g = {t: x for t, x in df.groupby("traj_id")}
    T = list(g)
    pt = df[df.inc][col].mean() - df[~df.inc][col].mean()
    bs = []
    for _ in range(B):
        s = pd.concat([g[t] for t in rng.choice(T, len(T))])
        if s.inc.any() and (~s.inc).any():
            bs.append(s[s.inc][col].mean() - s[~s.inc][col].mean())
    return float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def analyze():
    sel = json.load(open(os.path.join(OUT, "points.json")))
    inc = {tuple(x) for x in sel["included"]}
    rep = {"provenance": common.provenance(__file__, [os.path.join(OUT, f) for f in
                                                      ("scores_hf.parquet", "scores_chosen_hf.parquet",
                                                       "scores_vllm.parquet")])}
    s = pd.read_parquet(os.path.join(OUT, "scores_hf.parquet"))
    x = pd.read_parquet(os.path.join(OUT, "scores_chosen_hf.parquet"))
    ch = s[s.is_chosen][["traj_id", "turn", "logp_mean"]].rename(columns={"logp_mean": "bare"})
    for v in ("emptythink", "realized"):
        ch = ch.merge(x[x.variant == v][["traj_id", "turn", "logp_mean"]].rename(columns={"logp_mean": v}),
                      on=["traj_id", "turn"])
    ch["inc"] = [(t, u) in inc for t, u in zip(ch.traj_id, ch.turn)]
    rep["n_included"], rep["n_excluded"] = int(ch.inc.sum()), int((~ch.inc).sum())
    rep["conf_shift"] = {}
    for v in ("bare", "emptythink", "realized"):
        d, lo, hi = cluster_diff(ch, v)
        rep["conf_shift"][v] = {"included_minus_excluded": d, "ci95": [lo, hi],
                                "mean_included": float(ch[ch.inc][v].mean()),
                                "mean_excluded": float(ch[~ch.inc][v].mean()),
                                "median": float(ch[v].median())}
    rep["cwgt_reference"] = {"included_minus_excluded": -0.70, "ci95": [-0.81, -0.59]}

    v = pd.read_parquet(os.path.join(OUT, "scores_vllm.parquet"))
    m = v.merge(s, on=["traj_id", "turn", "action"], suffixes=("_v", "_h"))
    per = []
    for (t, u), g in m.groupby(["traj_id", "turn"]):
        per.append({"traj_id": t, "turn": int(u), "n_cand": len(g),
                    "spearman_bare": float(spearmanr(g.logp_sum_v, g.logp_sum_h)[0]),
                    "spearman_notask": float(spearmanr(g.logp_sum_notask_v, g.logp_sum_notask_h)[0]),
                    "top1_same": bool(g.logp_sum_v.idxmax() == g.logp_sum_h.idxmax()),
                    "max_abs_dmean": float((g.logp_mean_v - g.logp_mean_h).abs().max())})
    P = pd.DataFrame(per)
    rep["align"] = {"n_points": len(P), "n_pairs": int(len(m)),
                    "spearman_bare_median": float(P.spearman_bare.median()),
                    "spearman_bare_min": float(P.spearman_bare.min()),
                    "spearman_notask_median": float(P.spearman_notask.median()),
                    "spearman_notask_min": float(P.spearman_notask.min()),
                    "top1_agreement": float(P.top1_same.mean()),
                    "max_abs_dlogp_mean": float(P.max_abs_dmean.max()),
                    "gate_median_spearman_gt_0.95": bool(P.spearman_bare.median() > 0.95 and
                                                         P.spearman_notask.median() > 0.95),
                    "per_point": per}
    common.write_json(os.path.join(common.ROOT, "eval", "phase2_prereq.json"), rep)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("provenance",)}, indent=1, default=str)[:3000])


if __name__ == "__main__":
    {"select": select, "analyze": analyze}[sys.argv[1]]()
