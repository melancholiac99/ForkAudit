"""FINAL RULING 2026-10-26 §2.3 / CLAUDE.md §6.3a: select the 300 gold points (label-based selection; no features).

G-unbiased (200): 𝓛 stratified by n_draws tercile x progress tercile x task type, proportional allocation
                  (largest remainder), uniform within stratum. n_draws = bank-1 draft-sampler draws (forward-safe).
G-candidate (100): uniform from 𝓛 points with |A_3456| >= 0.2 that are not in G-unbiased (300 distinct points; D-043).
n_draws terciles are value-based: cut points = the 1/3 and 2/3 quantiles (inverted CDF); n_draws is heavily tied
(multiples of the 32-draft batch), so the bins are {<= q1}, (q1, q2], {> q2} and have unequal sizes.
Output eval/gold_points.json ({"points": [[traj_id, turn], ...]} is the 11_replay --subset_file format).
"""
import importlib
import json
import os

import numpy as np
import pandas as pd

import common

R = importlib.import_module("15_regimes")
SEED = 20261026
N_UNBIASED, N_CAND, A_MIN = 200, 100, 0.2


def main():
    rng = np.random.default_rng(SEED)
    df = R.load()
    fs, fn, as_, an = R.contrast(df, [1, 2])
    L = df[(fs + as_ > 0) & (fs + as_ < fn + an)].copy().reset_index(drop=True)
    f, n1, a, n2 = R.contrast(L, [3, 4, 5, 6])
    L["A3456"] = f / n1 - a / n2
    L["prog3"] = np.minimum((((L.turn - 1) / L.n_turns) * 3).astype(int), 2)
    rl = {}
    for t in L.traj_id.unique():
        rl[t] = json.load(open(os.path.join(common.ROOT, "data", "rollout", f"{t}.json")))["task_type"]
    L["task_type"] = L.traj_id.map(rl)
    L["n_draws"] = [json.load(open(os.path.join(common.ROOT, "data", "replay", t, f"t{u:02d}.json")))["n_draws"]
                    for t, u in zip(L.traj_id, L.turn)]
    q1, q2 = np.quantile(L.n_draws, [1 / 3, 2 / 3], method="inverted_cdf")
    L["nd3"] = np.where(L.n_draws <= q1, 0, np.where(L.n_draws <= q2, 1, 2))

    strata = L.groupby(["nd3", "prog3", "task_type"]).indices
    keys = sorted(strata)
    sizes = np.array([len(strata[k]) for k in keys])
    quota = N_UNBIASED * sizes / sizes.sum()
    alloc = np.floor(quota).astype(int)
    rem = N_UNBIASED - alloc.sum()
    alloc[np.argsort(-(quota - alloc), kind="stable")[:rem]] += 1
    unb = []
    for k, m in zip(keys, alloc):
        if m:
            unb.extend(rng.choice(strata[k], size=m, replace=False).tolist())
    unb_set = set(unb)
    cand_pool = [i for i in range(len(L)) if abs(L.A3456[i]) >= A_MIN and i not in unb_set]
    cand = rng.choice(cand_pool, size=N_CAND, replace=False).tolist()
    pt = lambda i: [L.traj_id[i], int(L.turn[i])]  # noqa: E731
    out = {"points": [pt(i) for i in unb + cand],
           "layers": {"unbiased": [pt(i) for i in unb], "candidate": [pt(i) for i in cand]},
           "settings": {"seed": SEED, "n_unbiased": N_UNBIASED, "n_candidate": N_CAND, "A_min": A_MIN,
                        "n_live": int(len(L)), "n_strata": len(keys), "n_strata_nonempty_alloc": int((alloc > 0).sum()),
                        "candidate_pool_size": len(cand_pool),
                        "n_draws_tercile_cuts": [int(q1), int(q2)], "n_draws_bin_sizes": [int((L.nd3 == b).sum()) for b in (0, 1, 2)],
                        "n_draws_value_counts": {str(k): int(v) for k, v in L.n_draws.value_counts().sort_index().items()}},
           "summary": {"unbiased_traj": int(L.loc[unb, "traj_id"].nunique()), "candidate_traj": int(L.loc[cand, "traj_id"].nunique()),
                       "candidate_A_abs_mean": float(L.loc[cand, "A3456"].abs().mean())},
           "provenance": common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}}
    assert len({tuple(p) for p in out["points"]}) == N_UNBIASED + N_CAND
    common.write_json(os.path.join(common.ROOT, "eval", "gold_points.json"), out)
    print(json.dumps({k: out[k] for k in ("settings", "summary")}, indent=1))


if __name__ == "__main__":
    main()
