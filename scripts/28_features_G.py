"""Phase 2 (FEATURES_FROZEN §3, features 37-40): sibling-trajectory features (R3) for every point of D.

Siblings = the 7 extra rollouts of the same task (data/siblings/s{seed}_k{k}.json, D-043/D-047). A sibling step
matches point t if its obs_before equals t's obs_before exactly (GiGPO anchor rule), at any turn.
  37 g_cover       = number of matched sibling steps
  38 g_succ_state  = mean over matched steps of the step's trajectory outcome (NaN if none)
  39 g_succ_alt_sd = SD (ddof 0) across action groups of the group success rate, matched steps grouped by the action
                     the sibling took there (NaN if < 2 groups)
  40 g_singleton   = 1[g_cover == 0]
Match rates are reported separately for same-turn and different-turn matches. No replay label is read.
Output data/features_G.parquet (+ .provenance.json).
"""
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import common


def main():
    t0 = time.time()
    sib = defaultdict(list)
    sd = os.path.join(common.ROOT, "data", "siblings")
    files = sorted(os.listdir(sd))
    for f in files:
        tr = json.load(open(os.path.join(sd, f)))
        sib[tr["seed"]].append(tr)
    assert all(len(v) == 7 for v in sib.values()) and len(sib) == 100, {k: len(v) for k, v in sib.items() if len(v) != 7}
    index = {}
    for seed, trs in sib.items():
        m = defaultdict(list)
        for tr in trs:
            for s in tr["steps"]:
                m[s["obs_before"]].append((s["turn"], s["action"], int(tr["won"])))
        index[seed] = m
    lab = pd.read_parquet(os.path.join(common.ROOT, "data", "labels_final.parquet"))[["traj_id", "turn", "seed"]]
    obs = {}
    for t in lab.traj_id.unique():
        tr = json.load(open(os.path.join(common.ROOT, "data", "rollout", f"{t}.json")))
        for s in tr["steps"]:
            obs[(t, s["turn"])] = s["obs_before"]
    rows, same, diff = [], 0, 0
    for t, u, seed in lab.itertuples(index=False):
        hits = index[seed].get(obs[(t, u)], [])
        same += sum(h[0] == u for h in hits)
        diff += sum(h[0] != u for h in hits)
        g = defaultdict(list)
        for _, a, w in hits:
            g[a].append(w)
        rates = [np.mean(v) for v in g.values()]
        rows.append({"traj_id": t, "turn": u, "g_cover": len(hits),
                     "g_succ_state": float(np.mean([h[2] for h in hits])) if hits else np.nan,
                     "g_succ_alt_sd": float(np.std(rates)) if len(rates) >= 2 else np.nan,
                     "g_singleton": int(len(hits) == 0)})
    df = pd.DataFrame(rows)
    out = os.path.join(common.ROOT, "data", "features_G.parquet")
    df.to_parquet(out)
    info = {"n_points": len(df), "n_sibling_trajectories": len(files),
            "sibling_success_rate": float(np.mean([tr["won"] for v in sib.values() for tr in v])),
            "matched_steps_same_turn": same, "matched_steps_different_turn": diff,
            "points_with_any_match": int((df.g_cover > 0).sum()), "singleton_rate": float(df.g_singleton.mean()),
            "g_succ_alt_sd_defined": int(df.g_succ_alt_sd.notna().sum())}
    common.write_json(out.replace(".parquet", ".provenance.json"),
                      {"provenance": common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))},
                       **info, "seconds": time.time() - t0})
    print(info)


if __name__ == "__main__":
    main()
