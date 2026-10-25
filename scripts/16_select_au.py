"""a'_u sensitivity point selection (RULING 2026-10-25 §7 row 3 / §8.4, D-037).

100 Indifferent points (regime from scripts/15_regimes.py logic), numpy seed 0. For each, one alternative drawn
uniformly from admissible \\ {help, chosen, the 4 bank-1 alternatives}, rng = default_rng(1_000_003 * task_seed + turn).
Points with an empty pool are skipped and replaced by the next draw; the count is reported.
Output eval/au_points.json, consumed by 11_replay.py --points_file.
"""
import importlib.util
import json
import os

import numpy as np

import common

spec = importlib.util.spec_from_file_location("reg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "15_regimes.py"))
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)


def main():
    df = reg.load()
    fs, fn, as_, an = reg.contrast(df, [1, 2])
    live = (fs + as_ > 0) & (fs + as_ < fn + an)
    L = df[live].copy()
    f, n1, a, n2 = reg.contrast(L, [3, 4, 5, 6])
    lo, hi = reg.newcombe(f.values, n1.values, a.values, n2.values)
    indiff = L[~((lo > 0) | (hi < 0))][["traj_id", "turn"]].values.tolist()
    indiff.sort()
    order = np.random.default_rng(0).permutation(len(indiff))
    rl = {}
    out, skipped = [], 0
    for k in order:
        t, u = indiff[k]
        if t not in rl:
            rl[t] = json.load(open(os.path.join(common.ROOT, "data", "rollout", f"{t}.json")))
        tr = rl[t]
        st = tr["steps"][u - 1]
        b1 = json.load(open(os.path.join(common.ROOT, "data", "replay", t, f"t{u:02d}.json")))
        pool = sorted(set(st["admissible"]) - {"help", st["action"]} - set(b1["alternatives"]))
        if not pool:
            skipped += 1
            continue
        rng = np.random.default_rng(1_000_003 * tr["seed"] + u)
        out.append([t, int(u), pool[int(rng.integers(len(pool)))], len(pool)])
        if len(out) == 100:
            break
    rec = {"provenance": common.provenance(__file__), "n_indifferent": len(indiff), "skipped_empty_pool": skipped,
           "points": [p[:3] for p in out], "pool_sizes": [p[3] for p in out]}
    common.write_json(os.path.join(common.ROOT, "eval", "au_points.json"), rec)
    print("indifferent", len(indiff), "selected", len(out), "skipped", skipped,
          "pool size median", int(np.median([p[3] for p in out])), "trajectories", len({p[0] for p in out}))
    print("examples:", out[:4])


if __name__ == "__main__":
    main()
