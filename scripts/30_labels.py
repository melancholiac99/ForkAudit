"""Frozen label table for Phase 3 (FEATURES_FROZEN §1; CLAUDE.md §6.0). One row per point of D (3,038).

Columns: traj_id, turn, seed, won, n_turns, prog3 (within-trajectory progress tercile), live (banks 1+2 pool
non-constant; B1 label), live_b1 (bank 1 only; y_state for M1), het (Fork, B2 main), credit_pivotal (B2 report),
cap (credit_pivotal AND het; B3 set), A_3456 (banks 3-6, live only), b3_pos, pivotal_cwgt (bank-1 A != 0), A_b1,
pool_b1 / pool_b2 (mean success of each bank's 15 continuations; V targets), crit_sd (SD of bank-1 outcomes),
sigma_floor (SD of bank-1 factual outcomes).
"""
import importlib
import os

import numpy as np
import pandas as pd

import common

R = importlib.import_module("15_regimes")


def main():
    df = R.load()
    fs, fn, as_, an = R.contrast(df, [1, 2])
    df["live"] = (fs + as_ > 0) & (fs + as_ < fn + an)
    f1, n1, a1, m1 = R.contrast(df, [1])
    df["live_b1"] = (f1 + a1 > 0) & (f1 + a1 < n1 + m1)
    df["A_b1"] = f1 / n1 - a1 / m1
    df["pivotal_cwgt"] = df.A_b1 != 0
    df["pool_b1"] = (f1 + a1) / (n1 + m1)
    f2, n2, a2, m2 = R.contrast(df, [2])
    df["pool_b2"] = (f2 + a2) / (n2 + m2)
    df["crit_sd"] = [float(np.std(np.concatenate([x, y.ravel()]))) for x, y in zip(df.f1, df.a1)]
    df["sigma_floor"] = df.f1.map(lambda x: float(np.std(x)))
    df["prog3"] = np.minimum((((df.turn - 1) / df.n_turns) * 3).astype(int), 2)
    hp = pd.read_parquet(os.path.join(common.ROOT, "eval", "het_points.parquet"))[["traj_id", "turn", "het", "piv_agg"]]
    df = df.merge(hp, on=["traj_id", "turn"], how="left")
    df["het"] = df.het.fillna(False).astype(bool)
    df["credit_pivotal"] = df.piv_agg.fillna(False).astype(bool)
    df["cap"] = df.het & df.credit_pivotal
    L = df.live.values
    f3, n3, a3, m3 = R.contrast(df[L], [3, 4, 5, 6])
    df["A_3456"] = np.nan
    df.loc[L, "A_3456"] = (f3 / n3 - a3 / m3).values
    df["b3_pos"] = df.A_3456 > 0
    df["seed"] = df.traj_id.str[1:].astype(int)
    assert (len(df), int(df.live.sum()), int(df.het.sum()), int(df.credit_pivotal.sum()), int(df.cap.sum())) == \
        (3038, 1478, 475, 229, 170)
    cols = ["traj_id", "turn", "seed", "won", "n_turns", "prog3", "live", "live_b1", "het", "credit_pivotal", "cap",
            "A_3456", "b3_pos", "pivotal_cwgt", "A_b1", "pool_b1", "pool_b2", "crit_sd", "sigma_floor"]
    out = os.path.join(common.ROOT, "data", "labels_final.parquet")
    df[cols].sort_values(["traj_id", "turn"]).to_parquet(out, index=False)
    common.write_json(out.replace(".parquet", ".provenance.json"),
                      {"provenance": common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))},
                       "counts": {"D": 3038, "live": 1478, "het": 475, "credit_pivotal": 229, "cap": 170,
                                  "cap_pos": int(df[df.cap].b3_pos.sum()), "pivotal_cwgt": int(df.pivotal_cwgt.sum())}})
    print("wrote", out, df[cols].shape, "pivotal_cwgt", int(df.pivotal_cwgt.sum()), "cap_pos", int(df[df.cap].b3_pos.sum()))


if __name__ == "__main__":
    main()
