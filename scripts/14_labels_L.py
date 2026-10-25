"""Phase 1C.4 (CLAUDE.md §6.0): build data/labels_L.parquet and report the L1 / L2 label reliability.

Labels only -- no feature, confidence or hidden state is read (§9.2 safe).
Banks: 1 = data/replay, 2 = data/replay_b1 (all D), 3..6 = data/replay_b2..b5 (live set only).
  live_b12  : pool of banks 1+2 (2 x 15 continuations) not constant           (primary 𝓛)
  live_b1   : pool of bank 1 not constant                                     (sensitivity 𝓛; = L1 y_state)
  A_L2      : A from banks 3-6 merged (12 per arm); sign_L2 = sign(A_L2), 0 excluded from M7
  A_h1/A_h2 : banks 3+4 vs 5+6 (6 per arm each) -> reliability ceiling
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import common

BANKS = ["replay", "replay_b1", "replay_b2", "replay_b3", "replay_b4", "replay_b5"]


def counts(r):
    """(factual successes, factual n, alt successes, alt n)"""
    f = r["factual_outcomes"]
    a = [o for x in r["alt_outcomes"] for o in x]
    return sum(f), len(f), sum(a), len(a)


def contrast(c):
    fs, fn, as_, an = c
    return fs / fn - as_ / an


def add(*cs):
    return tuple(sum(x) for x in zip(*cs))


def main():
    paths = {b: {} for b in BANKS}
    for b in BANKS:
        for p in glob.glob(os.path.join(common.ROOT, "data", b, "s*", "t*.json")):
            paths[b][(os.path.basename(os.path.dirname(p)), int(os.path.basename(p)[1:-5]))] = p
    keys = sorted(paths["replay"])
    rows, inputs = [], []
    for k in keys:
        r1 = json.load(open(paths["replay"][k]))
        if not r1["cf_defined"]:
            continue
        c = {}
        for i, b in enumerate(BANKS):
            if k in paths[b]:
                r = json.load(open(paths[b][k]))
                assert r["alternatives"] == r1["alternatives"] and r["factual_action"] == r1["factual_action"]
                c[i + 1] = counts(r)
        assert 2 in c, k
        c12 = add(c[1], c[2])
        live_b12 = 0 < c12[0] + c12[2] < c12[1] + c12[3]
        live_b1 = 0 < c[1][0] + c[1][2] < c[1][1] + c[1][3]
        has_deep = all(b in c for b in (3, 4, 5, 6))
        assert has_deep == live_b12, (k, has_deep, live_b12)  # banks 3-6 cover exactly the live set
        row = {"traj_id": k[0], "turn": k[1], "live_b1": live_b1, "live_b12": live_b12,
               "A_b1": contrast(c[1]), "A_b2": contrast(c[2]), "A_b12": contrast(c12)}
        for i in range(1, 7):
            row[f"n_succ_b{i}"] = (c[i][0], c[i][2]) if i in c else None
        if has_deep:
            d = add(c[3], c[4], c[5], c[6])
            h1, h2 = add(c[3], c[4]), add(c[5], c[6])
            row.update(A_L2=contrast(d), A_h1=contrast(h1), A_h2=contrast(h2), A_all=contrast(add(c12, d)),
                       pool_L2_succ=d[0] + d[2], pool_L2_n=d[1] + d[3])
        rows.append(row)
    df = pd.DataFrame(rows)
    df["sign_L2"] = np.sign(df["A_L2"])
    df.to_parquet(os.path.join(common.ROOT, "data", "labels_L.parquet"))

    L = df[df.live_b12]
    rep = {"provenance": common.provenance(__file__), "n_D": int(len(df)), "n_live_b12": int(len(L)),
           "n_live_b1": int(df.live_b1.sum()), "n_traj_live": int(L.traj_id.nunique()),
           "n_traj_live_ge5": int((L.groupby("traj_id").size() >= 5).sum()),
           "n_traj_live_ge4": int((L.groupby("traj_id").size() >= 4).sum())}
    # L1 two-bank consistency (all D)
    lb2 = df.apply(lambda x: 0 < sum(x.n_succ_b2) < 15, axis=1)
    rep["L1_live_b1_vs_b2_agreement"] = float((df.live_b1 == lb2).mean())
    rep["L1_live_rate_b1"], rep["L1_live_rate_b2"] = float(df.live_b1.mean()), float(lb2.mean())
    # L2 labels
    rep["L2_sign_counts"] = {"pos": int((L.A_L2 > 0).sum()), "neg": int((L.A_L2 < 0).sum()), "zero": int((L.A_L2 == 0).sum())}
    rep["L2_pool_constant_in_banks3_6"] = int(((L.pool_L2_succ == 0) | (L.pool_L2_succ == L.pool_L2_n)).sum())
    # reliability ceiling: banks 3+4 vs 5+6 on L
    nz = (L.A_h1 != 0) & (L.A_h2 != 0)
    rep["L2_reliability"] = {
        "n_both_nonzero": int(nz.sum()),
        "sign_agreement_both_nonzero": float((np.sign(L.A_h1[nz]) == np.sign(L.A_h2[nz])).mean()),
        "pearson": float(pearsonr(L.A_h1, L.A_h2)[0]), "spearman": float(spearmanr(L.A_h1, L.A_h2)[0]),
        # Spearman-Brown: reliability of the 12-per-arm label from the 6-vs-6 split
        "pearson_spearman_brown_12": float(2 * pearsonr(L.A_h1, L.A_h2)[0] / (1 + pearsonr(L.A_h1, L.A_h2)[0])),
    }
    # agreement of the 12-per-arm L2 label with the independent banks 1+2 (6 per arm)
    nz2 = (L.A_L2 != 0) & (L.A_b12 != 0)
    rep["L2_vs_b12"] = {"n": int(nz2.sum()),
                        "sign_agreement": float((np.sign(L.A_L2[nz2]) == np.sign(L.A_b12[nz2])).mean()),
                        "pearson": float(pearsonr(L.A_L2, L.A_b12)[0])}
    # sensitivity: L defined from bank 1 only, label from banks 3-6, restricted to points that have banks 3-6
    Ls = df[df.live_b1 & df.live_b12]
    rep["sens_L_b1_n_with_deep"] = int(len(Ls))
    rep["sens_L_b1_without_deep"] = int((df.live_b1 & ~df.live_b12).sum())
    common.write_json(os.path.join(common.ROOT, "eval", "phase1c4_labels.json"), rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    main()
