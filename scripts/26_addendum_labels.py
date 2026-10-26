"""FREEZE ADDENDUM 2026-10-26 §5 step 2 (label-only, §9.2 safe): new replay ceilings and renamed regimes.

Adds to eval/ceilings.json
  B2het_on_L_score_G      : label = Pivotal_het; score = five-arm G statistic of a banks 1+2 subsample (m per arm)
  B2het_on_L_score_absA   : same label, score = |A| of the subsample (comparison with the Credit-pivotal ceiling)
  B3cap_sign_in_cap_score_A: B3 set = Credit-pivotal AND het (170); label = sign(A_3456); score = subsample A
and to eval/regimes.json the field `fork_het`: Dead / Indifferent (Live AND NOT het) / Fork (het), Fork split into
Credit-pivotal AND het and Masked, with null-expected counts (D-042) and outcome / progress breakdowns.
m in {1, 2, 3, 6}, 200 subsamples (without replacement, per arm), banks 1+2 are not used by either label.
"""
import importlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common

R = importlib.import_module("15_regimes")
H = importlib.import_module("18_het_fork")
MS = (1, 2, 3, 6)
REPS = 200


def main():
    prov = common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}
    df = R.load()
    fs, fn, as_, an = R.contrast(df, [1, 2])
    df["live"] = (fs + as_ > 0) & (fs + as_ < fn + an)
    hp = pd.read_parquet(os.path.join(common.ROOT, "eval", "het_points.parquet"))[["traj_id", "turn", "het", "piv_agg"]]
    df = df.merge(hp, on=["traj_id", "turn"], how="left")
    df["het"] = df.het.fillna(False).astype(bool)
    df["piv_agg"] = df.piv_agg.fillna(False).astype(bool)
    assert int(df.het.sum()) == 475 and int(df.piv_agg.sum()) == 229 and int((df.het & df.piv_agg).sum()) == 170
    L = df[df.live].reset_index(drop=True)
    f12 = np.stack([np.concatenate([L.loc[i, "f1"], L.loc[i, "f2"]]) for i in range(len(L))])                 # (n, 6)
    a12 = np.stack([np.concatenate([L.loc[i, "a1"], L.loc[i, "a2"]], axis=1) for i in range(len(L))])         # (n, 4, 6)
    f3, n3, a3, m3 = R.contrast(L, [3, 4, 5, 6])
    L["A3456"] = f3 / n3 - a3 / m3
    rng = np.random.default_rng(0)

    def subsample(m):
        fi = np.argsort(rng.random(f12.shape), 1)[:, :m]
        ai = np.argsort(rng.random(a12.shape), 2)[:, :, :m]
        fsub = np.take_along_axis(f12, fi, 1)
        asub = np.take_along_axis(a12, ai, 2)
        c = np.concatenate([fsub.sum(1, keepdims=True), asub.sum(2)], axis=1)   # (n, 5) successes, m per arm
        A = fsub.mean(1) - asub.mean((1, 2))
        return H.g_stat(c.astype(float), n=m), A

    def summarize(v):
        return {"auroc_mean": float(np.mean(v)), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5))}

    y_het = L.het.values
    cap = (L.het & L.piv_agg).values
    y_b3 = (L.A3456[cap] > 0).values
    res = {"B2het_on_L_score_G": {}, "B2het_on_L_score_absA": {}, "B3cap_sign_in_cap_score_A": {}}
    for m in MS:
        vg, va, vb = [], [], []
        for _ in range(REPS):
            G, A = subsample(m)
            vg.append(roc_auc_score(y_het, G))
            va.append(roc_auc_score(y_het, np.abs(A)))
            vb.append(roc_auc_score(y_b3, A[cap]))
        res["B2het_on_L_score_G"][m] = summarize(vg)
        res["B2het_on_L_score_absA"][m] = summarize(va)
        res["B3cap_sign_in_cap_score_A"][m] = summarize(vb)
    res["n"] = {"live": int(len(L)), "het": int(y_het.sum()), "cap": int(cap.sum()), "cap_pos": int(y_b3.sum())}
    res["note"] = ("FREEZE ADDENDUM: B2 label = Pivotal_het; B3 set = Credit-pivotal AND het. Subsample = m runs per arm "
                   "without replacement from banks 1+2 (6 per arm), 200 repeats, rng default_rng(0); banks 1+2 define "
                   "only Live, not het / Credit-pivotal / A_3456.")
    res["provenance"] = prov
    cp = os.path.join(common.ROOT, "eval", "ceilings.json")
    ceil = json.load(open(cp))
    ceil["freeze_addendum"] = res
    common.write_json(cp, ceil)

    # ---------------- regimes under the addendum names
    df["regime3"] = np.where(~df.live, "dead", np.where(df.het, "fork", "indifferent"))
    df["fork_split"] = np.where(df.het & df.piv_agg, "credit_pivotal", np.where(df.het, "masked", ""))
    prog = (df.turn - 1) / df.n_turns
    df["prog3"] = np.minimum((prog * 3).astype(int), 2)
    tr = df.traj_id.values
    nul = json.load(open(os.path.join(common.ROOT, "eval", "het_fork.json")))["null_calibration"]
    out = {"n_D": int(len(df)),
           "dead": H.frac_ci(tr, df.regime3 == "dead"),
           "indifferent": H.frac_ci(tr, df.regime3 == "indifferent"),
           "fork": H.frac_ci(tr, df.regime3 == "fork"),
           "fork_credit_pivotal": H.frac_ci(tr, df.fork_split == "credit_pivotal"),
           "fork_masked": H.frac_ci(tr, df.fork_split == "masked"),
           "null_expected_counts": {"het": nul["E_het"], "credit_pivotal": nul["E_agg"],
                                    "credit_pivotal_and_het": nul["E_het"] - nul["E_masked"], "masked": nul["E_masked"]},
           "null_expected_fork_frac_of_D": nul["E_het"] / len(df)}
    for name, col in (("by_outcome", "won"), ("by_progress_tercile", "prog3")):
        out[name] = {str(k): {"n": int(len(g)),
                              **{r: H.frac_ci(g.traj_id.values, g.regime3 == r) for r in ("dead", "indifferent", "fork")},
                              "fork_of_live": H.frac_ci(g.traj_id.values, g.regime3 == "fork", g.live)}
                     for k, g in df.groupby(col)}
    out["note"] = "FREEZE ADDENDUM names: Indifferent := Live AND NOT het; Fork := het; a'_u (D-039) used the old Indifferent"
    out["provenance"] = prov
    rp = os.path.join(common.ROOT, "eval", "regimes.json")
    reg = json.load(open(rp))
    reg["fork_het"] = out
    common.write_json(rp, reg)
    print(json.dumps({k: v for k, v in res.items() if k != "provenance"}, indent=1))
    print(json.dumps({k: out[k] for k in ("dead", "indifferent", "fork", "fork_credit_pivotal", "fork_masked",
                                          "null_expected_counts", "by_progress_tercile")}, indent=1, default=float))


if __name__ == "__main__":
    main()
