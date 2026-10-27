"""REVIEW 2026-10-27 §3 addenda (report items and one exploratory row; nothing frozen changes). Output eval/review_addenda.json.

 #2 G coverage by regime: singleton rate and g_succ_state availability in Dead / Indifferent / Fork (is G blind to forks
    because of coverage holes or because the anchor merges histories?)
 #5 C->B decomposition (CLAUDE.md §4): R^2 of the C_sel out-of-fold score regressed on the B features, in-sample OLS
    and GroupKFold out-of-fold ridge, per boundary
 #7 exploratory_G_to_W1pure (E1): per trajectory take the 2k points with the highest G_multi B1 score, then the k with
    the highest W1-pure B2 score (fit on live training rows, as 41_eval_extras.w1pure_all); compared with uniform
    and with B->W1-pure under the same metrics
 #8 W1-J label counts including unparsed
"""
import importlib
import json
import os

import numpy as np
import pandas as pd

import common
import evalcore as E

X41 = importlib.import_module("41_eval_extras")
M40 = importlib.import_module("40_eval")


def main():
    lab = E.labels()
    F = pd.read_parquet(os.path.join(common.ROOT, "data", "features_B.parquet"))
    G = pd.read_parquet(os.path.join(common.ROOT, "data", "features_G.parquet"))
    df = lab.merge(F, on=["traj_id", "turn"], how="left").merge(G, on=["traj_id", "turn"], how="left")
    df["regime"] = np.where(~df.live, "dead", np.where(df.het, "fork", "indifferent"))
    out = {}
    # ---- #2
    cov = {}
    for r, g in df.groupby("regime"):
        cov[r] = {"n": int(len(g)), "singleton_rate": float(g.g_singleton.mean()),
                  "g_succ_state_defined": float(g.g_succ_state.notna().mean()), "g_cover_median": float(g.g_cover.median()),
                  "g_succ_alt_sd_defined": float(g.g_succ_alt_sd.notna().mean())}
    L = df[df.live]
    cov["live_singleton_ci_fork"] = X41.H.frac_ci(L.traj_id.values, L.g_singleton.values.astype(bool), L.het.values)
    cov["live_singleton_ci_indifferent"] = X41.H.frac_ci(L.traj_id.values, L.g_singleton.values.astype(bool), ~L.het.values)
    out["G_coverage_by_regime"] = cov
    # ---- #5
    Bf = list(dict.fromkeys(M40.F1 + M40.F2 + M40.F3 + M40.F4 + M40.F5 + ["entropy_sampled"])) + M40.B_CTRL
    dec = {}
    for b in ("B1", "B2", "B3"):
        S = pd.read_parquet(os.path.join(common.ROOT, "eval", f"scores_{b}.parquet")).merge(df[["traj_id", "turn"] + Bf],
                                                                                           on=["traj_id", "turn"])
        y = S["C_sel"].values
        X = E.Prep().fit(S[Bf].values.astype(float)).transform(S[Bf].values.astype(float))
        A = np.hstack([X, np.ones((len(X), 1))])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2_in = 1 - np.sum((A @ coef - y) ** 2) / np.sum((y - y.mean()) ** 2)
        fold = E.outer_folds(S.traj_id.values)
        gg = S.traj_id.values
        pred = np.full(len(y), np.nan)
        lams = []
        for k in range(5):
            tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
            mse = {al: 0.0 for al in E.C_GRID}
            from sklearn.model_selection import GroupKFold
            for a_, b_ in GroupKFold(5).split(tr, groups=gg[tr]):
                p = E.Prep().fit(S[Bf].values[tr[a_]].astype(float))
                pr = E.ridge_kernel_path(p.transform(S[Bf].values[tr[a_]].astype(float)), y[tr[a_]],
                                         p.transform(S[Bf].values[tr[b_]].astype(float)), E.C_GRID)
                for al in E.C_GRID:
                    mse[al] += float(np.mean((pr[al] - y[tr[b_]]) ** 2))
            al = min(mse, key=mse.get)
            lams.append(al)
            p = E.Prep().fit(S[Bf].values[tr].astype(float))
            w, ym = E.ridge_primal(p.transform(S[Bf].values[tr].astype(float)), y[tr], al)
            pred[te] = p.transform(S[Bf].values[te].astype(float)) @ w + ym
        r2_cv = 1 - np.sum((pred - y) ** 2) / np.sum((y - y.mean()) ** 2)
        dec[b] = {"n": int(len(y)), "R2_in_sample_OLS": float(r2_in), "R2_oof_ridge": float(r2_cv), "lambda_per_fold": lams}
    out["C_to_B_decomposition"] = dec
    # ---- #7
    S1 = pd.read_parquet(os.path.join(common.ROOT, "eval", "scores_B1.parquet"))
    d = df.sort_values(["traj_id", "turn"]).reset_index(drop=True)
    gs = d[["traj_id", "turn"]].merge(S1[["traj_id", "turn", "G_multi", "B_multi"]], on=["traj_id", "turn"], how="left")
    w1 = X41.w1pure_all(d)
    traj = d.traj_id.values
    groups = {t: np.where(traj == t)[0] for t in np.unique(traj)}
    A = d.A_3456.fillna(0).abs().values
    e1 = {}
    for k in (1, 2, 4, 8):
        def metrics(pick):
            sel = np.zeros(len(d), bool)
            for ix in pick.values():
                sel[ix] = True
            per = np.array([[d.het.values[ix][sel[ix]].sum(), d.het.values[ix].sum(), sel[ix].sum(), A[ix][sel[ix]].sum()]
                            for ix in groups.values()], float)
            return per, {"fork_precision": per[:, 0].sum() / per[:, 2].sum(), "fork_recall": per[:, 0].sum() / per[:, 1].sum(),
                         "captured_absA_per_fork": per[:, 3].sum() / per[:, 2].sum()}

        def two_stage(first):
            pk = {}
            for t, ix in groups.items():
                if len(ix) <= k:
                    pk[t] = ix
                    continue
                pre = ix[np.argsort(-first[ix], kind="stable")[:2 * k]]
                pk[t] = pre[np.argsort(-w1[pre], kind="stable")[:k]]
            return pk

        def uniform(ix):
            return ix if len(ix) <= k else ix[np.unique(np.round(np.linspace(0, len(ix) - 1, k)).astype(int))]
        pu, mu = metrics({t: uniform(ix) for t, ix in groups.items()})
        res = {"uniform": mu}
        for name, first in (("B->W1-pure", gs.B_multi.values), ("exploratory_G_to_W1pure", gs.G_multi.values)):
            per, m = metrics(two_stage(np.asarray(first, float)))
            rng = np.random.default_rng(1)
            diff = []
            for _ in range(2000):
                bi = rng.integers(0, len(per), len(per))
                diff.append(per[bi, 0].sum() / per[bi, 1].sum() - pu[bi, 0].sum() / pu[bi, 1].sum())
            m["fork_recall_minus_uniform_ci95"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
            res[name] = m
        e1[str(k)] = res
    out["E1_exploratory_G_to_W1pure"] = e1
    # ---- #8
    J = pd.read_parquet(os.path.join(common.ROOT, "data", "w1_judge.parquet"))
    out["W1J_labels"] = {"counts": J.label.value_counts().to_dict(), "n": int(len(J)),
                         "unparsed_rate": float((J.label == "unparsed").mean())}
    out["provenance"] = common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}
    common.write_json(os.path.join(common.ROOT, "eval", "review_addenda.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k != "provenance"}, indent=1, default=float))


if __name__ == "__main__":
    main()
