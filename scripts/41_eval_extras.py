"""Phase 3 extras (CLAUDE.md §6.2, §6.3 report items, §6.6, §8 gates; FEATURES_FROZEN §5-§6).

- Gates: M1 (arm A = -conf routes the top 13.1% of D; recall of pivotal_cwgt; cluster CI + Wilson); M4/T3 for H1
  (within-trajectory Delta = z(rho) - z(A_b1); Spearman(Delta, conf) per trajectory with >= 5 points; median + CI).
- Report: M5 sanity, M2 (within-trajectory Spearman of scores vs A_3456 on 𝓛), CARL delta (Cliff's delta of crit_sd,
  high vs low entropy by median split; entropy_sampled and entropy_q), family identification (CLAUDE.md §4 rule).
- E1 (FEATURES_FROZEN §6): per-trajectory selectors at k in {1,2,4,8}; precision/recall of Fork and Credit-pivotal,
  captured |A_3456| per fork, Pearson of the banks 1+2 (3/arm) A estimate vs A_3456 on selected live points;
  trajectory paired bootstrap vs uniform. exploratory_adaptive_replay as an extra row.
Inputs: eval/scores_{B1,B2,B2cp,B3,L2sign}.parquet from 40_eval.py. Output eval/extras.json.
"""
import importlib
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

import common
import evalcore as E

R = importlib.import_module("15_regimes")
H = importlib.import_module("18_het_fork")


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def traj_boot(vals, B=10_000, seed=0, stat=np.median):
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = [stat(rng.choice(vals, len(vals))) for _ in range(B)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def m1(df):
    s = -df.chosen_conf.values
    out = {}
    for r in (13.1, 100 * 780 / 3818):
        k = int(round(len(df) * r / 100))
        routed = np.zeros(len(df), bool)
        routed[np.argsort(-s, kind="stable")[:k]] = True
        y = df.pivotal_cwgt.values
        rec = (routed & y).sum() / y.sum()
        out[f"r={r:.1f}"] = {"recall": float(rec), "wilson95": wilson(int((routed & y).sum()), int(y.sum())),
                             "cluster95": H.cluster_ratio_ci(df.traj_id.values, (routed & y).astype(float), y.astype(float)),
                             "precision": float((routed & y).sum() / k), "contains_route_rate": None}
        lo, hi = out[f"r={r:.1f}"]["cluster95"]
        out[f"r={r:.1f}"]["contains_route_rate"] = bool(lo <= r / 100 <= hi)
    out["auroc_vs_live_b1"] = E.auroc(df.live_b1.values, s)
    return out


def m4(df, score, label="A_b1"):
    rows = []
    for t, g in df.groupby("traj_id"):
        g = g.dropna(subset=[score, label, "chosen_conf"])
        if len(g) < 5 or g[label].std() == 0 or g[score].std() == 0:
            continue
        z = lambda x: (x - x.mean()) / x.std()  # noqa: E731
        d = z(g[score]) - z(g[label])
        r = spearmanr(d, g.chosen_conf)[0]
        if np.isfinite(r):
            rows.append(r)
    return {"median": float(np.median(rows)), "ci95": traj_boot(rows), "n_traj": len(rows)}


def cliffs(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def family_id(bscores, bname, boot_seed=0):
    S = pd.read_parquet(os.path.join(common.ROOT, "eval", f"scores_{bname}.parquet"))
    y = S.y.values.astype(bool)
    boot = E.Boot(S.traj_id.values)
    d = {c: boot.auroc_dist(y, S[c].values) for c in S.columns if c.startswith(("B_family_only", "B_leave_out", "B_multi"))
         or c in ("V", "single:entropy_q")}
    out = {}
    full = "B_multi"
    if "B_family_only:F2" not in d or full not in d:
        return {"missing": True}
    for fam in ("F1", "F2", "F3", "F4", "F5"):
        fo, lo = d.get(f"B_family_only:{fam}"), d.get(f"B_leave_out:{fam}")
        if fo is None or lo is None:
            continue
        c1 = E.Boot.ci(fo - d["B_family_only:F2"]) if fam != "F2" else None
        c2 = E.Boot.ci(fo - d["V"])
        c3 = E.Boot.ci(d[full] - lo)
        ok = (fam != "F2" and c1[0] > 0) and c2[0] > 0 and c3[0] > 0
        out[fam] = {"family_only_minus_F2": c1, "family_only_minus_V": c2, "full_minus_leave_out": c3, "identified": bool(ok)}
    return out


def e1(df, scores_b1, ks=(1, 2, 4, 8), reps=200, seed=0):
    """Per-trajectory fork-budget allocation on D. Scores: entropy_q, B_multi (B1 OOF P(Live)), W1pure (B2 score on all
    of D, fit on live rows of the training folds), oracle (het)."""
    rng = np.random.default_rng(seed)
    base = {(r["traj_id"], r["turn"]): r for r in R.load().to_dict("records")}
    df = df.sort_values(["traj_id", "turn"]).reset_index(drop=True)
    A = df.A_3456.fillna(0).abs().values
    a12 = []
    for t, u, live in zip(df.traj_id, df.turn, df.live):
        b = base[(t, u)]
        f = np.concatenate([b["f1"], b["f2"]]); a = np.concatenate([b["a1"], b["a2"]], axis=1)
        fi = rng.permutation(6)[:3]; ai = rng.permutation(6)[:3]
        a12.append(f[fi].mean() - a[:, ai].mean())
    df["A12_m3"] = a12
    sel_scores = {"entropy-top-k": df.entropy_q.values, "B-top-k": scores_b1.get("B_multi"),
                  "oracle(het)": df.het.values.astype(float) + 1e-6 * df.A_3456.fillna(0).abs().values}
    w1 = w1pure_all(df)
    res = {}
    traj = df.traj_id.values
    groups = {t: np.where(traj == t)[0] for t in np.unique(traj)}
    for k in ks:
        res[k] = {}
        def evaluate(pick):  # pick: dict traj -> selected indices
            sel = np.zeros(len(df), bool)
            for ix in pick.values():
                sel[ix] = True
            per = []
            for t, ix in groups.items():
                s = sel[ix]
                per.append([df.het.values[ix][s].sum(), df.het.values[ix].sum(), df.credit_pivotal.values[ix][s].sum(),
                            df.credit_pivotal.values[ix].sum(), A[ix][s].sum(), s.sum()])
            per = np.array(per, float)
            L = sel & df.live.values
            pear = float(np.corrcoef(df.A12_m3.values[L], df.A_3456.values[L])[0, 1]) if L.sum() > 2 else np.nan
            return per, {"fork_precision": per[:, 0].sum() / per[:, 5].sum(), "fork_recall": per[:, 0].sum() / per[:, 1].sum(),
                         "cp_precision": per[:, 2].sum() / per[:, 5].sum(), "cp_recall": per[:, 2].sum() / per[:, 3].sum(),
                         "captured_absA_per_fork": per[:, 4].sum() / per[:, 5].sum(), "pearson_A12m3_vs_A3456": pear}
        def uniform(ix):
            n = len(ix)
            if n <= k:
                return ix
            return ix[np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))]
        per_u, res[k]["uniform"] = evaluate({t: uniform(ix) for t, ix in groups.items()})
        rr = [evaluate({t: (ix if len(ix) <= k else rng.choice(ix, k, replace=False)) for t, ix in groups.items()})[1]
              for _ in range(reps)]
        res[k]["random"] = {m: float(np.nanmean([r[m] for r in rr])) for m in rr[0]}
        picks = {}
        for name, sc in sel_scores.items():
            if sc is None:
                continue
            sc = np.asarray(sc, float)
            picks[name] = {t: (ix if len(ix) <= k else ix[np.argsort(-np.nan_to_num(sc[ix], nan=-np.inf), kind="stable")[:k]])
                           for t, ix in groups.items()}
        if w1 is not None and sel_scores["B-top-k"] is not None:
            bsc = np.asarray(sel_scores["B-top-k"], float)
            pk = {}
            for t, ix in groups.items():
                if len(ix) <= k:
                    pk[t] = ix
                    continue
                pre = ix[np.argsort(-bsc[ix], kind="stable")[:2 * k]]
                pk[t] = pre[np.argsort(-w1[pre], kind="stable")[:k]]
            picks["B->W1-pure"] = pk
        for name, pick in picks.items():
            per, m = evaluate(pick)
            # trajectory paired bootstrap vs uniform on fork recall
            diff = []
            brng = np.random.default_rng(1)
            for _ in range(2000):
                bi = brng.integers(0, len(per), len(per))
                diff.append(per[bi, 0].sum() / per[bi, 1].sum() - per_u[bi, 0].sum() / per_u[bi, 1].sum())
            m["fork_recall_minus_uniform_ci95"] = [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]
            res[k][name] = m
    return res


def w1pure_all(df):
    """W1-pure multi-feature B2 score for every point of D: L2 logistic fit on the live rows of the training folds
    (folds by trajectory on D), C by inner CV on those rows, predicted on all rows of the test fold."""
    p = os.path.join(common.ROOT, "data", "features_W1.parquet")
    if not os.path.exists(p):
        return None
    W = pd.read_parquet(p).drop(columns=["budget_left"], errors="ignore")
    X = df[["traj_id", "turn", "turn_index", "n_admissible", "budget_left"]].merge(W, on=["traj_id", "turn"], how="left")
    cols = ["turn_index", "n_admissible", "budget_left", "w1_nothing_frac", "w1_obs_len_sd", "w1_adm_delta_sd",
            "w1_task_sim_max", "w1_task_sim_mean", "target_revealed", "w1_div_emb", "w1_div_n_distinct",
            "w1_div_emb_x_budget", "w1_task_sim_max_x_budget", "target_revealed_x_budget"]
    Xv = X[cols].values.astype(float)
    g = df.traj_id.values
    fold = E.outer_folds(g)
    live = df.live.values
    y = df.het.values.astype(int)
    out = np.full(len(df), np.nan)
    for k in range(5):
        tr = np.where((fold != k) & live)[0]
        te = np.where(fold == k)[0]
        _, fi = E.fit_oof_logistic(Xv[tr], y[tr], g[tr], E.outer_folds(g[tr]), n_jobs=8)
        C = max(set(fi["C"]), key=fi["C"].count)
        pp = E.Prep().fit(Xv[tr])
        bas = E._Basis(pp.transform(Xv[tr]))
        out[te] = bas.fit(y[tr], C).decision_function(bas.project(pp.transform(Xv[te])))
    return out


def main():
    lab = E.labels()
    F = pd.read_parquet(os.path.join(common.ROOT, "data", "features_B.parquet"))
    df = lab.merge(F, on=["traj_id", "turn"], how="left")
    out = {"M1_gate_armA": m1(df),
           "M4_T3_gate_H1": m4(df, "rho_h1"), "M4_armA_mechanical": m4(df.assign(s=-df.chosen_conf), "s"),
           "M4_on_L_A3456_H1": m4(df[df.live], "rho_h1", "A_3456")}
    t3 = out["M4_T3_gate_H1"]["ci95"]
    out["gates"] = {"M1_contains_13.1": out["M1_gate_armA"]["r=13.1"]["contains_route_rate"],
                    "T3_compatible_[0.647,0.793]": bool(t3[1] >= 0.647 and t3[0] <= 0.793)}
    out["CARL_delta"] = {}
    for ent in ("entropy_sampled", "entropy_q"):
        hi = df[df[ent] > df[ent].median()].crit_sd.values
        lo = df[df[ent] <= df[ent].median()].crit_sd.values
        out["CARL_delta"][ent] = {"cliffs_delta": cliffs(hi, lo), "n_hi": len(hi), "n_lo": len(lo), "CARL_reference": 0.42}
    out["M5"] = {"D_over_total": 3038 / 3818, "pivotal_cwgt_frac": float(df.pivotal_cwgt.mean()),
                 "sigma_floor_zero_frac": float((df.sigma_floor == 0).mean()),
                 "conf_shift_incl_minus_excl": json.load(open(os.path.join(common.ROOT, "eval", "phase2_fp32_check.json")))
                 .get("conf_shift", "see D-040"), "CWGT": {"D_over_total": 0.869, "pivotal": 0.305, "sigma_floor_zero": 0.860,
                                                          "conf_shift": -0.70}}
    # M2 on 𝓛 for the main scores
    S2 = pd.read_parquet(os.path.join(common.ROOT, "eval", "scores_B2.parquet")).merge(
        lab[["traj_id", "turn", "A_3456"]], on=["traj_id", "turn"])
    out["M2"] = {}
    for c in ("A(-conf)", "H1(rho)", "V", "B_multi", "C_sel", "B+C", "W1pure_multi", "W1model_multi", "G_multi"):
        if c not in S2:
            continue
        rs = [spearmanr(g[c], g.A_3456)[0] for _, g in S2.groupby("traj_id") if len(g) >= 5 and g.A_3456.std() > 0 and g[c].std() > 0]
        rs = [r for r in rs if np.isfinite(r)]
        out["M2"][c] = {"median": float(np.median(rs)), "ci95": traj_boot(rs), "n_traj": len(rs)}
    out["family_identification"] = {b: family_id(None, b) for b in ("B1", "B2", "B3")}
    S1 = pd.read_parquet(os.path.join(common.ROOT, "eval", "scores_B1.parquet"))
    sc1 = {c: df[["traj_id", "turn"]].merge(S1[["traj_id", "turn", c]], on=["traj_id", "turn"], how="left")[c].values
           for c in ("B_multi",) if c in S1}
    out["E1"] = e1(df, sc1)
    out["L3"] = "optional; not run in this pass"
    out["provenance"] = common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}
    common.write_json(os.path.join(common.ROOT, "eval", "extras.json"), out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("provenance", "E1")}, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
