"""RULING 2026-10-25 §5.1-5.2 (label-only, §9.2 safe): three regimes, depth invariance, pivotal rule, replay ceilings.

Banks (per point: factual successes / n, alternative successes / n):
  b1 = data/replay, b2 = data/replay_b1 (all D);  b3..b6 = data/replay_b2..b5 (live set 𝓛 only), 3 per arm each.
Regimes on D:
  Dead       : banks 1+2 pooled 30 continuations constant
  Live (𝓛)   : not Dead
  Pivotal    : Live and the 90% Newcombe (hybrid score) interval for p_factual - p_alt from banks 3-6
               (factual 12 vs alternative pool 48) excludes 0.        Sensitivity: |A_3456| >= 0.25 and A_34, A_56 same
               sign, both nonzero.
  Indifferent: Live \\ Pivotal
Outputs eval/regimes.json, eval/pivotal_rule.json, eval/ceilings.json.
"""
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr
from sklearn.metrics import roc_auc_score

import common

BANKS = {1: "replay", 2: "replay_b1", 3: "replay_b2", 4: "replay_b3", 5: "replay_b4", 6: "replay_b5"}
Z90 = norm.ppf(0.95)


def wilson(x, n, z=Z90):
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def newcombe(x1, n1, x2, n2):
    """Newcombe (1998) method 10 hybrid score interval for p1 - p2."""
    l1, u1 = wilson(x1, n1)
    l2, u2 = wilson(x2, n2)
    p1, p2 = x1 / n1, x2 / n2
    d = p1 - p2
    lo = d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo, hi


def load():
    rows = []
    paths = {b: {} for b in BANKS}
    for b, d in BANKS.items():
        for p in glob.glob(os.path.join(common.ROOT, "data", d, "s*", "t*.json")):
            paths[b][(os.path.basename(os.path.dirname(p)), int(os.path.basename(p)[1:-5]))] = p
    rl = {os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(common.ROOT, "data", "rollout", "s*.json"))}
    for k, p1 in paths[1].items():
        r1 = json.load(open(p1))
        if not r1["cf_defined"]:
            continue
        row = {"traj_id": k[0], "turn": k[1], "won": rl[k[0]]["won"], "n_turns": rl[k[0]]["n_turns"]}
        for b in BANKS:
            if k in paths[b]:
                r = json.load(open(paths[b][k]))
                row[f"f{b}"] = np.array(r["factual_outcomes"])
                row[f"a{b}"] = np.array(r["alt_outcomes"])  # (4, 3)
        rows.append(row)
    return pd.DataFrame(rows)


def contrast(df, banks):
    fs = sum(df[f"f{b}"].map(np.sum) for b in banks)
    fn = sum(df[f"f{b}"].map(len) for b in banks)
    as_ = sum(df[f"a{b}"].map(np.sum) for b in banks)
    an = sum(df[f"a{b}"].map(np.size) for b in banks)
    return fs, fn, as_, an


def sign_agree(x, y):
    nz = (x != 0) & (y != 0)
    return float((np.sign(x[nz]) == np.sign(y[nz])).mean()), int(nz.sum())


def cluster_ci(df, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    g = {t: x for t, x in df.groupby("traj_id")}
    T = list(g)
    vals = [fn(pd.concat([g[t] for t in rng.choice(T, len(T))])) for _ in range(B)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def main():
    df = load()
    out_reg, out_piv, out_ceil = {}, {}, {}
    prov = common.provenance(__file__)
    fs, fn, as_, an = contrast(df, [1, 2])
    df["live"] = (fs + as_ > 0) & (fs + as_ < fn + an)
    L = df[df.live].copy()
    assert all(f"f{b}" in L and L[f"f{b}"].notna().all() for b in (3, 4, 5, 6))
    # ---------- A at different depths on 𝓛 (independent 6/arm halves and 3/arm single banks)
    for name, banks in {"12": [1, 2], "34": [3, 4], "56": [5, 6], "3456": [3, 4, 5, 6]}.items():
        f, n1, a, n2 = contrast(L, banks)
        L[f"A{name}"] = f / n1 - a / n2
        L[f"fs{name}"], L[f"fn{name}"], L[f"as{name}"], L[f"an{name}"] = f, n1, a, n2
    for b in (3, 4, 5, 6):
        f, n1, a, n2 = contrast(L, [b])
        L[f"A{b}"] = f / n1 - a / n2

    # ---------- 5.1 depth invariance on own data
    d3 = [sign_agree(L[f"A{x}"].values, L[f"A{y}"].values) + (pearsonr(L[f"A{x}"], L[f"A{y}"])[0],)
          for x, y in itertools.combinations((3, 4, 5, 6), 2)]
    d6 = []
    for (x1, x2), (y1, y2) in (((3, 4), (5, 6)), ((3, 5), (4, 6)), ((3, 6), (4, 5))):
        fa, na, aa, ma = contrast(L, [x1, x2])
        fb, nb, ab, mb = contrast(L, [y1, y2])
        Ax, Ay = fa / na - aa / ma, fb / nb - ab / mb
        d6.append(sign_agree(Ax.values, Ay.values) + (pearsonr(Ax, Ay)[0],))
    sel12 = [sign_agree(L.A12.values, L[c].values) + (pearsonr(L.A12, L[c])[0],) for c in ("A34", "A56")]
    out_reg["depth_invariance_own"] = {
        "note": "banks 3-6 are independent of the 𝓛 selection; pairs involving banks 1+2 are selection-biased",
        "d3_single_bank_pairs": {"sign_agree_mean": float(np.mean([x[0] for x in d3])), "n_mean": float(np.mean([x[1] for x in d3])),
                                 "pearson_mean": float(np.mean([x[2] for x in d3])), "pairs": d3},
        "d6_half_splits": {"sign_agree_mean": float(np.mean([x[0] for x in d6])), "n_mean": float(np.mean([x[1] for x in d6])),
                           "pearson_mean": float(np.mean([x[2] for x in d6])), "splits": d6},
        "banks12_vs_34_56_selection_biased": sel12,
        "cwgt_reference_D023": {"d": [3, 6, 9, 12, 16], "sign_agree": [0.641, 0.628, 0.625, 0.626, 0.624],
                                "pearson": [0.352, 0.516, 0.617, 0.679, 0.736]},
    }
    a6 = out_reg["depth_invariance_own"]["d6_half_splits"]["sign_agree_mean"]
    out_reg["mixture_fit"] = {f"q={q}": float((a6 - 0.5) / (q - 0.5)) for q in (0.9, 0.95, 1.0)}
    out_reg["mixture_fit"]["note"] = "agreement = q*p + 0.5*(1-p) on both-nonzero live points; p = fraction with a real effect"

    # ---------- 5.2 pivotal rule
    lo, hi = newcombe(L.fs3456.values, L.fn3456.values, L.as3456.values, L.an3456.values)
    L["pivotal"] = (lo > 0) | (hi < 0)
    L["pivotal_sens"] = (L.A3456.abs() >= 0.25) & (L.A34 != 0) & (L.A56 != 0) & (np.sign(L.A34) == np.sign(L.A56))
    df["pivotal"] = False
    df.loc[L.index, "pivotal"] = L.pivotal
    df["regime"] = np.where(~df.live, "dead", np.where(df.pivotal, "pivotal", "indifferent"))
    prog = (df.turn - 1) / df.n_turns
    df["prog3"] = np.minimum((prog * 3).astype(int), 2)
    frac = lambda d, r: float((d.regime == r).mean())  # noqa: E731
    out_reg["regimes"] = {r: {"n": int((df.regime == r).sum()), "frac": frac(df, r),
                              "ci95_cluster": cluster_ci(df, lambda d, r=r: frac(d, r))}
                          for r in ("dead", "indifferent", "pivotal")}
    out_reg["regimes"]["n_D"] = int(len(df))
    out_reg["by_outcome"] = {str(w): {r: frac(g, r) for r in ("dead", "indifferent", "pivotal")} | {"n": len(g)}
                             for w, g in df.groupby("won")}
    out_reg["by_progress_tercile"] = {str(k): {r: frac(g, r) for r in ("dead", "indifferent", "pivotal")} | {"n": len(g)}
                                      for k, g in df.groupby("prog3")}
    out_reg["dead_pool_value"] = {"all_fail": int(((fs + as_) == 0).sum()), "all_success": int(((fs + as_) == fn + an).sum())}
    P = L[L.pivotal]
    out_piv = {"n_live": int(len(L)), "n_pivotal_newcombe90": int(L.pivotal.sum()), "n_pivotal_sensitivity": int(L.pivotal_sens.sum()),
               "overlap": int((L.pivotal & L.pivotal_sens).sum()),
               "pivotal_sign": {"pos": int((P.A3456 > 0).sum()), "neg": int((P.A3456 < 0).sum())},
               "n_traj_with_pivotal": int(P.traj_id.nunique()), "n_traj_pivotal_ge3": int((P.groupby("traj_id").size() >= 3).sum()),
               "gate_n_pivotal_ge_150": bool(L.pivotal.sum() >= 150),
               "within_pivotal_split_sign_agree_34_56": sign_agree(P.A34.values, P.A56.values),
               "within_indifferent_split_sign_agree_34_56": sign_agree(L[~L.pivotal].A34.values, L[~L.pivotal].A56.values)}

    # ---------- 5.2 replay ceilings B_m (banks 1+2 subsamples) -- B2 / B3 / L2-sign;  B1 from bank 2 vs bank-1 label
    rng = np.random.default_rng(0)

    def sub(df_, m, banks=(1, 2)):
        f = np.stack([np.concatenate([x[f"f{b}"] for b in banks]) for _, x in df_.iterrows()])
        a = np.stack([np.concatenate([x[f"a{b}"] for b in banks], axis=1) for _, x in df_.iterrows()])
        fi = np.argsort(rng.random(f.shape), 1)[:, :m]
        ai = np.argsort(rng.random(a.shape), 2)[:, :, :m]
        fm = np.take_along_axis(f, fi, 1).mean(1)
        am = np.take_along_axis(a, ai, 2).mean((1, 2))
        pool = (np.take_along_axis(f, fi, 1).sum(1) + np.take_along_axis(a, ai, 2).sum((1, 2))) / (5 * m)
        return fm - am, pool

    def ceil(df_, y, kind, ms, banks=(1, 2), R=200):
        res = {}
        for m in ms:
            v = []
            for _ in range(R):
                A, pool = sub(df_, m, banks)
                s = {"abs": np.abs(A), "signed": A, "poolvar": pool * (1 - pool)}[kind]
                v.append(roc_auc_score(y, s))
            res[m] = {"auroc_mean": float(np.mean(v)), "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5))}
        return res

    out_ceil["B2_pivotal_on_L_score_absA"] = ceil(L, L.pivotal.values, "abs", (1, 2, 3, 6))
    Ls = L[L.A3456 != 0]
    out_ceil["L2sign_on_L_score_A"] = ceil(Ls, (Ls.A3456 > 0).values, "signed", (1, 2, 3, 6))
    out_ceil["B3_sign_in_pivotal_score_A"] = ceil(P, (P.A3456 > 0).values, "signed", (1, 2, 3, 6))
    # B1: label = bank-1 pool non-constant (15), score = bank-2 subsample pool variance (independent bank)
    f1, n1, a1, m1 = contrast(df, [1])
    y1 = ((f1 + a1) > 0) & ((f1 + a1) < n1 + m1)
    out_ceil["B1_live_bank1_label_score_bank2_poolvar"] = ceil(df, y1.values, "poolvar", (1, 2, 3), banks=(2,))
    out_ceil["note"] = ("B_m = per-arm m continuations subsampled without replacement (200 repeats). B2/B3/L2-sign use banks 1+2 "
                        "(not used to define Pivotal or the L2 label; 𝓛 itself is selected on banks 1+2). B1 deviates from "
                        "RULING §2.4: the Live label uses banks 1+2, so the B1 ceiling uses label = bank 1 and score = bank 2.")
    for o, name in ((out_reg, "regimes"), (out_piv, "pivotal_rule"), (out_ceil, "ceilings")):
        o["provenance"] = prov
        common.write_json(os.path.join(common.ROOT, "eval", f"{name}.json"), o)
    print(json.dumps({"regimes": out_reg["regimes"], "depth": {k: v for k, v in out_reg["depth_invariance_own"].items()
                                                               if k in ("d3_single_bank_pairs", "d6_half_splits")},
                      "mixture": out_reg["mixture_fit"], "pivotal": out_piv,
                      "by_outcome": out_reg["by_outcome"], "by_progress": out_reg["by_progress_tercile"]},
                     indent=1, default=lambda x: round(float(x), 3))[:6000])
    print(json.dumps({k: v for k, v in out_ceil.items() if k not in ("provenance",)}, indent=1)[:3000])


if __name__ == "__main__":
    main()
