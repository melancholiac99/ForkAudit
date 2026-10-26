"""FINAL RULING 2026-10-26 §6.2 (label-only, §9.2 safe): heterogeneity forks, pair-level contrasts, reference
dependence, regime cross-table.

On 𝓛 (live by banks 1+2), with banks 3-6 (factual 12 runs, each bank-1 alternative j 12 runs):
  Pivotal_agg : factual(12) vs alternative pool(48) Newcombe 90% excludes 0            (= 15_regimes, n must be 229)
  Pivotal_het : five-arm likelihood-ratio G test, p from 10,000 permutations of the 60 outcomes across arms
                (conditional on margins), alpha = 0.10
  masked fork : het AND NOT agg
  C_j / sig_j : factual(12) vs alternative j (12) Newcombe 90%
  T-ref       : among points with >= 2 sig_j, fraction whose sig_j signs disagree
Null calibration (reported, not used in any definition): under the same permutation null, the expected number of
het flags, agg flags, het AND NOT agg, and sig_j. A permuted data set's het p-value is its rank in the same
permutation distribution (the null distribution is identical for every permutation, conditional on margins).
Outputs eval/het_fork.json, eval/pairs.json, eval/het_points.parquet, eval/pairs_points.parquet.
"""
import importlib
import json
import os

import numpy as np
import pandas as pd

import common

R = importlib.import_module("15_regimes")
N_PERM = 10_000
N_PERM_PAIR = 2_000
ALPHA = 0.10
B_BOOT = 10_000


def g_stat(c, n=12):
    """c (..., 5) successes per arm, n runs per arm -> likelihood-ratio G statistic."""
    S = c.sum(-1, keepdims=True)
    p = S / (n * c.shape[-1])
    E1, E0 = n * p, n * (1 - p)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(c > 0, c * np.log(c / E1), 0.0)
        t0 = np.where(n - c > 0, (n - c) * np.log((n - c) / E0), 0.0)
    return 2 * (t1 + t0).sum(-1)


def agg_flag(c, n=12):
    """c (..., 5): arm 0 = factual, arms 1-4 = alternatives -> Newcombe 90% excludes 0 (factual 12 vs pool 48)."""
    lo, hi = R.newcombe(c[..., 0], n, c[..., 1:].sum(-1), 4 * n)
    return (lo > 0) | (hi < 0)


def cluster_ratio_ci(traj, num, den=None, B=B_BOOT, seed=0):
    """Trajectory cluster bootstrap 95% CI of sum(num)/sum(den) (den defaults to ones)."""
    den = np.ones(len(num)) if den is None else np.asarray(den, float)
    t, inv = np.unique(np.asarray(traj), return_inverse=True)
    nt = np.bincount(inv, weights=np.asarray(num, float), minlength=len(t))
    dt = np.bincount(inv, weights=den, minlength=len(t))
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(t), np.full(len(t), 1 / len(t)), size=B)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (w @ nt) / (w @ dt)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def frac_ci(traj, mask, sub=None):
    mask = np.asarray(mask, bool)
    sub = np.ones(len(mask), bool) if sub is None else np.asarray(sub, bool)
    n = int(sub.sum())
    k = int((mask & sub).sum())
    return {"k": k, "n": n, "frac": (k / n) if n else None,
            "ci95_cluster": cluster_ratio_ci(np.asarray(traj)[sub], mask[sub].astype(float)) if n else None}


def main():
    df = R.load()
    prov = common.provenance(__file__)
    prov["script_sha256"] = common.file_sha256(os.path.abspath(__file__))
    fs, fn, as_, an = R.contrast(df, [1, 2])
    df["live"] = (fs + as_ > 0) & (fs + as_ < fn + an)
    prog = (df.turn - 1) / df.n_turns
    df["prog3"] = np.minimum((prog * 3).astype(int), 2)
    L = df[df.live].reset_index(drop=True)
    f = np.stack([np.concatenate([L.loc[i, f"f{b}"] for b in (3, 4, 5, 6)]) for i in range(len(L))])           # (n, 12)
    a = np.stack([np.concatenate([L.loc[i, f"a{b}"] for b in (3, 4, 5, 6)], axis=1) for i in range(len(L))])  # (n, 4, 12)
    assert f.shape[1] == 12 and a.shape[1:] == (4, 12)
    c_obs = np.concatenate([f.sum(1, keepdims=True), a.sum(2)], axis=1)  # (n, 5)
    L["piv_agg"] = agg_flag(c_obs)
    assert int(L.piv_agg.sum()) == 229, int(L.piv_agg.sum())
    L["G"] = g_stat(c_obs)

    # ---------------- permutation p-values and null calibration (per point)
    p_het, null_het, null_agg, null_masked = (np.zeros(len(L)) for _ in range(4))
    for i in range(len(L)):
        rng = np.random.default_rng(1_000_000 + 1000 * int(L.traj_id[i][1:]) + int(L.turn[i]))
        o = np.concatenate([f[i], a[i].ravel()])  # 60 outcomes; arm k = positions 12k..12k+11
        perm = rng.permuted(np.tile(o, (N_PERM, 1)), axis=1)
        cp = perm.reshape(N_PERM, 5, 12).sum(2)
        Gp = g_stat(cp)
        p_het[i] = (1 + np.sum(Gp >= L.G[i] - 1e-9)) / (N_PERM + 1)
        srt = np.sort(Gp)
        p_perm = (1 + (N_PERM - np.searchsorted(srt, Gp - 1e-9, side="left"))) / (N_PERM + 1)
        het_k = p_perm <= ALPHA
        agg_k = agg_flag(cp)
        null_het[i], null_agg[i], null_masked[i] = het_k.mean(), agg_k.mean(), (het_k & ~agg_k).mean()
    L["p_het"] = p_het
    L["het"] = L.p_het <= ALPHA
    L["masked"] = L.het & ~L.piv_agg
    tr = L.traj_id.values

    het_out = {
        "n_live": int(len(L)),
        "pivotal_agg": int(L.piv_agg.sum()),
        "pivotal_het": frac_ci(tr, L.het),
        "overlap_2x2": {"het&agg": int((L.het & L.piv_agg).sum()), "het&~agg": int((L.het & ~L.piv_agg).sum()),
                        "~het&agg": int((~L.het & L.piv_agg).sum()), "~het&~agg": int((~L.het & ~L.piv_agg).sum())},
        "masked_fork": {"of_live": frac_ci(tr, L.masked),
                        "of_het (P5 statistic)": frac_ci(tr, L.masked, L.het)},
        "null_calibration": {
            "note": "expected counts if every live point had no arm difference (same permutation null); reported only",
            "E_het": float(null_het.sum()), "E_agg": float(null_agg.sum()), "E_masked": float(null_masked.sum()),
            "observed": {"het": int(L.het.sum()), "agg": int(L.piv_agg.sum()), "masked": int(L.masked.sum())}},
        "settings": {"alpha": ALPHA, "n_perm": N_PERM, "rng": "default_rng(1e6 + 1000*task_seed + turn)",
                     "G": "likelihood ratio over 5 arms x {success, failure}, 12 runs per arm, banks 3-6"},
    }

    # ---------------- pair-level C_j
    rows = []
    for i in range(len(L)):
        b1 = json.load(open(os.path.join(common.ROOT, "data", "replay", L.traj_id[i], f"t{int(L.turn[i]):02d}.json")))
        rng = np.random.default_rng(2_000_000 + 1000 * int(L.traj_id[i][1:]) + int(L.turn[i]))
        for j in range(4):
            x1, x2 = int(f[i].sum()), int(a[i, j].sum())
            lo, hi = R.newcombe(x1, 12, x2, 12)
            o = np.concatenate([f[i], a[i, j]])
            perm = rng.permuted(np.tile(o, (N_PERM_PAIR, 1)), axis=1)
            plo, phi = R.newcombe(perm[:, :12].sum(1), 12, perm[:, 12:].sum(1), 12)
            rows.append({"traj_id": L.traj_id[i], "turn": int(L.turn[i]), "j": j, "alt_action": b1["alternatives"][j],
                         "f_succ": x1, "a_succ": x2, "C_j": (x1 - x2) / 12, "lo90": float(lo), "hi90": float(hi),
                         "sig": bool(lo > 0 or hi < 0), "null_sig_prob": float(((plo > 0) | (phi < 0)).mean()),
                         "piv_agg": bool(L.piv_agg[i]), "het": bool(L.het[i])})
    P = pd.DataFrame(rows)
    nsig = P.groupby(["traj_id", "turn"]).sig.sum()
    pairs_out = {
        "n_pairs": int(len(P)), "n_points": int(len(L)),
        "sig": frac_ci(P.traj_id.values, P.sig),
        "null_expected_sig": float(P.null_sig_prob.sum()),
        "n_sig_per_point_hist": {str(k): int(v) for k, v in nsig.value_counts().sort_index().items()},
        "sig_by_regime": {"agg_pivotal": frac_ci(P.traj_id.values, P.sig, P.piv_agg),
                          "indifferent": frac_ci(P.traj_id.values, P.sig, ~P.piv_agg)},
        "sig_sign": {"pos (factual better)": int((P.sig & (P.C_j > 0)).sum()), "neg": int((P.sig & (P.C_j < 0)).sum())},
    }

    # ---------------- T-ref: reference dependence
    g = P[P.sig].groupby(["traj_id", "turn"]).C_j
    tref = pd.DataFrame({"n": g.size(), "mixed": g.apply(lambda s: bool((s > 0).any() and (s < 0).any()))}).reset_index()
    tref = tref[tref.n >= 2]
    au = json.load(open(os.path.join(common.ROOT, "eval", "au_sensitivity.json")))
    het_out["T_ref"] = {"points_with_ge2_sig": int(len(tref)),
                        "sign_disagreement": frac_ci(tref.traj_id.values, tref.mixed) if len(tref) else None,
                        "a_u_flip_rate_D039": {k: au[k] for k in au if k != "provenance"}}

    # ---------------- regimes: outcome x progress tercile (agg-based; het-based alongside)
    df["piv_agg"] = False
    df["het"] = False
    key = dict(zip(zip(L.traj_id, L.turn), zip(L.piv_agg, L.het)))
    for k, (ag, ht) in key.items():
        m = (df.traj_id == k[0]) & (df.turn == k[1])
        df.loc[m, "piv_agg"], df.loc[m, "het"] = ag, ht
    df["regime"] = np.where(~df.live, "dead", np.where(df.piv_agg, "pivotal", "indifferent"))
    df["regime_het"] = np.where(~df.live, "dead", np.where(df.het, "pivotal_het", "live_not_het"))
    cross = {}
    for (w, pg), gdf in df.groupby(["won", "prog3"]):
        cell = {"n": int(len(gdf))}
        for r in ("dead", "indifferent", "pivotal"):
            cell[r] = frac_ci(gdf.traj_id.values, gdf.regime == r)
        cell["pivotal_het"] = frac_ci(gdf.traj_id.values, gdf.regime_het == "pivotal_het")
        cell["pivotal_of_live"] = frac_ci(gdf.traj_id.values, gdf.regime == "pivotal", gdf.live)
        cross[f"won={w}|prog3={pg}"] = cell
    het_out["regimes_outcome_x_progress"] = cross
    het_out["regimes_progress_only_pivotal_of_live"] = {
        str(pg): frac_ci(gdf.traj_id.values, gdf.regime == "pivotal", gdf.live) for pg, gdf in df.groupby("prog3")}

    for o, name in ((het_out, "het_fork"), (pairs_out, "pairs")):
        o["provenance"] = prov
        common.write_json(os.path.join(common.ROOT, "eval", f"{name}.json"), o)
    L.drop(columns=[c for c in L.columns if c[:1] in "fa" and c[1:].isdigit()]).to_parquet(
        os.path.join(common.ROOT, "eval", "het_points.parquet"))
    P.to_parquet(os.path.join(common.ROOT, "eval", "pairs_points.parquet"))
    show = {k: het_out[k] for k in ("pivotal_het", "overlap_2x2", "masked_fork", "null_calibration", "T_ref",
                                     "regimes_progress_only_pivotal_of_live")}
    print(json.dumps(show, indent=1, ensure_ascii=False, default=float)[:5000])
    print(json.dumps({k: v for k, v in pairs_out.items() if k != "provenance"}, indent=1, default=float))


if __name__ == "__main__":
    main()
