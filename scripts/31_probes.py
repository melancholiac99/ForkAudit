"""Phase 2 (FEATURES_FROZEN §4): state-value baseline V and arm-C hidden-state probes, out-of-fold by trajectory.

V  : ridge on [turn_index, n_admissible, prompt_end hidden(L)], target = continuation-pool mean success of bank 1 and of
     bank 2 (two fits, cross-fitted, V = their mean); layer L in {14, 21, 26} and lambda in 10^{-4..4} chosen per outer
     fold by inner 5-fold GroupKFold MSE. Evaluated on D. Per-fold models saved for W1-V (peek states).
C  : one L2-logistic probe per (position, layer) and boundary (B1 on D, B2-het and B2-Credit-pivotal on 𝓛, B3 on
     Credit-pivotal AND het, L2-sign on 𝓛 \\ {A = 0}); C chosen by inner CV (evalcore.fit_oof_logistic). The inner
     AUROC tables are kept so the B+C model can pick the prompt_end layer by inner CV.
Outputs data/probes/{V_oof.parquet, V_models.npz, C_oof.parquet, probes.json}.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common
import evalcore as E

LAYERS = (14, 21, 26)
POS = ("prompt_end", "pre_action")
ALPHAS = E.C_GRID


def hidden(pos, L, keys):
    idx = pd.read_parquet(os.path.join(common.ROOT, "hidden", "index.parquet"))
    row = dict(zip(zip(idx.traj_id, idx.turn), idx.row))
    H = np.load(os.path.join(common.ROOT, "hidden", f"{pos}_{L}.npy"), mmap_mode="r")
    return np.asarray(H[[row[k] for k in keys]], dtype=np.float64)


def v_fit(D, F, n_jobs):
    from joblib import Parallel, delayed
    keys = list(zip(D.traj_id, D.turn))
    ctrl = F.loc[keys, ["turn_index", "n_admissible"]].values.astype(float)
    Xs = {L: np.hstack([ctrl, hidden("prompt_end", L, keys)]) for L in LAYERS}
    g = D.traj_id.values
    fold = E.outer_folds(g)

    def inner(L, a, b, y):
        p = E.Prep().fit(Xs[L][a])
        pr = E.ridge_kernel_path(p.transform(Xs[L][a]), y[a], p.transform(Xs[L][b]), ALPHAS)
        return {al: float(np.mean((pr[al] - y[b]) ** 2)) for al in ALPHAS}

    out, models = {}, {}
    for tgt in ("pool_b1", "pool_b2"):
        y = D[tgt].values.astype(float)
        oof = np.full(len(y), np.nan)
        for k in range(5):
            tr = np.where(fold != k)[0]
            splits = list(GroupKFold(5).split(tr, groups=g[tr]))
            res = Parallel(n_jobs=n_jobs)(delayed(inner)(L, tr[a], tr[b], y) for L in LAYERS for a, b in splits)
            best, bl = np.inf, None
            for li, L in enumerate(LAYERS):
                for al in ALPHAS:
                    m = np.mean([res[li * 5 + j][al] for j in range(5)])
                    if m < best:
                        best, bl = m, (L, al)
            L, al = bl
            te = np.where(fold == k)[0]
            p = E.Prep().fit(Xs[L][tr])
            w, ym = E.ridge_primal(p.transform(Xs[L][tr]), y[tr], al)
            oof[te] = p.transform(Xs[L][te]) @ w + ym
            models[f"{tgt}_k{k}"] = {"layer": L, "alpha": al, "mu": p.mu, "sd": p.sd, "keep": p.keep, "w": w, "ym": ym}
        out[tgt] = oof
    D = D[["traj_id", "turn"]].copy()
    D["V_b1"], D["V_b2"] = out["pool_b1"], out["pool_b2"]
    D["V"] = 0.5 * (D.V_b1 + D.V_b2)
    D["fold"] = fold
    return D, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_jobs", type=int, default=12)
    ap.add_argument("--only", default=None, help="comma list of parts: V,B1,B2het,B2cp,B3,L2sign (default all)")
    args = ap.parse_args()
    parts = set((args.only or "V,B1,B2het,B2cp,B3,L2sign").split(","))
    t0 = time.time()
    od = os.path.join(common.ROOT, "data", "probes")
    os.makedirs(od, exist_ok=True)
    prov = common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}
    lab = E.labels()
    F = pd.read_parquet(os.path.join(common.ROOT, "data", "features_B.parquet")).set_index(["traj_id", "turn"])
    info = json.load(open(os.path.join(od, "probes.json"))) if os.path.exists(os.path.join(od, "probes.json")) else {}
    if "V" in parts:
        Vdf, models = v_fit(lab, F, args.n_jobs)
        Vdf.to_parquet(os.path.join(od, "V_oof.parquet"))
        np.savez(os.path.join(od, "V_models.npz"),
                 **{f"{k}__{f}": np.asarray(v[f]) for k, v in models.items() for f in v})
        info["V"] = {"models": {k: {"layer": v["layer"], "alpha": v["alpha"]} for k, v in models.items()},
                     "corr_V_b1_b2": float(np.corrcoef(Vdf.V_b1, Vdf.V_b2)[0, 1]), "seconds": time.time() - t0}
        print("V done", info["V"], flush=True)
    sets = {"B1": (lab, "live"), "B2het": (lab[lab.live], "het"), "B2cp": (lab[lab.live], "credit_pivotal"),
            "B3": (lab[lab.cap], "b3_pos"), "L2sign": (lab[lab.live & (lab.A_3456 != 0)], None)}
    cpath = os.path.join(od, "C_oof.parquet")
    C = pd.read_parquet(cpath) if os.path.exists(cpath) else lab[["traj_id", "turn"]].copy()
    for name, (S, ycol) in sets.items():
        if name not in parts:
            continue
        y = (S.A_3456 > 0).values if ycol is None else S[ycol].values
        keys = list(zip(S.traj_id, S.turn))
        g = S.traj_id.values
        fold = E.outer_folds(g)
        info.setdefault("C", {})[name] = {"n": len(S), "pos": int(np.sum(y))}
        for pos in POS:
            for L in LAYERS:
                t1 = time.time()
                X = hidden(pos, L, keys)
                oof, fi = E.fit_oof_logistic(X, y, g, fold, n_jobs=args.n_jobs)
                col = f"C_{name}_{pos}_{L}"
                C[col] = pd.Series(oof, index=pd.MultiIndex.from_tuples(keys)).reindex(
                    pd.MultiIndex.from_frame(C[["traj_id", "turn"]])).values
                info["C"][name][col] = fi | {"oof_auroc": E.auroc(y, oof), "seconds": time.time() - t1}
                print(name, col, f"AUROC {E.auroc(y, oof):.3f}", fi, f"{time.time() - t1:.0f}s", flush=True)
        C.to_parquet(cpath)
    info["provenance"] = prov
    common.write_json(os.path.join(od, "probes.json"), info)
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
