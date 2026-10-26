"""Shared evaluation machinery (FEATURES_FROZEN §4-§5, CLAUDE.md §6.1/§6.3).

- labels(): the frozen label table (data/labels_final.parquet, built by 30_labels.py)
- outer_folds(): GroupKFold(5) by trajectory on a given evaluation set (deterministic default split)
- fit_oof_logistic / fit_oof_ridge: out-of-fold predictions with an inner 5-fold GroupKFold over the C grid
  10^{-4..4}; training-fold standardisation, median imputation + missing indicators, zero-variance columns dropped
- weighted AUROC with trajectory-cluster bootstrap weights (common random numbers per evaluation set), paired
  differences, stratified within-trajectory shuffle (progress terciles), normalised effect, REV
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import common

C_GRID = [10.0 ** k for k in range(-4, 5)]
B_BOOT = 10_000
N_SHUF = 1_000


def labels():
    return pd.read_parquet(os.path.join(common.ROOT, "data", "labels_final.parquet"))


def outer_folds(groups, n=5):
    fold = np.full(len(groups), -1)
    for k, (_, te) in enumerate(GroupKFold(n_splits=n).split(np.zeros(len(groups)), groups=groups)):
        fold[te] = k
    return fold


# ---------------------------------------------------------------- preprocessing inside a training fold
class Prep:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.miss = np.isnan(X).any(0)
        self.med = np.where(self.miss, np.nanmedian(X, 0), 0.0)
        Z = self._fill(X)
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.keep = self.sd > 1e-12
        return self

    def _fill(self, X):
        return np.where(np.isnan(X), self.med, X)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        ind = np.isnan(X[:, self.miss]).astype(float)
        Z = (self._fill(X) - self.mu) / np.where(self.sd > 1e-12, self.sd, 1.0)
        Z = Z[:, self.keep]
        return np.hstack([Z, ind]) if ind.shape[1] else Z


class _Basis:
    """liblinear L2 logistic with its penalised intercept (intercept_scaling = 1) is the L2 problem on [Z, 1]; when
    n < d the optimum lies in the row space of [Z, 1], so it is solved exactly on U*S of the thin SVD, computed once
    per training set and shared by every C (D-048). Verified against the full fit: max |Δ decision| ≈ 1e-4."""

    def __init__(self, Z):
        A = np.hstack([Z, np.ones((len(Z), 1))])
        self.reduce = A.shape[0] < A.shape[1]
        if self.reduce:
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            self.V = Vt.T
            A = U * S
        self.A = A

    def project(self, Z):
        A = np.hstack([Z, np.ones((len(Z), 1))])
        return A @ self.V if self.reduce else A

    def fit(self, y, C):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.linear_model import LogisticRegression
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m = LogisticRegression(C=C, solver="liblinear", fit_intercept=False).fit(self.A, y)  # sklearn defaults
        m.converged_ = not any(issubclass(x.category, ConvergenceWarning) for x in w)
        return m


def _inner_task(X, y, a, b):
    """One inner split: -> AUROC per C (NaN if a side has one class) and convergence flags."""
    if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
        return [np.nan] * len(C_GRID), [True] * len(C_GRID)
    p = Prep().fit(X[a])
    bas = _Basis(p.transform(X[a]))
    Zb = bas.project(p.transform(X[b]))
    sc, cv = [], []
    for C in C_GRID:
        m = bas.fit(y[a], C)
        sc.append(auroc(y[b], m.decision_function(Zb)))
        cv.append(m.converged_)
    return sc, cv


def _refit_task(X, y, tr, te, C):
    p = Prep().fit(X[tr])
    bas = _Basis(p.transform(X[tr]))
    m = bas.fit(y[tr], C)
    return m.decision_function(bas.project(p.transform(X[te]))), m.converged_


def fit_oof_logistic(X, y, groups, fold, n_inner=5, n_jobs=1):
    """-> (oof decision scores, info{chosen C per outer fold, non-converged fit count})."""
    from joblib import Parallel, delayed
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    ks = list(np.unique(fold))
    tasks = []
    for k in ks:
        tr = np.where(fold != k)[0]
        for itr, iva in GroupKFold(n_splits=n_inner).split(np.zeros(len(tr)), groups=groups[tr]):
            tasks.append((k, tr[itr], tr[iva]))
    res = Parallel(n_jobs=n_jobs)(delayed(_inner_task)(X, y, a, b) for _, a, b in tasks)
    chosen, nonconv = {}, sum(not c for _, cv in res for c in cv)
    for k in ks:
        sc = np.array([r[0] for (kk, _, _), r in zip(tasks, res) if kk == k], float)
        chosen[k] = C_GRID[int(np.nanargmax(np.nanmean(sc, 0)))]
    ref = Parallel(n_jobs=n_jobs)(delayed(_refit_task)(X, y, np.where(fold != k)[0], np.where(fold == k)[0], chosen[k])
                                  for k in ks)
    oof = np.full(len(y), np.nan)
    for k, (d, c) in zip(ks, ref):
        oof[fold == k] = d
        nonconv += (not c)
    return oof, {"C": [chosen[k] for k in ks], "nonconverged_fits": int(nonconv),
                 "n_fits": len(tasks) * len(C_GRID) + len(ks)}


def ridge_kernel_path(Ztr, ytr, Zva, alphas):
    """Ridge predictions on Zva for every alpha, via the eigen-decomposition of the training Gram matrix."""
    ym = ytr.mean()
    K = Ztr @ Ztr.T
    s, U = np.linalg.eigh(K)
    s = np.clip(s, 0, None)
    Uy = U.T @ (ytr - ym)
    Kva = Zva @ Ztr.T @ U
    return {a: Kva @ (Uy / (s + a)) + ym for a in alphas}


def ridge_primal(Ztr, ytr, alpha):
    ym = ytr.mean()
    K = Ztr @ Ztr.T
    coef = np.linalg.solve(K + alpha * np.eye(len(K)), ytr - ym)
    return Ztr.T @ coef, ym


def auroc(y, s):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.all() or (~y).all():
        return np.nan
    from scipy.stats import rankdata
    r = rankdata(s)
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (~y).sum()))


# ---------------------------------------------------------------- cluster bootstrap with common weights
class Boot:
    """Trajectory-cluster bootstrap weights for one evaluation set; all scores on that set share the weights."""

    def __init__(self, groups, B=B_BOOT, seed=0):
        self.t, self.inv = np.unique(np.asarray(groups), return_inverse=True)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(len(self.t), np.full(len(self.t), 1 / len(self.t)), size=B).astype(np.float32)

    def auroc_dist(self, y, s):
        """Weighted AUROC for every bootstrap replicate (points with non-finite s dropped)."""
        y = np.asarray(y).astype(bool)
        s = np.asarray(s, float)
        ok = np.isfinite(s)
        y, s, inv = y[ok], s[ok], self.inv[ok]
        w = self.W[:, inv]                                   # (B, n)
        order = np.argsort(s, kind="mergesort")
        s, y, w = s[order], y[order], w[:, order]
        # tie groups
        _, start, cnt = np.unique(s, return_index=True, return_counts=True)
        wp = w * y
        wn = w * (~y)
        gp = np.add.reduceat(wp, start, axis=1)
        gn = np.add.reduceat(wn, start, axis=1)
        cneg = np.cumsum(gn, axis=1) - gn                     # negatives strictly below each tie group
        num = (gp * (cneg + 0.5 * gn)).sum(1)
        den = gp.sum(1) * gn.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    @staticmethod
    def ci(v):
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def shuffle_null(y, s, groups, strata, n=N_SHUF, seed=1):
    """AUROC under within-trajectory (x stratum) permutation of the scores; -> array of n AUROCs, changed fraction."""
    rng = np.random.default_rng(seed)
    s = np.asarray(s, float)
    key = pd.Series(list(zip(groups, strata))).astype(str).values
    idx = pd.Series(np.arange(len(s))).groupby(key).apply(lambda x: x.values).tolist()
    out = np.empty(n)
    changed = []
    for i in range(n):
        p = np.arange(len(s))
        for ix in idx:
            if len(ix) > 1:
                p[ix] = rng.permutation(ix)
        out[i] = auroc(y, s[p])
        changed.append(np.mean(p != np.arange(len(s))))
    return out, float(np.mean(changed))


def summarize(y, s, boot, groups, strata, ceiling=None, ref=None):
    """Point AUROC, cluster CI, shuffle band, normalised effect; ref = {name: score} for paired differences."""
    a = auroc(y, s)
    d = boot.auroc_dist(y, s)
    sh, ch = shuffle_null(y, s, groups, strata)
    out = {"auroc": a, "ci95": Boot.ci(d), "n": int(np.isfinite(np.asarray(s, float)).sum()),
           "shuffle_p97.5": float(np.nanpercentile(sh, 97.5)), "shuffle_p2.5": float(np.nanpercentile(sh, 2.5)),
           "shuffle_changed_frac": ch}
    if ceiling:
        out["norm_effect"] = (a - 0.5) / (ceiling - 0.5)
        out["norm_effect_ci95"] = Boot.ci((d - 0.5) / (ceiling - 0.5))
    out["_dist"] = d
    if ref:
        out["paired"] = {}
        for k, (rs, rd) in ref.items():
            diff = d - rd
            out["paired"][k] = {"diff": a - auroc(y, rs), "ci95": Boot.ci(diff)}
    return out
