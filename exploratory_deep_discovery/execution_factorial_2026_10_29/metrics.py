"""Exploratory action contrasts; unbiased across distinct continuation repeats.

All arrays are [repeat, action]. Conditions may share random numbers within a
repeat, so quadratic moments exclude same-repeat products. No clipping at zero.
"""
import numpy as np


def cross_action_cov(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 2 or len(x) < 2:
        raise ValueError('Need matching [repeat, action] arrays, at least two repeats')
    n, k = x.shape
    xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    return float((xc.sum(0) @ yc.sum(0) - np.sum(xc * yc)) / (k * n * (n - 1)))


def outcome_components(x):
    x = np.asarray(x, float)
    m = x.mean(1)
    n = len(m)
    mean = float(m.mean())
    total = float(mean - (m.sum() ** 2 - m @ m) / (n * (n - 1)))
    c = cross_action_cov(x, x)
    return dict(mean_success=mean, action_variance=c,
                outcome_variance=total, residual_variance=total-c)


def completion_effect(native, completed):
    native, completed = np.asarray(native, float), np.asarray(completed, float)
    d = completed - native
    c0 = cross_action_cov(native, native)
    c1 = cross_action_cov(completed, completed)
    cd = cross_action_cov(d, d)
    cov = cross_action_cov(native, d)
    if not np.isclose(c1-c0, 2*cov+cd):
        raise AssertionError('Variance identity failed')
    return dict(mean_lift=float(d.mean()), native_C=c0, completed_C=c1,
                C_change=c1-c0, C_reduction=c0-c1,
                baseline_lift_covariance=cov, lift_variance=cd,
                equalization_term=2*cov)


def bootstrap(values, seed=20261028, reps=4000):
    x = np.asarray(values, float)
    if not len(x):
        return dict(n=0, mean=None, ci95=None)
    rng = np.random.default_rng(seed)
    samples = x[rng.integers(0, len(x), size=(reps, len(x)))].mean(1)
    return dict(n=len(x), mean=float(x.mean()),
                ci95=np.quantile(samples, [.025, .975]).tolist())
