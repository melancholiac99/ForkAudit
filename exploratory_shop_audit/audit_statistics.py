"""Saved-data estimators; no environment or model imports."""
import numpy as np


def reliability_components(values, weights=None):
    """All ordered cross-repeat pairs, one statistical unit per state.

    values: candidate x independent repeat. Deterministic terminal arms are
    excluded by the caller from the primary estimator.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 2 or not np.isfinite(y).all():
        raise ValueError('Need >=2 candidates and >=2 finite independent repetitions')
    w = np.ones(y.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (y.shape[0],) or (w < 0).any() or np.count_nonzero(w) < 2:
        raise ValueError('Need >=2 supported candidates')
    w = w / w.sum()
    z = y - (w[:, None] * y).sum(axis=0, keepdims=True)
    k = y.shape[1]
    cross = (z.sum(axis=1) ** 2 - (z ** 2).sum(axis=1)) / (k * (k - 1))
    return float(w @ cross), float(w @ z.var(axis=1, ddof=1))


def reliability(states):
    if not states:
        return {'states': 0, 'S': None, 'N': None, 'rho': {}, 'identifiable': False}
    s, n = np.mean(np.asarray(states), axis=0)
    rho = {}
    for depth in (1, 3, 6):
        denominator = s + n / depth
        rho[str(depth)] = float(s / denominator) if abs(denominator) > 1e-12 else None
    return {'states': len(states), 'S': float(s), 'N': float(n), 'rho': rho,
            'identifiable': bool(s > 0 and all(s + n / d > 1e-12 for d in (1, 3, 6))),
            'finite_sample_out_of_range': any(v is not None and not 0 <= v <= 1 for v in rho.values())}


def bootstrap_mean(values, seed, repetitions=2000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {'n_tasks': 0, 'estimate': None, 'ci95': None}
    rng = np.random.default_rng(seed)
    draws = x[rng.integers(0, len(x), size=(repetitions, len(x)))].mean(axis=1)
    return {'n_tasks': len(x), 'estimate': float(x.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist()}


def bootstrap_reliability(components, seed, repetitions=2000):
    point = reliability(components)
    if not components:
        return point
    rng = np.random.default_rng(seed)
    x = np.asarray(components)
    draws = [reliability(x[index].tolist()) for index in rng.integers(0, len(x), (repetitions, len(x)))]
    point['ci95'] = {key: np.quantile([d[key] for d in draws], [.025, .975]).tolist() for key in ('S', 'N')}
    point['rho_ci95'] = {}
    for depth in ('1', '3', '6'):
        finite = [d['rho'][depth] for d in draws if d['rho'][depth] is not None]
        point['rho_ci95'][depth] = np.quantile(finite, [.025, .975]).tolist() if finite else None
    point['bootstrap_nonpositive_S_fraction'] = sum(d['S'] <= 0 for d in draws) / repetitions
    return point


def funnel(surface, opened, item, actual):
    values = [1 - surface, surface - opened, opened - item, item - actual]
    if abs(sum(values) - (1 - actual)) > 1e-8:
        raise ValueError('Funnel identity failed')
    if min(values[1:]) < -1e-8:
        raise ValueError('Funnel sets/option maximum do not contain actual purchase')
    return values
