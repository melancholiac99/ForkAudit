"""E4b shared definitions (PREREG_E4b.md). Every rule that decides classes, pairs, sampling and the reading lives here.

E4b reuses E4's frozen inputs (points, 32B F judge, 3-run replay), E4's pair rules (pair_judge, pair_explore) and
E4's training / evaluation code unchanged; only which points feed which arm is new. Unit tests: test_e4b.py (CPU).
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
E4_DIR = os.path.join(ROOT, "exploratory_E4")
sys.path.insert(0, E4_DIR)
import e4_common as E                                   # noqa: E402  (also puts scripts/ and post_e2/ on sys.path)

DATA = os.path.join(HERE, "data")                        # E4b outputs (pairs, adapters, rollouts)
E4_DATA = os.path.join(E4_DIR, "data")                   # E4 inputs, read-only
PRIMARY = ["Azero", "Agap", "Dzero", "Dgap"]             # 2 x 2: labeller {A, D} x replay class {zero, gap}
ARMS = PRIMARY + ["Bfacgap"]
SEEDS = [0, 1, 2]
LR = 5e-06                                               # E4's selected learning rate; not re-tuned
N_MAX, N_MULT = 1000, 16                                 # E4 cap; N is a multiple of the effective batch
DELTA = 4.0                                              # non-inferiority margin, pp (PREREG_E4b §5)


# ----------------------------------------------------------------------------- classes and pair rules

def classify(outcomes):
    """outcomes: five lists of 3 ints (arm 0 = factual). Z3 = all 15 zero; V3 = arm means not all equal;
    E3 = all equal and not all zero (counted, never used). Returns (class, q) with q = per-arm 3-run means."""
    q = np.array([np.mean(o) for o in outcomes], float)
    if q.max() == q.min():
        return ("Z3" if q.max() == 0 else "E3"), q
    return "V3", q


def pair_bfac(q):
    """Bfac: best replayed alternative strictly above the factual mean -> (alt, 0); ties -> lowest arm index."""
    alt = np.asarray(q, float)[1:]
    j = int(np.argmax(alt))
    return (1 + j, 0) if alt[j] > q[0] else None


def arm_pair(arm, cls, q, pA, pD):
    """Pair (preferred, rejected) of `arm` at a point, or None. pA, pD: E4's pair_judge / pair_explore results."""
    if arm == "Azero":
        return pA if cls == "Z3" else None
    if arm == "Agap":
        return pA if cls == "V3" else None
    if arm == "Dzero":
        return pD if cls == "Z3" else None
    if arm == "Dgap":
        return pD if cls == "V3" else None
    if arm == "Bfacgap":
        return pair_bfac(q) if cls == "V3" else None
    raise KeyError(arm)


# ----------------------------------------------------------------------------- task-type-matched sampling

def target_shares(v3_task_types):
    """Common task-type mix for every arm = the mix of all V3 points (the population a replay filter keeps)."""
    types, counts = np.unique(np.asarray(v3_task_types), return_counts=True)
    return {t: c / counts.sum() for t, c in zip(types.tolist(), counts.tolist())}


def task_quotas(shares, n):
    """Largest-remainder split of n over task types; ties by type name."""
    types = sorted(shares)
    raw = {t: n * shares[t] for t in types}
    q = {t: int(np.floor(raw[t])) for t in types}
    rest = n - sum(q.values())
    for t in sorted(types, key=lambda t: (-(raw[t] - q[t]), t))[:rest]:
        q[t] += 1
    return q


def common_n(avail, shares, n_max=N_MAX, mult=N_MULT):
    """avail: {arm: {task_type: n_available}}. Largest multiple of `mult` <= n_max whose quotas every arm can fill."""
    for n in range(n_max - n_max % mult, 0, -mult):
        q = task_quotas(shares, n)
        if all(avail[a].get(t, 0) >= q[t] for a in avail for t in q):
            return n
    return 0


def draw(arm, rows, quotas):
    """rows: candidate pair rows of one arm (each with 'point_id', 'task_type'). Seeded draw of quotas[t] rows per
    task type without replacement; output sorted by point_id. Fails loudly if a quota cannot be met."""
    out = []
    for t in sorted(quotas):
        pool = sorted((r for r in rows if r["task_type"] == t), key=lambda r: r["point_id"])
        assert len(pool) >= quotas[t], (arm, t, len(pool), quotas[t])
        idx = E.seeded_rng("E4b-draw", arm, t, len(pool), quotas[t]).choice(len(pool), size=quotas[t], replace=False)
        out += [pool[i] for i in sorted(idx.tolist())]
    return sorted(out, key=lambda r: r["point_id"])


# ----------------------------------------------------------------------------- reading (first match)

def reading(lo_azero_base, lo_dzero_base, zg_ci, delta=DELTA):
    """zg_ci: 95% CI of Z - G = (Azero + Dzero)/2 - (Agap + Dgap)/2 on valid_unseen (pp)."""
    if not (lo_azero_base > 0 and lo_dzero_base > 0):
        return "E4b-NULL"
    if zg_ci[0] > -delta:
        return "E4b-SUPPORT"
    if zg_ci[1] < 0:
        return "E4b-COUNTER"
    return "E4b-PARTIAL"
