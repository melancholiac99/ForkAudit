"""Pure, data-only rules for E4d: do not read post-training outcomes during selection.

The original E4b Z3/V3 definitions, E4 search-phase rule, E4 model seeds,
and common game-cluster inference are reused; all E4d decisions are frozen
before any pi1 continuation. See PREREG_E4d.md.
"""
import hashlib
import json
import math
from collections import defaultdict

import numpy as np

K_ALT, MAX_DRAWS, DRAFT_BATCH = 4, 300, 32
MAX_STEPS, MIN_REMAINING = 50, 10
MAX_PER_GAME = 3
SEEDS, REPS = (0, 1, 2), 8
PRIMARY_ARMS = ("Azero", "Dzero")
SECONDARY_ARMS = ("Agap", "Dgap")
ALL_ARMS = PRIMARY_ARMS + SECONDARY_ARMS
SCREEN_OFFSET = 6 * 10**12
FORMAL_OFFSET = 7 * 10**12
FIRST_OFFSET = 8 * 10**12
PROBE_OFFSET = 9 * 10**12
PROBE_N = 24
MIN_Z3, MIN_GAMES = 80, 40
N_BOOT = 10000


def digest(data):
    if isinstance(data, (dict, list)):
        data = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    elif isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()


def evenly_spaced(rows, k=MAX_PER_GAME):
    """Order is the ascending qualifying turn, not all steps; choose bin midpoints."""
    rows = sorted(rows, key=lambda p: (p["turn"], p["point_id"]))
    n = len(rows)
    if n <= k:
        return rows
    indices = [int((j + 0.5) * n / k) for j in range(k)]
    assert len(set(indices)) == k
    return [rows[j] for j in indices]


def classify(outcomes):
    """Exactly 5 arms of 3 independent pi0 outcomes, identical to E4b.
    Z3: all 15 fail; V3: 3-run arm means differ; E3: all equal but not zero.
    """
    assert len(outcomes) == 5 and all(len(a) == 3 for a in outcomes)
    assert all(y in (0, 1) for arm in outcomes for y in arm)
    q = np.asarray(outcomes, dtype=float).mean(axis=1)
    return "Z3" if np.all(q == 0) else "E3" if np.all(q == q[0]) else "V3"


def run_path(root, arm, cond, pid, u):
    from pathlib import Path
    assert arm == "base" and cond == "00" or arm in ALL_ARMS and cond in ("01", "10", "11")
    return Path(root) / "results" / arm / cond / f"{pid}_u{u:02d}.json"


def fresh_seed(task_seed, turn, global_rep, role=1, offset=FORMAL_OFFSET):
    # Same pairing as scripts/11_replay.seed_of; role=1 used in formal conditions.
    # The within-continuation seed additionally depends on the actual continuation turn.
    return ((task_seed * 100 + turn) * 10 + role) * 100000 + global_rep * 100 + turn - 1 + offset


def game_contrasts(state_rows):
    """Input rows: state-level {game, remaining, arm: {condition: probability}}.
    Average within game, then average over games; each game is one bootstrap cluster.
    All primary cells must be present; no silent missing-seed averaging.
    """
    groups = defaultdict(list)
    for s in state_rows:
        for arm in PRIMARY_ARMS:
            assert all(c in s[arm] for c in ("00", "01", "10", "11")), (s["point_id"], arm)
        groups[s["game"]].append(s)
    result = []
    for g, rows in sorted(groups.items()):
        def m(arm, c):
            return float(np.mean([s[arm][c] for s in rows]))
        base = float(np.mean([m(a, "00") for a in PRIMARY_ARMS]))
        after = float(np.mean([m(a, "01") for a in PRIMARY_ARMS]))
        step = float(np.mean([m(a, "10") for a in PRIMARY_ARMS]))
        full = float(np.mean([m(a, "11") for a in PRIMARY_ARMS]))
        result.append(dict(game=g, n_states=len(rows), base=base, after=after, step=step, full=full,
                           rescue=after - base, one_step=step - base, primary=after - step,
                           full_gain=full - base, interaction=full - after - step + base))
    return result


def bootstrap_game(rows, field, n_boot=N_BOOT, seed=20261106):
    """Paired game-cluster percentile CI in pp; state and policy-seed means first."""
    x = np.asarray([r[field] for r in rows], dtype=float)
    if x.size == 0:
        raise ValueError("no games")
    rng = np.random.default_rng(seed)
    sums = []
    # Chunk samples to cap memory on large game sets.
    for _ in range(0, n_boot, 1000):
        n = min(1000, n_boot - len(sums))
        z = rng.integers(0, len(x), (n, len(x)))
        sums.extend(x[z].mean(axis=1).tolist())
    return {"mean_pp": float(100 * x.mean()), "ci95_pp": [float(100 * np.percentile(sums, 2.5)),
                                                         float(100 * np.percentile(sums, 97.5))],
            "n_games": int(len(rows))}


def reading(rescue_ci, main_ci, n_states, n_games):
    """Hierarchy frozen before formal pi1 results."""
    if n_states < MIN_Z3 or n_games < MIN_GAMES:
        return "E4d-UNDERPOWERED"
    if rescue_ci[0] > 0 and main_ci[0] > 0:
        return "E4d-SUPPORT"
    if main_ci[1] < 0:
        return "E4d-COUNTER"
    if rescue_ci[0] > 0:
        return "E4d-RESCUE-ONLY"
    return "E4d-INCONCLUSIVE"


def remaining_bin(remaining):
    assert remaining >= MIN_REMAINING
    return "10-19" if remaining < 20 else "20-29" if remaining < 30 else "30+"


def probe_ids(z_ids, n=PROBE_N):
    """Deterministic subset, selected after pi0 screening but before ANY pi1 observations."""
    return sorted(sorted(z_ids, key=lambda p: (digest("E4d-armprobe:" + p), p))[:n])


def crossfit_range(outcomes):
    """Five arms x 24 binary outcomes. Balance halves within all three blocks of 8.
    Select best/worst on first 4 of each block, measure their gap on held-out 4.
    Unlike the in-sample max-min range, expected gap is 0 if all arm q values coincide.
    """
    a = np.asarray(outcomes, float)
    assert a.shape == (5, 24) and np.all((a == 0) | (a == 1))
    train_idx = [u for u in range(24) if u % 8 < 4]
    test_idx = [u for u in range(24) if u % 8 >= 4]
    fit = a[:, train_idx].mean(axis=1)
    hi, lo = int(np.argmax(fit)), int(np.argmin(fit))
    held = a[:, test_idx].mean(axis=1)
    return dict(mean=float(a.mean()), range_naive=float(a.mean(axis=1).max() - a.mean(axis=1).min()),
                heldout_gap=float(held[hi] - held[lo]), selected_high=hi, selected_low=lo)
