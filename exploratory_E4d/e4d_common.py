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
PROBE_N = 72
PROBE_REPS = 8  # per policy, pooled across E4b training seeds (3+3+2)
PROBE_N_OPTIONS = (72, 96)
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
    """Deterministic subset from locked Z3, chosen before ANY pi1 outcomes."""
    if n not in PROBE_N_OPTIONS:
        raise ValueError(f"Probe size must be frozen from {PROBE_N_OPTIONS}: {n}")
    return sorted(sorted(z_ids, key=lambda p: (digest("E4d-armprobe:" + p), p))[:n])


def probe_seed_for(pid, rep, n_seeds=3):
    """Rotate training seeds without reading outcomes; exactly 8 trajectories per
    policy/point/arm, not 8 per training seed. Both halves contain all seeds.
    """
    assert 0 <= rep < PROBE_REPS and n_seeds in (1, 3)
    return (rep + int(digest("E4d-probe-seed:" + pid)[:8], 16)) % n_seeds


def target_peek_flags(observations, target_word):
    """One-step observation asymmetry O(h), computed in candidates before screening.
    The current prompt contains no target instance by the search-phase gate.
    """
    import re
    assert len(observations) == 5 and target_word
    pat = re.compile(r"\b" + re.escape(target_word) + r"\s+[0-9]+\b", re.I)
    flags = [bool(pat.search(text)) for text in observations]
    return dict(O=int(any(flags) and not all(flags)), flags=flags)


def seed_balanced_arm_means(values, seed_ids=None, columns=None):
    """Arm means are computed within checkpoint seed before equal seed pooling.
    For eight mixed-seed reps (3+3+2), this avoids overweighting the seed
    appearing three times. Both crossfit halves have all three seeds.
    """
    a = np.asarray(values, dtype=float)
    assert a.shape == (5, PROBE_REPS)
    ids = np.zeros(PROBE_REPS, dtype=int) if seed_ids is None else np.asarray(seed_ids, dtype=int)
    assert ids.shape == (PROBE_REPS,)
    cols = np.arange(PROBE_REPS) if columns is None else np.asarray(columns, dtype=int)
    chosen = ids[cols]
    if len(set(chosen)) != len(set(ids)):
        raise ValueError("All checkpoint seeds must occur in each crossfit half")
    return np.mean([a[:, cols[chosen == seed]].mean(axis=1) for seed in sorted(set(ids))], axis=0)


def crossfit_range(outcomes, seed_ids=None):
    """5 arms x 8 binary rewards, two-way 4+4 heldout top-minus-bottom.
    Select high/low on A, evaluate on B; reverse and average. Each half's
    per-arm estimate equally pools checkpoint seeds, even when allocation is
    3+3+2 for exactly eight trajectories.
    """
    a = np.asarray(outcomes, float)
    assert a.shape == (5, PROBE_REPS) and np.all((a == 0) | (a == 1))
    fit_a = seed_balanced_arm_means(a, seed_ids, range(4))
    fit_b = seed_balanced_arm_means(a, seed_ids, range(4,8))
    hi_a, lo_a = int(np.argmax(fit_a)), int(np.argmin(fit_a))
    hi_b, lo_b = int(np.argmax(fit_b)), int(np.argmin(fit_b))
    gap_ab, gap_ba = float(fit_b[hi_a] - fit_b[lo_a]), float(fit_a[hi_b] - fit_a[lo_b])
    means = seed_balanced_arm_means(a, seed_ids)
    return dict(mean=float(means.mean()), range_naive=float(means.max() - means.min()),
                heldout_gap=(gap_ab + gap_ba)/2,
                crossfit_ab=gap_ab, crossfit_ba=gap_ba,
                selected_high=[hi_a, hi_b], selected_low=[lo_a, lo_b])


def probe_triage(mean_gain_ci, gap_gain_ci, O_contrast_ci, n_o1, n_o0, n_games_o1, n_games_o0, O1_gap_ci=None):
    """Secondary decision framework; not a revision to E4d.reading.
    Three actionable outcomes + UNRESOLVED. Lack of significance is not
    proof of absent differentiation. O prediction requires both strata.
    """
    if (gap_gain_ci[0] > 0 and min(n_o1, n_o0) >= 10 and min(n_games_o1, n_games_o0) >= 5
        and O_contrast_ci is not None and O_contrast_ci[0] > 0
        and O1_gap_ci is not None and O1_gap_ci[0] > 0):
        return "MCR-PREDICTIVE-CANDIDATE"
    if gap_gain_ci[0] > 0:
        return "EMERGENCE-NOT-PREDICTED"
    if mean_gain_ci[0] > 0 and gap_gain_ci[0] >= -10.0 and gap_gain_ci[1] <= 10.0:
        return "UPLIFT-WITHOUT-RESOLVED-FORK"
    return "UNRESOLVED"


def bootstrap_O_contrast(state_rows, field="gap_gain", n_boot=N_BOOT, seed=20261107):
    """Within the same pre-frozen probe cohort, compare O=1 against O=0.
    Cluster by game and use a shared resampling of game IDs (so a game
    carrying both O values is not treated as independent across strata).
    Per-game means come before the across-game mean, as in E4d's primary.
    Returns None if either group is empty. This is descriptive/secondary.
    """
    groups = defaultdict(lambda: {0: [], 1: []})
    for row in state_rows:
        if row["O_h"] not in (0, 1):
            raise ValueError("O(h) missing or not binary")
        groups[row["game"]][row["O_h"]].append(float(row[field]))
    ids = sorted(groups)
    if not ids or not any(groups[g][0] for g in ids) or not any(groups[g][1] for g in ids):
        return None
    a = np.asarray([[np.mean(groups[g][bit]) if groups[g][bit] else float("nan") for bit in (0, 1)] for g in ids])
    def group_mean(sample, bit):
        vals = sample[:, bit]
        return float(np.mean(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else float("nan")
    observed = group_mean(a, 1) - group_mean(a, 0)
    rng = np.random.default_rng(seed)
    dist = []
    for _ in range(n_boot):
        sample = a[rng.integers(0, len(ids), size=len(ids))]
        m1, m0 = group_mean(sample, 1), group_mean(sample, 0)
        if np.isfinite(m1) and np.isfinite(m0):
            dist.append(m1 - m0)
    if len(dist) < max(20, int(n_boot * 0.9)):
        return None
    return dict(mean_pp=100*observed,
                ci95_pp=[100*float(np.percentile(dist, 2.5)), 100*float(np.percentile(dist, 97.5))],
                n_games_total=len(ids), n_games_O0=sum(bool(groups[g][0]) for g in ids),
                n_games_O1=sum(bool(groups[g][1]) for g in ids))
