"""E4 shared definitions (PREREG_E4.md). Pure functions here are unit-tested in test_e4.py without GPU or ALFWorld.

Everything that decides which points are labelled, which arm is preferred, how cost is matched and how results are
summarised lives in this file, so the frozen rules are in one place.
"""
import hashlib
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "exploratory_P4", "post_e2"))

GAMES_PATH = os.path.join(HERE, "games.json")
SEED_BASE = 1000          # E4 task seeds start here (M0 used 0-99)
N_TRAIN, N_TUNE = 400, 40
QUOTA_SEED = 20261031
ARMS_TRAINED = ["A", "Aprime", "H", "Rnd", "D", "Bfull", "Bmatch", "Cmatch", "Esft"]
ARMS_ALL = ["base"] + ARMS_TRAINED
REPLAY_MIN_GAP = 1 / 3    # B/C: pair only if max-min of 3-run estimates >= 1/3
CAP = 1000                # max training examples per arm (seeded subsample); keeps training cost bounded
LR_GRID = [5e-6, 2e-5, 5e-5]

# ----------------------------------------------------------------------------- target and search phase
D_ = r"(?:a |an |some |two |the )?"
TARGET_PATS = [rf"^(?:clean|heat|cool) {D_}(\w+) and put", rf"^find {D_}(\w+) and put",
               rf"^put {D_}(?:clean |hot |cool |cold )?(\w+) (?:in|on|into)\b",
               rf"^(?:look at|examine) {D_}(\w+) (?:under|with)\b"]


def target_word(task):
    """Identical to scripts/27_features_W1.py target_word (FEATURES_FROZEN §3)."""
    t = task.lower().strip().rstrip(".")
    for p in TARGET_PATS:
        m = re.search(p, t)
        if m:
            return m.group(1)
    return None


def target_regex(word):
    return re.compile(rf"\b{re.escape(word)} \d+\b")


def search_phase(prompt_text, word):
    """Point is in the search phase iff the decision-time prompt contains no instance of the target object
    (same marker as P4's prompt_target = False). Unknown target word -> not selected."""
    return word is not None and not target_regex(word).search(prompt_text.lower())


# ----------------------------------------------------------------------------- deterministic helpers

def seeded_rng(*parts):
    h = hashlib.sha256(("E4-v1:" + ":".join(map(str, parts))).encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def display_order(point_id):
    """Candidate display order for judge prompts (same construction as P4 E2)."""
    seed = int(hashlib.sha256(f"P4-v1:30102026:{point_id}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed).permutation(5).tolist()


# ----------------------------------------------------------------------------- arm labelling (frozen rules)
# Every rule returns None (no pair at this point) or (preferred_arm_index, rejected_arm_index); arm 0 = factual.

def pair_judge(probs):
    """A (32B, F) and H (32B, L): argmax of the judge's candidate distribution; pair if it is not the factual arm."""
    j = int(np.argmax(probs))
    return None if j == 0 else (j, 0)


def pair_rule(peek_obs, word):
    """A': arms whose one-step response names a target instance. Pair if the factual arm does not and some
    alternative does; preferred = first such alternative in arm order."""
    if word is None:
        return None
    rx = target_regex(word)
    hits = [bool(rx.search(o.lower())) for o in peek_obs]
    if hits[0] or not any(hits[1:]):
        return None
    return (1 + hits[1:].index(True), 0)


def pair_random(point_id, has_pair):
    """Rnd: at exactly the points where A has a pair, a uniformly random alternative is preferred."""
    if not has_pair:
        return None
    return (int(seeded_rng("Rnd", point_id).integers(1, 5)), 0)


def pair_explore(point_id, classes):
    """D: if the factual arm is not E_new and some alternative is, a seeded-random E_new alternative is preferred."""
    if classes[0] == "E_new":
        return None
    e = [j for j in range(1, 5) if classes[j] == "E_new"]
    if not e:
        return None
    return (e[int(seeded_rng("D", point_id).integers(0, len(e)))], 0)


def pair_replay(qhat):
    """B/C: preferred = highest 3-run estimate, rejected = lowest (ties -> lowest index); pair if gap >= 1/3."""
    q = np.asarray(qhat, float)
    hi, lo = int(np.argmax(q)), int(np.argmin(q))
    if q[hi] - q[lo] < REPLAY_MIN_GAP - 1e-12:
        return None
    return (hi, lo)


def sampled_entropy(draft_actions):
    """C selector: entropy of distinct valid admissible draft actions in the first batch of 32 (CARL-style)."""
    if not draft_actions:
        return float("nan")
    _, c = np.unique(draft_actions, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def matched_count(gpu_sec_judgeF, gpu_sec_replay_total, n_replayed):
    """Number of replay-labelled points affordable at A's labelling cost (GPU seconds), capped at n_replayed."""
    per_point = gpu_sec_replay_total / n_replayed
    return int(min(n_replayed, np.floor(gpu_sec_judgeF / per_point)))


def matched_subsets(point_ids, entropies, n):
    """Bmatch: seeded uniform subset of size n. Cmatch: the n highest sampled-entropy points (ties by point id)."""
    ids = sorted(point_ids)
    rng = seeded_rng("Bmatch", len(ids), n)
    b = sorted(rng.choice(ids, size=n, replace=False).tolist()) if n else []
    order = sorted(ids, key=lambda p: (-(entropies[p] if np.isfinite(entropies[p]) else -1), p))
    return b, sorted(order[:n])


# ----------------------------------------------------------------------------- statistics (game-level paired)

def game_bootstrap_diff(succ_a, succ_b, B=10000, seed=31102026):
    """succ_x: array [n_games, n_reps] of 0/1 (reps = training seeds x eval seeds, same layout for both arms).
    Returns the mean difference (pp) and a 95% percentile CI from resampling games."""
    a, b = np.asarray(succ_a, float).mean(1), np.asarray(succ_b, float).mean(1)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    dist = d[idx].mean(1)
    return dict(diff=float(100 * d.mean()), ci95=[float(100 * np.percentile(dist, 2.5)), float(100 * np.percentile(dist, 97.5))],
                n_games=int(len(d)))


def cluster_bootstrap_mean(values, clusters, B=10000, seed=30102026):
    """Mean (pp) with a 95% percentile CI resampling clusters (M0 trajectories) with replacement."""
    v = np.asarray(values, float)
    cl, inv = np.unique(np.asarray(clusters), return_inverse=True)
    W = np.random.default_rng(seed).multinomial(len(cl), np.full(len(cl), 1 / len(cl)), size=B).astype(float)
    dist = (W @ np.bincount(inv, v, len(cl))) / (W @ np.bincount(inv, None, len(cl)))
    return dict(mean=float(100 * v.mean()), ci95=[float(100 * np.percentile(dist, 2.5)), float(100 * np.percentile(dist, 97.5))],
                n=int(len(v)), n_clusters=int(len(cl)))


def reading(pl1, pl2, pl3):
    """E4 readings, first match (PREREG_E4 §6). PL-4 / PL-5 are reported alongside, not part of the reading."""
    if not pl1:
        return "R-L-NULL"
    if not pl3:
        return "R-L-HEURISTIC"
    if not pl2:
        return "R-L-COSTLY"
    return "R-L-WIN"


def quotas(n, n_types):
    """Split n over sorted task types as evenly as possible, earlier types first (e.g. 400/6 -> 67,67,67,67,66,66)."""
    base, extra = divmod(n, n_types)
    return [base + (1 if i < extra else 0) for i in range(n_types)]


def stratified_draw(games, n, exclude, seed):
    """games: sorted [(game_file, task_type)]. Draw n games stratified by type from those not in `exclude`."""
    types = sorted({t for _, t in games})
    rng = np.random.default_rng(seed)
    picked = []
    for t, q in zip(types, quotas(n, len(types))):
        pool = [g for g, tt in games if tt == t and g not in exclude]
        idx = rng.choice(len(pool), size=q, replace=False)
        picked += [(pool[i], t) for i in sorted(idx)]
    order = rng.permutation(len(picked))
    return [picked[k] for k in order]


def cap_rows(arm, rows, cap=CAP):
    """Seeded subsample to at most `cap` rows, original order kept."""
    if len(rows) <= cap:
        return rows
    idx = sorted(seeded_rng("cap", arm, len(rows)).choice(len(rows), size=cap, replace=False).tolist())
    return [rows[i] for i in idx]


def select_lr(success_by_lr):
    """success_by_lr: {lr: mean tune success}. Highest success; ties -> smaller lr."""
    return min(success_by_lr, key=lambda lr: (-round(success_by_lr[lr], 12), lr))


# ----------------------------------------------------------------------------- io

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
