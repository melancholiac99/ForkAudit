"""CTHA v2: student-observable, task-agnostic interaction templates.

Design rule: no environment hidden state, future observation, reward, episode
outcome or rollout position may enter features at decision time. Generic habit
templates and task-progress indicators are explicitly separable. Hashing is
deterministic; no Python built-in hash randomization.
"""
from __future__ import annotations

import hashlib
import re
from math import log
from typing import Mapping

import numpy as np

from habits import FEATURES as MANUAL_NAMES, observable_features as manual_features

HASH_DIM = 128
HABIT_SCALARS = (
    "is_legal", "n_legal_log", "verb_legal_frequency",
    "same_as_recent", "verb_in_recent", "recent_action_distinctness",
    "action_length_log", "action_in_current_observation",
    "action_object_in_current_observation", "legal_verb_entropy",
)
PROGRESS_NAMES = (
    "action_overlap_with_goal", "object_overlap_with_goal",
    "verb_overlap_with_goal", "target_in_action", "target_visible_now",
    "goal_and_action_have_shared_number", "goal_related_object_action",
)
STOPWORDS = frozenset(
    "a an the to and or of for on in at by from with into out as it is are be "
    "your you this that current now task step observation action available "
    "admissible find put look go do need want then after before some two one "
    "there here have has can should".split()
)
ACTION_HISTORY = re.compile(r"Action\s+\d+:\s*'([^']+)'", flags=re.IGNORECASE)
TOKEN = re.compile(r"[a-z][a-z0-9_-]*", flags=re.IGNORECASE)


def word_tokens(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


def words(text: str, *, cap: int = 60) -> list[str]:
    result = [x for x in word_tokens(text) if x not in STOPWORDS and not x.isdigit()]
    return list(dict.fromkeys(result))[:cap]


def _hash(key: str) -> tuple[int, float]:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8, person=b"CTHA-v2").digest()
    number = int.from_bytes(digest, "little")
    return number % HASH_DIM, (1.0 if number & (1 << 10) else -1.0)


def _features_observable(step: Mapping, episode: Mapping):
    for field in ("prompt", "action", "admissible", "obs_before"):
        if field not in step:
            raise ValueError(f"Missing decision-time feature field: {field}")
    if "task" not in episode:
        raise ValueError("Episode has no task description")
    action = " ".join(str(step["action"]).lower().split())
    legal = [" ".join(str(x).lower().split()) for x in step["admissible"]]
    prompt = str(step["prompt"])
    obs = str(step["obs_before"]).lower()
    return action, legal, prompt, obs, str(episode["task"]).lower()


def feature_names(family="habit") -> tuple[str, ...]:
    if family == "habit":
        return tuple(f"hashed_interaction_{i:03d}" for i in range(HASH_DIM)) + HABIT_SCALARS
    if family == "progress":
        return PROGRESS_NAMES
    if family == "both":
        return feature_names("habit") + feature_names("progress")
    if family == "action":
        return tuple(f"action_hash_{i:03d}" for i in range(HASH_DIM))
    if family == "manual":
        return tuple("manual_" + name for name in MANUAL_NAMES)
    raise ValueError("feature family must be habit, progress, both, action, or manual")


def _habit(step, episode, *, action_only=False) -> np.ndarray:
    action, legal, prompt, obs, task = _features_observable(step, episode)
    tokens = word_tokens(action)
    verb = tokens[0] if tokens else "empty"
    ob_words = words(obs, cap=32)
    # One-step history is already visible in the Student's actual prompt.
    history = [" ".join(x.lower().split()) for x in ACTION_HISTORY.findall(prompt)]
    hist_verbs = [word_tokens(x)[:1] for x in history]
    features = []
    features.append("verb:" + verb)
    features += ["action:" + v for v in tokens[:8]]
    features += ["verb_object:" + verb + "|" + w for w in tokens[1:6]]
    features += ["action_pair:" + a + "|" + b for a, b in zip(tokens[:7], tokens[1:8])]
    if not action_only:
        # Reusable context/action interactions, not task identities as standalone features.
        features += [f"obs_verb:{verb}|{w}" for w in ob_words[:20]]
        features += [f"obs_action:{obj}|{w}" for obj in tokens[1:4]
                     for w in ob_words[:8]]
        features += [f"legal_verb:{verb}|{v}" for v in
                     list(dict.fromkeys(word_tokens(a)[0] for a in legal if word_tokens(a)))[:10]]
        features += [f"history_verb:{verb}|{x[0]}" for x in hist_verbs if x]
        features += ["same_recent_action" if action in history else "new_vs_recent_action"]
    v = np.zeros(HASH_DIM, dtype=float)
    # Frequency-normalized signed hashing. Avoid rewarding longer strings by default.
    norm = max(1, len(features)) ** 0.5
    for name in features:
        slot, sign = _hash(name)
        v[slot] += sign / norm
    if action_only:
        return v
    verbs = [word_tokens(a)[:1] for a in legal]
    verbs = [z[0] for z in verbs if z]
    legal_counts = {z: verbs.count(z) for z in set(verbs)}
    p = np.asarray([count / max(1, len(verbs)) for count in legal_counts.values()])
    legal_entropy = float(-np.sum(p * np.log(np.maximum(p, 1e-12)))) if len(p) else 0
    obj_terms = set(words(" ".join(tokens[1:]), cap=8))
    obs_terms = set(ob_words)
    scalars = [
        float(action in legal),
        log(1 + len(legal)) / 5.0,
        legal_counts.get(verb, 0) / max(1, len(verbs)),
        float(action in history),
        float(any(x == verb for history_verb in hist_verbs for x in history_verb)),
        len(set(history)) / max(1, len(history)),
        log(1 + len(tokens)) / 4.0,
        float(action in obs),
        len(obj_terms & obs_terms) / max(1, len(obj_terms)),
        legal_entropy / 3.0,
    ]
    return np.concatenate([v, np.asarray(scalars, dtype=float)])


def _progress(step, episode) -> np.ndarray:
    action, legal, prompt, obs, task = _features_observable(step, episode)
    tokens = word_tokens(action)
    verb = tokens[0] if tokens else ""
    a = set(words(action))
    obj = set(words(" ".join(tokens[1:])))
    g = set(words(task))
    word_overlap = len(a & g) / max(1, len(a))
    obj_overlap = len(obj & g) / max(1, len(obj))
    verb_overlap = float(verb in g)
    # Generic goal-object clue; neither target location nor future result is read.
    from habits import task_target
    goal_target = task_target(task)
    has_target = bool(goal_target and re.search(r"\b" + re.escape(goal_target) + r"(?:\s+\d+)?\b", action))
    visible_target = bool(goal_target and re.search(r"\b" + re.escape(goal_target) + r"\s+\d+\b", obs))
    goal_nums = set(re.findall(r"\b\d+\b", task))
    action_nums = set(re.findall(r"\b\d+\b", action))
    return np.asarray([
        word_overlap, obj_overlap, verb_overlap, float(has_target),
        float(visible_target), float(bool(goal_nums & action_nums)),
        float(has_target and verb in {"take", "put", "move", "heat", "cool", "clean", "use"}),
    ], dtype=float)


def step_features(step: Mapping, episode: Mapping, family="habit") -> np.ndarray:
    if family == "habit":
        return _habit(step, episode)
    if family == "progress":
        return _progress(step, episode)
    if family == "both":
        return np.concatenate([_habit(step, episode), _progress(step, episode)])
    if family == "action":
        return _habit(step, episode, action_only=True)
    if family == "manual":
        a, legal, prompt, obs, task = _features_observable(step, episode)
        return manual_features(prompt, a, legal, task)
    raise ValueError(f"Unknown feature family: {family}")


def with_action(step: Mapping, action: str) -> dict:
    # Candidate scoring changes only the action; preserves the actual decision input.
    modified = dict(step)
    modified["action"] = action
    return modified


def prefix_feature_rate(episode: Mapping, n_steps=8, family="habit") -> np.ndarray:
    if n_steps < 1:
        raise ValueError("Prefix window must be >=1")
    steps = episode["steps"][:n_steps]
    if not steps:
        raise ValueError("Episode cannot have empty steps")
    return np.stack([step_features(s, episode, family) for s in steps]).mean(axis=0)


def score_candidate(beta: np.ndarray, step: dict, ep: dict, *, family="habit",
                    baseline="legal") -> float:
    if len(beta) != len(feature_names(family)):
        raise ValueError("Habit regression coefficients have wrong feature dimension")
    val = float(beta @ step_features(step, ep, family))
    if baseline == "zero":
        return val
    if baseline != "legal":
        raise ValueError("Unknown action baseline")
    actions = list(dict.fromkeys(step["admissible"]))
    if not actions:
        return val
    return val - float(np.mean([beta @ step_features(with_action(step, a), ep, family)
                               for a in actions]))
