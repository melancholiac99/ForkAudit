"""Observable habit features and cross-task within-group return regression.

The coefficients are predictive reward-decomposition statistics, NOT causal Q
values. Student-visible prompt, action and legal actions are the only inputs
used at decision time. No hidden simulator state, peek or future feedback.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURES = (
    "examine", "goto", "goto_outside_recent", "goto_recent",
    "open", "take_target", "take_non_target", "take_unknown",
    "put", "close", "toggle", "use", "inventory", "other", "invalid_action",
)
TARGET_PATTERNS = (
    r"^(?:find|clean|heat|cool)\s+(?:(?:a|an|the|some|two)\s+)?(\w+)\s+and\s+put",
    r"^put\s+(?:(?:a|an|the|some|two)\s+)?(?:(?:clean|hot|cool|cold)\s+)?(\w+)\b",
    r"^(?:look at|examine)\s+(?:(?:a|an|the|some|two)\s+)?(\w+)\b",
)
HISTORY_ACTION = re.compile(r"Action\s+\d+:\s*'([^']+)'", re.IGNORECASE)


def task_target(task: str) -> str | None:
    for pat in TARGET_PATTERNS:
        match = re.search(pat, task.strip().lower())
        if match:
            return match.group(1)
    return None


def observable_features(prompt: str, action: str, admissible: list[str], task: str) -> np.ndarray:
    """Extract only information available at the decision, including the proposed action."""
    action = " ".join(action.lower().split())
    legal = {" ".join(a.lower().split()) for a in admissible}
    f = np.zeros(len(FEATURES), dtype=np.float64)
    def setf(name):
        f[FEATURES.index(name)] = 1.0
    if action not in legal:
        setf("invalid_action")
    if action.startswith("go to "):
        setf("goto")
        recent = [" ".join(s.lower().split()) for s in HISTORY_ACTION.findall(prompt)]
        setf("goto_recent" if action in recent else "goto_outside_recent")
    elif action.startswith(("examine ", "look ", "look at ")):
        setf("examine")
    elif action.startswith("open "):
        setf("open")
    elif action.startswith("take "):
        target = task_target(task)
        # Only the item, not "from <receptacle>", may match the task goal.
        obj = re.split(r"\s+(?:from|out of)\s+", action[5:], maxsplit=1)[0]
        if target is None:
            setf("take_unknown")
        elif re.search(r"\b" + re.escape(target) + r"(?:\s+\d+)?\b", obj):
            setf("take_target")
        else:
            setf("take_non_target")
    elif action.startswith(("put ", "move ")):
        setf("put")
    elif action.startswith("close "):
        setf("close")
    elif action.startswith("toggle "):
        setf("toggle")
    elif action.startswith("use "):
        setf("use")
    elif action.startswith("inventory"):
        setf("inventory")
    else:
        setf("other")
    return f


def episode_vector(ep: dict, horizon: int) -> np.ndarray:
    accum = np.zeros(len(FEATURES), dtype=float)
    for step in ep["steps"][:horizon]:
        accum += observable_features(step["prompt"], step["action"],
                                     step["admissible"], ep["task"])
    return accum


def grouping(ep: dict) -> tuple[int, str]:
    return int(ep["iteration"]), ep["game_file"]


def centered_design(episodes: list[dict], horizon: int, center=True) -> tuple[np.ndarray, np.ndarray, int]:
    """One row per *episode*, grouped by (iteration, game), not action or token."""
    groups = defaultdict(list)
    for ep in episodes:
        groups[grouping(ep)].append(ep)
    X, y, mixed = [], [], 0
    for key in sorted(groups):
        group = groups[key]
        if len(group) < 2:
            continue
        rewards = np.asarray([float(e["won"]) for e in group])
        # Constant-reward groups provide no contrast for within-task differences.
        if np.ptp(rewards) <= 0:
            continue
        mixed += 1
        features = np.stack([episode_vector(e, horizon) for e in group])
        if center:
            features = features - features.mean(axis=0)
            rewards = rewards - rewards.mean()
        X.extend(features)
        y.extend(rewards)
    if not X:
        return np.zeros((0, len(FEATURES))), np.zeros(0), 0
    return np.asarray(X, dtype=float), np.asarray(y, dtype=float), mixed


def ridge(episodes: list[dict], horizon: int, alpha: float, center=True,
          min_mixed=8, cap=0.5) -> tuple[np.ndarray, dict]:
    if alpha <= 0:
        raise ValueError("ridge alpha must be > 0")
    X, y, mixed = centered_design(episodes, horizon, center)
    if mixed < min_mixed:
        return np.zeros(len(FEATURES)), dict(mixed_groups=mixed, active=False,
                                             reason="too_few_mixed_groups")
    beta = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ y)
    if not np.isfinite(beta).all():
        raise ValueError("Nonfinite fitted habit coefficient")
    beta = np.clip(beta, -cap, cap)
    pred = X @ beta
    mse = float(np.mean((y - pred) ** 2))
    return beta, dict(mixed_groups=mixed, active=True,
                      train_mse=mse, centered=bool(center), rows=len(y))


def sign_stability(episodes: list[dict], beta: np.ndarray, horizon: int,
                   alpha: float, *, center=True, folds=4, min_mixed=8,
                   required_fraction=0.75) -> tuple[np.ndarray, list]:
    """Drop coefficients whose signs cannot be reproduced across held-out games.

    This is a heuristic stability gate and does not confer causal validity.
    """
    games = sorted({e["game_file"] for e in episodes})
    if len(games) < folds or not np.any(beta):
        return np.zeros_like(beta), []
    copies, per_fold = [], []
    for k in range(folds):
        subset = [e for e in episodes
                  if int(hashlib.sha256(e["game_file"].encode()).hexdigest(), 16) % folds != k]
        b, diag = ridge(subset, horizon, alpha, center=center, min_mixed=min_mixed)
        copies.append(b)
        per_fold.append(dict(fold=k, **diag))
    betas = np.stack(copies)
    # A zero coefficient is never promoted to stable by sign(0)=0.
    match = np.mean(np.sign(betas) == np.sign(beta)[None, :], axis=0)
    mask = (np.abs(beta) > 1e-9) & (match >= required_fraction)
    return beta * mask, per_fold


def per_action_score(beta: np.ndarray, prompt: str, action: str,
                     admissible: list[str], task: str, baseline="legal") -> float:
    val = float(beta @ observable_features(prompt, action, admissible, task))
    if baseline == "zero" or not admissible:
        return val
    if baseline != "legal":
        raise ValueError("baseline must be legal or zero")
    # This is an action-independent baseline; it need not equal E_pi[q(h,a)].
    legal = list(dict.fromkeys(admissible))
    avg = sum(float(beta @ observable_features(prompt, a, admissible, task))
              for a in legal) / len(legal)
    return val - avg


def group_returns(episodes: list[dict], normalize=True) -> dict[str, float]:
    groups = defaultdict(list)
    for e in episodes:
        groups[e["game_file"]].append(e)
    advantages = {}
    for game, group in groups.items():
        if len(group) < 2:
            raise ValueError(f"GRPO group has fewer than 2 episodes for {game}")
        rewards = np.asarray([float(e["won"]) for e in group])
        std = float(rewards.std())
        for e, r in zip(group, rewards):
            advantages[e["episode_id"]] = float((r - rewards.mean()) / (std + 1e-6)) if normalize and std > 0 else (
                float(r - rewards.mean()) if std > 0 else 0.0)
    return advantages


def load_episodes(folder: Path, iteration: int) -> list[dict]:
    paths = sorted(folder.glob("s*_r*.json"))
    if not paths:
        raise FileNotFoundError(f"No rollout episodes under {folder}")
    episodes = []
    for path in paths:
        ep = json.loads(path.read_text())
        for key in ("game_file", "task", "seed", "rep", "won", "steps", "provenance"):
            if key not in ep:
                raise ValueError(f"{path} lacks key {key}")
        for st in ep["steps"]:
            for key in ("prompt", "response", "response_token_ids", "admissible", "action"):
                if key not in st:
                    raise ValueError(f"{path}: full rollout missing {key}")
        ep["iteration"] = iteration
        ep["episode_id"] = f"{iteration}:{ep['seed']}:{ep['rep']}"
        episodes.append(ep)
    by_game = defaultdict(set)
    for ep in episodes:
        if ep["rep"] in by_game[ep["game_file"]]:
            raise ValueError("Duplicate game/rep episode")
        by_game[ep["game_file"]].add(ep["rep"])
    return episodes


def make_batch(episodes: list[dict], prior: list[dict], *,
               mode="ctha", horizon=8, alpha=5.0, min_mixed=8,
               omega=0.30, clip=2.0, sign_fraction=0.75,
               center=True, baseline="legal") -> tuple[list[dict], dict]:
    """Produce one policy-update batch with task-held-out CTHA predictions.

    A held-out game never contributes its own outcome to its coefficient fit,
    even when there are earlier iterations of the same game.
    """
    modes = {"ctha", "grpo", "fixed", "ctha_all", "ctha_nogate", "ctha_nocenter"}
    if mode not in modes:
        raise ValueError(f"Unknown mode {mode}; expected {sorted(modes)}")
    if len(episodes) < 2 or horizon < 1 or omega < 0:
        raise ValueError("Invalid input size or horizon/omega")
    group_adv = group_returns(episodes)
    groups = defaultdict(list)
    for ep in episodes:
        groups[ep["game_file"]].append(ep)
    pool = prior + episodes  # exclude same game during each fit
    betas, fit_diag, folds = {}, {}, {}
    for game in sorted(groups):
        if mode == "grpo":
            beta, diag, fold_diag = np.zeros(len(FEATURES)), {"active": False, "reason": "grpo"}, []
        elif mode == "fixed":
            beta = np.zeros(len(FEATURES))
            # A deliberately nonlearned, fixed heuristic baseline.
            for name, value in {"examine": -0.10, "take_non_target": -0.30,
                                "goto_outside_recent": +0.15}.items():
                beta[FEATURES.index(name)] = value
            diag, fold_diag = {"active": True, "reason": "fixed_heuristic"}, []
        else:
            others = [e for e in pool if e["game_file"] != game]
            use_center = center and mode != "ctha_nocenter"
            beta, diag = ridge(others, horizon, alpha, center=use_center, min_mixed=min_mixed)
            fold_diag = []
            if diag["active"] and mode != "ctha_nogate":
                beta, fold_diag = sign_stability(others, beta, horizon, alpha,
                                                 center=use_center, min_mixed=min_mixed,
                                                 required_fraction=sign_fraction)
                diag["active_coefficients"] = int(np.count_nonzero(beta))
        betas[game], fit_diag[game], folds[game] = beta, diag, fold_diag
    raw, samples = [], []
    for ep in episodes:
        game = ep["game_file"]
        A_e = group_adv[ep["episode_id"]]
        all_failed = not any(other["won"] for other in groups[game])
        for index, st in enumerate(ep["steps"][:horizon]):
            ids = st["response_token_ids"]
            if not ids or not all(isinstance(x, int) and x >= 0 for x in ids):
                raise ValueError(f"Missing actual vLLM response token IDs at {ep['episode_id']}:{index}")
            a_h = per_action_score(betas[game], st["prompt"], st["action"],
                                   st["admissible"], ep["task"], baseline=baseline)
            if mode not in ("ctha_all", "fixed") and not all_failed:
                a_h = 0.0
            raw.append(a_h)
            samples.append(dict(episode_id=ep["episode_id"], game_file=game,
                                turn=int(st["turn"]), prompt=st["prompt"],
                                response=st["response"], response_token_ids=ids,
                                advantage_episode=A_e, raw_advantage_habit=a_h,
                                group_all_failed=bool(all_failed)))
    # Center/scaling is across **this rollout batch**; it never changes the score
    # direction, and no outcome of the focal game enters beta estimation.
    raw = np.asarray(raw, dtype=float)
    std = float(np.std(raw))
    scale = 1.0 / std if std > 1e-10 else 0.0
    for s, x in zip(samples, raw):
        shaped = float(np.clip(x * scale, -clip, clip)) if scale else 0.0
        s["advantage_habit"] = shaped
        s["advantage"] = float(s["advantage_episode"] + omega * shaped)
    report = dict(mode=mode, horizon=horizon, omega=omega, alpha=alpha,
                  n_episodes=len(episodes), n_samples=len(samples),
                  mixed_groups_current=sum(len({float(e["won"]) for e in group}) > 1 for group in groups.values()),
                  all_failed_groups=sum(not any(e["won"] for e in group) for group in groups.values()),
                  habit_raw_std=std, habit_scale=scale,
                  nonzero_aux_samples=int(np.count_nonzero(raw)),
                  beta_by_game={game: dict(zip(FEATURES, map(float, beta)))
                                for game, beta in betas.items()},
                  diagnostics_by_game=fit_diag, stability_folds=folds,
                  summary_advantages=dict(
                      episode_mean=float(np.mean([s["advantage_episode"] for s in samples])) if samples else 0,
                      habit_mean=float(np.mean([s["advantage_habit"] for s in samples])) if samples else 0,
                      combined_mean=float(np.mean([s["advantage"] for s in samples])) if samples else 0))
    if not samples:
        raise ValueError("No usable training steps: check completion token IDs and horizon")
    return samples, report
