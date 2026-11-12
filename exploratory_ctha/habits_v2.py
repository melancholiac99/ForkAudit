"""CTHA v2: mixed-outcome group regression, task-type OOF gate, GiGPO-style anchor credit.

The habit model estimates predictive conditional return contrasts, not causal
per-action Q. All fitting inputs come from train-split on-policy episodes.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np

from features_v2 import (feature_names, prefix_feature_rate, score_candidate,
                         step_features, with_action)

MODES = ("ctha", "grpo", "fixed", "ctha_all", "ctha_nogate", "ctha_nocenter")


def _groups(episodes):
    groups = defaultdict(list)
    for e in episodes:
        if "task_type" not in e:
            raise ValueError("CTHA v2 requires task_type for type-held-out evaluation")
        groups[(int(e["iteration"]), e["game_file"])].append(e)
    return groups


def _mixed_episodes(episodes):
    return [g for g in _groups(episodes).values()
            if len(g) >= 2 and 0 < sum(int(e["won"]) for e in g) < len(g)]


def num_pairs(episodes):
    groups = _mixed_episodes(episodes)
    return sum(sum(int(e["won"]) for e in g) *
               (len(g) - sum(int(e["won"]) for e in g)) for g in groups)


def fit_ridge(episodes, *, family="habit", window=8, alpha=5.,
              min_mixed=2, shuffle=False, seed=20261112, center=True,
              cap=0.5):
    """Within-(iteration,game) contrasts; successes vs failures only.

    Centering across trajectories of same game removes constant task offset.
    Rate-normalized prefix features mitigate outcome-dependent trajectory length.
    """
    groups = _mixed_episodes(episodes)
    d = len(feature_names(family))
    n_pair = num_pairs(episodes)
    if len(groups) < min_mixed:
        return np.zeros(d), dict(active=False, mixed_groups=len(groups),
                                 pairs=n_pair, reason="insufficient_mixed_groups")
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for g in groups:
        x = np.stack([prefix_feature_rate(e, window, family) for e in g])
        y = np.asarray([float(e["won"]) for e in g])
        if shuffle:
            y = rng.permutation(y)
        if center:
            x -= x.mean(axis=0)
            y -= y.mean()
        X.append(x)
        Y.append(y)
    matrix = np.concatenate(X)
    targets = np.concatenate(Y)
    beta = np.linalg.solve(matrix.T @ matrix + alpha * np.eye(d), matrix.T @ targets)
    if not np.isfinite(beta).all():
        raise ValueError("Nonfinite ridge coefficients")
    beta = np.clip(beta, -cap, cap)
    return beta, dict(active=True, mixed_groups=len(groups), pairs=n_pair,
                      fit_mse=float(np.mean((matrix @ beta-targets)**2)))


def pair_scores(test_episodes, beta, *, family="habit", window=8):
    """Per game-pair sums. Pairs are success x failure within an iteration."""
    records = defaultdict(lambda: [0., 0])
    for g in _mixed_episodes(test_episodes):
        scored = [(e, float(beta @ prefix_feature_rate(e, window, family))) for e in g]
        succ = [val for e, val in scored if e["won"]]
        fail = [val for e, val in scored if not e["won"]]
        for a in succ:
            for b in fail:
                records[g[0]["game_file"]][0] += float(a > b) + 0.5*float(a == b)
                records[g[0]["game_file"]][1] += 1
    return records


def _collect_predictions(episodes, *, family, window, alpha, min_mixed,
                         permutations, seed, center, train_exclude_types=True):
    """Leave entire task types out for evaluation, not just individual games."""
    types = sorted({e["task_type"] for e in episodes})
    by_game = defaultdict(lambda: [0., 0])
    nulls = [defaultdict(lambda: [0., 0]) for _ in range(permutations)]
    folds = []
    for typ in types:
        train = [e for e in episodes if e["task_type"] != typ]
        test = [e for e in episodes if e["task_type"] == typ]
        beta, fit = fit_ridge(train, family=family, window=window,
                              alpha=alpha, min_mixed=min_mixed, center=center)
        real = pair_scores(test, beta, family=family, window=window)
        for game, values in real.items():
            by_game[game][0] += values[0]
            by_game[game][1] += values[1]
        folds.append(dict(task_type=typ, fitted=fit, heldout_pairs=num_pairs(test)))
        for p in range(permutations):
            # Shuffle outcomes in TRAIN groups, never in held-out test groups.
            null_beta, _ = fit_ridge(train, family=family, window=window, alpha=alpha,
                                     min_mixed=min_mixed, shuffle=True,
                                     seed=seed + p * 10007, center=center)
            for game, values in pair_scores(test, null_beta, family=family, window=window).items():
                nulls[p][game][0] += values[0]
                nulls[p][game][1] += values[1]
    return by_game, nulls, folds


def oof_audit(episodes, *, family="habit", window=8, alpha=5.,
              min_mixed=8, min_pairs=200, permutations=12, bootstraps=300,
              seed=20261112, min_delta=0.0, center=True):
    """Predictive gate: real model must beat shuffled training-label null on OOF task types.

    Cluster-bootstrap by game; require positive heldout pairwise excess over null.
    Never interpret pairwise accuracy as a causal advantage estimate.
    """
    if permutations < 1 or bootstraps < 1 or min_pairs < 1:
        raise ValueError("Permutations, bootstraps and min_pairs must be >=1")
    mixed = len(_mixed_episodes(episodes))
    pairs = num_pairs(episodes)
    types = sorted({e["task_type"] for e in episodes})
    result = dict(family=family, total_mixed_groups=mixed, total_pairs=pairs,
                  heldout_types=types, n_types=len(types), permutations=permutations,
                  bootstraps=bootstraps, min_pairs=min_pairs, min_mixed=min_mixed)
    if mixed < min_mixed or pairs < min_pairs or len(types) < 3:
        return dict(result, passed=False, reason="warmup_or_insufficient_task_types",
                    accuracy=None, null_accuracy=None, delta=None, ci95=None)
    real, nulls, folds = _collect_predictions(
        episodes, family=family, window=window, alpha=alpha,
        min_mixed=max(2, min_mixed//3), permutations=permutations,
        seed=seed, center=center)
    games = sorted(g for g, val in real.items() if val[1] > 0)
    if not games:
        return dict(result, passed=False, reason="no_heldout_mixed_groups",
                    accuracy=None, null_accuracy=None, delta=None, ci95=None,
                    folds=folds)
    count = np.array([real[g][1] for g in games])
    score = np.array([real[g][0] for g in games])
    null_score = np.array([np.mean([p[g][0] for p in nulls]) for g in games])
    total = float(np.sum(count))
    observed = float(score.sum()/total)
    baseline = float(null_score.sum()/total)
    delta = observed-baseline
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(games), size=(bootstraps, len(games)))
    boot = ((score[ids]-null_score[ids]).sum(axis=1) /
            np.maximum(1., count[ids].sum(axis=1)))
    ci = np.percentile(boot, [2.5, 97.5]).tolist()
    passed = bool(delta > min_delta and ci[0] > min_delta and observed > 0.5)
    return dict(result, passed=passed, reason="passed" if passed else "not_above_shuffle",
                accuracy=observed, null_accuracy=baseline, delta=delta,
                ci95=[float(x) for x in ci], heldout_games=len(games),
                heldout_pairs=int(total), folds=folds)


def type_heldout_betas(episodes, *, family, window, alpha, min_mixed,
                       stable_sign=0.75, center=True):
    """Produce one vector per target task TYPE, fitting on all *other* types."""
    types = sorted({e["task_type"] for e in episodes})
    betas, diagnostics = {}, {}
    for typ in types:
        other = [e for e in episodes if e["task_type"] != typ]
        beta, diag = fit_ridge(other, family=family, window=window,
                               alpha=alpha, min_mixed=min_mixed, center=center)
        # Leave-task-type-out sign stability within the permissible training types.
        if diag["active"] and stable_sign > 0:
            alternatives = []
            for sub_type in sorted({e["task_type"] for e in other}):
                sub = [e for e in other if e["task_type"] != sub_type]
                b, sub_diag = fit_ridge(sub, family=family, window=window,
                                        alpha=alpha, min_mixed=min_mixed, center=center)
                if sub_diag["active"]:
                    alternatives.append(b)
            if alternatives:
                v = np.stack(alternatives)
                agree = np.mean(np.sign(v) == np.sign(beta)[None, :], axis=0)
                beta = beta * (agree >= stable_sign)
                diag["retained_coefficients"] = int(np.count_nonzero(beta))
                diag["stability_folds"] = len(alternatives)
            else:
                beta *= 0.
                diag["reason"] = "no_stable_subtypes"
        betas[typ] = beta
        diagnostics[typ] = diag
    return betas, diagnostics


def action_base_advantages(episodes, *, base="gigpo", local_weight=1.,
                           gamma=1.0):
    """GiGPO-style exact-observation anchor grouping, or episode GRPO baseline.

    For sparse final reward gamma=1 means step return=terminal R.
    This matches GiGPO's episode + anchor-state mean-normalized structure, not
    the distributed reference implementation or its training hyperparameters.
    """
    if base not in ("grpo", "gigpo"):
        raise ValueError("Expected grpo or gigpo")
    if not 0 < gamma <= 1 or local_weight < 0:
        raise ValueError("gamma in (0,1], local_weight >=0 required")
    groups = defaultdict(list)
    for e in episodes:
        groups[e["game_file"]].append(e)
    ret, all_failed, anchor_groups = {}, {}, defaultdict(list)
    for game, group in groups.items():
        rewards = np.asarray([float(e["won"]) for e in group], dtype=float)
        mean = float(rewards.mean())
        std = float(rewards.std())
        all_failed[game] = bool(np.max(rewards) == 0)
        for ep, r in zip(group, rewards):
            # GRPO has std normalization; default GiGPO "mean_norm" does not.
            episodic = ((float(r)-mean)/(std+1e-6) if base == "grpo" and std > 0
                        else float(r)-mean)
            for i, step in enumerate(ep["steps"]):
                key = (ep["episode_id"], i)
                ret[key] = episodic
                if base == "gigpo":
                    obs = str(step["obs_before"])
                    # Keep anchor group restricted to SAME task and exact visible observation.
                    value = float(r) * gamma ** (len(ep["steps"])-1-i)
                    anchor_groups[(game, obs)].append((key, value))
    n_anchor = 0
    nonzero = 0
    if base == "gigpo":
        for values in anchor_groups.values():
            if len(values) < 2:
                continue
            n_anchor += 1
            avg = float(np.mean([v for _, v in values]))
            for key, v in values:
                delta = local_weight * (v-avg)
                ret[key] += delta
                nonzero += int(abs(delta) > 1e-12)
    return ret, all_failed, dict(base=base, local_weight=local_weight,
                                  n_repeated_anchors=n_anchor,
                                  n_local_nonzero=nonzero)


def fixed_rule_score(step, ep) -> float:
    """No learned weight; descriptive fixed habit-shaping negative control."""
    a = str(step["action"]).lower().strip()
    prompt = str(step["prompt"]).lower()
    if a.startswith(("examine ", "look ")):
        return -0.1
    if a.startswith("take "):
        from habits import task_target
        goal = task_target(ep["task"])
        obj = a[5:].split(" from ", 1)[0]
        if goal and goal not in obj:
            return -0.3
    if a.startswith("go to ") and a not in prompt.split("action history:")[-1][:120]:
        return 0.15
    return 0.0


def fixed_step_score(step, ep, baseline):
    if baseline == "zero":
        return fixed_rule_score(step, ep)
    if baseline != "legal":
        raise ValueError("Unknown baseline")
    candidates = list(dict.fromkeys(step["admissible"]))
    if not candidates:
        return fixed_rule_score(step, ep)
    score = fixed_rule_score(step, ep)
    baseline_score = float(np.mean([fixed_rule_score(with_action(step, a), ep)
                                    for a in candidates]))
    return score-baseline_score


def make_batch(episodes, prior, *, mode="ctha", family="habit",
               feature_window=8, aux_horizon=8, train_horizon=0,
               alpha=5., min_mixed=8, min_pairs=200,
               permutations=12, bootstraps=300, min_delta=0.0,
               sign_fraction=0.75, omega=0.30, clip=2.0,
               base="gigpo", local_weight=1., gamma=1.,
               baseline="legal"):
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode}")
    if feature_window < 1 or train_horizon < 0 or aux_horizon < 0:
        raise ValueError("Negative training window; 0=full for train and auxiliary")
    if omega < 0 or clip <= 0:
        raise ValueError("Invalid auxiliary coefficient/clip")
    current_games = {e["game_file"] for e in episodes}
    pool = prior + episodes
    observed_types = {e["task_type"] for e in pool}
    base_adv, all_failed, base_diag = action_base_advantages(
        episodes, base=base, local_weight=local_weight, gamma=gamma)
    if mode in ("grpo", "fixed"):
        gate = dict(passed=False, reason="no_learned_credit")
        betas = {}
        fit_diag = {}
    elif mode == "ctha_nogate":
        gate = dict(passed=True, reason="deliberately_bypassed",
                    total_mixed_groups=len(_mixed_episodes(pool)), total_pairs=num_pairs(pool))
        betas, fit_diag = type_heldout_betas(pool, family=family,
            window=feature_window, alpha=alpha, min_mixed=max(2, min_mixed//3),
            stable_sign=0.)
    else:
        gate = oof_audit(pool, family=family, window=feature_window,
                         alpha=alpha, min_mixed=min_mixed, min_pairs=min_pairs,
                         permutations=permutations, bootstraps=bootstraps,
                         min_delta=min_delta, center=(mode!="ctha_nocenter"))
        if gate["passed"]:
            betas, fit_diag = type_heldout_betas(
                pool, family=family, window=feature_window, alpha=alpha,
                min_mixed=max(2, min_mixed//3),
                stable_sign=sign_fraction, center=(mode!="ctha_nocenter"))
        else:
            betas, fit_diag = {}, {}
    samples, raw = [], []
    d = len(feature_names(family))
    for ep in episodes:
        typ = ep["task_type"]
        beta = betas.get(typ, np.zeros(d))
        train_steps = ep["steps"] if train_horizon == 0 else ep["steps"][:train_horizon]
        for i, st in enumerate(train_steps):
            tokens = st.get("response_token_ids")
            if not isinstance(tokens, list) or not tokens or not all(isinstance(t, int) and t >= 0 for t in tokens):
                raise ValueError(f"Missing real vLLM completion token IDs: {ep['episode_id']}:{i}")
            effective = 0.
            allowed = (aux_horizon == 0 or i < aux_horizon)
            if mode == "ctha_all" or all_failed[ep["game_file"]]:
                if allowed and mode == "fixed":
                    effective = fixed_step_score(st, ep, baseline=baseline)
                elif allowed and mode not in ("grpo", "fixed") and gate["passed"]:
                    effective = score_candidate(beta, st, ep, family=family, baseline=baseline)
            samples.append(dict(episode_id=ep["episode_id"], game_file=ep["game_file"],
                                task_type=typ, turn=int(st["turn"]),
                                prompt=st["prompt"], response=st["response"],
                                response_token_ids=tokens,
                                advantage_episode=float(base_adv[(ep["episode_id"], i)]),
                                group_all_failed=bool(all_failed[ep["game_file"]])))
            raw.append(effective)
    if not samples:
        raise ValueError("No policy-update samples")
    raw = np.asarray(raw, dtype=float)
    # Scaling only over nonzero auxiliary rows: adding more base-only rows cannot
    # make the bonus explode or shrink by changing denominator.
    nz = raw[np.abs(raw) > 1e-12]
    sigma = float(nz.std()) if len(nz) > 1 else 0.
    scale = 1. / sigma if sigma > 1e-10 else 0.
    for row, value in zip(samples, raw):
        bonus = float(np.clip(value*scale, -clip, clip)) if scale else 0.
        row["raw_advantage_habit"] = float(value)
        row["advantage_habit"] = bonus
        row["advantage"] = float(row["advantage_episode"] + omega*bonus)
    report = dict(mode=mode, base=base, family=family,
                  n_episodes=len(episodes), n_samples=len(samples),
                  n_games=len(current_games), full_episode_policy_updates=train_horizon == 0,
                  feature_window=feature_window, train_horizon=train_horizon,
                  aux_horizon=aux_horizon, omega_effective=omega,
                  mixed_groups_current=len(_mixed_episodes(episodes)),
                  all_failed_groups=sum(all_failed.values()),
                  nonzero_aux_samples=int(np.count_nonzero(raw)), aux_std_nonzero=sigma,
                  gate=gate, base_diagnostics=base_diag,
                  betas_by_task_type={typ: dict(zip(feature_names(family), map(float, beta)))
                                      for typ, beta in betas.items()},
                  beta_fit_diagnostics=fit_diag,
                  n_progress_features=len(feature_names("progress")),
                  n_habit_features=len(feature_names("habit")),
                  base_advantage_mean=float(np.mean([x["advantage_episode"] for x in samples])),
                  combined_advantage_mean=float(np.mean([x["advantage"] for x in samples])))
    return samples, report
