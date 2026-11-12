#!/usr/bin/env python3
"""CTHA v2 P0: train-split only, task-TYPE-held-out habit vs progress prediction.

Never fit on valid_seen/valid_unseen. Training-label permutation is within
(rollout iteration, game) and evaluation labels stay unchanged.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from habits import load_episodes
from habits_v2 import _collect_predictions, oof_audit
from prepare import validate_train


def accuracy_gap(episodes, *, family_a="both", family_b="progress",
                 window=8, alpha=5., bootstraps=300, seed=20261112):
    kwargs = dict(window=window, alpha=alpha, min_mixed=2,
                  permutations=0, seed=seed, center=True)
    a, _, _ = _collect_predictions(episodes, family=family_a, **kwargs)
    b, _, _ = _collect_predictions(episodes, family=family_b, **kwargs)
    ids = sorted(g for g, val in a.items() if val[1] > 0 and b[g][1] == val[1])
    if not ids:
        return dict(delta=None, ci95=None, reason="no_shared_mixed_test_groups")
    vals_a = np.array([a[k][0] for k in ids])
    vals_b = np.array([b[k][0] for k in ids])
    count = np.array([a[k][1] for k in ids])
    dif = float((vals_a.sum()-vals_b.sum())/count.sum())
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(ids), size=(bootstraps, len(ids)))
    dist = (vals_a[take]-vals_b[take]).sum(axis=1) / np.maximum(1, count[take].sum(axis=1))
    return dict(delta=dif, ci95=[float(x) for x in np.percentile(dist, [2.5, 97.5])],
                games=len(ids), pairs=int(count.sum()),
                accuracy_a=float(vals_a.sum()/count.sum()),
                accuracy_b=float(vals_b.sum()/count.sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="Completed CTHA train-only rounds")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--upto", type=int, help="Inclusive last rollout round")
    p.add_argument("--feature-window", type=int, default=8)
    p.add_argument("--ridge-alpha", type=float, default=5.)
    p.add_argument("--min-mixed", type=int, default=20)
    p.add_argument("--min-pairs", type=int, default=200)
    p.add_argument("--permutations", type=int, default=12)
    p.add_argument("--bootstraps", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261112)
    args = p.parse_args()
    if args.feature_window < 1 or args.min_mixed < 1 or args.min_pairs < 1:
        p.error("Invalid P0 parameters")
    episodes = []
    for path in sorted(args.run_dir.glob("round_[0-9][0-9][0-9]")):
        iteration = int(path.name.split("_")[1])
        if args.upto is not None and iteration > args.upto:
            continue
        if not (path / "rollouts" / "summary.json").exists():
            continue
        current = load_episodes(path / "rollouts", iteration)
        validate_train(current, iteration)
        episodes.extend(current)
    if not episodes:
        raise FileNotFoundError("No completed training rollout rounds")
    results = {}
    for family in ("manual", "action", "habit", "progress", "both"):
        results[family] = oof_audit(
            episodes, family=family, window=args.feature_window,
            alpha=args.ridge_alpha, min_mixed=args.min_mixed,
            min_pairs=args.min_pairs, permutations=args.permutations,
            bootstraps=args.bootstraps, seed=args.seed)
        print(json.dumps(dict(family=family,
             pairs=results[family]["total_pairs"], passed=results[family]["passed"],
             accuracy=results[family]["accuracy"],
             null_accuracy=results[family]["null_accuracy"],
             ci95=results[family]["ci95"])), flush=True)
    incremental = accuracy_gap(episodes, window=args.feature_window,
                               alpha=args.ridge_alpha,
                               bootstraps=args.bootstraps, seed=args.seed)
    result = dict(source="train-only", n_episodes=len(episodes),
                  n_games=len({x["game_file"] for x in episodes}),
                  n_task_types=len({x["task_type"] for x in episodes}),
                  feature_window=args.feature_window, methods=results,
                  both_minus_progress=incremental,
                  caveats=[
                    "The metric is out-of-task-type, within-game success/failure ranking, not causal credit.",
                    "Progress features must be compared before claiming habit-only gain.",
                    "Permuted training outcomes remain group-centered; heldout outcomes are never permuted.",
                    "If the CI includes zero or mixed pairs are scarce, report inconclusive, not a discovery."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(output=str(args.out),
                          both_minus_progress=result["both_minus_progress"])), flush=True)


if __name__ == "__main__":
    main()
