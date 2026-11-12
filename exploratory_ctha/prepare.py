#!/usr/bin/env python3
"""CTHA v2 training-only advantage preparation, with immutable source hashes.

Important: feature_window controls the predictor's prefix. train_horizon=0 means
ALL rollout steps receive the base GRPO/GiGPO advantage. An auxiliary bonus
can separately be restricted to the first aux_horizon steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from features_v2 import feature_names
from habits import load_episodes
from habits_v2 import make_batch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, data: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


def validate_train(episodes, iteration):
    if not episodes:
        raise ValueError("No training episodes")
    per_game = {}
    for e in episodes:
        prov = e["provenance"]
        if prov.get("split") != "train" or prov.get("iteration") != iteration or prov.get("dry"):
            raise ValueError("Refusing non-train/dry or wrong-iteration episode")
        if not e.get("task_type") or not e.get("steps"):
            raise ValueError("CTHA v2 requires nonempty task type and trajectory")
        if not all("obs_before" in s and "response_token_ids" in s for s in e["steps"]):
            raise ValueError("CTHA v2 requires decision-time observations and actual vLLM tokens")
        per_game.setdefault(e["game_file"], []).append(e)
    K = {len(group) for group in per_game.values()}
    if len(K) != 1 or min(K) < 2:
        raise ValueError(f"GRPO task groups must have equal K>=2: got {K}")
    for game, group in per_game.items():
        reps = [e["rep"] for e in group]
        if len(set(reps)) != len(reps):
            raise ValueError(f"Duplicate game/repetition in {game}")


def effective_omega(base_omega, iteration, start, end):
    """Late annealing; remains constant until start, reaches zero at end."""
    if end <= start or start < 0:
        raise ValueError("Require 0 <= anneal_start < anneal_end")
    return base_omega * max(0., min(1., (end-iteration)/(end-start)))


def prepare(args):
    cur = load_episodes(args.rollouts, args.iteration)
    validate_train(cur, args.iteration)
    past = []
    for iteration in range(max(0, args.iteration - args.window), args.iteration):
        folder = args.run_dir.resolve() / f"round_{iteration:03d}" / "rollouts"
        ep = load_episodes(folder, iteration)
        validate_train(ep, iteration)
        past.extend(ep)
    omega = effective_omega(args.omega, args.iteration,
                            args.anneal_start, args.anneal_end)
    samples, report = make_batch(
        cur, past, mode=args.mode, family=args.family, base=args.base_advantage,
        feature_window=args.feature_window, train_horizon=args.train_horizon,
        aux_horizon=args.aux_horizon, alpha=args.ridge_alpha,
        min_mixed=args.min_mixed, min_pairs=args.min_pairs,
        permutations=args.permutations, bootstraps=args.bootstraps,
        min_delta=args.min_delta, sign_fraction=args.sign_fraction,
        omega=omega, clip=args.aux_clip, baseline=args.baseline,
        gamma=args.gamma, local_weight=args.local_weight)
    source_hashes = {p.name: sha256(p) for p in sorted(args.rollouts.glob("s*_r*.json"))}
    report.update(iteration=args.iteration, source="CTHA-v2-online-train",
                  train_only=True, n_history_episodes=len(past),
                  n_current_games=len({e["game_file"] for e in cur}),
                  n_tokens=sum(len(s["response_token_ids"]) for s in samples),
                  feature_names=list(feature_names(args.family)),
                  rollouts_sha256=source_hashes,
                  config=dict(mode=args.mode, family=args.family,
                    base_advantage=args.base_advantage,
                    feature_window=args.feature_window,
                    train_horizon=args.train_horizon, aux_horizon=args.aux_horizon,
                    ridge_alpha=args.ridge_alpha, min_mixed=args.min_mixed,
                    min_pairs=args.min_pairs, window=args.window,
                    omega=args.omega, omega_effective=omega,
                    anneal_start=args.anneal_start, anneal_end=args.anneal_end,
                    aux_clip=args.aux_clip, sign_fraction=args.sign_fraction,
                    baseline=args.baseline, permutations=args.permutations,
                    bootstraps=args.bootstraps, min_delta=args.min_delta,
                    local_weight=args.local_weight, gamma=args.gamma))
    args.out.mkdir(parents=True, exist_ok=True)
    data = "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n"
    report["samples_sha256"] = hashlib.sha256(data.encode()).hexdigest()
    samples_path, report_path = args.out / "samples.jsonl", args.out / "advantage_report.json"
    if samples_path.exists() or report_path.exists():
        if (samples_path.is_file() and report_path.is_file()
                and sha256(samples_path) == report["samples_sha256"]
                and json.loads(report_path.read_text()) == report):
            print(f"Verified existing preparation: {samples_path}")
            return report
        raise RuntimeError(f"Existing preparation differs from this run: {args.out}")
    atomic_write(samples_path, data)
    atomic_write(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(iteration=args.iteration, base=args.base_advantage,
          mode=args.mode, family=args.family, n_samples=len(samples),
          n_current=len(cur), n_history=len(past), gate=report["gate"]["reason"],
          gate_pass=report["gate"]["passed"],
          n_aux=report["nonzero_aux_samples"],
          all_failed=report["all_failed_groups"], omega=omega)), flush=True)
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=("ctha", "grpo", "fixed", "ctha_all",
                   "ctha_nogate", "ctha_nocenter"), default="ctha")
    p.add_argument("--family", choices=("habit", "progress", "both", "action", "manual"),
                   default="habit")
    p.add_argument("--base-advantage", choices=("grpo", "gigpo"), default="gigpo")
    p.add_argument("--feature-window", type=int, default=8)
    p.add_argument("--train-horizon", type=int, default=0, help="0=all rollout steps")
    p.add_argument("--aux-horizon", type=int, default=8, help="0=all rollout steps")
    p.add_argument("--ridge-alpha", type=float, default=5.)
    p.add_argument("--min-mixed", type=int, default=20)
    p.add_argument("--min-pairs", type=int, default=200)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--permutations", type=int, default=12)
    p.add_argument("--bootstraps", type=int, default=300)
    p.add_argument("--min-delta", type=float, default=0.)
    p.add_argument("--omega", type=float, default=0.30)
    p.add_argument("--anneal-start", type=int, default=4)
    p.add_argument("--anneal-end", type=int, default=16)
    p.add_argument("--aux-clip", type=float, default=2.)
    p.add_argument("--sign-fraction", type=float, default=0.75)
    p.add_argument("--baseline", choices=("legal", "zero"), default="legal")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--local-weight", type=float, default=1.0)
    a = p.parse_args()
    if (a.iteration < 0 or a.feature_window < 1 or a.train_horizon < 0
        or a.aux_horizon < 0 or a.ridge_alpha <= 0 or a.min_mixed < 1
        or a.min_pairs < 1 or a.window < 0 or a.permutations < 1 or a.bootstraps < 1
        or not 0 <= a.sign_fraction <= 1 or a.omega < 0
        or a.aux_clip <= 0 or not 0 < a.gamma <= 1 or a.local_weight < 0
        or not 0 <= a.anneal_start < a.anneal_end):
        p.error("Invalid CTHA v2 preparation configuration")
    return a


if __name__ == "__main__":
    prepare(parse_args())
