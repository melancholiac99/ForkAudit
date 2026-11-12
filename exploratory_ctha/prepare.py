#!/usr/bin/env python3
"""Prepare on-policy GRPO / CTHA advantages from fresh, train-split episodes.

Never read E4 validation rollouts for fitting. For the focal game, habit
coefficients are trained on OTHER games only, including prior iterations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from habits import FEATURES, load_episodes, make_batch


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_write(path: Path, data: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(data)
    os.replace(tmp, path)


def validate_train(episodes, expected_iteration):
    for ep in episodes:
        prov = ep["provenance"]
        if prov.get("split") != "train" or prov.get("iteration") != expected_iteration:
            raise ValueError("Refuse fitting on non-train rollouts or mismatched iteration")
        if not ep["steps"]:
            raise ValueError("Empty trajectory has no action for policy optimization")
        if prov.get("dry"):
            raise ValueError("Dry-mode rollouts have fake token IDs; cannot train")
    games = {}
    for ep in episodes:
        games.setdefault(ep["game_file"], []).append(ep)
    K = {len(v) for v in games.values()}
    if len(K) != 1 or min(K) < 2:
        raise ValueError(f"GRPO requires equal K>=2 episodes per game: got {K}")
    if len(set(ep["episode_id"] for ep in episodes)) != len(episodes):
        raise ValueError("Duplicate train episodes")


def prepare(args):
    cur = load_episodes(args.rollouts, args.iteration)
    validate_train(cur, args.iteration)
    history = []
    origin = args.run_dir.resolve()
    for prev in range(max(0, args.iteration - args.window), args.iteration):
        folder = origin / f"round_{prev:03d}" / "rollouts"
        episodes = load_episodes(folder, prev)
        validate_train(episodes, prev)
        history.extend(episodes)
    samples, report = make_batch(
        cur, history, mode=args.mode, horizon=args.horizon,
        alpha=args.ridge_alpha, min_mixed=args.min_mixed,
        omega=args.omega, clip=args.aux_clip,
        sign_fraction=args.sign_fraction,
        baseline=args.baseline,
    )
    report.update(iteration=args.iteration, train_only=True, n_history_episodes=len(history),
                  n_current_games=len({ep["game_file"] for ep in cur}),
                  n_tokens=sum(len(s["response_token_ids"]) for s in samples),
                  features=list(FEATURES), rollouts_sha256={
                      p.name: sha256(p) for p in sorted(args.rollouts.glob("s*_r*.json"))},
                  config=dict(mode=args.mode, horizon=args.horizon, ridge_alpha=args.ridge_alpha,
                              min_mixed=args.min_mixed, window=args.window,
                              omega=args.omega, aux_clip=args.aux_clip,
                              sign_fraction=args.sign_fraction, baseline=args.baseline))
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    data = "\n".join(json.dumps(s, ensure_ascii=False) for s in samples) + "\n"
    report["samples_sha256"] = hashlib.sha256(data.encode()).hexdigest()
    samples_path, report_path = output / "samples.jsonl", output / "advantage_report.json"
    if samples_path.exists() or report_path.exists():
        if (samples_path.is_file() and report_path.is_file()
                and sha256(samples_path) == report["samples_sha256"]
                and json.loads(report_path.read_text()) == report):
            print(f"Verified existing preparation: {samples_path}")
            return report
        raise RuntimeError(f"Preparation already exists with different inputs: {output}")
    atomic_write(samples_path, data)
    atomic_write(report_path, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(dict(iteration=args.iteration, mode=args.mode,
                          episodes=len(cur), history_episodes=len(history),
                          samples=len(samples), nonzero_aux=report["nonzero_aux_samples"],
                          mixed=report["mixed_groups_current"],
                          all_failed=report["all_failed_groups"])), flush=True)
    return report


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", default="ctha", choices=[
        "ctha", "grpo", "fixed", "ctha_all", "ctha_nogate", "ctha_nocenter"])
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--ridge-alpha", type=float, default=5.)
    p.add_argument("--min-mixed", type=int, default=8)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--omega", type=float, default=0.30)
    p.add_argument("--aux-clip", type=float, default=2.)
    p.add_argument("--sign-fraction", type=float, default=0.75)
    p.add_argument("--baseline", choices=("legal", "zero"), default="legal")
    a = p.parse_args()
    if (a.iteration < 0 or a.horizon < 1 or a.min_mixed < 1 or a.window < 0
            or a.ridge_alpha <= 0 or a.omega < 0 or a.aux_clip <= 0
            or not 0 <= a.sign_fraction <= 1):
        p.error("Invalid preparation parameters")
    return a


if __name__ == "__main__":
    prepare(parse_args())
