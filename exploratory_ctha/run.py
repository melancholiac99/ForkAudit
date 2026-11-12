#!/usr/bin/env python3
"""Reproducible single-GPU CTHA online RL driver (rollout -> advantages -> LoRA).

Run each experimental arm in its OWN --run-dir. No changes to E4 frozen files,
no automatic server shutdown, no reuse of evaluation data for updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GAMES = ROOT / "exploratory_E4" / "games.json"
CODE_FILES = ("habits.py", "habits_v2.py", "features_v2.py", "rollout.py", "prepare.py", "train.py", "run.py")


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, data):
    content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text() != content:
            raise RuntimeError(f"Refusing to overwrite mismatching run metadata: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


def select_games(all_games, seed, iteration, count):
    if count > len(all_games):
        raise ValueError("tasks_per_round exceeds available train games")
    # A fixed seeded permutation is cycled in contiguous batches. Each round
    # contains distinct games, including when crossing the permutation boundary.
    order = list(all_games)
    random.Random(seed).shuffle(order)
    start = iteration * count
    return [order[(start + j) % len(order)] for j in range(count)]


def config_for(a):
    return dict(
        experiment="CTHA-v2-type-heldout-online-RL",
        model=str(a.model.resolve()),
        init_adapter=str(a.init_adapter.resolve()) if a.init_adapter else None,
        initial_adapter_sha=sha(a.init_adapter / "adapter_model.safetensors") if a.init_adapter else None,
        seed=a.seed, mode=a.mode, tasks_per_round=a.tasks_per_round,
        reps=a.reps, feature_window=a.feature_window, aux_horizon=a.aux_horizon,
        train_horizon=a.train_horizon, base_advantage=a.base_advantage,
        gamma=a.gamma, local_weight=a.local_weight, family=a.family,
        window=a.window,
        ridge_alpha=a.ridge_alpha, min_mixed=a.min_mixed, min_pairs=a.min_pairs,
        permutations=a.permutations, bootstraps=a.bootstraps, min_delta=a.min_delta,
        omega=a.omega, anneal_start=a.anneal_start, anneal_end=a.anneal_end,
        aux_clip=a.aux_clip, sign_fraction=a.sign_fraction,
        baseline=a.baseline, temperature=a.temperature,
        max_steps=a.max_steps, max_model_len=a.max_model_len,
        workers=a.workers, chunk_games=a.chunk_games, gpu_util=a.gpu_util,
        lr=a.lr, epsilon=a.epsilon, epochs=a.epochs, accum=a.accum,
        grad_clip=a.grad_clip, action_only=a.action_only,
        no_grad_checkpoint=a.no_grad_checkpoint,
        eval_every=a.eval_every, eval_reps=a.eval_reps,
        eval_games=a.eval_games, games_sha=sha(GAMES),
        code_sha={name: sha(HERE / name) for name in CODE_FILES})


def call(cmd, dry_run=False):
    print("+ " + " ".join(map(str, cmd)), flush=True)
    if not dry_run:
        subprocess.run([str(x) for x in cmd], check=True, cwd=ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", type=Path,
                   default=Path(os.environ.get("M0_MODEL", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")))
    p.add_argument("--init-adapter", type=Path)
    p.add_argument("--rounds", type=int, default=10, help="Total rounds; safe to increase on resume")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=("ctha", "grpo", "fixed", "ctha_all",
                                      "ctha_nogate", "ctha_nocenter"), default="ctha")
    p.add_argument("--tasks-per-round", type=int, default=16)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--feature-window", type=int, default=8,
                   help="Only this many early steps enter habit regression")
    p.add_argument("--train-horizon", type=int, default=0, help="0 means train all rollout steps")
    p.add_argument("--aux-horizon", type=int, default=8, help="0 means shape all steps")
    p.add_argument("--base-advantage", choices=("auto", "grpo", "gigpo"), default="auto")
    p.add_argument("--family", choices=("habit", "progress", "both", "action", "manual"), default="habit")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--local-weight", type=float, default=1.0)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--ridge-alpha", type=float, default=5.0)
    p.add_argument("--min-mixed", type=int, default=20)
    p.add_argument("--min-pairs", type=int, default=200)
    p.add_argument("--permutations", type=int, default=12)
    p.add_argument("--bootstraps", type=int, default=300)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--omega", type=float, default=0.30)
    p.add_argument("--aux-clip", type=float, default=2.0)
    p.add_argument("--sign-fraction", type=float, default=0.75)
    p.add_argument("--anneal-start", type=int, default=4)
    p.add_argument("--anneal-end", type=int, default=16)
    p.add_argument("--baseline", choices=("legal", "zero"), default="legal")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--max-model-len", type=int, default=6144)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--chunk-games", type=int, default=16)
    p.add_argument("--gpu-util", type=float, default=0.75)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--action-only", action="store_true")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--eval-every", type=int, default=0,
                   help="Optional tune-only evaluation cadence; 0 disables it")
    p.add_argument("--eval-reps", type=int, default=2)
    p.add_argument("--eval-games", type=int, default=40)
    p.add_argument("--dry-run", action="store_true", help="Print commands without GPU or environment")
    a = p.parse_args()
    if a.rounds < 1 or a.reps < 2 or a.tasks_per_round < 2 or a.window < 0 or a.eval_every < 0:
        p.error("Need rounds>=1, reps>=2, tasks_per_round>=2, window>=0")
    if a.eval_every and (a.eval_reps < 1 or a.eval_games < 1):
        p.error("Invalid tune evaluation settings")
    if (a.feature_window < 1 or a.train_horizon < 0 or a.aux_horizon < 0
            or a.min_mixed < 1 or a.min_pairs < 1 or a.permutations < 1
            or a.bootstraps < 1 or not 0 <= a.anneal_start < a.anneal_end):
        p.error("Invalid CTHA v2 configuration")
    if a.base_advantage == "auto":
        a.base_advantage = "grpo" if a.mode == "grpo" else "gigpo"
    a.run_dir = a.run_dir.resolve()
    a.model = a.model.resolve()
    a.init_adapter = a.init_adapter.resolve() if a.init_adapter else None
    if not a.dry_run and not a.model.is_dir():
        p.error(f"Local model directory missing: {a.model}")
    if a.init_adapter and not (a.init_adapter / "adapter_model.safetensors").is_file():
        p.error(f"Missing initial adapter weights: {a.init_adapter}")
    games = json.loads(GAMES.read_text())
    if a.tasks_per_round > len(games["train"]) or a.eval_games > len(games["tune"]):
        p.error("Requested game count exceeds frozen E4 train/tune pool")
    cfg = config_for(a)
    if not a.dry_run:
        write_once(a.run_dir / "config.json", cfg)
    for step in range(a.rounds):
        rnd = a.run_dir / f"round_{step:03d}"
        selected = select_games(games["train"], a.seed, step, a.tasks_per_round)
        if not a.dry_run:
            write_once(rnd / "games.json", selected)
        prior_adapter = a.init_adapter if step == 0 else (
            a.run_dir / f"round_{step-1:03d}" / "adapter")
        if step == 0 and not a.init_adapter:
            prior_adapter = None
        rollout_cmd = [
            sys.executable, HERE / "rollout.py", "--split", "train", "--games", rnd / "games.json",
            "--iteration", step, "--model", a.model, "--reps", a.reps,
            "--out", rnd / "rollouts", "--temperature", a.temperature,
            "--workers", a.workers, "--chunk-games", a.chunk_games,
            "--gpu-util", a.gpu_util, "--max-model-len", a.max_model_len,
            "--max-steps", a.max_steps]
        if prior_adapter:
            rollout_cmd.extend(["--adapter", prior_adapter])
        call(rollout_cmd, a.dry_run)
        prep_cmd = [
            sys.executable, HERE / "prepare.py", "--run-dir", a.run_dir,
            "--rollouts", rnd / "rollouts", "--iteration", step,
            "--out", rnd / "prepared", "--mode", a.mode,
            "--family", a.family, "--base-advantage", a.base_advantage,
            "--feature-window", a.feature_window, "--train-horizon", a.train_horizon,
            "--aux-horizon", a.aux_horizon, "--window", a.window,
            "--ridge-alpha", a.ridge_alpha, "--min-mixed", a.min_mixed,
            "--min-pairs", a.min_pairs, "--permutations", a.permutations,
            "--bootstraps", a.bootstraps, "--min-delta", a.min_delta,
            "--omega", a.omega, "--anneal-start", a.anneal_start,
            "--anneal-end", a.anneal_end, "--aux-clip", a.aux_clip,
            "--sign-fraction", a.sign_fraction, "--gamma", a.gamma,
            "--local-weight", a.local_weight, "--baseline", a.baseline]
        call(prep_cmd, a.dry_run)
        train_cmd = [
            sys.executable, HERE / "train.py",
            "--data", rnd / "prepared" / "samples.jsonl",
            "--advantage-report", rnd / "prepared" / "advantage_report.json",
            "--rollout-summary", rnd / "rollouts" / "summary.json",
            "--model", a.model, "--out", rnd / "adapter",
            "--temperature", a.temperature, "--max-model-len", a.max_model_len,
            "--lr", a.lr, "--epsilon", a.epsilon, "--epochs", a.epochs,
            "--accum", a.accum, "--grad-clip", a.grad_clip,
            "--seed", a.seed + step]
        if prior_adapter:
            train_cmd.extend(["--adapter", prior_adapter])
        if a.action_only:
            train_cmd.append("--action-only")
        if a.no_grad_checkpoint:
            train_cmd.append("--no-grad-checkpoint")
        call(train_cmd, a.dry_run)
        if a.eval_every and (step + 1) % a.eval_every == 0:
            eval_selected = games["tune"][:a.eval_games]
            if not a.dry_run:
                write_once(a.run_dir / "eval_tune_games.json", eval_selected)
            call([sys.executable, HERE / "rollout.py", "--split", "tune",
                  "--games", a.run_dir / "eval_tune_games.json",
                  "--iteration", step + 1, "--model", a.model,
                  "--adapter", rnd / "adapter", "--reps", a.eval_reps,
                  "--out", rnd / "eval_tune", "--temperature", a.temperature,
                  "--workers", a.workers, "--chunk-games", a.chunk_games,
                  "--gpu-util", a.gpu_util, "--max-model-len", a.max_model_len,
                  "--max-steps", a.max_steps], a.dry_run)
    print(json.dumps(dict(done=not a.dry_run, rounds=a.rounds, mode=a.mode,
                          run_dir=str(a.run_dir))), flush=True)


if __name__ == "__main__":
    main()
