#!/usr/bin/env python3
"""Summarize CTHA/GRPO runs without touching the raw training files."""
import argparse
import csv
import json
from pathlib import Path


def inspect(label, root):
    history, total_steps, total_updates = [], 0, 0
    for rnd in sorted(root.glob("round_[0-9][0-9][0-9]")):
        path = rnd / "rollouts" / "summary.json"
        tr = rnd / "adapter" / "train_log.json"
        ad = rnd / "prepared" / "advantage_report.json"
        if not path.exists() or not tr.exists() or not ad.exists():
            continue
        rollout, train, adv = [json.loads(p.read_text()) for p in (path, tr, ad)]
        total_steps += rollout["env_steps"]
        total_updates += train["optimizer_steps"]
        ev = rnd / "eval_tune" / "summary.json"
        val = json.loads(ev.read_text()) if ev.is_file() else None
        history.append(dict(arm=label, iteration=rollout["iteration"],
                            training_games=rollout["n_games"],
                            train_episodes=rollout["n_episodes"],
                            train_env_steps=rollout["env_steps"],
                            cumulative_env_steps=total_steps,
                            train_success=rollout["success_rate"],
                            mixed_groups=adv["mixed_groups_current"],
                            all_failed_groups=adv["all_failed_groups"],
                            aux_samples=adv["nonzero_aux_samples"],
                            optimizer_steps=train["optimizer_steps"],
                            cumulative_optimizer_steps=total_updates,
                            tune_success=val["success_rate"] if val else None,
                            tune_games=val["n_games"] if val else None))
    return history


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, help="name=/absolute/path/to/run_dir")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    a = ap.parse_args()
    if not 0 <= a.threshold <= 1:
        ap.error("threshold should be a success rate between 0 and 1")
    rows, threshold = [], {}
    for entry in a.run:
        if "=" not in entry:
            ap.error("--run must be name=directory")
        name, path = entry.split("=", 1)
        if not name or not path:
            ap.error("--run must contain nonempty name/path")
        part = inspect(name, Path(path))
        if not part:
            raise RuntimeError(f"No completed online RL rounds in {path}")
        rows += part
        first = next((x for x in part
                      if x["tune_success"] is not None and x["tune_success"] >= a.threshold), None)
        threshold[name] = (first["cumulative_env_steps"] if first else None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    p = a.out.with_suffix(".json")
    p.write_text(json.dumps(dict(
        threshold=a.threshold,
        env_steps_to_threshold=threshold,
        caveats=["Evaluation is tune split only; valid_unseen must be held out for final analysis.",
                 "Environment steps include training rollouts but not tune or optional model warm-start.",
                 "Comparisons must use same game selection, sampling budget and seeds.",
                 "CTHA auxiliary credit is predictive shaping and not guaranteed unbiased."],
        rows=rows), indent=2) + "\n")
    print(json.dumps(dict(out=str(a.out), n_records=len(rows),
                          env_steps_to_threshold=threshold)), flush=True)


if __name__ == "__main__":
    main()
