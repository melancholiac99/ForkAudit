"""Offline per-task milestone funnel from existing E4 / E4c episode JSONs.

Example:
 python exploratory_E4e/funnel.py --source base=exploratory_E4/data/rollout/valid_unseen/base \
      --source aprime=exploratory_E4/data/rollout/valid_unseen/Aprime_s0 \
      --out /tmp/funnel.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

from stage_logic import milestones


def paths_for(spec: str) -> tuple[str, list[Path]]:
    if "=" not in spec:
        raise ValueError(f"Use NAME=GLOB_OR_DIRECTORY, got {spec}")
    name, location = spec.split("=", 1)
    if not name or not location:
        raise ValueError(f"Invalid --source: {spec}")
    p = Path(location)
    matches = sorted(p.glob("s*_r*.json")) if p.is_dir() else [Path(x) for x in sorted(glob.glob(location, recursive=True))]
    matches = [p for p in matches if p.is_file() and p.name.startswith("s") and "_r" in p.stem]
    if not matches:
        raise FileNotFoundError(f"No episode JSONs for {spec}; summary.json alone is insufficient")
    return name, matches


def funnel_for(episodes: list[dict]) -> dict:
    groups = defaultdict(list)
    unrecognized = defaultdict(int)
    for ep in episodes:
        ms = milestones(ep)
        if not ms["recognized"]:
            unrecognized[ep.get("task_type", "<missing>")] += 1
            continue
        groups[ep.get("task_type", "<missing>")].append((ep, ms))
    ans = {}
    for kind, rows in sorted(groups.items()):
        names = list(rows[0][1]["stages"])
        n = len(rows)
        stages = []
        eligible = n
        for name in names:
            reached = sum(int(m["stages"].get(name, False)) for _, m in rows)
            if reached > eligible:
                raise ValueError(f"Milestone order violated in {kind}: {name} {reached} > {eligible}")
            stages.append(dict(stage=name, eligible=eligible, passed=reached,
                               conditional_rate=reached / eligible if eligible else None,
                               first_missing=max(0, eligible - reached)))
            eligible = reached
        ans[kind] = {"episodes": n, "success": sum(bool(ep.get("won")) for ep, _ in rows) / n,
                     "stages": stages}
    return {"by_task_type": ans, "unrecognized": dict(unrecognized),
            "n_total": len(episodes), "n_recognized": sum(x["episodes"] for x in ans.values()),
            "note": "first_missing = first unobserved milestone, NOT causal recoverable failures or a gain upper bound."}


def paired_ci(a: list[dict], b: list[dict], n_boot: int = 2000, seed: int = 20261110) -> dict:
    """Game-cluster paired contrast, tolerates different model/training-seed reps."""
    by = []
    for episodes in (a, b):
        g = defaultdict(list)
        for ep in episodes:
            if "seed" in ep:
                g[str(ep["seed"])].append(float(bool(ep.get("won", False))))
        by.append({key: sum(val) / len(val) for key, val in g.items()})
    ids = sorted(set(by[0]) & set(by[1]))
    if not ids:
        return {"n_games": 0, "difference_pp": None, "ci95_pp": None}
    d = [by[1][i] - by[0][i] for i in ids]
    avg = sum(d) / len(d)
    rng = random.Random(seed)
    boots = sorted(sum(d[rng.randrange(len(d))] for _ in d) / len(d) for _ in range(n_boot))
    return {"n_games": len(ids), "difference_pp": round(avg * 100, 4),
            "ci95_pp": [round(100 * boots[int(.025 * (n_boot - 1))], 4),
                        round(100 * boots[int(.975 * (n_boot - 1))], 4)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", action="append", required=True, help="Repeat NAME=DIRECTORY or NAME='GLOB'")
    p.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"), help="Optional paired contrast")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    data = defaultdict(list)
    input_files = defaultdict(list)
    for s in args.source:
        name, paths = paths_for(s)
        for path in paths:
            data[name].append(json.loads(path.read_text(encoding="utf-8")))
            input_files[name].append(str(path))
    report = {"sources": {name: {"input_files": len(input_files[name]), **funnel_for(episodes)}
                          for name, episodes in sorted(data.items())}}
    if args.compare:
        base, new = args.compare
        if base not in data or new not in data:
            p.error("--compare names must appear in --source")
        report["paired_success"] = {"baseline": base, "candidate": new, **paired_ci(data[base], data[new])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, summary in report["sources"].items():
        print(f"\n{name}: {summary['n_recognized']}/{summary['n_total']} recognized episodes")
        for task_type, d in summary["by_task_type"].items():
            print(f"  {task_type} n={d['episodes']} success={d['success']:.1%}")
            for stage in d["stages"]:
                rate = "NA" if stage["conditional_rate"] is None else f"{stage['conditional_rate']:.1%}"
                print(f"    {stage['stage']:10} {stage['passed']}/{stage['eligible']} ({rate})")
    if "paired_success" in report:
        print("\npaired_success", report["paired_success"])
    print("\nSaved", args.out)


if __name__ == "__main__":
    main()
