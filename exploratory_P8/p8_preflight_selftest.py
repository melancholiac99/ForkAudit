"""P8 preflight + synthetic-only tests. NO real-data scientific analysis.

Safe before P8 reading rules are frozen:
  python exploratory_P8/p8_preflight_selftest.py --selftest
  python exploratory_P8/p8_preflight_selftest.py --preflight --repo-root .

Preflight computes file metadata and hashes; it never decodes actual rewards,
Fork labels, action probabilities, or samples. No P1/P2/A2 estimates are made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


REQUIRED_FILES = [
    "data/labels_final.parquet",
    "data/features_W1.parquet",
    "data/scores_fp32.parquet",
    "exploratory_P4/strata.parquet",
    "exploratory_P4/strata_audit.json",
    "exploratory_P4/e1.json",
    "exploratory_P4/PREREG_X.md",
    "exploratory_P4/p4_x_robust.py",
    "exploratory_P4/p4_strata.py",
    "scripts/evalcore.py",
]
REQUIRED_DIRS = [
    "data/rollout",
    "data/replay",
    "data/replay_b1",
]


def gain(p, q):
    """One point, a fixed finite candidate list, no learned or model data."""
    assert len(p) == len(q) and len(p) > 0
    assert all(0 <= x <= 1 for x in p)
    assert math.isclose(sum(p), 1, abs_tol=1e-12)
    return sum(x * y for x, y in zip(p, q)) - sum(q) / len(q)


def mean(xs):
    if not xs:
        raise ValueError("empty group")
    return sum(xs) / len(xs)


def live_split(rows):
    """Exact identity on the *same* Live population and estimator."""
    live = [r for r in rows if r["live"]]
    forks = [r for r in live if r["fork"]]
    other = [r for r in live if not r["fork"]]
    if not forks or not other:
        raise ValueError("need both groups in synthetic test")
    weight = len(forks) / len(live)
    gf = mean([r["g"] for r in forks])
    gn = mean([r["g"] for r in other])
    total = mean([r["g"] for r in live])
    decomposed = weight * gf + (1 - weight) * gn
    assert math.isclose(total, decomposed, abs_tol=1e-12)
    return {"total_live": total, "fork": gf, "nonfork_live": gn,
            "w_fork": weight, "no_compensation": weight * gf,
            "nonfork_contribution": (1 - weight) * gn}


def synthetic_cases():
    """No repository files or external data are read by these tests."""
    p = (0.8, 0.2)

    # Common world: small but real a advantage; rare world: large b advantage.
    common = gain(p, (1.0, 0.9))
    rare = gain(p, (0.0, 1.0))
    assert math.isclose(common, 0.03, abs_tol=1e-12)
    assert math.isclose(rare, -0.3, abs_tol=1e-12)
    rows = ([{"g": common, "live": True, "fork": False} for _ in range(95)] +
            [{"g": rare, "live": True, "fork": True} for _ in range(5)])
    d = live_split(rows)
    assert d["total_live"] > 0 and d["fork"] < 0 and d["nonfork_live"] > 0
    assert math.isclose(d["total_live"], 0.0135, abs_tol=1e-12)
    # History expectation Q(a)=0.95, Q(b)=0.905; a is actually better.
    assert 0.95 > (0.95 * 0.9 + 0.05 * 1.0)

    # Genuine population-level negative G_u: both world types prefer b.
    mis = [gain(p, (0.0, 1.0)), gain(p, (0.5, 1.0))]
    assert mean(mis) < 0

    # Dead under a finite replay estimate: same q for all arms -> G_u exactly 0.
    dead = gain(p, (0.4, 0.4))
    assert math.isclose(dead, 0, abs_tol=1e-12)
    diluted = rows + [{"g": dead, "live": False, "fork": False} for _ in range(100)]
    assert math.isclose(mean([r["g"] for r in diluted]),
                        0.5 * d["total_live"], abs_tol=1e-12)

    # A control where the Fork label is assigned equally in each world type:
    # conditioning on Fork does not change world composition or mean gain.
    control = (
        [{"g": common, "live": True, "fork": i < 95} for i in range(190)] +
        [{"g": rare, "live": True, "fork": i < 5} for i in range(10)])
    dc = live_split(control)
    assert math.isclose(dc["fork"], dc["total_live"], abs_tol=1e-12)
    assert math.isclose(dc["nonfork_live"], dc["total_live"], abs_tol=1e-12)
    return {"passed": True, "cases": 5, "source": "synthetic only",
            "no_repository_data_read": True}


def sha256_file(path):
    dig = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            dig.update(chunk)
    return dig.hexdigest()


def preflight(root):
    """Only path existence/size/content hashes. No dataset parsing."""
    root = root.resolve()
    out = {"mode": "preflight_no_science", "repo_root": str(root),
           "files": {}, "dirs": {}, "missing": []}
    try:
        out["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        out["git_head"] = None

    for rel in REQUIRED_FILES:
        file = root / rel
        if not file.is_file():
            out["missing"].append(rel)
            continue
        out["files"][rel] = {"bytes": file.stat().st_size,
                             "sha256": sha256_file(file)}
    for rel in REQUIRED_DIRS:
        directory = root / rel
        if not directory.is_dir():
            out["missing"].append(rel)
        else:
            # Only count immediate entries; never read scientific contents.
            out["dirs"][rel] = {"exists": True,
                                "immediate_entries": sum(1 for _ in directory.iterdir())}
    out["passed"] = len(out["missing"]) == 0
    out["analysis_performed"] = False
    return out


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--preflight", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # The selftest is safe on a machine without any project data.
    report = synthetic_cases() if args.selftest else preflight(args.repo_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
