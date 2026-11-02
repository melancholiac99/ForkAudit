"""E4b analysis (PREREG_E4b §5): primary contrasts on valid_unseen, replication on valid_seen, reading.

Per game, success is averaged over all rollouts of an arm (3 training seeds x 3 reps; base: 3 reps); differences are
paired by game, 10,000-game bootstrap, 95% percentile CI (E4's game_bootstrap_diff, unchanged).
Z - G pools the 2 x 2 by concatenating columns, i.e. per game (Azero + Dzero)/2 - (Agap + Dgap)/2.
  python e4b_analyze.py [--splits valid_unseen]   # valid_seen is optional so the primary can be read first
Output: e4b.json, E4b_REPORT.txt
"""
import argparse
import json
import os

import numpy as np

import e4b_common as B
from e4b_common import E

CONTRASTS = [("Azero", "base"), ("Dzero", "base"), ("Agap", "base"), ("Dgap", "base"), ("Bfacgap", "base"),
             ("Z", "G"), ("Azero", "Agap"), ("Dzero", "Dgap"), ("Bfacgap", "Agap"), ("Bfacgap", "Dgap"), ("Azero", "Dzero")]


def matrix(split, arm, games, data=B.DATA):
    tags = ["base"] if arm == "base" else [f"{arm}_s{s}" for s in B.SEEDS]
    cols = []
    for tag in tags:
        sm = json.load(open(os.path.join(data, "rollout", split, tag, "summary.json")))
        cols.append(np.array([sm["per_game"][str(g["seed"])] for g in games], float))
    return np.concatenate(cols, axis=1)


def complete(split, arm, data=B.DATA):
    """All training seeds (or base) evaluated; partial arms are never analysed."""
    tags = ["base"] if arm == "base" else [f"{arm}_s{s}" for s in B.SEEDS]
    return all(os.path.exists(os.path.join(data, "rollout", split, t, "summary.json")) for t in tags)


def base_drift(split, games):
    """Same-seed agreement of E4b's base re-evaluation with E4's base run (common random numbers)."""
    if not (complete(split, "base") and complete(split, "base", B.E4_DATA)):
        return None
    old, new = matrix(split, "base", games, B.E4_DATA), matrix(split, "base", games)
    return dict(episode_agreement=float((old == new).mean()), success_E4=float(100 * old.mean()),
                success_E4b=float(100 * new.mean()), diff=E.game_bootstrap_diff(new, old))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["valid_unseen", "valid_seen"])
    args = ap.parse_args()
    G = json.load(open(E.GAMES_PATH))
    res = dict(delta_pp=B.DELTA, splits={})
    for split in args.splits:
        games = G[split]
        M = {arm: matrix(split, arm, games) for arm in ["base"] + B.ARMS if complete(split, arm)}
        if all(a in M for a in B.PRIMARY):
            M["Z"] = np.concatenate([M["Azero"], M["Dzero"]], axis=1)
            M["G"] = np.concatenate([M["Agap"], M["Dgap"]], axis=1)
        r = dict(success={a: float(100 * M[a].mean()) for a in ["base"] + B.ARMS if a in M},
                 diff={f"{a}-{b}": E.game_bootstrap_diff(M[a], M[b]) for a, b in CONTRASTS if a in M and b in M},
                 missing=[a for a in ["base"] + B.ARMS if a not in M], base_drift_vs_E4=base_drift(split, games))
        res["splits"][split] = r
    if "valid_unseen" in res["splits"] and "Z-G" in res["splits"]["valid_unseen"]["diff"]:
        d = res["splits"]["valid_unseen"]["diff"]
        res["reading"] = B.reading(d["Azero-base"]["ci95"][0], d["Dzero-base"]["ci95"][0], d["Z-G"]["ci95"])
        res["flags"] = dict(gap_significantly_better=d["Z-G"]["ci95"][1] < 0, zero_significantly_better=d["Z-G"]["ci95"][0] > 0)
    res["pairs_manifest_sha256"] = E.file_sha256(os.path.join(B.DATA, "pairs", "manifest.json"))
    E.write_json(os.path.join(B.HERE, "e4b.json"), res)
    lines = [f"E4b reading: {res.get('reading')}  (delta = {B.DELTA} pp)  flags: {res.get('flags')}", ""]
    for split, r in res["splits"].items():
        lines.append(f"== {split}: success (%)" + (f"   [not yet evaluated: {r['missing']}]" if r["missing"] else ""))
        lines += [f"  {a:8s} {v:6.2f}" for a, v in r["success"].items()]
        lines.append("  paired differences (pp, 95% CI over games)")
        lines += [f"  {k:16s} {v['diff']:+6.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for k, v in r["diff"].items()]
        if r["base_drift_vs_E4"]:
            bd = r["base_drift_vs_E4"]
            lines.append(f"  base re-run vs E4 base: episode agreement {bd['episode_agreement']:.3f}, "
                         f"{bd['success_E4b']:.2f} vs {bd['success_E4']:.2f}")
    open(os.path.join(B.HERE, "E4b_REPORT.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
