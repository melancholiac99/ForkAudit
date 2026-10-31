"""E4 step 9: predictions PL-1..PL-5 and the reading (PREREG_E4 §6). Primary split valid_unseen; valid_seen replicates.

Per game, success is averaged over all rollouts of that model family (training seeds x rollout reps; base: reps);
arm differences are paired by game and bootstrapped over games (10,000). PL-5 uses data/mechanism.json.
Output: e4.json, E4_REPORT.txt (in exploratory_E4/).
"""
import json
import os

import numpy as np

import e4_common as E

SEEDS = [0, 1, 2]


def matrix(split, arm, games):
    tags = ["base"] if arm == "base" else [f"{arm}_s{s}" for s in SEEDS]
    cols = []
    for tag in tags:
        sm = json.load(open(os.path.join(E.DATA, "rollout", split, tag, "summary.json")))
        cols.append(np.array([sm["per_game"][str(g["seed"])] for g in games], float))
    return np.concatenate(cols, axis=1)


def main():
    G = json.load(open(E.GAMES_PATH))
    res = dict(splits={})
    for split in ["valid_unseen", "valid_seen"]:
        games = G[split]
        M = {arm: matrix(split, arm, games) for arm in E.ARMS_ALL}
        r = dict(success={arm: float(100 * m.mean()) for arm, m in M.items()}, diff={})
        for a, b in [("A", "base"), ("A", "Bmatch"), ("A", "Cmatch"), ("A", "D"), ("A", "Rnd"), ("A", "H"), ("A", "Bfull"),
                     ("A", "Aprime"), ("A", "Esft"), ("Aprime", "base"), ("H", "base"), ("Bfull", "base"), ("Bmatch", "base"),
                     ("Cmatch", "base"), ("D", "base"), ("Rnd", "base"), ("Esft", "base")]:
            r["diff"][f"{a}-{b}"] = E.game_bootstrap_diff(M[a], M[b])
        res["splits"][split] = r
    d = res["splits"]["valid_unseen"]["diff"]
    lo = lambda k: d[k]["ci95"][0]
    pl1 = lo("A-base") > 0
    pl2 = lo("A-Bmatch") > -2 and lo("A-Cmatch") > 0
    pl3 = lo("A-D") > 0 and lo("A-Rnd") > 0
    pl4 = lo("A-H") > 0
    mech_path = os.path.join(E.DATA, "mechanism.json")
    pl5 = None
    if os.path.exists(mech_path):
        m = json.load(open(mech_path))
        I = [k for k, z in enumerate(m["layer"]) if z == "I"]
        traj = [m["points"][k].split("_t")[0] for k in I]
        a_mean = np.mean([[m["models"][f"A_s{s}"][k] for k in I] for s in SEEDS], axis=0)
        base = np.array([m["models"]["base"][k] for k in I])
        res["mechanism_I"] = dict(A_minus_base=E.cluster_bootstrap_mean(a_mean - base, traj),
                                  base=E.cluster_bootstrap_mean(base, traj), A=E.cluster_bootstrap_mean(a_mean, traj))
        pl5 = res["mechanism_I"]["A_minus_base"]["ci95"][0] > 0
    res["predictions"] = {"PL-1": pl1, "PL-2": pl2, "PL-3": pl3, "PL-4": pl4, "PL-5": pl5}
    res["reading"] = E.reading(pl1, pl2, pl3)
    res["pairs_manifest"] = json.load(open(os.path.join(E.DATA, "pairs", "manifest.json")))
    res["pairs_manifest"].pop("Bmatch_points", None); res["pairs_manifest"].pop("Cmatch_points", None)
    E.write_json(os.path.join(E.HERE, "e4.json"), res)
    lines = [f"E4 reading: {res['reading']}", f"predictions: {res['predictions']}", ""]
    for split, r in res["splits"].items():
        lines.append(f"== {split}: success (%)")
        lines += [f"  {a:7s} {v:6.2f}" for a, v in r["success"].items()]
        lines.append("  paired differences (pp, 95% CI over games)")
        lines += [f"  {k:14s} {v['diff']:+6.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for k, v in r["diff"].items()]
    open(os.path.join(E.HERE, "E4_REPORT.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
