"""E4b step 1 (CPU): build the five training sets from E4's frozen inputs (PREREG_E4b §3–§4) and the CPU gate report.

Reads E4's points, 32B F judge scores and 3-run replay outcomes; no E4b outcome exists or is read.
  python e4b_pairs.py                       # server: E4 inputs at ../exploratory_E4/data, writes data/pairs/
  python e4b_pairs.py --e4data DIR --gate-only --out gate.json   # counts only (any machine with the E4 archive)
Output: data/pairs/{arm}.jsonl (prompt, chosen, rejected, point_id, preferred, rejected_arm, task_type, cls, turn)
        data/pairs/manifest.json (availability, N, quotas, covariates, input and output SHA256)
"""
import argparse
import collections
import glob
import json
import os

import numpy as np

import e4b_common as B
from e4b_common import E
from post_e2_mechanism import arm_classes


def load(e4data):
    points = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(e4data, "points", "*.json")))]
    points = [r for r in points if r["cf_defined"]]
    judge = {}
    for line in open(os.path.join(e4data, "judge", "32b_F", "scores.jsonl")):
        r = json.loads(line)
        judge[r["point_id"]] = np.array(r["candidate_prob"])
    rows = []
    for r in points:
        rp = json.load(open(os.path.join(e4data, "replay", f"s{r['seed']}", f"t{r['turn']:02d}.json")))
        assert rp["factual_action"] == r["factual"] and rp["alternatives"] == r["alternatives"], r["point_id"]
        cls, q = B.classify([rp["factual_outcomes"]] + rp["alt_outcomes"])
        actions = [r["factual"]] + r["alternatives"]
        classes = arm_classes(actions, [(h["executed_action"], h["observation_after"]) for h in r["history"]])
        pA, pD = E.pair_judge(judge[r["point_id"]]), E.pair_explore(r["point_id"], classes)
        rows.append(dict(rec=r, cls=cls, q=q, classes=classes, pairs={a: B.arm_pair(a, cls, q, pA, pD) for a in B.ARMS},
                         pA=pA))
    return rows


def describe(rows):
    if not rows:
        return dict(n=0)
    return dict(n=len(rows), n_games=len({r["point_id"].split("_t")[0] for r in rows}),
                frac_traj_won=round(float(np.mean([r["traj_won"] for r in rows])), 3),
                turn_median=float(np.median([r["turn"] for r in rows])),
                preferred_E_new=round(float(np.mean([r["preferred_class"] == "E_new" for r in rows])), 3),
                task_type={k: v for k, v in sorted(collections.Counter(r["task_type"] for r in rows).items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e4data", default=B.E4_DATA)
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pts = load(a.e4data)
    cand = {arm: [] for arm in B.ARMS}
    for p in pts:
        r = p["rec"]
        for arm, pr in p["pairs"].items():
            if pr is None:
                continue
            win, lose = pr
            assert lose == 0 and win != 0, (arm, r["point_id"], pr)          # every arm: non-factual > factual
            cand[arm].append(dict(point_id=r["point_id"], prompt=r["prompt"], chosen=r["arm_responses"][win],
                                  rejected=r["arm_responses"][lose], preferred=win, rejected_arm=lose,
                                  task_type=r["task_type"], cls=p["cls"], turn=r["turn"],
                                  traj_won=int(r["final_reward"] > 0), preferred_class=p["classes"][win]))
    shares = B.target_shares([p["rec"]["task_type"] for p in pts if p["cls"] == "V3"])
    avail = {arm: dict(collections.Counter(x["task_type"] for x in rows)) for arm, rows in cand.items()}
    n = B.common_n(avail, shares)
    quotas = B.task_quotas(shares, n)
    sel = {arm: B.draw(arm, rows, quotas) for arm, rows in cand.items()}

    def agree(cls_set):
        s = [int(np.sign(p["q"][p["pA"][0]] - p["q"][0])) for p in pts if p["pA"] and p["cls"] in cls_set]
        c = collections.Counter(s)
        return dict(n=len(s), agree=round(c[1] / max(1, len(s)), 3), tie=round(c[0] / max(1, len(s)), 3),
                    opposite=round(c[-1] / max(1, len(s)), 3))

    man = dict(
        n_points=len(pts), classes=dict(collections.Counter(p["cls"] for p in pts)),
        V3_with_fac_alternative=sum(p["pairs"]["Bfacgap"] is not None for p in pts),
        target_shares={t: round(s, 4) for t, s in shares.items()}, N=n, quotas=quotas,
        available={arm: describe(rows) for arm, rows in cand.items()},
        selected={arm: describe(rows) for arm, rows in sel.items()},
        overlap_selected_Azero_Dzero=len({r["point_id"] for r in sel["Azero"]} & {r["point_id"] for r in sel["Dzero"]}),
        overlap_selected_Agap_Dgap=len({r["point_id"] for r in sel["Agap"]} & {r["point_id"] for r in sel["Dgap"]}),
        A_label_vs_3run_replay={"Z3": agree({"Z3"}), "V3": agree({"V3"})},
        inputs=dict(judgeF_sha256=E.file_sha256(os.path.join(a.e4data, "judge", "32b_F", "scores.jsonl")),
                    replay_manifest_sha256=E.file_sha256(os.path.join(a.e4data, "replay_manifest.json")),
                    points_manifest_sha256=E.file_sha256(os.path.join(a.e4data, "points_manifest.json"))),
        note="Z3 = 15/15 observed zeros under pi0 (3 runs x 5 arms); not 'Q = 0'.")
    if not a.gate_only:
        os.makedirs(os.path.join(B.DATA, "pairs"), exist_ok=True)
        man["sha256"] = {}
        for arm, rows in sel.items():
            path = os.path.join(B.DATA, "pairs", f"{arm}.jsonl")
            with open(path, "w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            man["sha256"][arm] = E.file_sha256(path)
    out = a.out or os.path.join(B.DATA, "pairs", "manifest.json")
    E.write_json(out, man)
    print(json.dumps({k: man[k] for k in ("n_points", "classes", "V3_with_fac_alternative", "N", "quotas")}, indent=1))
    print({arm: d["n"] for arm, d in man["available"].items()})


if __name__ == "__main__":
    main()
