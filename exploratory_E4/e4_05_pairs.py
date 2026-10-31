"""E4 step 5: build the training set of every arm with the frozen rules in e4_common (PREREG_E4 §3–§4).

Inputs: data/points/*.json, data/judge/32b_{F,L}/scores.jsonl, data/replay/**, data/replay_manifest.json,
        data/rollout/train/base/*.json (Esft).
Output: data/pairs/{arm}.jsonl  (DPO arms: prompt, chosen, rejected, point_id, preferred, rejected_arm;
                                 Esft: prompt, response), data/pairs/manifest.json (counts, costs, N_match, SHA256).
"""
import glob
import json
import os

import numpy as np

import e4_common as E
from post_e2_mechanism import arm_classes


def load_judge(cond):
    run = json.load(open(os.path.join(E.DATA, "judge", f"32b_{cond}", "run.json")))
    assert run["complete"], cond
    out = {}
    for line in open(os.path.join(E.DATA, "judge", f"32b_{cond}", "scores.jsonl")):
        r = json.loads(line)
        out[r["point_id"]] = np.array(r["candidate_prob"])
    return out, run["scoring_seconds"]


def load_replay(rec):
    p = os.path.join(E.DATA, "replay", f"s{rec['seed']}", f"t{rec['turn']:02d}.json")
    if not os.path.exists(p):
        return None
    r = json.load(open(p))
    assert r["factual_action"] == rec["factual"] and r["alternatives"] == rec["alternatives"]
    return [np.mean(r["factual_outcomes"])] + [np.mean(o) for o in r["alt_outcomes"]]


def build(points, judgeF, judgeL, replay, n_match):
    """Pure given loaded inputs -> {arm: [pair dicts]} plus the B/C subsets. Unit-tested in test_e4.py."""
    arms = {a: [] for a in ["A", "Aprime", "H", "Rnd", "D", "Bfull", "Bmatch", "Cmatch"]}
    replayed = [r["point_id"] for r in points if replay.get(r["point_id"]) is not None]
    ent = {r["point_id"]: r["entropy_sampled"] for r in points}
    bsub, csub = E.matched_subsets(replayed, ent, n_match)
    bsub, csub = set(bsub), set(csub)
    for r in points:
        pid = r["point_id"]
        actions = [r["factual"]] + r["alternatives"]
        hist = [(h["executed_action"], h["observation_after"]) for h in r["history"]]
        rules = {
            "A": E.pair_judge(judgeF[pid]),
            "Aprime": E.pair_rule([p["obs"] for p in r["peeks"]], r["target_word"]),
            "H": E.pair_judge(judgeL[pid]),
            "D": E.pair_explore(pid, arm_classes(actions, hist)),
        }
        rules["Rnd"] = E.pair_random(pid, rules["A"] is not None)
        q = replay.get(pid)
        if q is not None:
            pr = E.pair_replay(q)
            rules["Bfull"] = pr
            rules["Bmatch"] = pr if pid in bsub else None
            rules["Cmatch"] = pr if pid in csub else None
        for arm, pr in rules.items():
            if pr is not None:
                win, lose = pr
                arms[arm].append(dict(point_id=pid, prompt=r["prompt"], chosen=r["arm_responses"][win],
                                      rejected=r["arm_responses"][lose], preferred=win, rejected_arm=lose))
    return arms, sorted(bsub), sorted(csub)


def main():
    points = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(E.DATA, "points", "*.json")))]
    points = [r for r in points if r["cf_defined"]]
    judgeF, secF = load_judge("F")
    judgeL, _ = load_judge("L")
    replay = {r["point_id"]: load_replay(r) for r in points}
    man_r = json.load(open(os.path.join(E.DATA, "replay_manifest.json")))
    n_rep = sum(v is not None for v in replay.values())
    assert n_rep == man_r["n_points"], (n_rep, man_r["n_points"])
    n_match = E.matched_count(secF, man_r["gpu_sec"], n_rep)
    arms, bsub, csub = build(points, judgeF, judgeL, replay, n_match)
    # Esft: outcome-only rejection-sampling fine-tuning, example count = |A|
    sft = []
    for f in sorted(glob.glob(os.path.join(E.DATA, "rollout", "train", "base", "s*_r0.json"))):
        ep = json.load(open(f))
        if ep["won"]:
            sft += [dict(point_id=f"s{ep['seed']}_t{s['turn']:02d}", prompt=s["prompt"], response=s["response"])
                    for s in ep["steps"] if s["valid"]]
    n_sft = min(len(sft), len(arms["A"]), E.CAP)
    rng = E.seeded_rng("Esft", len(sft), n_sft)
    idx = sorted(rng.choice(len(sft), size=n_sft, replace=False).tolist())
    arms["Esft"] = [sft[i] for i in idx]
    precap = {arm: len(rows) for arm, rows in arms.items()}
    capped = {arm: E.cap_rows(arm, rows) for arm, rows in arms.items()}
    keep = {r["point_id"] for r in capped["A"]}                 # Rnd stays on exactly A's points
    capped["Rnd"] = [r for r in arms["Rnd"] if r["point_id"] in keep]
    assert len(capped["Rnd"]) == len(capped["A"])
    arms = capped
    os.makedirs(os.path.join(E.DATA, "pairs"), exist_ok=True)
    man = dict(n_points=len(points), n_replayed=n_rep, gpu_sec_judgeF=secF, gpu_sec_replay=man_r["gpu_sec"], n_match=n_match,
               Bmatch_points=bsub, Cmatch_points=csub, counts_precap=precap, cap=E.CAP, counts={}, sha256={})
    for arm, rows in arms.items():
        p = os.path.join(E.DATA, "pairs", f"{arm}.jsonl")
        with open(p, "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        man["counts"][arm] = len(rows)
        man["sha256"][arm] = E.file_sha256(p)
    E.write_json(os.path.join(E.DATA, "pairs", "manifest.json"), man)
    print({k: v for k, v in man.items() if k not in ("Bmatch_points", "Cmatch_points", "sha256")})


if __name__ == "__main__":
    main()
