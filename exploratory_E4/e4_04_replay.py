"""E4 step 4: executed replay for the B/C arms — factual + 4 alternatives x 3 continuations at every cf-defined point,
with the M0 async replay engine (scripts/11_replay.py run_async, same seeds formula, parser and step cap).

Seeds: 11_replay.seed_of(task_seed, turn, role, idx) + 5 * 10**12 (disjoint from M0 banks and E4 drafts).
Output: data/replay/s{seed}/t{turn:02d}.json (M0 replay schema), data/replay_manifest.json with GPU wall seconds
(used for cost matching, PREREG_E4 §4).
Usage: python e4_04_replay.py [--dry --limit N --out DIR]
"""
import argparse
import glob
import importlib
import json
import os
import sys
import time

import e4_common as E
import common
from envpool import EnvPool

OFFSET = 5 * 10 ** 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max_concurrent", type=int, default=600)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--points", default=os.path.join(E.DATA, "points"))
    ap.add_argument("--out", default=os.path.join(E.DATA, "replay"))
    ap.add_argument("--rollouts", default=os.path.join(E.DATA, "rollout", "train", "base"))
    args = ap.parse_args()
    R = importlib.import_module("11_replay")
    pool = EnvPool(args.workers)
    import alfenv
    pol = R.DryPolicy() if args.dry else importlib.import_module("policy").Policy()
    stream = R.DryStream(pol) if args.dry else R.VllmStream(pol)
    trajs = {}
    dps, conts = [], []
    files = sorted(glob.glob(os.path.join(args.points, "*.json")))
    if args.limit:
        files = files[:args.limit]
    for f in files:
        rec = json.load(open(f))
        if not rec["cf_defined"]:
            continue
        path = os.path.join(args.out, f"s{rec['seed']}", f"t{rec['turn']:02d}.json")
        if os.path.exists(path):
            continue
        seed = rec["seed"]
        if seed not in trajs:
            ep = json.load(open(os.path.join(args.rollouts, f"s{seed}_r0.json")))
            trajs[seed] = dict(traj_id=f"s{seed}", task_id=ep["game_file"], task_type=ep["task_type"], seed=seed,
                               game_file=ep["game_file"], task=ep["task"],
                               steps=[dict(obs_before=s["obs_before"], action=s["action"], valid=s["valid"]) for s in ep["steps"]],
                               obs_after=[s["obs_after"] for s in ep["steps"]])
        tr = trajs[seed]
        i = rec["turn"] - 1
        assert tr["steps"][i]["action"] == rec["factual"]
        d = dict(tr=tr, i=i, turn=rec["turn"], path=path, prefix=[s["action"] for s in tr["steps"][:i]],
                 expected=tr["obs_after"][:i], factual=rec["factual"], alts=rec["alternatives"], drafts=[], draws=rec["n_draws"],
                 cf_defined=True, n_open=0)
        j = len(dps)
        dps.append(d)
        for role, a in [(1, d["factual"])] + [(2 + k, x) for k, x in enumerate(d["alts"])]:
            for r in range(R.N_REP):
                conts.append(R.Cont(f"{j}:{role}:{r}", j, role, r, a))
                d["n_open"] += 1
    print(f"{len(dps)} points, {len(conts)} continuations", flush=True)
    prov = dict(script="exploratory_E4/e4_04_replay.py", commit=common.git_commit(), argv=sys.argv, seed_offset=OFFSET,
                vllm=pol.vllm_version, engine="async")
    t0 = time.time()
    n_gen, div = R.run_async(dps, conts, pool, stream, args, OFFSET, prov, alfenv, t0) if dps else (0, 0)
    sec = time.time() - t0
    man_path = os.path.join(os.path.dirname(args.out.rstrip("/")), "replay_manifest.json")
    prev = json.load(open(man_path)) if os.path.exists(man_path) else dict(gpu_sec=0.0, n_points=0, n_gen_steps=0, refeed_div=0)
    E.write_json(man_path, dict(gpu_sec=prev["gpu_sec"] + sec, n_points=prev["n_points"] + len(dps),
                                n_gen_steps=prev["n_gen_steps"] + n_gen, refeed_div=prev["refeed_div"] + div, provenance=prov))
    pool.shutdown()
    pol.close()
    print(f"DONE {len(dps)} points {n_gen} generated steps div {div} {sec:.0f}s")


if __name__ == "__main__":
    main()
