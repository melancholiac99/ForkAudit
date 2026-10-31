"""E4 step 7: evaluate the base model and every adapter on one split with a single LoRA-enabled vLLM engine.

Each model: every game of the split x 3 rollout reps (common random numbers across models, e4_01_rollout.seed_for).
Usage:
  python e4_07_eval.py --split tune --adapters Aprime_s0_lr5e-06 Aprime_s0_lr2e-05 Aprime_s0_lr5e-05   # lr selection
  python e4_07_eval.py --split valid_unseen --all                                                      # all arms x seeds + base
Output: data/rollout/{split}/{tag}/ + summary.json per model (same format as e4_01_rollout).
"""
import argparse
import glob
import importlib
import json
import os
import sys

import e4_common as E
import common
from envpool import EnvPool


def summarize(out_dir, games, reps, prov, split, tag):
    per = {str(g["seed"]): [int(json.load(open(os.path.join(out_dir, f"s{g['seed']}_r{r}.json")))["won"]) for r in range(reps)]
           for g in games}
    n = sum(len(v) for v in per.values())
    E.write_json(os.path.join(out_dir, "summary.json"), dict(split=split, tag=tag, n=n, per_game=per, provenance=prov,
                                                             success=sum(map(sum, per.values())) / n))
    return sum(map(sum, per.values())) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["tune", "valid_seen", "valid_unseen"])
    ap.add_argument("--adapters", nargs="*", default=[], help="adapter dir names under data/adapters")
    ap.add_argument("--all", action="store_true", help="base + every non-smoke adapter")
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    games = json.load(open(E.GAMES_PATH))[args.split]
    names = args.adapters
    if args.all:
        names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(E.DATA, "adapters", "*"))
                       if not p.endswith("_smoke") and "_lr" not in os.path.basename(p))
    models = ([] if args.no_base else [("base", None)]) + [(n, os.path.join(E.DATA, "adapters", n)) for n in names]
    pool = EnvPool(args.workers)
    import alfenv
    from e4_policy import LoraPolicy
    pol = LoraPolicy()
    roll = importlib.import_module("e4_01_rollout")
    res = {}
    for k, (tag, path) in enumerate(models, 1):
        pol.set_adapter(path, k)
        prov = dict(script="exploratory_E4/e4_07_eval.py", commit=common.git_commit(), argv=sys.argv, adapter=path,
                    adapter_sha256=E.file_sha256(os.path.join(path, "adapter_model.safetensors")) if path else None,
                    vllm=pol.vllm_version, games_sha256=E.file_sha256(E.GAMES_PATH))
        out_dir = os.path.join(E.DATA, "rollout", args.split, tag)
        roll.run(games, args.reps, pol, pool, out_dir, False, alfenv, prov)
        res[tag] = summarize(out_dir, games, args.reps, prov, args.split, tag)
        print(tag, res[tag], flush=True)
    pool.shutdown()
    pol.close()
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
