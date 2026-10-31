"""E4 rollouts: training-data collection (train split, base model) and evaluation (tune / valid splits, any adapter).

Same template, history length, temperature, parser and 50-step cap as M0 (scripts/10_rollout.py).
Sample seed = 1_000_000 * task_seed + turn + 10**11 * (rep + 1) for rep >= 0 (distinct from all M0 seeds; identical
across arms for the same game, rep and turn -> common random numbers between arms).

Usage:
  python e4_01_rollout.py --split train --tag base --reps 1 --full          # training data (prompts and responses kept)
  python e4_01_rollout.py --split valid_unseen --tag A_s0 --adapter PATH --reps 3
  python e4_01_rollout.py --split train --dry --limit 4 --out /tmp/x        # CPU plumbing test (random policy)
Output: data/rollout/{split}/{tag}/s{seed}_r{rep}.json (+ summary.json). Resumable per episode file.
"""
import argparse
import json
import os
import sys
import time

import e4_common as E
import common
from envpool import EnvPool


def seed_for(task_seed, turn, rep):
    return 1_000_000 * task_seed + turn + 10 ** 11 * (rep + 1)


def run(games, reps, pol, pool, out_dir, full, alfenv, prov):
    jobs = [(g, r) for g in games for r in range(reps)
            if not os.path.exists(os.path.join(out_dir, f"s{g['seed']}_r{r}.json"))]
    if not jobs:
        return
    keys = [f"{g['seed']}:{r}" for g, r in jobs]
    init = pool.open([(k, os.path.join(alfenv.ALFWORLD_DATA, g["game_file"]), [], None) for k, (g, _) in zip(keys, jobs)])
    st = {}
    for k, (g, r) in zip(keys, jobs):
        obs, adm, _, _ = init[k]
        st[k] = dict(seed=g["seed"], rep=r, game_file=g["game_file"], task_type=g["task_type"],
                     task=common.extract_task(obs), initial_obs=obs, obs=obs, adm=adm, mem=[], steps=[], won=False, done=False)
    t0 = time.time()
    for turn in range(1, common.MAX_STEPS + 1):
        active = [k for k in keys if not st[k]["done"]]
        if not active:
            break
        prompts = [common.build_prompt(st[k]["task"], st[k]["mem"], st[k]["obs"], st[k]["adm"]) for k in active]
        seeds = [seed_for(st[k]["seed"], turn, st[k]["rep"]) for k in active]
        outs = pol.sample(prompts, seeds)
        acts = {}
        for k, p, sd, o in zip(active, prompts, seeds, outs):
            a, valid, fmt = common.parse(o[0]["text"])
            acts[k] = a
            step = dict(turn=turn, action=a, valid=valid, sample_seed=sd)
            if full:
                step.update(prompt=p, obs_before=st[k]["obs"], admissible=st[k]["adm"], response=o[0]["text"])
            st[k]["steps"].append(step)
        res = pool.step(acts)
        for k in active:
            obs, adm, won, done = res[k]
            x = st[k]
            x["mem"].append({"text_obs": x["obs"], "action": acts[k]})
            x["steps"][-1].update(obs_after=obs, won_after=won)
            x["obs"], x["adm"] = obs, adm
            if won or done:
                x["won"], x["done"] = won, True
        print(f"turn {turn} active {len(active)} won {sum(st[k]['won'] for k in keys)} ({time.time() - t0:.0f}s)", flush=True)
    for k in keys:
        x = st[k]
        rec = {kk: v for kk, v in x.items() if kk not in ("obs", "adm", "mem")}
        rec.update(n_turns=len(x["steps"]), provenance=prov)
        E.write_json(os.path.join(out_dir, f"s{x['seed']}_r{x['rep']}.json"), rec)
    pool.close(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "tune", "valid_seen", "valid_unseen"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--adapter_id", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--full", action="store_true", help="keep prompts, responses, observations (training data)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=400, help="episodes per lockstep batch")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    games = json.load(open(E.GAMES_PATH))[args.split]
    if args.limit:
        games = games[:args.limit]
    out_dir = args.out or os.path.join(E.DATA, "rollout", args.split, args.tag)
    pool = EnvPool(args.workers)  # fork before CUDA
    import alfenv
    if args.dry:
        import importlib
        pol = importlib.import_module("11_replay").DryPolicy()
    elif args.split == "train":
        assert args.adapter is None, "training data is collected with the base policy"
        from policy import Policy
        pol = Policy()                      # same engine as M0 rollouts, drafts and replay
    else:
        from e4_policy import LoraPolicy      # every evaluated model (base included) uses the same LoRA-enabled engine
        pol = LoraPolicy()
        pol.set_adapter(args.adapter, args.adapter_id)
    prov = dict(script="exploratory_E4/e4_01_rollout.py", commit=common.git_commit(), argv=sys.argv,
                adapter=args.adapter, adapter_sha256=(E.file_sha256(os.path.join(args.adapter, "adapter_model.safetensors"))
                                                      if args.adapter else None),
                vllm=pol.vllm_version, games_sha256=E.file_sha256(E.GAMES_PATH))
    jobs = [(g, r) for g in games for r in range(args.reps)]
    step = max(1, args.chunk // args.reps)
    for i in range(0, len(games), step):
        run(games[i:i + step], args.reps, pol, pool, out_dir, args.full, alfenv, prov)
    eps = [json.load(open(os.path.join(out_dir, f"s{g['seed']}_r{r}.json"))) for g, r in jobs]
    E.write_json(os.path.join(out_dir, "summary.json"), dict(
        split=args.split, tag=args.tag, n=len(eps), success=sum(e["won"] for e in eps) / len(eps),
        per_game={str(g["seed"]): [int(json.load(open(os.path.join(out_dir, f"s{g['seed']}_r{r}.json")))["won"])
                                   for r in range(args.reps)] for g in games}, provenance=prov))
    pool.shutdown()
    if hasattr(pol, "close"):
        pol.close()
    print("success", sum(e["won"] for e in eps) / len(eps))


if __name__ == "__main__":
    main()
