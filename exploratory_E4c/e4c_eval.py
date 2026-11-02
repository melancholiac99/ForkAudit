"""E4c evaluation (PREREG_E4c §3): base weights under prompt / decoding controls, E4's evaluation protocol otherwise.

Same games (E4 games.json), 3 rollout reps, LoRA-enabled engine with no adapter, sampling parameters, parser, 50-step
cap and first-draw seeds (e4_01_rollout.seed_for -> common random numbers with every E4 / E4b evaluation). Only:
  P / PR : the HCAPO template with one hint line (e4c_common.HINT)
  R / PR : a draw whose parsed action equals the previous executed action is redrawn
  X      : a draw that repeats the previous action or is examine / look / inventory is redrawn
Redraws use seed + k * 10**13 (k = 1..8); if all 9 draws are blocked the last is kept. Every step records the draw used
and the blocked actions.
  python e4c_eval.py --split valid_unseen --conds base P R PR X
  python e4c_eval.py --split valid_unseen --conds R --dry --limit 2 --out /tmp/e4c_dry     # CPU plumbing
Output: data/rollout/{split}/{cond}/s{seed}_r{rep}.json + summary.json (E4 format plus step fields draw_index,
blocked). Resumable per episode file.
"""
import argparse
import importlib
import json
import os
import sys
import time

import e4c_common as C
from e4c_common import E, common


def run(games, reps, pol, pool, out_dir, alfenv, prov, cond):
    roll = importlib.import_module("e4_01_rollout")
    jobs = [(g, r) for g in games for r in range(reps) if not os.path.exists(os.path.join(out_dir, f"s{g['seed']}_r{r}.json"))]
    if not jobs:
        return
    template = C.template_for(cond)
    keys = [f"{g['seed']}:{r}" for g, r in jobs]
    init = pool.open([(k, os.path.join(alfenv.ALFWORLD_DATA, g["game_file"]), [], None) for k, (g, _) in zip(keys, jobs)])
    st = {}
    for k, (g, r) in zip(keys, jobs):
        obs, adm, _, _ = init[k]
        st[k] = dict(seed=g["seed"], rep=r, game_file=g["game_file"], task_type=g["task_type"], task=common.extract_task(obs),
                     initial_obs=obs, obs=obs, adm=adm, mem=[], steps=[], won=False, done=False)
    t0 = time.time()
    for turn in range(1, common.MAX_STEPS + 1):
        active = [k for k in keys if not st[k]["done"]]
        if not active:
            break
        prompts = {k: common.build_prompt(st[k]["task"], st[k]["mem"], st[k]["obs"], st[k]["adm"], template=template) for k in active}
        seeds = {k: roll.seed_for(st[k]["seed"], turn, st[k]["rep"]) for k in active}
        draws = {k: [o[0]["text"]] for k, o in zip(active, pol.sample([prompts[k] for k in active], [seeds[k] for k in active]))}
        prev = {k: (st[k]["steps"][-1]["action"] if st[k]["steps"] else None) for k in active}
        for j in range(1, C.K_RESAMPLE + 1):
            need = [k for k in active if C.is_blocked(cond, common.parse(draws[k][-1])[0], prev[k])
                    and len(draws[k]) == j]
            if not need:
                break
            outs = pol.sample([prompts[k] for k in need], [C.resample_seed(seeds[k], j) for k in need])
            for k, o in zip(need, outs):
                draws[k].append(o[0]["text"])
        acts = {}
        for k in active:
            i, a, valid, fmt = C.choose(cond, draws[k], prev[k])
            acts[k] = a
            kept_blocked = C.is_blocked(cond, a, prev[k])
            blocked = [common.parse(t)[0] for t in draws[k][:i]] + ([a] if kept_blocked else [])
            st[k]["steps"].append(dict(turn=turn, action=a, valid=valid, sample_seed=seeds[k] if i == 0 else C.resample_seed(seeds[k], i),
                                       draw_index=i, n_draws=len(draws[k]), blocked=blocked, kept_blocked=kept_blocked))
        res = pool.step(acts)
        for k in active:
            obs, adm, won, done = res[k]
            x = st[k]
            x["mem"].append({"text_obs": x["obs"], "action": acts[k]})
            x["steps"][-1].update(obs_after=obs, won_after=won)
            x["obs"], x["adm"] = obs, adm
            if won or done:
                x["won"], x["done"] = won, True
        redrawn = sum(st[k]["steps"][-1]["draw_index"] > 0 for k in active)
        print(f"[{cond}] turn {turn} active {len(active)} redrawn {redrawn} won {sum(st[k]['won'] for k in keys)} "
              f"({time.time() - t0:.0f}s)", flush=True)
    for k in keys:
        x = st[k]
        rec = {kk: v for kk, v in x.items() if kk not in ("obs", "adm", "mem")}
        rec.update(n_turns=len(x["steps"]), condition=cond, provenance=prov)
        E.write_json(os.path.join(out_dir, f"s{x['seed']}_r{x['rep']}.json"), rec)
    pool.close(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["valid_seen", "valid_unseen"])
    ap.add_argument("--conds", nargs="+", default=C.CONDITIONS, choices=C.CONDITIONS)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    games = json.load(open(E.GAMES_PATH))[args.split][:args.limit or None]
    from envpool import EnvPool
    pool = EnvPool(args.workers)                     # fork before CUDA
    import alfenv
    if args.dry:
        pol = importlib.import_module("11_replay").DryPolicy()
        pol.vllm_version = "dry"
    else:
        from e4_policy import LoraPolicy
        pol = LoraPolicy()
        pol.set_adapter(None, 0)                     # base weights, same engine configuration as every E4 evaluation
    root = args.out or C.DATA
    for cond in args.conds:
        prov = dict(script="exploratory_E4c/e4c_eval.py", commit=common.git_commit(), argv=sys.argv, condition=cond,
                    hint=C.HINT if cond in ("P", "PR") else None, k_resample=C.K_RESAMPLE, vllm=pol.vllm_version,
                    games_sha256=E.file_sha256(E.GAMES_PATH))
        out_dir = os.path.join(root, "rollout", args.split, cond)
        step = max(1, args.chunk // args.reps)
        for i in range(0, len(games), step):
            run(games[i:i + step], args.reps, pol, pool, out_dir, alfenv, prov, cond)
        per = {str(g["seed"]): [int(json.load(open(os.path.join(out_dir, f"s{g['seed']}_r{r}.json")))["won"]) for r in range(args.reps)]
               for g in games}
        n = sum(len(v) for v in per.values())
        E.write_json(os.path.join(out_dir, "summary.json"), dict(split=args.split, tag=cond, n=n, per_game=per, provenance=prov,
                                                                 success=sum(map(sum, per.values())) / n))
        print(cond, sum(map(sum, per.values())) / n, flush=True)
    pool.shutdown()
    if hasattr(pol, "close"):
        pol.close()


if __name__ == "__main__":
    main()
