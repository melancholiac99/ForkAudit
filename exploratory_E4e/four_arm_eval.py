"""E4-style four-arm evaluation of carrying-state visibility and legal recovery actions.

Only inference; NEVER changes existing E4/E4d registries, samples, or adapters.

python exploratory_E4e/four_arm_eval.py --split valid_unseen --policy-tag base --conditions base state hard both
python exploratory_E4e/four_arm_eval.py --split valid_unseen --policy-tag aprime_s0 \
    --adapter exploratory_E4/data/adapters/Aprime_s0 --conditions base state hard both
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for directory in (ROOT / "scripts", ROOT / "exploratory_E4"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from stage_logic import CarryState, forced_release_action
import common
import e4_common as E

CONDITIONS = ("base", "state", "hard", "both")


def prompt_for(task: str, mem: list, obs: str, adm: list[str], carry: CarryState, cond: str) -> str:
    text = common.build_prompt(task, mem, obs, adm)
    if cond in ("state", "both"):
        anchor = "Now it's your turn to take an action."
        if text.count(anchor) != 1:
            raise RuntimeError("HCAPO template anchor changed; refuse to silently move state line")
        text = text.replace(anchor, carry.description() + "\n" + anchor)
    return text


def run(games, reps, pol, pool, out_dir, cond, alfenv, prov):
    """Uses E4 seeds, 50-step cap and exact same games; only prompt/decoding differs."""
    if cond not in CONDITIONS:
        raise ValueError(cond)
    from e4_01_rollout import seed_for

    os.makedirs(out_dir, exist_ok=True)
    jobs = [(g, r) for g in games for r in range(reps)
            if not os.path.exists(os.path.join(out_dir, f"s{g['seed']}_r{r}.json"))]
    if not jobs:
        return
    keys = [f"{g['seed']}:{rep}" for g, rep in jobs]
    opened = pool.open([(key, os.path.join(alfenv.ALFWORLD_DATA, g["game_file"]), [], None)
                        for key, (g, _) in zip(keys, jobs)])
    active_state = {}
    for key, (g, rep) in zip(keys, jobs):
        obs, adm, won, div = opened[key]
        if div:
            raise RuntimeError(f"Unexpected prefix divergence on reset: {key}")
        cs = CarryState()
        cs.update(obs)
        active_state[key] = dict(seed=g["seed"], rep=rep, game_file=g["game_file"],
                                 task_type=g["task_type"], task=common.extract_task(obs),
                                 initial_obs=obs, obs=obs, adm=adm, mem=[], steps=[],
                                 carry=cs, won=bool(won), done=bool(won))
    try:
        for turn in range(1, common.MAX_STEPS + 1):
            active = [key for key in keys if not active_state[key]["done"]]
            if not active:
                break
            prompts = [prompt_for(active_state[key]["task"], active_state[key]["mem"],
                                  active_state[key]["obs"], active_state[key]["adm"],
                                  active_state[key]["carry"], cond) for key in active]
            seeds = [seed_for(active_state[key]["seed"], turn, active_state[key]["rep"]) for key in active]
            model_out = pol.sample(prompts, seeds, logprobs=False)
            to_execute = {}
            for key, prompt, sample_seed, out in zip(active, prompts, seeds, model_out):
                x = active_state[key]
                proposed, valid, fmt = common.parse(out[0]["text"])
                forced = (forced_release_action(x["carry"], x["task"], x["task_type"], x["adm"])
                          if cond in ("hard", "both") else None)
                chosen = forced or proposed
                if forced is not None and forced not in x["adm"]:
                    raise RuntimeError("Recovery rule chose a non-admissible action")
                to_execute[key] = chosen
                x["steps"].append(dict(turn=turn, action=chosen, model_action=proposed,
                                        rule_triggered=forced is not None,
                                        overridden=forced is not None and proposed != forced,
                                        held_before=x["carry"].item if x["carry"].known else None,
                                        held_known_before=x["carry"].known,
                                        valid=valid, fmt=fmt, sample_seed=sample_seed,
                                        obs_before=x["obs"], admissible=x["adm"]))
            responses = pool.step(to_execute)
            for key in active:
                x = active_state[key]
                obs, adm, won, done = responses[key]
                x["carry"].update(obs)
                x["steps"][-1].update(obs_after=obs, won_after=bool(won),
                                       held_after=x["carry"].item if x["carry"].known else None,
                                       held_known_after=x["carry"].known)
                x["mem"].append(dict(text_obs=x["obs"], action=to_execute[key]))
                x["obs"], x["adm"] = obs, adm
                x["won"], x["done"] = bool(won), bool(won or done)
            print(f"[{cond}] turn={turn} active={len(active)} won={sum(active_state[k]['won'] for k in keys)}", flush=True)
        for key in keys:
            x = active_state[key]
            result = {k: v for k, v in x.items() if k not in ("carry", "obs", "adm", "mem")}
            result.update(n_turns=len(x["steps"]), condition=cond, provenance=prov)
            E.write_json(os.path.join(out_dir, f"s{x['seed']}_r{x['rep']}.json"), result)
    finally:
        pool.close(keys)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", required=True, choices=["train", "tune", "valid_seen", "valid_unseen"])
    p.add_argument("--policy-tag", required=True, help="Safe name for this model, e.g. base or gigpo7b")
    p.add_argument("--model-path", default=common.MODEL_PATH, help="Full model path; defaults to E4 base")
    p.add_argument("--adapter", default=None, help="Optional existing LoRA adapter (e.g. Aprime_s0)")
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--chunk", type=int, default=160)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--dry", action="store_true", help="Use E4's mock policy; still requires ALFWorld")
    p.add_argument("--out", default=str(ROOT / "exploratory_E4e" / "data" / "rollout"))
    args = p.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.policy_tag):
        p.error("--policy-tag may only contain ASCII letters/digits/-/_")
    if args.reps < 1 or args.workers < 1 or args.chunk < args.reps:
        p.error("Require reps>=1, workers>=1, chunk>=reps")
    games = json.loads(Path(E.GAMES_PATH).read_text())[args.split]
    if args.limit:
        games = games[:args.limit]
    # Environment workers must fork BEFORE initializing CUDA/vLLM.
    from envpool import EnvPool
    pool = EnvPool(args.workers)
    try:
        import alfenv
        if args.dry:
            pol = importlib.import_module("11_replay").DryPolicy()
            pol.vllm_version = "dry"
        else:
            from e4_policy import LoraPolicy
            pol = LoraPolicy(model_path=args.model_path)
            pol.set_adapter(args.adapter, 1)
        try:
            for cond in args.conditions:
                dest = os.path.join(args.out, args.split, args.policy_tag, cond)
                prov = dict(script="exploratory_E4e/four_arm_eval.py", commit=common.git_commit(), argv=sys.argv,
                            model=args.model_path, adapter=args.adapter, condition=cond,
                            vllm=pol.vllm_version, games_sha256=E.file_sha256(E.GAMES_PATH))
                batch_size = max(1, args.chunk // args.reps)
                for i in range(0, len(games), batch_size):
                    run(games[i:i + batch_size], args.reps, pol, pool, dest, cond, alfenv, prov)
                per_game = {str(g["seed"]): [int(json.loads(Path(dest, f"s{g['seed']}_r{r}.json").read_text())["won"])
                                             for r in range(args.reps)] for g in games}
                n = sum(map(len, per_game.values()))
                E.write_json(os.path.join(dest, "summary.json"), dict(
                    split=args.split, tag=args.policy_tag, condition=cond, n=n,
                    success=sum(map(sum, per_game.values())) / max(n, 1),
                    per_game=per_game, provenance=prov))
                print(f"[{args.policy_tag}/{cond}] success={sum(map(sum, per_game.values())) / max(n, 1):.4f}")
        finally:
            if hasattr(pol, "close"):
                pol.close()
    finally:
        pool.shutdown()


if __name__ == "__main__":
    main()
