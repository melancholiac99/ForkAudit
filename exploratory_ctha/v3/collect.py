#!/usr/bin/env python3
"""Fresh multi-turn ALFWorld rollouts with vLLM token IDs for on-policy PPO/GRPO.

Each invocation loads exactly one frozen policy checkpoint. Never train with
validation splits; use the E4 fixed game list without changing frozen E4 files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "exploratory_E4")]
import common
import e4_common as E


def digest(path: Path | None) -> str | None:
    if path is None:
        return None
    return E.file_sha256(path)


def provenance(args, games: list[dict]) -> dict:
    adapter = Path(args.adapter) if args.adapter else None
    if adapter is not None and not (adapter / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Adapter weights unavailable: {adapter}")
    frozen = dict(script="exploratory_ctha/rollout.py", split=args.split,
                  iteration=args.iteration, model=str(Path(args.model).resolve()),
                  adapter_sha256=digest(adapter / "adapter_model.safetensors") if adapter else None,
                  games_sha256=E.file_sha256(E.GAMES_PATH),
                  selected_game_ids=[g["game_file"] for g in games],
                  max_steps=args.max_steps, temperature=args.temperature,
                  max_model_len=args.max_model_len, max_tokens=common.MAX_RESPONSE_TOKENS,
                  history_length=common.HISTORY_LENGTH, reps=args.reps, dry=args.dry,
                  source_sha256=E.file_sha256(__file__),
                  schema="ctha-v3", sampling_logprobs="processed_logprobs",
                  batch_invariant=os.environ.get("VLLM_BATCH_INVARIANT"),
                  enforce_eager=True, seed_namespace=args.seed_namespace)
    frozen["fingerprint"] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    return frozen


SEED_NAMESPACE = "train"
def sample_seed(game_seed, iteration, repeat, turn):
    payload=f"{SEED_NAMESPACE}:{game_seed}:{iteration}:{repeat}:{turn}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4],"little")


class SamplingPolicy:
    def __init__(self, model, adapter, max_model_len, temperature, gpu_util):
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        if os.environ.get("VLLM_BATCH_INVARIANT")!="1":
            raise RuntimeError("v3 requires explicit batch-invariant configuration")
        self.llm=LLM(model=model,dtype="bfloat16",seed=0,
            gpu_memory_utilization=gpu_util,max_model_len=max_model_len,
            enforce_eager=True,enable_prefix_caching=True,generation_config="vllm",
            enable_lora=True,max_lora_rank=16,max_loras=1,
            logprobs_mode="processed_logprobs")
        self.tok=self.llm.get_tokenizer();self.temp=temperature
        self.lora=LoRARequest("ctha_v3",1,str(adapter)) if adapter else None
    def sample(self,prompts,seeds):
        from vllm import SamplingParams
        params=[SamplingParams(temperature=self.temp,top_p=1.,top_k=-1,min_p=0.,
            repetition_penalty=1.,presence_penalty=0.,frequency_penalty=0.,
            max_tokens=common.MAX_RESPONSE_TOKENS,seed=s,logprobs=0) for s in seeds]
        texts=[self.tok.apply_chat_template(common.chat_messages(p),tokenize=False,add_generation_prompt=True) for p in prompts]
        kw={} if self.lora is None else {"lora_request":self.lora}
        results=self.llm.generate(texts,params,use_tqdm=False,**kw)
        out=[]
        for result in results:
            c=result.outputs[0];ids=list(c.token_ids)
            lp=[float(d[t].logprob) for t,d in zip(ids,c.logprobs)]
            assert len(lp)==len(ids) and all(math.isfinite(x) for x in lp)
            out.append([dict(text=c.text,token_ids=ids,behavior_logprobs=lp,
                prompt_token_ids=list(result.prompt_token_ids),finish_reason=c.finish_reason)])
        return out
    def close(self):
        self.llm.llm_engine.engine_core.shutdown()
        del self.llm


class DryPolicy:
    def __init__(self):
        import importlib
        self.inner = importlib.import_module("11_replay").DryPolicy()

    def sample(self, prompts, seeds):
        return self.inner.sample(prompts, seeds, logprobs=False)

    def close(self):
        pass


def write_atomically(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run_chunk(games, args, pool, policy, prov, alfenv):
    pending = []
    for game in games:
        for rep in range(args.reps):
            dest = args.out / f"s{game['seed']}_r{rep}.json"
            if dest.exists():
                old = json.loads(dest.read_text())
                if old.get("provenance", {}).get("fingerprint") != prov["fingerprint"]:
                    raise RuntimeError(f"Refusing to reuse rollouts from another model/config: {dest}")
                if old["game_file"] != game["game_file"] or old["rep"] != rep:
                    raise RuntimeError(f"Existing episode identity mismatch: {dest}")
            else:
                pending.append((game, rep))
    if not pending:
        return
    keys = [f"{g['seed']}:{rep}" for g, rep in pending]
    opened = pool.open([(key, os.path.join(alfenv.ALFWORLD_DATA, g["game_file"]), [], None)
                        for key, (g, rep) in zip(keys, pending)])
    states = {}
    for key, (game, rep) in zip(keys, pending):
        obs, adm, won, div = opened[key]
        if div:
            raise RuntimeError(f"Unexpected environment divergence at reset: {key}")
        states[key] = dict(seed=game["seed"], rep=rep, game_file=game["game_file"],
                           task_type=game["task_type"], task=common.extract_task(obs),
                           initial_obs=obs, obs=obs, adm=adm, mem=[], steps=[],
                           won=bool(won), done=bool(won))
    saved = set()
    def persist(key):
        if key in saved:
            return
        e = states[key]
        record = {k: v for k, v in e.items() if k not in ("mem", "adm", "obs")}
        record["n_turns"] = len(e["steps"])
        record["provenance"] = prov
        write_atomically(args.out / f"s{e['seed']}_r{e['rep']}.json", record)
        saved.add(key)
    try:
        for turn in range(1, args.max_steps + 1):
            live = [key for key in keys if not states[key]["done"]]
            if not live:
                break
            prompts = [common.build_prompt(states[key]["task"], states[key]["mem"],
                                           states[key]["obs"], states[key]["adm"]) for key in live]
            seeds = [sample_seed(states[key]["seed"], args.iteration,
                                 states[key]["rep"], turn) for key in live]
            results = policy.sample(prompts, seeds)
            actions = {}
            for key, prompt, seed, candidates in zip(live, prompts, seeds, results):
                sample = candidates[0]
                action, valid, fmt = common.parse(sample["text"])
                tokens = list(sample["token_ids"])
                if not args.dry and not tokens:
                    raise RuntimeError("vLLM produced an empty completion; cannot run PPO")
                actions[key] = action
                e = states[key]
                e["steps"].append(dict(turn=turn, prompt=prompt, obs_before=e["obs"],
                                       admissible=e["adm"], action=action, valid=bool(valid),
                                       fmt=fmt, response=sample["text"],
                                       response_token_ids=tokens, sample_seed=seed,
                                       behavior_logprobs=sample["behavior_logprobs"],
                                       prompt_token_ids=sample["prompt_token_ids"],
                                       finish_reason=sample["finish_reason"]))
            result = pool.step(actions)
            for key in live:
                e = states[key]
                obs, adm, won, done = result[key]
                e["steps"][-1]["obs_after"] = obs
                e["steps"][-1]["won_after"] = bool(won)
                e["mem"].append({"text_obs": e["obs"], "action": actions[key]})
                e["obs"], e["adm"] = obs, adm
                e["won"], e["done"] = bool(won), bool(won or done)
                if e["done"]:
                    persist(key)
            print(json.dumps(dict(phase="rollout", iteration=args.iteration,
                                  turn=turn, active=len(live),
                                  completed=sum(states[k]["done"] for k in keys))), flush=True)
        for key in keys:
            persist(key)
    finally:
        pool.close(keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True, choices=("train", "tune", "valid_seen", "valid_unseen"))
    ap.add_argument("--games", type=Path, help="JSON list of E4 games; train selection written by run.py")
    ap.add_argument("--iteration", required=True, type=int)
    ap.add_argument("--model", default=os.environ.get("M0_MODEL", common.MODEL_PATH))
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu-util", type=float, default=0.75)
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk-games", type=int, default=16)
    ap.add_argument("--dry", action="store_true", help="Environment-only plumbing test; no usable PPO token IDs")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-namespace", default="train")
    args = ap.parse_args()
    global SEED_NAMESPACE
    SEED_NAMESPACE=args.seed_namespace
    if args.dry:raise ValueError("v3 validation requires actual GPU behavior logprobs")
    if (args.iteration < 0 or args.reps < 1 or args.max_steps < 1 or args.workers < 1
            or args.chunk_games < 1 or not 0 < args.temperature <= 2.0
            or not 0 < args.gpu_util < 1.0):
        ap.error("Invalid rollout configuration")
    registered = json.loads(Path(E.GAMES_PATH).read_text())[args.split]
    known = {x["game_file"]: x for x in registered}
    games = json.loads(args.games.read_text()) if args.games else registered
    if not games or len({g["game_file"] for g in games}) != len(games):
        ap.error("Games must be a nonempty set of distinct games")
    if any(g.get("seed") != known.get(g["game_file"], {}).get("seed") for g in games):
        ap.error("Selected game/seed mismatch or cross-split data leakage")
    prov = provenance(args, games)
    args.out.mkdir(parents=True, exist_ok=True)
    # Fork environment workers before initializing any CUDA engine.
    from envpool import EnvPool
    pool = EnvPool(args.workers)
    try:
        import alfenv
        policy = DryPolicy() if args.dry else SamplingPolicy(args.model, args.adapter,
                                                             args.max_model_len,
                                                             args.temperature, args.gpu_util)
        try:
            for start in range(0, len(games), args.chunk_games):
                run_chunk(games[start:start+args.chunk_games], args, pool, policy, prov, alfenv)
        finally:
            policy.close()
    finally:
        pool.shutdown()
    eps = [json.loads((args.out / f"s{g['seed']}_r{rep}.json").read_text())
           for g in games for rep in range(args.reps)]
    write_atomically(args.out / "summary.json", dict(
        split=args.split, iteration=args.iteration, n_games=len(games), n_episodes=len(eps),
        success_rate=sum(e["won"] for e in eps) / len(eps),
        env_steps=sum(e["n_turns"] for e in eps), provenance=prov))
    print(json.dumps(dict(done=True, split=args.split, iteration=args.iteration,
                          success_rate=sum(e["won"] for e in eps)/len(eps),
                          env_steps=sum(e["n_turns"] for e in eps))), flush=True)


if __name__ == "__main__":
    main()
