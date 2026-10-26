"""Phase 1B.1: collect Qwen2.5-7B-Instruct trajectories on the frozen task list (lockstep across tasks).

Writes data/rollout/{traj_id}.json and prompts/{traj_id}/t{turn:02d}.txt (filled HCAPO template = the true input text).
Usage: python 10_rollout.py [--seeds 0-49]
"""
import argparse
import json
import os
import time

import common
from envpool import EnvPool


def parse_range(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--siblings", default=None,
                    help="k range, e.g. 1-7: sibling rollouts of the same tasks (R3, D-043); traj_id s{seed}_k{k}, "
                         "sample seeds + (20+k)*1e11, output data/siblings/")
    args = ap.parse_args()

    tl_path = os.path.join(common.ROOT, "data", "task_list.json")
    tasks = {t["seed"]: t for t in json.load(open(tl_path))["tasks"]}
    ks = parse_range(args.siblings) if args.siblings else [0]
    assert args.siblings is None or min(ks) >= 1
    seeds = [(s, k) for s in parse_range(args.seeds) for k in ks]  # k = 0: the original single-rollout run
    pool = EnvPool(min(args.workers, len(seeds)))  # fork before CUDA init

    import alfenv
    from policy import Policy
    pol = Policy()
    prov = common.provenance(__file__, [tl_path])
    prov["vllm"] = pol.vllm_version
    prov["siblings"] = args.siblings

    st = {}
    init = pool.open([(sk, os.path.join(alfenv.ALFWORLD_DATA, tasks[sk[0]]["game_file"]), [], None) for sk in seeds])
    for sk in seeds:
        s, k = sk
        obs, adm, won, _ = init[sk]
        st[sk] = {"traj_id": f"s{s:03d}" + (f"_k{k}" if k else ""), "seed": s, "sibling": k,
                  "task_id": tasks[s]["task_id"], "task_type": tasks[s]["task_type"],
                  "game_file": tasks[s]["game_file"], "task": common.extract_task(obs), "initial_obs": obs,
                  "obs": obs, "adm": adm, "mem": [], "steps": [], "won": False, "done": False}

    t0 = time.time()
    for turn in range(1, common.MAX_STEPS + 1):
        active = [s for s in seeds if not st[s]["done"]]
        if not active:
            break
        prompts, rseeds = [], []
        for s in active:
            x = st[s]
            p = common.build_prompt(x["task"], x["mem"], x["obs"], x["adm"])
            d = os.path.join(common.ROOT, "prompts", *(["siblings"] if s[1] else []), x["traj_id"])
            os.makedirs(d, exist_ok=True)
            pp = os.path.join(d, f"t{turn:02d}.txt")
            with open(pp, "w") as f:
                f.write(p)
            x["_prompt_path"] = os.path.relpath(pp, common.ROOT)
            prompts.append(p)
            rseeds.append(1_000_000 * s[0] + turn + ((20 + s[1]) * 10 ** 11 if s[1] else 0))
        outs = pol.sample(prompts, rseeds)
        acts = {}
        for s, rs, o in zip(active, rseeds, outs):
            c = o[0]
            a, valid, fmt = common.parse(c["text"])
            acts[s] = a
            x = st[s]
            x["steps"].append({"turn": turn, "prompt_path": x["_prompt_path"], "obs_before": x["obs"],
                               "admissible": x["adm"], "response": c["text"], "response_token_ids": c["token_ids"],
                               "response_logprobs": c["logprobs"], "finish_reason": c["finish_reason"],
                               "sample_seed": rs, "action": a, "valid": valid, "fmt": fmt,
                               "admissible_hit": a in x["adm"]})
        res = pool.step(acts)
        for s in active:
            obs, adm, won, done = res[s]
            x = st[s]
            x["mem"].append({"text_obs": x["obs"], "action": acts[s]})
            x["steps"][-1].update({"obs_after": obs, "won_after": won})
            x["obs"], x["adm"] = obs, adm
            if won or done:
                x["won"], x["done"] = won, True
        n_done = sum(st[s]["done"] for s in seeds)
        print(f"turn {turn}: active {len(active)} done {n_done} won {sum(st[s]['won'] for s in seeds)} "
              f"elapsed {time.time() - t0:.0f}s", flush=True)

    outdir = os.path.join(common.ROOT, "data", "siblings" if args.siblings else "rollout")
    for s in seeds:
        x = st[s]
        rec = {k: v for k, v in x.items() if k not in ("obs", "adm", "mem", "_prompt_path")}
        rec["n_turns"] = len(x["steps"])
        rec["truncated"] = not x["won"] and len(x["steps"]) >= common.MAX_STEPS
        rec["provenance"] = prov
        common.write_json(os.path.join(outdir, f"{x['traj_id']}.json"), rec)
    pool.shutdown()
    pol.close()
    print("success rate", sum(st[s]["won"] for s in seeds) / len(seeds),
          "mean turns", sum(len(st[s]["steps"]) for s in seeds) / len(seeds))


if __name__ == "__main__":
    main()
