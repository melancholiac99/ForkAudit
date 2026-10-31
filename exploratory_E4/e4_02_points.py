"""E4 step 2: select search-phase decision points of the training trajectories, draw policy drafts until four distinct
alternatives exist (M0 protocol, scripts/11_replay.py phase A), and record the one-step environment response of all
five arms (factual + 4 alternatives). No outcome beyond one step is observed here.

Selection (frozen, PREREG_E4 §2): every step of every train trajectory whose decision-time prompt contains no instance
of the task's target object (E.search_phase), with a known target word. Points with < 4 alternatives after 300 draws
are kept in the manifest as `lt4_alternatives` and receive no label.

Output: data/points/s{seed}_t{turn:02d}.json, data/points_manifest.json (counts, CPU seconds of peeks).
Usage: python e4_02_points.py [--dry --limit N --out DIR]
"""
import argparse
import glob
import json
import os
import sys
import time

import e4_common as E
import common
from envpool import EnvPool

K_ALT, DRAFT_BATCH, MAX_DRAWS = 4, 32, 300


def draft_seed(task_seed, turn, batch):
    """scripts/11_replay.py seed_of(task_seed, turn, role=0, idx=batch)."""
    return ((task_seed * 100 + turn) * 10 + 0) * 100_000 + batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N trajectories")
    ap.add_argument("--rollouts", default=os.path.join(E.DATA, "rollout", "train", "base"))
    ap.add_argument("--out", default=os.path.join(E.DATA, "points"))
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.rollouts, "s*_r0.json")))
    if args.limit:
        files = files[:args.limit]
    pool = EnvPool(args.workers)
    import alfenv
    if args.dry:
        import importlib
        pol = importlib.import_module("11_replay").DryPolicy()
    else:
        from policy import Policy
        pol = Policy()

    # ---------------- candidate points
    pts = []
    n_steps = 0
    for f in files:
        tr = json.load(open(f))
        word = E.target_word(tr["task"])
        for i, st in enumerate(tr["steps"]):
            n_steps += 1
            pid = f"s{tr['seed']}_t{st['turn']:02d}"
            if os.path.exists(os.path.join(args.out, pid + ".json")) or not E.search_phase(st["prompt"], word):
                continue
            pts.append(dict(pid=pid, tr=tr, i=i, st=st, word=word, drafts=[], alts=[], draws=0))
    print(f"{len(files)} trajectories, {n_steps} steps, {len(pts)} search-phase points to process", flush=True)

    # ---------------- drafts (phase A of the M0 replay protocol)
    pending, batch = list(range(len(pts))), 0
    t0 = time.time()
    while pending:
        prompts = [pts[j]["st"]["prompt"] for j in pending]
        seeds = [draft_seed(pts[j]["tr"]["seed"], pts[j]["st"]["turn"], batch) for j in pending]
        outs = pol.sample(prompts, seeds, n=DRAFT_BATCH, logprobs=False)
        still = []
        for j, o in zip(pending, outs):
            d = pts[j]
            adm, fac = d["st"]["admissible"], d["st"]["action"]
            for c in o:
                a, valid, fmt = common.parse(c["text"])
                d["drafts"].append(dict(batch=batch, action=a, valid=valid, admissible=a in adm, text=c["text"]))
                if valid and a in adm and a != fac and a not in d["alts"] and len(d["alts"]) < K_ALT:
                    d["alts"].append(a)
            d["draws"] += len(o)
            if len(d["alts"]) < K_ALT and d["draws"] < MAX_DRAWS:
                still.append(j)
        pending, batch = still, batch + 1
        print(f"draft batch {batch}: {len(pending)} points still short ({time.time() - t0:.0f}s)", flush=True)
    gpu_sec_drafts = time.time() - t0

    # ---------------- one-step peeks of the five arms
    t1 = time.time()
    ok = [d for d in pts if len(d["alts"]) == K_ALT]
    for c0 in range(0, len(ok), 200):
        chunk = ok[c0:c0 + 200]
        jobs, acts = [], {}
        for d in chunk:
            prefix = [s["action"] for s in d["tr"]["steps"][:d["i"]]]
            expected = [s["obs_after"] for s in d["tr"]["steps"][:d["i"]]]
            for j, a in enumerate([d["st"]["action"]] + d["alts"]):
                key = f"{d['pid']}:{j}"
                jobs.append((key, os.path.join(alfenv.ALFWORLD_DATA, d["tr"]["game_file"]), prefix, expected))
                acts[key] = a
        opened = pool.open(jobs)
        res = pool.step(acts)
        pool.close(list(acts))
        for d in chunk:
            d["peeks"], d["refeed_div"] = [], 0
            for j in range(5):
                key = f"{d['pid']}:{j}"
                d["refeed_div"] += opened[key][3]
                obs, adm, won, done = res[key]
                d["peeks"].append(dict(obs=obs, won=bool(won), done=bool(done)))
            assert d["peeks"][0]["obs"] == d["st"]["obs_after"], ("factual peek mismatch", d["pid"])
    cpu_sec_peeks = time.time() - t1

    # ---------------- write
    for d in pts:
        tr, st = d["tr"], d["st"]
        first32 = [x["action"] for x in d["drafts"] if x["batch"] == 0 and x["valid"] and x["admissible"]]
        texts = [st["response"]]
        for a in d["alts"]:
            texts.append(next(x["text"] for x in d["drafts"] if x["action"] == a and x["valid"] and x["admissible"]))
        rec = dict(point_id=d["pid"], seed=tr["seed"], turn=st["turn"], game_file=tr["game_file"], task_type=tr["task_type"],
                   task=tr["task"], target_word=d["word"], initial_obs=tr["initial_obs"], final_reward=int(tr["won"]),
                   prompt=st["prompt"], factual=st["action"], alternatives=d["alts"],
                   cf_defined=len(d["alts"]) == K_ALT, n_draws=d["draws"],
                   entropy_sampled=E.sampled_entropy(first32),
                   arm_responses=texts if len(d["alts"]) == K_ALT else None,
                   peeks=d.get("peeks"), refeed_div=d.get("refeed_div"),
                   history=[dict(turn=s["turn"], executed_action=s["action"], observation_after=s["obs_after"])
                            for s in tr["steps"][:d["i"]]],
                   episode=[dict(turn=s["turn"], response=s["response"], executed_action=s["action"],
                                 observation_after=s["obs_after"]) for s in tr["steps"]],
                   drafts=[{k: v for k, v in x.items() if k != "text"} for x in d["drafts"]])
        E.write_json(os.path.join(args.out, d["pid"] + ".json"), rec)
    man = dict(n_trajectories=len(files), n_steps=n_steps, n_selected=len(pts), n_cf_defined=len(ok),
               gpu_sec_drafts=gpu_sec_drafts, cpu_sec_peeks=cpu_sec_peeks, total_refeed_div=sum(d.get("refeed_div") or 0 for d in pts),
               commit=common.git_commit(), argv=sys.argv, vllm=pol.vllm_version)
    E.write_json(os.path.join(os.path.dirname(args.out.rstrip("/")), "points_manifest.json"), man)
    pool.shutdown()
    if hasattr(pol, "close"):
        pol.close()
    print(man, flush=True)


if __name__ == "__main__":
    main()
