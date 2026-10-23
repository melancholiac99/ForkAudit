"""Phase 1B.2: CWGT executed-replay protocol at every decision point of the collected trajectories.

Per decision point (traj, turn t), state s_t = fresh game + re-fed prefix a_1..a_{t-1}:
  * drafts: first batch of 32 policy samples at s_t (T=0.7, seeded) -> `entropy_sampled` input (D-008);
    alternatives = distinct, valid, admissible parsed actions != factual, in draw order; if < 4, further batches of
    32 until 300 draws total. < 4 found -> cf undefined (exclude_reason="lt4_alternatives"), no imputation.
  * continuations: factual action x3, each of the 4 alternatives x3; each = execute first action, then the policy
    (T=0.7, fresh seeds) continues until won / env done / total step count reaches MAX_STEPS (same cap as the
    original trajectory, counted from the episode start). outcome = 1[won].
Outputs data/replay/{traj_id}/t{turn:02d}.json (all drafts, all continuation action sequences + outcomes, seeds,
re-feed divergence counts). Resumable: finished decision points are skipped.

Continuations of many decision points run concurrently in lockstep (one vLLM batch per env step).
Usage: python 11_replay.py --seeds 0-4 [--max_concurrent 600]

Additional banks (CLAUDE.md Phase 1C, RULING 2026-10-23 §7.5):
  python 11_replay.py --bank 1 --points D --seeds 0-99      # bank 2 = data/replay_b1/, all cf-defined points
  python 11_replay.py --bank 2 --points live --seeds 0-99   # banks 3+ = data/replay_b{2..}/, only the live set
For --bank N >= 1 the 4 alternatives are copied from bank 1 (data/replay/), no drafts are drawn, and every seed is
seed_of(...) + N * 10**11 (D-024). "live" = pool of banks 1+2 (30 continuations) not constant.
--dry replaces the model by a random admissible-action policy (CPU plumbing test; write with --out_root to a tmp dir).
"""
import argparse
import glob
import json
import os
import time

import common
from envpool import EnvPool

K_ALT = 4
N_REP = 3
DRAFT_BATCH = 32
MAX_DRAWS = 300


def seed_of(task_seed, turn, role, idx):
    """Deterministic per-request seeds. role: 0=draft batch, 1=factual cont., 2..5=alternative k cont.
    idx = batch index (drafts) or replica*100 + continuation step (continuations)."""
    return ((task_seed * 100 + turn) * 10 + role) * 100_000 + idx


BANK_OFFSET = 10 ** 11  # > max seed_of (~1e10 for task seeds < 100): distinct across banks (D-024)


class DryPolicy:
    """CPU stand-in for Policy: picks a uniformly random admissible action listed in the prompt."""
    vllm_version = "dry"

    def __init__(self):
        import random
        self.random = random

    def sample(self, prompts, seeds, n=1, logprobs=False):
        out = []
        for p, sd in zip(prompts, seeds):
            seg = p.split("Your admissible actions of the current situation are: [", 1)[1].rsplit("].", 1)[0]
            adm = [x.strip().strip("'") for x in seg.split("\n")]
            rng = self.random.Random(sd)
            out.append([{"text": f"<think>dry</think>\n<action>{rng.choice(adm)}</action>", "token_ids": [],
                         "logprobs": None, "finish_reason": "stop"} for _ in range(n)])
        return out

    def close(self):
        pass


def live_b12(traj_id, turn):
    """Pool of banks 1+2 (factual + 4 alternatives, 3 each per bank) is not constant."""
    outs = []
    for root in ("replay", "replay_b1"):
        r = json.load(open(os.path.join(common.ROOT, "data", root, traj_id, f"t{turn:02d}.json")))
        outs += r["factual_outcomes"] + [o for a in r["alt_outcomes"] for o in a]
    return 0 < sum(outs) < len(outs)


def parse_range(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


class Cont:
    """One continuation rollout from s_t."""
    __slots__ = ("key", "dp", "role", "rep", "first_action", "task", "mem", "obs", "adm", "steps_taken",
                 "actions", "valids", "won", "done", "div", "n_invalid", "seeds")

    def __init__(self, key, dp, role, rep, first_action):
        self.key, self.dp, self.role, self.rep, self.first_action = key, dp, role, rep, first_action
        self.actions, self.valids, self.seeds = [], [], []
        self.won = self.done = False
        self.n_invalid = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-49")
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max_concurrent", type=int, default=600, help="max live continuations")
    ap.add_argument("--bank", type=int, default=0, help="0 = bank 1 (drafts + alternatives); N>=1 = bank N+1")
    ap.add_argument("--points", choices=["all", "D", "live"], default="all")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out_root", default=None)
    args = ap.parse_args()
    assert (args.bank == 0) == (args.points == "all"), "bank 1 runs all points; banks >= 2 need --points D|live"

    pool = EnvPool(args.workers)
    import alfenv
    if args.dry:
        pol = DryPolicy()
    else:
        from policy import Policy
        pol = Policy()
    offset = args.bank * BANK_OFFSET

    out_root = args.out_root or os.path.join(common.ROOT, "data", "replay" if args.bank == 0 else f"replay_b{args.bank}")
    trajs = []
    for s in parse_range(args.seeds):
        p = os.path.join(common.ROOT, "data", "rollout", f"s{s:03d}.json")
        trajs.append((p, json.load(open(p))))
    prov = common.provenance(__file__, [p for p, _ in trajs])
    prov["vllm"] = pol.vllm_version
    prov["bank"] = args.bank
    prov["points"] = args.points
    prov["seed_offset"] = offset

    # ---------------- decision points to do
    dps = []
    for _, tr in trajs:
        for i, stp in enumerate(tr["steps"]):
            path = os.path.join(out_root, tr["traj_id"], f"t{stp['turn']:02d}.json")
            if os.path.exists(path):
                continue
            d = {"tr": tr, "i": i, "turn": stp["turn"], "path": path,
                 "prefix": [s["action"] for s in tr["steps"][:i]],
                 "expected": [s["obs_after"] for s in tr["steps"][:i]],
                 "factual": stp["action"]}
            if args.bank >= 1:
                src = os.path.join(common.ROOT, "data", "replay", tr["traj_id"], f"t{stp['turn']:02d}.json")
                b1 = json.load(open(src))
                if not b1["cf_defined"]:
                    continue
                if args.points == "live" and not live_b12(tr["traj_id"], stp["turn"]):
                    continue
                assert b1["factual_action"] == stp["action"]
                d.update(alts=list(b1["alternatives"]), drafts=[], draws=0,
                         alt_source={"path": os.path.relpath(src, common.ROOT), "sha256": common.file_sha256(src)})
            dps.append(d)
    print(f"{len(dps)} decision points to replay", flush=True)

    # ---------------- phase A: drafts (prompt at s_t is the stored prompt; no env needed)
    t0 = time.time()
    pending = list(range(len(dps))) if args.bank == 0 else []
    for d in dps:
        if args.bank == 0:
            d["drafts"], d["alts"], d["draws"] = [], [], 0
    batch_idx = 0
    while pending:
        prompts, seeds, n = [], [], []
        for j in pending:
            d = dps[j]
            prompts.append(open(os.path.join(common.ROOT, d["tr"]["steps"][d["i"]]["prompt_path"])).read())
            seeds.append(seed_of(d["tr"]["seed"], d["turn"], 0, batch_idx) + offset)
            n.append(min(DRAFT_BATCH, MAX_DRAWS - d["draws"]))
        # vLLM n differs per request only via separate SamplingParams; group by n
        outs = {}
        for nn in sorted(set(n)):
            idx = [k for k, v in enumerate(n) if v == nn]
            res = pol.sample([prompts[k] for k in idx], [seeds[k] for k in idx], n=nn, logprobs=False)
            for k, r in zip(idx, res):
                outs[k] = r
        still = []
        for k, j in enumerate(pending):
            d = dps[j]
            adm = d["tr"]["steps"][d["i"]]["admissible"]
            for c in outs[k]:
                a, valid, fmt = common.parse(c["text"])
                ok = valid and a in adm
                d["drafts"].append({"batch": batch_idx, "action": a, "valid": valid, "fmt": fmt, "admissible": a in adm,
                                    "text": c["text"]})
                if ok and a != d["factual"] and a not in d["alts"] and len(d["alts"]) < K_ALT:
                    d["alts"].append(a)
            d["draws"] += len(outs[k])
            if len(d["alts"]) < K_ALT and d["draws"] < MAX_DRAWS:
                still.append(j)
        pending = still
        batch_idx += 1
        print(f"drafts batch {batch_idx}: {len(pending)} points still need alternatives "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---------------- phase B: continuations
    conts = []
    for j, d in enumerate(dps):
        d["cf_defined"] = len(d["alts"]) == K_ALT
        roles = [(1, d["factual"])] + ([(2 + k, a) for k, a in enumerate(d["alts"])] if d["cf_defined"] else [])
        d["n_open"] = 0
        for role, a in roles:
            for r in range(N_REP):
                conts.append(Cont(f"{j}:{role}:{r}", j, role, r, a))
                d["n_open"] += 1
    print(f"{len(conts)} continuations", flush=True)

    queue = list(conts)
    live = {}
    finished = {j: [] for j in range(len(dps))}
    total_div = 0
    n_gen_steps = 0
    while queue or live:
        # open new continuations up to the concurrency cap
        room = args.max_concurrent - len(live)
        if room > 0 and queue:
            new, queue = queue[:room], queue[room:]
            jobs = []
            for c in new:
                d = dps[c.dp]
                jobs.append((c.key, os.path.join(alfenv.ALFWORLD_DATA, d["tr"]["game_file"]), d["prefix"],
                             d["expected"]))
            opened = pool.open(jobs)
            # first action of every new continuation
            first = {}
            for c in new:
                d = dps[c.dp]
                obs, adm, won, div = opened[c.key]
                c.div = div
                total_div += div
                c.task = d["tr"]["task"]
                c.mem = [{"text_obs": s["obs_before"], "action": s["action"]} for s in d["tr"]["steps"][:d["i"]]]
                c.obs, c.adm = obs, adm
                c.steps_taken = d["i"]
                first[c.key] = c.first_action
                live[c.key] = c
            res = pool.step(first)
            for key, a in first.items():
                c = live[key]
                d = dps[c.dp]
                first_valid = d["tr"]["steps"][d["i"]]["valid"] if c.role == 1 else True
                _apply(c, a, first_valid, res[key])
        # one policy step for all live, not-done continuations
        act = [c for c in live.values() if not c.done]
        if act:
            prompts = [common.build_prompt(c.task, c.mem, c.obs, c.adm) for c in act]
            seeds = [seed_of(dps[c.dp]["tr"]["seed"], dps[c.dp]["turn"], c.role, c.rep * 100 + c.steps_taken) + offset
                     for c in act]
            for c, sd in zip(act, seeds):
                c.seeds.append(sd)
            outs = pol.sample(prompts, seeds, logprobs=False)
            acts = {}
            for c, o in zip(act, outs):
                a, valid, _ = common.parse(o[0]["text"])
                acts[c.key] = (a, valid)
            res = pool.step({k: v[0] for k, v in acts.items()})
            for k, (a, valid) in acts.items():
                _apply(live[k], a, valid, res[k])
            n_gen_steps += len(act)
        # retire finished continuations, write finished decision points
        done_keys = [k for k, c in live.items() if c.done]
        if done_keys:
            pool.close(done_keys)
            for k in done_keys:
                c = live.pop(k)
                finished[c.dp].append(c)
                d = dps[c.dp]
                if len(finished[c.dp]) == d["n_open"]:
                    _write(d, finished[c.dp], prov)
                    finished[c.dp] = None
        print(f"live {len(live)} queued {len(queue)} gen_steps {n_gen_steps} div {total_div} "
              f"({time.time() - t0:.0f}s)", flush=True)
    pool.shutdown()
    pol.close()
    print(f"DONE {len(dps)} points, {len(conts)} continuations, {n_gen_steps} generated steps, "
          f"refeed divergences {total_div}, {time.time() - t0:.0f}s")


def _apply(c, a, valid, res):
    obs, adm, won, done = res
    c.mem.append({"text_obs": c.obs, "action": a})
    c.actions.append(a)
    c.valids.append(valid)
    c.n_invalid += (not valid)
    c.obs, c.adm = obs, adm
    c.steps_taken += 1
    if won or done or c.steps_taken >= common.MAX_STEPS:
        c.won, c.done = bool(won), True


def _write(d, conts, prov):
    tr = d["tr"]
    fac = sorted([c for c in conts if c.role == 1], key=lambda c: c.rep)
    alt = {k: sorted([c for c in conts if c.role == 2 + k], key=lambda c: c.rep) for k in range(len(d["alts"]))}
    rec = {
        "traj_id": tr["traj_id"], "turn": d["turn"], "task_id": tr["task_id"], "task_type": tr["task_type"],
        "seed": tr["seed"], "factual_action": d["factual"], "alternatives": d["alts"],
        "cf_defined": d["cf_defined"], "exclude_reason": None if d["cf_defined"] else "lt4_alternatives",
        "n_draws": d["draws"], "drafts": d["drafts"],
        "factual_outcomes": [int(c.won) for c in fac],
        "alt_outcomes": [[int(c.won) for c in alt[k]] for k in range(len(d["alts"]))],
        "continuations": [{"role": c.role, "rep": c.rep, "first_action": c.first_action, "actions": c.actions,
                           "valids": c.valids, "won": c.won, "len": c.steps_taken, "refeed_div": c.div,
                           "seeds": c.seeds}
                          for c in conts],
        "provenance": prov,
    }
    if "alt_source" in d:
        rec["alternatives_source"] = d["alt_source"]
    common.write_json(d["path"], rec)


if __name__ == "__main__":
    main()
