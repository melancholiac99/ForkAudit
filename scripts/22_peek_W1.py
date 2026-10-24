"""P2 groundwork (DECISIONS D-029): world-side one-step peeks (W1) and privileged state (W0) at every decision point.

For each rollout step with a bank-1 replay file: reset + re-feed the action prefix (verified against the stored
observations), then execute ONE action and record what the environment returns. Nothing here is shown to the policy,
no LLM is run, and no replay outcome is read (bank-1 files are opened only for their alternative-action list).

--set alts        peek the factual action and the bank-1 alternatives; also store the full fact set at s_t (W0)
--set admissible  peek every admissible action (minus 'help') plus the chosen one; facts as deltas only
Every peek costs one reset + re-feed (TextWorld PDDL envs cannot be copied: native pointers).
Determinism: factual peek obs must equal the stored obs_after; on points with (hash % 20 == 0) every peek is run
twice and compared (`repeat_match`).
Output: data/peek_{set}/{traj_id}/t{turn:02d}.json (resumable per point).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time

import common

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["alts", "admissible"], default="alts")
ap.add_argument("--seeds", default="0-99")
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--limit", type=int, default=0, help="debug: only the first N points")
ap.add_argument("--out-root", default=None)
args = ap.parse_args()


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def facts_of(gs):
    return sorted(str(p) for p in gs["facts"])


_game = None


def _env(game_file):
    """One facts-enabled game per worker, reused across points (same wrapper stack as alfenv.AlfGame)."""
    global _game
    import alfenv  # noqa: F401  (sets ALFWORLD_DATA)
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    if _game is None:
        infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True, extras=["gamefile"])
        _game = [textworld.start(game_file, request_infos=infos,
                                 wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]), game_file]
    elif _game[1] != game_file:
        _game[0].load(game_file)
        _game[1] = game_file
    return _game[0]


def _refeed(env, prefix, expected):
    gs = env.reset()
    div = 0
    for a, e in zip(prefix, expected):
        gs, _, _ = env.step(a)
        div += gs.feedback != e
    return gs, div


def _peek(env, prefix, expected, action, base_facts):
    gs, div = _refeed(env, prefix, expected)
    g2, _, done = env.step(action)
    f2 = set(facts_of(g2))
    return {"obs": g2.feedback, "admissible_after": [a for a in g2["admissible_commands"] if a != "help"],
            "won": bool(g2["won"]), "done": bool(done),
            "facts_added": sorted(f2 - base_facts), "facts_removed": sorted(base_facts - f2), "refeed_div": div}


def work(job):
    t0 = time.time()
    env = _env(job["game_file"])
    prefix, expected = job["prefix"], job["expected"]
    gs, div = _refeed(env, prefix, expected)
    base = facts_of(gs)
    base_set = set(base)
    state = {"obs": gs.feedback, "admissible": [a for a in gs["admissible_commands"] if a != "help"],
             "obs_match": gs.feedback == job["obs_before"], "refeed_div": div}
    if job["set"] == "alts":
        state["facts"] = base
    repeat = int(hashlib.sha256(f"{job['traj_id']}|{job['turn']}".encode()).hexdigest(), 16) % 20 == 0
    peeks = []
    for role, idx, a in job["actions"]:
        p = _peek(env, prefix, expected, a, base_set)
        p.update(action=a, role=role, alt_index=idx)
        if repeat:
            q = _peek(env, prefix, expected, a, base_set)
            p["repeat_match"] = all(p[k] == q[k] for k in ("obs", "admissible_after", "won", "facts_added",
                                                            "facts_removed"))
        peeks.append(p)
    fac = [p for p in peeks if p["role"] == "factual"]
    out = {"traj_id": job["traj_id"], "turn": job["turn"], "task_id": job["task_id"], "set": job["set"],
           "chosen_action": job["chosen"], "state": state, "peeks": peeks,
           "checks": {"factual_obs_match": bool(fac) and fac[0]["obs"] == job["obs_after"],
                      "refeed_div_total": div + sum(p["refeed_div"] for p in peeks),
                      "repeat_checked": repeat,
                      "repeat_all_match": all(p.get("repeat_match", True) for p in peeks)},
           "alternatives_source": job["alt_source"], "seconds": round(time.time() - t0, 2),
           "provenance": job["prov"]}
    common.write_json(job["path"], out)
    return job["path"], out["checks"], out["state"]["obs_match"]


def main():
    import alfenv
    out_root = args.out_root or os.path.join(common.ROOT, "data", f"peek_{args.set}")
    rollouts = [os.path.join(common.ROOT, "data", "rollout", f"s{s:03d}.json") for s in parse_range(args.seeds)]
    prov = common.provenance(__file__, rollouts)
    prov["script_sha256"] = common.file_sha256(os.path.abspath(__file__))
    prov["set"] = args.set
    jobs = []
    for rp in rollouts:
        tr = json.load(open(rp))
        for i, stp in enumerate(tr["steps"]):
            src = os.path.join(common.ROOT, "data", "replay", tr["traj_id"], f"t{stp['turn']:02d}.json")
            path = os.path.join(out_root, tr["traj_id"], f"t{stp['turn']:02d}.json")
            if not os.path.exists(src) or os.path.exists(path):
                continue
            b1 = json.load(open(src))
            assert b1["factual_action"] == stp["action"]
            alts = list(b1["alternatives"])
            acts = [("factual", None, stp["action"])] + [("alt", j, a) for j, a in enumerate(alts)]
            if args.set == "admissible":
                seen = {a for _, _, a in acts}
                acts += [("other", None, a) for a in stp["admissible"] if a != "help" and a not in seen]
            jobs.append({"traj_id": tr["traj_id"], "turn": stp["turn"], "task_id": tr["task_id"], "set": args.set,
                         "game_file": os.path.join(alfenv.ALFWORLD_DATA, tr["game_file"]),
                         "prefix": [s["action"] for s in tr["steps"][:i]],
                         "expected": [s["obs_after"] for s in tr["steps"][:i]],
                         "obs_before": stp["obs_before"], "obs_after": stp["obs_after"], "chosen": stp["action"],
                         "actions": acts, "path": path,
                         "alt_source": {"path": os.path.relpath(src, common.ROOT), "sha256": common.file_sha256(src)},
                         "prov": prov})
    if args.limit:
        jobs = jobs[:args.limit]
    n_peeks = sum(len(j["actions"]) for j in jobs)
    print(f"{len(jobs)} points, {n_peeks} peeks, set={args.set}, workers={args.workers}", flush=True)
    t0 = time.time()
    bad = 0
    with mp.get_context("fork").Pool(args.workers) as pool:
        for k, (path, chk, om) in enumerate(pool.imap_unordered(work, jobs, chunksize=1), 1):
            ok = om and chk["factual_obs_match"] and chk["refeed_div_total"] == 0 and chk["repeat_all_match"]
            if not ok:
                bad += 1
                print(f"CHECK FAIL {path} {chk} state_obs_match={om}", flush=True)
            if k % 50 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} points, {bad} check failures ({time.time() - t0:.0f}s)", flush=True)
    print(f"DONE {len(jobs)} points, {n_peeks} peeks, {bad} check failures, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
