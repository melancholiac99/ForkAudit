"""E4d independent dead-state rescue study. See PREREG_E4d.md and README.md.

Stages: candidates -> screen [--limit 20 pilot] -> lock --approve -> first -> run -> analyze.
Original E4/E4b data and methods are read-only; no retraining occurs in E4d.
"""
import argparse
import collections
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
E4 = ROOT / "exploratory_E4"
E4B = ROOT / "exploratory_E4b"
for p in (str(ROOT / "scripts"), str(E4), str(E4B)):
    if p not in sys.path:
        sys.path.insert(0, p)
import e4d_common as D  # noqa: E402

DATA = HERE / "data"
CANDIDATES = DATA / "candidates.json"
SELECTED = DATA / "selected.json"
LOCK = DATA / "lock.json"
FIRST_MANIFEST = DATA / "first_manifest.json"
BASE_DIR = E4 / "data" / "rollout" / "valid_unseen" / "base"
ADAPTER_DIR = E4B / "data" / "adapters"


def io():
    import common
    import e4_common as E
    return common, E


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, content):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(content, f, indent=2, ensure_ascii=False, sort_keys=True)
    tmp.replace(p)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ensure_absent(*paths):
    for path in paths:
        if Path(path).exists():
            raise RuntimeError(f"Frozen output exists (refusing to overwrite): {path}")


def ensure_unstarted():
    if LOCK.exists() or FIRST_MANIFEST.exists() or list((DATA / "first").glob("*.json")) or list((DATA / "results").rglob("*.json")):
        raise RuntimeError("E4d already locked or pi1 has started; do not change candidate/screen inputs")


def base_episodes():
    """Exactly the existing E4 valid_unseen base r0 artifacts. Never generate replacement trajectories."""
    games = load(E4 / "games.json")["valid_unseen"]
    episodes = []
    for g in games:
        path = BASE_DIR / f"s{g['seed']}_r0.json"
        if not path.is_file():
            raise FileNotFoundError(f"Required E4 baseline is not available: {path}")
        tr = load(path)
        assert tr["seed"] == g["seed"] and tr["game_file"] == g["game_file"]
        assert tr["rep"] == 0 and tr["task_type"] == g["task_type"]
        episodes.append((tr, path))
    return episodes


def prefix_of(tr, i):
    steps = tr["steps"]
    prefix = [s["action"] for s in steps[:i]]
    expected = [s["obs_after"] for s in steps[:i]]
    before = tr["initial_obs"] if i == 0 else steps[i-1]["obs_after"]
    mem = [dict(text_obs=(tr["initial_obs"] if j == 0 else steps[j-1]["obs_after"]),
                action=steps[j]["action"]) for j in range(i)]
    return prefix, expected, before, mem


def stage_candidates(args):
    """Pre-outcome qualification: only failed rep0, search phase, remaining >=10,
    deterministic refeed; draw original E4-style 4 distinct legal alternatives.
    Then equally spaced choose <=3 *among fully qualified points* per failed game.
    """
    ensure_unstarted()
    ensure_absent(CANDIDATES, SELECTED)
    common, E = io()
    from envpool import EnvPool
    import alfenv
    from policy import Policy
    eps = base_episodes()
    files = {str(tr["seed"]): sha_file(path) for tr, path in eps}
    preliminary = []
    n_failed = sum(not tr["won"] for tr, _ in eps)
    for tr, path in eps:
        if tr["won"]:
            continue
        task_word = E.target_word(tr["task"])
        if task_word is None:
            continue
        assert tr["n_turns"] == len(tr["steps"])
        assert tr["task"] == common.extract_task(tr["initial_obs"])
        for i, s in enumerate(tr["steps"]):
            assert s["turn"] == i + 1
            remaining = D.MAX_STEPS - s["turn"]
            if remaining < D.MIN_REMAINING:
                continue
            prefix, expected, before, mem = prefix_of(tr, i)
            preliminary.append(dict(point_id=f"s{tr['seed']}_t{s['turn']:02d}", game=str(tr["seed"]),
                                    seed=tr["seed"], turn=s["turn"], remaining=remaining, i=i,
                                    game_file=tr["game_file"], task=tr["task"], task_type=tr["task_type"],
                                    initial_obs=tr["initial_obs"], prefix=prefix, expected=expected,
                                    obs_before=before, mem=mem, factual=s["action"], factual_valid=s["valid"],
                                    factual_obs_after=s["obs_after"], target_word=task_word,
                                    baseline_sha256=files[str(tr["seed"])], n_draws=0, alternatives=[]))

    # Workers are forked before creating CUDA/vLLM objects.
    pool = EnvPool(args.workers)
    eligible = []
    for start in range(0, len(preliminary), 100):
        chunk = preliminary[start:start+100]
        jobs = [(p["point_id"], os.path.join(alfenv.ALFWORLD_DATA, p["game_file"]),
                 p["prefix"], p["expected"]) for p in chunk]
        opened = pool.open(jobs)
        for p in chunk:
            obs, adm, won, div = opened[p["point_id"]]
            if div or obs != p["obs_before"] or won:
                raise RuntimeError(f"Prefix refeed divergence at {p['point_id']}: {div}")
            prompt = common.build_prompt(p["task"], p["mem"], obs, adm)
            if not E.search_phase(prompt, p["target_word"]):
                continue
            p["prompt"], p["admissible"] = prompt, adm
            p.pop("mem")
            eligible.append(p)
        pool.close([p["point_id"] for p in chunk])
    pool.shutdown()
    print(f"found {len(eligible)} search-phase/remaining eligible positions in {n_failed} failed r0 episodes", flush=True)
    if not eligible:
        raise RuntimeError("No eligible states; do not freeze or start pi1")

    pol = Policy()
    pending = list(range(len(eligible)))
    batch = 0
    while pending:
        n = min(D.DRAFT_BATCH, D.MAX_DRAWS - eligible[pending[0]]["n_draws"])
        assert n > 0
        outs = pol.sample([eligible[j]["prompt"] for j in pending],
                          [((eligible[j]["seed"] * 100 + eligible[j]["turn"]) * 10) * 100000 + batch
                           for j in pending], n=n, logprobs=False)
        more = []
        for j, row in zip(pending, outs):
            p = eligible[j]
            for c in row:
                a, valid, fmt = common.parse(c["text"])
                if valid and a in p["admissible"] and a != p["factual"] and a not in p["alternatives"] and len(p["alternatives"]) < 4:
                    p["alternatives"].append(a)
            p["n_draws"] += len(row)
            if len(p["alternatives"]) < 4 and p["n_draws"] < D.MAX_DRAWS:
                more.append(j)
        pending = more
        batch += 1
        print(f"draft batch {batch}: {len(pending)} states still need four alternatives", flush=True)
    pol.close()

    by_game = collections.defaultdict(list)
    for p in eligible:
        p["cf_defined"] = len(p["alternatives"]) == 4
        if p["cf_defined"]:
            by_game[p["game"]].append(p)
    selected = []
    for g in sorted(by_game):
        selected.extend(D.evenly_spaced(by_game[g]))
    selected.sort(key=lambda p: (p["seed"], p["turn"]))
    if not selected:
        raise RuntimeError("No valid four-arm candidates")
    root = dict(protocol="E4d-v1", source="E4 valid_unseen base rep0 (already evaluated)",
                n_games=len(eps), n_failed=n_failed, n_preliminary=len(preliminary),
                n_search_phase=len(eligible), n_cf_defined=sum(x["cf_defined"] for x in eligible),
                n_selected=len(selected), baseline_shas=files)
    save_json(CANDIDATES, dict(meta=root, points=eligible))
    save_json(SELECTED, dict(meta={**root, "candidates_sha256": sha_file(CANDIDATES)}, points=selected))
    print(f"PREOUTCOME FROZEN CANDIDATES {len(selected)} states across {len(by_game)} games, sha {sha_file(SELECTED)}")


def build_dp(p, forced, valid, path, u):
    """Construct a replay engine decision point from the E4 base prefix and fixed intervention.
    E4 base eval records lack prompts and obs_before. They are reconstructed, not guessed.
    """
    prior = [dict(obs_before=(p["initial_obs"] if j == 0 else p["expected"][j-1]),
                  action=a, valid=True) for j, a in enumerate(p["prefix"])]
    steps = prior + [dict(obs_before=p["obs_before"], action=forced, valid=valid)]
    tr = dict(traj_id=f"s{p['seed']}", seed=p["seed"], task_id=p["game_file"], game_file=p["game_file"],
              task_type=p["task_type"], task=p["task"], steps=steps)
    return dict(tr=tr, i=p["i"], turn=p["turn"], path=str(path), prefix=p["prefix"], expected=p["expected"],
                factual=forced, alts=[], draws=0, drafts=[], cf_defined=True, n_open=1), u


def stream_for(pol):
    """R.VllmStream silently omits LoRA; this adapter-aware stream MUST pass the LoRA request."""
    class LoraStream:
        def __init__(self, policy):
            self.policy = policy
            self.engine = policy.llm.llm_engine
        def add(self, rid, prompt, seed):
            kw = {"lora_request": self.policy.lora} if self.policy.lora is not None else {}
            self.engine.add_request(rid, self.policy.chat_text(prompt), self.policy.params(seed), **kw)
        def step(self):
            return [(o.request_id, o.outputs[0].text) for o in self.engine.step() if o.finished]
        def busy(self):
            return self.engine.has_unfinished_requests()
    return LoraStream(pol)


def replay_run(jobs, policy, pool, offset, args, provenance):
    """Run one forced action per record with R.run_async. Each record has its *own*
    first_valid, so a sampled invalid pi1 action cannot be misreported as valid.
    """
    R = importlib.import_module("11_replay")
    import alfenv
    dps, conts = [], []
    for j, (p, a, valid, path, u) in enumerate(jobs):
        dp, _ = build_dp(p, a, valid, path, u)
        dps.append(dp)
        conts.append(R.Cont(f"{j}:1:{u}", j, 1, u, a))
    if not dps:
        return dict(n=0, gen_steps=0, divergences=0, seconds=0)
    started = time.time()
    gen, div = R.run_async(dps, conts, pool, stream_for(policy), args, offset, provenance, alfenv, started)
    if div:
        raise RuntimeError(f"Refeed divergence during E4d continuation: {div}; outputs invalid")
    return dict(n=len(dps), gen_steps=gen, divergences=div, seconds=time.time()-started)


def stage_screen(args):
    ensure_unstarted()
    from envpool import EnvPool
    from policy import Policy
    sel = load(SELECTED)["points"]
    validate_source_files(sel)
    if args.limit:
        # Hashed, not chronological, pi0-only pilot; full run still uses the full frozen selection.
        sel = sorted(sel, key=lambda p: (D.digest("E4d-pi0-pilot:" + p["point_id"]), p["point_id"]))[:args.limit]
    jobs = []
    for p in sel:
        path = DATA / "screen" / p["point_id"] / "replay.json"
        if path.exists():
            r = load(path)
            if r["factual_action"] != p["factual"] or r["alternatives"] != p["alternatives"]:
                raise RuntimeError(f"Stale or mismatched screening output: {path}")
            continue
        jobs.append((p, path))
    pool = EnvPool(args.workers)
    pol = Policy()
    R = importlib.import_module("11_replay")
    dps, conts = [], []
    for j, (p, path) in enumerate(jobs):
        dp, _ = build_dp(p, p["factual"], p["factual_valid"], path, 0)
        dp.update(alts=p["alternatives"], n_open=15)
        dps.append(dp)
        for role, act in enumerate([p["factual"]] + p["alternatives"], start=1):
            for rep in range(3):
                conts.append(R.Cont(f"{j}:{role}:{rep}", j, role, rep, act))
    started = time.time()
    prov = dict(protocol="E4d-screen-v1", source_sha256=sha_file(SELECTED), seed_offset=D.SCREEN_OFFSET,
                model="base", commit=io()[0].git_commit())
    # The normal engine receives alfenv.ALFWORLD_DATA as the final path locator.
    import alfenv
    gen, div = R.run_async(dps, conts, pool, R.VllmStream(pol), args, D.SCREEN_OFFSET, prov, alfenv, started) if dps else (0, 0)
    pool.shutdown(); pol.close()
    if div:
        raise RuntimeError(f"Screen refeed divergence: {div}")
    summary = dict(stage="screen", n=len(jobs), seconds=time.time()-started, gen_steps=gen, refeed_div=div,
                   limit=args.limit, selected_sha256=sha_file(SELECTED))
    with (DATA / "screen_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
    print(summary)


def stage_lock(args):
    ensure_absent(LOCK)
    if not args.approve:
        raise RuntimeError("Explicit --approve required: freeze pre/post policy hypotheses BEFORE any pi1 outcome")
    if FIRST_MANIFEST.exists() or list((DATA / "first").glob("*.json")) or list((DATA / "results").rglob("*.json")):
        raise RuntimeError("Pi1 first actions or outcomes exist before lock. Stop; do not retrospectively freeze")
    content = load(SELECTED)
    sel = content["points"]
    validate_source_files(sel)
    counts = collections.Counter()
    z = []
    screen_shas = {}
    for p in sel:
        path = DATA / "screen" / p["point_id"] / "replay.json"
        if not path.exists():
            raise RuntimeError(f"Missing pi0 screening record: {path}")
        rec = load(path)
        assert rec["factual_action"] == p["factual"] and rec["alternatives"] == p["alternatives"]
        assert rec["cf_defined"] and len(rec["factual_outcomes"]) == 3 and len(rec["alt_outcomes"]) == 4
        if any(c["refeed_div"] for c in rec["continuations"]):
            raise RuntimeError(f"Nondeterministic replay at {p['point_id']}")
        cls = D.classify([rec["factual_outcomes"]] + rec["alt_outcomes"])
        counts[cls] += 1
        screen_shas[p["point_id"]] = sha_file(path)
        if cls == "Z3":
            z.append(p["point_id"])
    if not z:
        raise RuntimeError("No Z3 points")
    adapters = {}
    for arm in D.ALL_ARMS:
        for seed in D.SEEDS:
            path = ADAPTER_DIR / f"{arm}_s{seed}" / "adapter_model.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"Frozen E4b model missing: {path}")
            adapters[f"{arm}_s{seed}"] = sha_file(path)
    groups = {p["game"] for p in sel if p["point_id"] in set(z)}
    lock = dict(protocol="E4d-v1", selected_sha256=sha_file(SELECTED), probe_ids=D.probe_ids(z), source_metadata_sha256=D.digest(content["meta"]),
                screen_shas=screen_shas, classes=dict(counts), Z3=sorted(z), n_Z3=len(z), n_games_Z3=len(groups),
                adapters=adapters, n_seeds=len(D.SEEDS), reps_per_seed=D.REPS,
                original_policy="Qwen2.5-7B-Instruct E4 base", pi1_models=list(D.ALL_ARMS),
                first_offset=D.FIRST_OFFSET, formal_offset=D.FORMAL_OFFSET,
                min_for_power=dict(n_states=D.MIN_Z3, n_games=D.MIN_GAMES),
                underpowered=len(z) < D.MIN_Z3 or len(groups) < D.MIN_GAMES,
                note="No pi1 first action/continuation has been generated; pi0 screening observed before freezing.")
    save_json(LOCK, lock)
    print("LOCKED", lock["n_Z3"], "Z3 states across", lock["n_games_Z3"], "games", sha_file(LOCK),
          "underpowered=", lock["underpowered"])


def locked():
    if not LOCK.exists():
        raise RuntimeError("No immutable E4d lock; run screen and lock --approve first")
    lock = load(LOCK)
    if lock["selected_sha256"] != sha_file(SELECTED):
        raise RuntimeError("Selected candidate list changed after lock")
    for arm, seedsha in lock["adapters"].items():
        path = ADAPTER_DIR / arm / "adapter_model.safetensors"
        if not path.is_file() or sha_file(path) != seedsha:
            raise RuntimeError(f"Trained model missing or SHA mismatch: {path}")
    return lock


def validate_source_files(points):
    seen = {}
    for p in points:
        s = p["seed"]
        if s not in seen:
            path = BASE_DIR / f"s{s}_r0.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            seen[s] = sha_file(path)
        if seen[s] != p["baseline_sha256"]:
            raise RuntimeError(f"Source E4 base trajectory changed: s{s}_r0.json")


def zpoints(lock):
    z = set(lock["Z3"])
    pts = [p for p in load(SELECTED)["points"] if p["point_id"] in z]
    assert len(pts) == len(z)
    validate_source_files(pts)
    return pts


def first_path(arm, seed):
    return DATA / "first" / f"{arm}_s{seed}.json"


def frozen_first_manifest(lock):
    """Freeze all 12 action files, once, BEFORE any pi1 continuation can begin."""
    paths = {f"{arm}_s{seed}": first_path(arm, seed) for arm in D.ALL_ARMS for seed in D.SEEDS}
    sha = {}
    for key, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"Must sample all first steps before formal runs: {path}")
        f = load(path)
        if f["lock_sha256"] != sha_file(LOCK) or f["adapter_sha256"] != lock["adapters"][key]:
            raise RuntimeError(f"First action file has wrong lock or adapter: {path}")
        sha[key] = sha_file(path)
    if FIRST_MANIFEST.exists():
        m = load(FIRST_MANIFEST)
        if m["lock_sha256"] != sha_file(LOCK) or m["sha256"] != sha:
            raise RuntimeError("First action file changed after formal start: locked SHA mismatch")
    else:
        if list((DATA / "results").rglob("*.json")) or list((DATA / "armprobe").rglob("*.json")):
            raise RuntimeError("Formal results exist before first actions were frozen; refuse retrospective lock")
        save_json(FIRST_MANIFEST, dict(lock_sha256=sha_file(LOCK), sha256=sha))
    return sha_file(FIRST_MANIFEST)


def stage_first(args):
    lock = locked()
    if args.arm not in D.ALL_ARMS:
        raise ValueError("first must use a trained E4b arm")
    if args.seed not in D.SEEDS:
        raise ValueError("seed must be 0,1,2")
    if FIRST_MANIFEST.exists():
        raise RuntimeError("All first actions already frozen; cannot regenerate a model's current-step actions")
    out = first_path(args.arm, args.seed)
    ensure_absent(out)
    from e4_policy import LoraPolicy
    common, E = io()
    pol = LoraPolicy()
    adapter = ADAPTER_DIR / f"{args.arm}_s{args.seed}"
    pol.set_adapter(str(adapter), 1 + D.ALL_ARMS.index(args.arm)*3 + args.seed)
    assert pol.lora is not None, "LoRA adapter not activated"
    ps = zpoints(lock)
    outputs = {}
    for start in range(0, len(ps), 100):
        chunk = ps[start:start+100]
        prompts, seeds = [], []
        for p in chunk:
            for rep in range(D.REPS):
                prompts.append(p["prompt"])
                # Seed is deliberately disjoint from pi0 screening and all continuation seeds.
                R = importlib.import_module("11_replay")
                seeds.append(R.seed_of(p["seed"], p["turn"], 7, args.seed*D.REPS+rep) + D.FIRST_OFFSET)
        sampled = pol.sample(prompts, seeds, logprobs=False)
        k = 0
        for p in chunk:
            seq = []
            for rep in range(D.REPS):
                txt = sampled[k][0]["text"]
                action, valid, fmt = common.parse(txt)
                seq.append(dict(rep=rep, action=action, valid=bool(valid), fmt=fmt, sampling_seed=seeds[k],
                                same_as_factual=action == p["factual"],
                                in_screen_arms=action in [p["factual"]] + p["alternatives"]))
                k += 1
            outputs[p["point_id"]] = seq
    pol.close()
    save_json(out, dict(lock_sha256=sha_file(LOCK), arm=args.arm, seed=args.seed,
                        adapter_sha256=lock["adapters"][f"{args.arm}_s{args.seed}"], actions=outputs))
    print("FIRST", out, len(outputs), "sha", sha_file(out))


def stage_run(args):
    lock = locked()
    if args.arm == "base":
        assert args.cond == "00", "base only supports 00"
    else:
        assert args.arm in D.ALL_ARMS and args.cond in ("01", "10", "11")
    pts = zpoints(lock)
    by_id = {p["point_id"]: p for p in pts}
    first_manifest_sha = frozen_first_manifest(lock)
    # First-step actions for *all* arms/seeds are frozen before ANY formal continuation.
    for a in D.ALL_ARMS:
        for s in D.SEEDS:
            path = first_path(a, s)
            if not path.exists():
                raise RuntimeError(f"All pi1 first-action files must exist before formal rollout: {path}")
            f = load(path)
            if f["lock_sha256"] != sha_file(LOCK) or f["adapter_sha256"] != lock["adapters"][f"{a}_s{s}"]:
                raise RuntimeError(f"First-action file mismatches frozen lock: {path}")
            if set(f["actions"]) != set(by_id) or any(len(v) != D.REPS for v in f["actions"].values()):
                raise RuntimeError(f"Incomplete first-action file: {path}")
    actions = {}
    if args.arm != "base" and args.cond in ("10", "11"):
        for seed in D.SEEDS:
            rec = load(first_path(args.arm, seed))
            assert rec["lock_sha256"] == sha_file(LOCK)
            assert rec["adapter_sha256"] == lock["adapters"][f"{args.arm}_s{seed}"]
            assert set(rec["actions"]) == set(by_id)
            actions[seed] = rec["actions"]
    post_policy = args.arm if args.cond in ("01", "11") else "base"
    from envpool import EnvPool
    from e4_policy import LoraPolicy
    pool = EnvPool(args.workers)  # critical: fork before CUDA
    pol = LoraPolicy()
    summaries = []
    seed_sets = D.SEEDS if post_policy != "base" else (None,)
    for ps in seed_sets:
        if ps is not None:
            pol.set_adapter(str(ADAPTER_DIR / f"{args.arm}_s{ps}"), 1+D.ALL_ARMS.index(args.arm)*3+ps)
            assert pol.lora is not None, "post-policy LoRA not enabled"
        else:
            pol.set_adapter(None, 0)
        jobs = []
        for p in pts:
            indices = (range(ps*D.REPS, (ps+1)*D.REPS) if ps is not None else range(len(D.SEEDS)*D.REPS))
            for u in indices:
                path = D.run_path(DATA, args.arm, args.cond, p["point_id"], u)
                if path.exists():
                    rec = load(path)
                    if rec.get("provenance", {}).get("lock_sha256") != sha_file(LOCK):
                        raise RuntimeError(f"Stale formal output under new lock: {path}")
                    continue
                a, valid = p["factual"], p["factual_valid"]
                if args.cond in ("10", "11"):
                    r = actions[u//D.REPS][p["point_id"]][u%D.REPS]
                    assert r["rep"] == u%D.REPS
                    a, valid = r["action"], r["valid"]
                jobs.append((p, a, valid, path, u))
        prov = dict(protocol="E4d-formal-v1", arm=args.arm, cond=args.cond,
                    first_policy=(args.arm if args.cond in ("10", "11") else "base"),
                    post_policy=post_policy, train_seed=ps, lock_sha256=sha_file(LOCK),
                    adapter_sha256=(lock["adapters"][f"{args.arm}_s{ps}"] if ps is not None else None),
                    selected_sha256=lock["selected_sha256"], first_manifest_sha256=first_manifest_sha,
                    seed_offset=D.FORMAL_OFFSET,
                    commit=io()[0].git_commit(), lo_ra_explicit=True)
        result = replay_run(jobs, pol, pool, D.FORMAL_OFFSET, args, prov)
        summaries.append(dict(train_seed=ps, **result))
        print(args.arm, args.cond, "post_seed", ps, result, flush=True)
    pool.shutdown(); pol.close()
    print("FORMAL RUN COMPLETE", args.arm, args.cond, summaries)


def result_prob(arm, cond, p, lock):
    outcomes = []
    for u in range(len(D.SEEDS)*D.REPS):
        path = D.run_path(DATA, arm, cond, p["point_id"], u)
        if not path.exists():
            raise FileNotFoundError(f"Missing required formal result: {path}")
        rec = load(path)
        provenance = rec["provenance"]
        if (provenance["lock_sha256"] != sha_file(LOCK) or
            provenance["first_manifest_sha256"] != sha_file(FIRST_MANIFEST) or
            provenance["arm"] != arm or provenance["cond"] != cond):
            raise RuntimeError(f"Lock or intervention condition mismatch in {path}")
        if (provenance["post_policy"] != (arm if cond in ("01", "11") else "base") or
            provenance["first_policy"] != (arm if cond in ("10", "11") else "base")):
            raise RuntimeError(f"Incorrect policy routing in {path}")
        expected_first = p["factual"]
        if cond in ("10", "11"):
            fa = load(first_path(arm, u // D.REPS))
            expected_first = fa["actions"][p["point_id"]][u % D.REPS]["action"]
        if rec["factual_action"] != expected_first or rec["continuations"][0]["first_action"] != expected_first:
            raise RuntimeError(f"Wrong forced action in {path}")
        if any(c["refeed_div"] for c in rec["continuations"]):
            raise RuntimeError(f"Refeed divergence in {path}")
        assert len(rec["factual_outcomes"]) == 1
        assert rec["seed"] == p["seed"] and rec["turn"] == p["turn"]
        outcomes.append(rec["factual_outcomes"][0])
    return float(np.mean(outcomes))


def stage_analyze(args):
    lock = locked()
    pts = zpoints(lock)
    frozen_first_manifest(lock)
    rows = []
    for p in pts:
        b = result_prob("base", "00", p, lock)
        row = dict(point_id=p["point_id"], game=p["game"], remaining=p["remaining"], base=b)
        for arm in D.ALL_ARMS:
            row[arm] = {"00": b}
            for c in ("01", "10", "11"):
                row[arm][c] = result_prob(arm, c, p, lock)
        rows.append(row)
    game_rows = D.game_contrasts(rows)
    d = {k: D.bootstrap_game(game_rows, k) for k in ("base", "after", "step", "full", "rescue", "one_step",
                                                        "primary", "full_gain", "interaction")}
    verdict = D.reading(d["rescue"]["ci95_pp"], d["primary"]["ci95_pp"], len(rows), len(game_rows))
    bins = {}
    for b in ("10-19", "20-29", "30+"):
        sub = [x for x in rows if D.remaining_bin(x["remaining"]) == b]
        if sub:
            gr = D.game_contrasts(sub)
            bins[b] = dict(n_states=len(sub), n_games=len(gr),
                           after=D.bootstrap_game(gr, "rescue"), one_step=D.bootstrap_game(gr, "one_step"),
                           primary=D.bootstrap_game(gr, "primary"))
    # Primary pooled models are Azero, Dzero only. Gap-trained models are compulsory secondary comparisons.
    secondary = {}
    for arm in D.ALL_ARMS:
        g = []
        grouped = collections.defaultdict(list)
        for p in rows:
            grouped[p["game"]].append(p)
        for game, ss in sorted(grouped.items()):
            g.append(dict(game=game, rescue=float(np.mean([x[arm]["01"]-x["base"] for x in ss])),
                          one_step=float(np.mean([x[arm]["10"]-x["base"] for x in ss])),
                          main=float(np.mean([x[arm]["01"]-x[arm]["10"] for x in ss]))))
        secondary[arm] = {f: D.bootstrap_game(g, f) for f in ("rescue", "one_step", "main")}
    # Paired zero-versus-gap trained models are secondary and cannot change the main reading.
    zero_gap = {}
    for zero_arm, gap_arm in (("Azero", "Agap"), ("Dzero", "Dgap")):
        group = collections.defaultdict(list)
        for p in rows:
            group[p["game"]].append(p)
        gr = []
        for game, ss in sorted(group.items()):
            gr.append(dict(game=game,
                           rescue=float(np.mean([p[zero_arm]["01"] - p[gap_arm]["01"] for p in ss])),
                           main=float(np.mean([(p[zero_arm]["01"] - p[zero_arm]["10"]) -
                                               (p[gap_arm]["01"] - p[gap_arm]["10"]) for p in ss]))))
        zero_gap[f"{zero_arm}-{gap_arm}"] = {metric: D.bootstrap_game(gr, metric)
                                             for metric in ("rescue", "main")}
    actions = {}
    for arm in D.ALL_ARMS:
        vals = []
        for seed in D.SEEDS:
            fa = load(first_path(arm, seed))
            assert fa["lock_sha256"] == sha_file(LOCK)
            vals += [a for seq in fa["actions"].values() for a in seq]
        actions[arm] = dict(n=len(vals), same_as_factual=float(np.mean([v["same_as_factual"] for v in vals])),
                            in_screen_arms=float(np.mean([v["in_screen_arms"] for v in vals])),
                            invalid=float(np.mean([not v["valid"] for v in vals])))
    output = dict(protocol="E4d-v1", status="FORMAL", reading=verdict, n_states=len(rows), n_games=len(game_rows),
                  lock_sha256=sha_file(LOCK), pooled=d, remaining_bins=bins, per_arm=secondary,
                  zero_vs_gap=zero_gap, first_actions=actions,
                  warning="Z3 is 15/15 observed zeros, not proof Q=0. Rescue need not be unique to zero-trained models.")
    save_json(HERE / "e4d.json", output)
    lines = [f"E4d reading: {verdict} (n={len(rows)} Z3 states; {len(game_rows)} games)",
             f"lock_sha256: {sha_file(LOCK)}", "", "Primary contrasts (pp, 95% game bootstrap CI):"]
    for field in ("rescue", "one_step", "primary", "full_gain", "interaction"):
        stat = d[field]
        lines.append(f"  {field:13s}: {stat['mean_pp']:+.2f}  [{stat['ci95_pp'][0]:+.2f}, {stat['ci95_pp'][1]:+.2f}]")
    lines += ["", "Per trained arm:"]
    for arm, vals in secondary.items():
        lines.append(f"  {arm}: after {vals['rescue']['mean_pp']:+.2f}; one-step {vals['one_step']['mean_pp']:+.2f}; "
                     f"contrast {vals['main']['mean_pp']:+.2f}")
    lines += ["", "Secondary zero-trained vs gap-trained (paired game contrasts):"]
    for pair, vals in zero_gap.items():
        lines.append(f"  {pair}: rescue diff {vals['rescue']['mean_pp']:+.2f}; "
                     f"primary diff {vals['main']['mean_pp']:+.2f}")
    lines += ["", "Remaining-step bins (descriptive, NOT independent hypotheses):"]
    for k, x in bins.items():
        lines.append(f"  {k}: {x['n_states']} states, {x['n_games']} games, "
                     f"after {x['after']['mean_pp']:+.2f}, one-step {x['one_step']['mean_pp']:+.2f}")
    (HERE / "E4d_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))




def probe_path(arm, pid, j, u):
    return DATA / "armprobe" / arm / f"{pid}_a{j}_u{u:02d}.json"


def stage_armprobe(args):
    """Optional *predeclared* secondary: follow the same 5 screened actions with pi0 / pi1.
    NOT used in the registered E4d primary reading. Both uniform uplift and
    between-arm differentiation are compatible with the Section 7 proposal.
    """
    lock = locked()
    frozen_first_manifest(lock)
    if args.arm not in ("base",) + D.PRIMARY_ARMS:
        raise ValueError("Arm probe supports base/Azero/Dzero only")
    ps = {p["point_id"]: p for p in zpoints(lock)}
    subset = [ps[pid] for pid in lock["probe_ids"]]
    from envpool import EnvPool
    from e4_policy import LoraPolicy
    pool = EnvPool(args.workers)
    pol = LoraPolicy()
    summaries = []
    loops = (None,) if args.arm == "base" else D.SEEDS
    for seed in loops:
        if seed is None:
            pol.set_adapter(None, 0)
        else:
            pol.set_adapter(str(ADAPTER_DIR / f"{args.arm}_s{seed}"),
                            1 + D.ALL_ARMS.index(args.arm)*3 + seed)
            assert pol.lora is not None
        jobs = []
        for p in subset:
            forced = [p["factual"]] + p["alternatives"]
            indices = range(24) if seed is None else range(seed*8, (seed+1)*8)
            for j, a in enumerate(forced):
                for u in indices:
                    path = probe_path(args.arm, p["point_id"], j, u)
                    if path.exists():
                        rec = load(path)
                        if rec.get("provenance", {}).get("lock_sha256") != sha_file(LOCK):
                            raise RuntimeError(f"Stale arm probe result: {path}")
                        continue
                    jobs.append((p, a, p["factual_valid"] if j == 0 else True, path, u))
        prov = dict(protocol="E4d-armprobe-v1", arm=args.arm, train_seed=seed,
                    lock_sha256=sha_file(LOCK), probe_sha256=D.digest(lock["probe_ids"]),
                    seed_offset=D.PROBE_OFFSET, adapter_sha256=(
                        lock["adapters"][f"{args.arm}_s{seed}"] if seed is not None else None),
                    commit=io()[0].git_commit(), lo_ra_explicit=True)
        s = replay_run(jobs, pol, pool, D.PROBE_OFFSET, args, prov)
        summaries.append(dict(seed=seed, **s))
        print("PROBE", args.arm, seed, s, flush=True)
    pool.shutdown(); pol.close()
    print("ARM PROBE DONE", args.arm, summaries)


def stage_probe_analyze(args):
    lock = locked()
    ps = {p["point_id"]: p for p in zpoints(lock)}
    out = dict(lock_sha256=sha_file(LOCK), selected_probe_ids=lock["probe_ids"], arms={})
    for arm in ("base",) + D.PRIMARY_ARMS:
        per = []
        for pid in lock["probe_ids"]:
            p = ps[pid]
            outcomes = []
            for j in range(5):
                rewards = []
                for u in range(24):
                    path = probe_path(arm, pid, j, u)
                    if not path.exists():
                        raise FileNotFoundError(path)
                    r = load(path)
                    assert r["provenance"]["lock_sha256"] == sha_file(LOCK)
                    assert len(r["factual_outcomes"]) == 1
                    if any(c["refeed_div"] for c in r["continuations"]):
                        raise RuntimeError(f"Refeed failed in {path}")
                    assert r["factual_action"] == ([p["factual"]] + p["alternatives"])[j]
                    rewards.append(r["factual_outcomes"][0])
                outcomes.append(rewards)
            per.append(dict(point_id=pid, game=p["game"], **D.crossfit_range(outcomes)))
        games = collections.defaultdict(list)
        for x in per:
            games[x["game"]].append(x)
        g = [dict(game=k, **{key: float(np.mean([x[key] for x in vv]))
                            for key in ("mean", "range_naive", "heldout_gap")})
             for k, vv in sorted(games.items())]
        out["arms"][arm] = dict(n_states=len(per), n_games=len(g),
                                summaries={field: D.bootstrap_game(g, field)
                                           for field in ("mean", "range_naive", "heldout_gap")})
    out["interpretation"] = ("Mean value can rise with or without action differentiation. In-sample range is "+
        "upward biased by 24 finite draws per arm; crossfit heldout_gap is the noise-controlled secondary. "+
        "This analysis is descriptive, does not enter E4d's primary reading.")
    save_json(HERE / "e4d_armprobe.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["candidates", "screen", "lock", "first", "run", "analyze", "armprobe", "probe-analyze"])
    ap.add_argument("--arm", choices=["base"]+list(D.ALL_ARMS), default=None)
    ap.add_argument("--seed", type=int, choices=D.SEEDS, default=None)
    ap.add_argument("--cond", choices=["00", "01", "10", "11"], default=None)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max_concurrent", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="screen only: first N selected points (pi0 pilot)")
    ap.add_argument("--approve", action="store_true", help="lock only: explicit pre-pi1 research freeze")
    args = ap.parse_args()
    f = dict(candidates=stage_candidates, screen=stage_screen, lock=stage_lock,
             first=stage_first, run=stage_run, analyze=stage_analyze,
             armprobe=stage_armprobe, **{"probe-analyze": stage_probe_analyze})[args.stage]
    f(args)


if __name__ == "__main__":
    main()
