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
import re
import sys
import time
from contextlib import contextmanager
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
PROBE_REGISTRY = DATA / "probe_registrations"


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


def sha_file_bytes(data):
    return hashlib.sha256(data).hexdigest()


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


def shard_points(points, args):
    n, i = getattr(args, 'shard_count', 1), getattr(args, 'shard_index', 0)
    if not 1 <= n <= 2 or not 0 <= i < n:
        raise ValueError('Invalid deterministic point shard')
    return [p for p in points if int(D.digest('E4d-gpu-shard:'+p['point_id']), 16) % n == i]


def execution_metadata():
    return dict(gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                execution_plan_sha256=os.environ.get('E4D_EXECUTION_PLAN_SHA256'),
                source_sha256=sha_file(Path(__file__)))


@contextmanager
def durable_target_writer(replay, tracker):
    """No final result path is visible until its target log is also durable."""
    original = replay._write
    def write(dp, continuations, provenance):
        if any(c.div for c in continuations):
            raise RuntimeError('Divergent replay must not be published')
        final = Path(dp['path'])
        staged = final.with_name(final.name+'.pending-target')
        original(dict(dp, path=str(staged)), continuations, provenance)
        # Reuse the original attachment schema, but only for this completed record.
        proxies = [type('TargetRef', (), dict(dp=0, key=c.key, role=c.role, rep=c.rep))() for c in continuations]
        attach_target_logs([dict(path=str(staged))], proxies, tracker)
        verify_target_log(load(staged))
        if final.exists():raise RuntimeError('Refusing to replace complete replay '+str(final))
        staged.replace(final)
    replay._write = write
    try:yield
    finally:replay._write = original


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
    # CPU-only one-step environment peeks BEFORE observing any screening outcome.
    # The originals are deterministic; store each observation and the target bit
    # because 11_replay continuation files record actions, not tool observations.
    pool = EnvPool(args.workers)
    for start in range(0, len(selected), 60):
        chunk = selected[start:start+60]
        opened_jobs, actions = [], {}
        for state in chunk:
            for j, act in enumerate([state["factual"]] + state["alternatives"]):
                k = f"{state['point_id']}:peek:{j}"
                opened_jobs.append((k, os.path.join(alfenv.ALFWORLD_DATA, state["game_file"]),
                                    state["prefix"], state["expected"]))
                actions[k] = act
        opened = pool.open(opened_jobs)
        for k, (obs, adm, won, div) in opened.items():
            if div or won:
                raise RuntimeError(f"CPU peek prefix invalid: {k}, div={div}, won={won}")
        results = pool.step(actions)
        for state in chunk:
            observations = [results[f"{state['point_id']}:peek:{j}"][0] for j in range(5)]
            if observations[0] != state["factual_obs_after"]:
                raise RuntimeError(f"CPU peek mismatches factual observation: {state['point_id']}")
            signal = D.target_peek_flags(observations, state["target_word"])
            state.update(O_h=signal["O"], peek_target_flags=signal["flags"],
                         peek_observations=observations,
                         peek_obs_sha256=[sha_file_bytes(o.encode("utf-8")) for o in observations],
                         peek_source="CPU-deterministic-one-step-v1")
        pool.close(list(actions))
    pool.shutdown()
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
                initial_obs=p["initial_obs"], target_word=p["target_word"],
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


class TargetLoggingPool:
    """Passive instrumentation for R.run_async, no changed environment actions.

    Captures observations before R.run_async discards them. The original pool is
    still the exact E4/ALFWorld pool; this wrapper affects only the saved logs.
    """
    def __init__(self, pool, dps, conts):
        self.pool = pool
        self.log = {}
        for c in conts:
            d = dps[c.dp]
            word = d["target_word"]
            rx = re.compile(r"\b" + re.escape(word) + r"\s+[0-9]+\b", re.I)
            first = 0 if rx.search(d["initial_obs"]) else None
            if first is None:
                first = next((j+1 for j, obs in enumerate(d["expected"]) if rx.search(obs)), None)
            self.log[c.key] = dict(rx=rx, current_turn=d["turn"], last=d["turn"]-1,
                                   first_seen_turn=first, first_seen_after_decision=None,
                                   seen_before_decision=first is not None, seen_current_step=False,
                                   observations=0)

    def __getattr__(self, name):
        return getattr(self.pool, name)

    def poll(self, timeout=0.0):
        events = self.pool.poll(timeout=timeout)
        for cmd, result in events:
            if cmd not in ("open_step", "step"):
                continue
            for key, value in result.items():
                o = value[1][0] if cmd == "open_step" else value[0]
                log = self.log[key]
                log["last"] += 1
                log["observations"] += 1
                if log["rx"].search(o):
                    if log["first_seen_turn"] is None:
                        log["first_seen_turn"] = log["last"]
                    if log["first_seen_after_decision"] is None:
                        log["first_seen_after_decision"] = log["last"]
                    if log["last"] == log["current_turn"]:
                        log["seen_current_step"] = True
        return events

    def metadata(self, key):
        l = self.log[key]
        return dict(schema="credit-grok-target-v1", target_first_seen_turn=l["first_seen_turn"],
                    first_seen_after_decision=l["first_seen_after_decision"],
                    seen_before_decision=l["seen_before_decision"],
                    seen_on_forced_step=l["seen_current_step"],
                    ever_seen=l["first_seen_turn"] is not None,
                    seen_after_decision=l["first_seen_after_decision"] is not None,
                    observed_steps=l["observations"])


def attach_target_logs(dps, conts, tracker):
    """Reopen R._write output and append per-continuation first-target-turn.
    Every file is independently updated atomically after the replay completes.
    """
    by_dp = collections.defaultdict(list)
    for c in conts:
        by_dp[c.dp].append(c)
    for j, d in enumerate(dps):
        path = Path(d["path"])
        rec = load(path)
        assert len(rec["continuations"]) == len(by_dp[j]), path
        records = {(r["role"], r["rep"]): r for r in rec["continuations"]}
        for c in by_dp[j]:
            records[(c.role, c.rep)]["target_observation"] = tracker.metadata(c.key)
        rec["target_logging_schema"] = "credit-grok-target-v1"
        if len(rec["continuations"]) == 1:
            rec["target_observation"] = rec["continuations"][0]["target_observation"]
        save_json(path, rec)


def verify_target_log(rec):
    if rec.get("target_logging_schema") != "credit-grok-target-v1":
        raise RuntimeError("Replay file lacks target-first-seen logging; use a fresh E4d data directory")
    for c in rec["continuations"]:
        if c.get("target_observation", {}).get("schema") != "credit-grok-target-v1":
            raise RuntimeError("Incomplete per-continuation target logging")


def replay_run(jobs, policy, pool, offset, args, provenance):
    """Run one forced action per record with R.run_async. Each record has its *own*
    first_valid, so a sampled invalid pi1 action cannot be misreported as valid.
    """
    R = importlib.import_module("11_replay")
    provenance = dict(provenance, execution=execution_metadata(), vllm=policy.vllm_version)
    import alfenv
    dps, conts = [], []
    for j, (p, a, valid, path, u) in enumerate(jobs):
        dp, _ = build_dp(p, a, valid, path, u)
        dps.append(dp)
        conts.append(R.Cont(f"{j}:1:{u}", j, 1, u, a))
    if not dps:
        return dict(n=0, gen_steps=0, divergences=0, seconds=0)
    started = time.time()
    tracker = TargetLoggingPool(pool, dps, conts)
    with durable_target_writer(R, tracker):
        gen, div = R.run_async(dps, conts, tracker, stream_for(policy), args, offset, provenance, alfenv, started)
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
    sel = shard_points(sel, args)
    jobs = []
    for p in sel:
        path = DATA / "screen" / p["point_id"] / "replay.json"
        if path.exists():
            r = load(path)
            if r["factual_action"] != p["factual"] or r["alternatives"] != p["alternatives"]:
                raise RuntimeError(f"Stale or mismatched screening output: {path}")
            verify_target_log(r)
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
                model="base", commit=io()[0].git_commit(), execution=execution_metadata(), vllm=pol.vllm_version)
    # The normal engine receives alfenv.ALFWORLD_DATA as the final path locator.
    import alfenv
    tracker = TargetLoggingPool(pool, dps, conts)
    with durable_target_writer(R, tracker):
        gen, div = R.run_async(dps, conts, tracker, R.VllmStream(pol), args, D.SCREEN_OFFSET, prov, alfenv, started) if dps else (0, 0)
    pool.shutdown(); pol.close()
    if div:
        raise RuntimeError(f"Screen refeed divergence: {div}")
    summary = dict(stage="screen", n=len(jobs), seconds=time.time()-started, gen_steps=gen, refeed_div=div,
                   limit=args.limit, selected_sha256=sha_file(SELECTED))
    shard_suffix = f"_shard{args.shard_index}" if getattr(args, 'shard_count', 1) > 1 else ''
    with (DATA / ("screen_runs"+shard_suffix+".jsonl")).open("a", encoding="utf-8") as f:
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
        verify_target_log(rec)
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
    if any(p.get("peek_source") != "CPU-deterministic-one-step-v1" or len(p.get("peek_target_flags", [])) != 5
           or p.get("peek_obs_sha256") != [sha_file_bytes(o.encode("utf-8")) for o in p["peek_observations"]]
           or p.get("peek_target_flags") != D.target_peek_flags(p["peek_observations"], p["target_word"])["flags"]
           or p.get("O_h") != D.target_peek_flags(p["peek_observations"], p["target_word"])["O"] for p in sel):
        raise RuntimeError("One-step O(h) was not frozen in candidates; cannot lock")
    adapters = {}
    for arm in D.ALL_ARMS:
        for seed in D.SEEDS:
            path = ADAPTER_DIR / f"{arm}_s{seed}" / "adapter_model.safetensors"
            if not path.is_file():
                raise FileNotFoundError(f"Frozen E4b model missing: {path}")
            adapters[f"{arm}_s{seed}"] = sha_file(path)
    groups = {p["game"] for p in sel if p["point_id"] in set(z)}
    probe_n = args.probe_n if args.probe_n is not None else D.PROBE_N
    if probe_n not in D.PROBE_N_OPTIONS:
        raise ValueError(f"Probe size must be one of {D.PROBE_N_OPTIONS}")
    probe = D.probe_ids(z, n=probe_n)
    signal = {p["point_id"]: int(p["O_h"]) for p in sel if p["point_id"] in set(z)}
    lock = dict(protocol="E4d-v1", selected_sha256=sha_file(SELECTED), probe_ids=probe,
                probe_requested=probe_n, probe_reps=D.PROBE_REPS,
                peek_signal_O=signal, O_counts=dict(collections.Counter(signal.values())), source_metadata_sha256=D.digest(content["meta"]),
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
    selected = {p["point_id"]: p for p in load(SELECTED)["points"]}
    for pid, O in lock["peek_signal_O"].items():
        if pid not in selected or selected[pid].get("O_h") != O:
            raise RuntimeError(f"Frozen O(h) changed: {pid}")
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
                        adapter_sha256=lock["adapters"][f"{args.arm}_s{args.seed}"], actions=outputs,
                        execution=execution_metadata()))
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
        for p in shard_points(pts, args):
            indices = (range(ps*D.REPS, (ps+1)*D.REPS) if ps is not None else range(len(D.SEEDS)*D.REPS))
            for u in indices:
                path = D.run_path(DATA, args.arm, args.cond, p["point_id"], u)
                if path.exists():
                    rec = load(path)
                    if rec.get("provenance", {}).get("lock_sha256") != sha_file(LOCK):
                        raise RuntimeError(f"Stale formal output under new lock: {path}")
                    verify_target_log(rec)
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
        verify_target_log(rec)
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


def target_mechanism_summary(pts):
    """Secondary non-causal decomposition of 01-vs-00 rescue into evidence
    discovery and execution conditional on discovery. Every replay already
    stores the first target-visible turn; no additional environment steps.
    """
    all_data = {"00": [], "01": []}
    game_data = collections.defaultdict(lambda: {"00": [], "01": []})
    for state in pts:
        for condition, arms in (("00", ("base",)), ("01", D.PRIMARY_ARMS)):
            for arm in arms:
                for u in range(len(D.SEEDS)*D.REPS):
                    path = D.run_path(DATA, arm, condition, state["point_id"], u)
                    rec = load(path)
                    verify_target_log(rec)
                    obs = rec["target_observation"]
                    got = int(obs["ever_seen"])
                    new = int(obs["seen_after_decision"])
                    succ = int(rec["factual_outcomes"][0])
                    row = dict(game=state["game"], state_id=state["point_id"],
                               target_seen=got, seen_after_decision=new,
                               seen_before_decision=int(obs["seen_before_decision"]),
                               first_seen_turn=obs["target_first_seen_turn"],
                               success=succ, seen_and_success=got*succ,
                               unseen_and_success=(1-got)*succ)
                    all_data[condition].append(row)
                    game_data[state["game"]][condition].append(row)
    metrics = ("target_seen", "seen_after_decision", "success",
               "seen_and_success", "unseen_and_success")
    def summarize(rows):
        n = len(rows)
        found = [r for r in rows if r["target_seen"]]
        unseen = [r for r in rows if not r["target_seen"]]
        seen_turns = [r["first_seen_turn"] for r in found if r["first_seen_turn"] is not None]
        return dict(n=n, **{k: float(np.mean([r[k] for r in rows])) for k in metrics},
                    success_if_seen=float(np.mean([r["success"] for r in found])) if found else None,
                    success_if_not_seen=float(np.mean([r["success"] for r in unseen])) if unseen else None,
                    first_seen_turn_mean=float(np.mean(seen_turns)) if seen_turns else None,
                    frac_seen_before_decision=float(np.mean([r["seen_before_decision"] for r in rows])))
    by_cond = {c: summarize(rows) for c, rows in all_data.items()}
    per_game = []
    for game, rec in sorted(game_data.items()):
        m = {c: summarize(rec[c]) for c in ("00", "01")}
        per_game.append(dict(game=game, **{k: m["01"][k] - m["00"][k] for k in metrics}))
    contrasts = {k: D.bootstrap_game(per_game, k) for k in metrics}
    # Always-exact unconditional split; it is NOT a causal mediation estimate.
    prob = by_cond
    p0, p1 = prob["00"]["target_seen"], prob["01"]["target_seen"]
    q0, q1 = prob["00"]["success_if_seen"], prob["01"]["success_if_seen"]
    r0, r1 = prob["00"]["success_if_not_seen"], prob["01"]["success_if_not_seen"]
    if all(v is not None for v in (q0,q1,r0,r1)):
        # Symmetric (Shapley) algebraic decomposition of success = p*q+(1-p)*r.
        prevalence = (p1-p0)*((q1-r1)+(q0-r0))/2
        conditional = ((p1+p0)/2)*(q1-q0) + (1-(p1+p0)/2)*(r1-r0)
        shapley = dict(discovery_prevalence_term_pp=100*prevalence,
                       conditional_execution_term_pp=100*conditional,
                       total_pp=100*(prevalence+conditional),
                       note="Algebraic decomposition, not causal mediation; observed conditional groups can differ")
    else:
        shapley = dict(estimable=False,
                       note="At least one conditional P(success|found/notfound) is undefined; no imputation")
    return dict(conditional=by_cond, differences=contrasts,
                # Independently estimable even when some condition never finds the target.
                exact_success_split_pp={"found_and_succeeded":100*(prob["01"]["seen_and_success"]-prob["00"]["seen_and_success"]),
                                        "not_found_and_succeeded":100*(prob["01"]["unseen_and_success"]-prob["00"]["unseen_and_success"])},
                shapley_descriptive=shapley,
                caveat="First target seen = first ALFWorld numbered-target mention in prefix or continuation observation. "
                       "The discovery/conditional execution split is descriptive and not a causal mediation claim.")


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
    target_summary = target_mechanism_summary(pts)
    output = dict(protocol="E4d-v1", status="FORMAL", reading=verdict, n_states=len(rows), n_games=len(game_rows),
                  lock_sha256=sha_file(LOCK), pooled=d, remaining_bins=bins, per_arm=secondary,
                  zero_vs_gap=zero_gap, first_actions=actions, target_discovery=target_summary,
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
    lines += ["", "Target discovery (secondary descriptive; 01 minus 00, pp):"]
    for key in ("target_seen", "seen_after_decision", "seen_and_success", "unseen_and_success"):
        stat = target_summary["differences"][key]
        lines.append(f"  {key}: {stat['mean_pp']:+.2f} [{stat['ci95_pp'][0]:+.2f}, {stat['ci95_pp'][1]:+.2f}]")
    lines += ["", "Remaining-step bins (descriptive, NOT independent hypotheses):"]
    for k, x in bins.items():
        lines.append(f"  {k}: {x['n_states']} states, {x['n_games']} games, "
                     f"after {x['after']['mean_pp']:+.2f}, one-step {x['one_step']['mean_pp']:+.2f}")
    (HERE / "E4d_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))





# ----------------------------------------------------------------------------- optional credit-grok five-arm probe

def probe_path(policy_id, pid, j, u):
    return DATA / "armprobe" / policy_id / f"{pid}_a{j}_u{u:02d}.json"


def _validate_policy_id(name):
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{1,63}", name) is None:
        raise ValueError("Policy ID must be 2..64 ASCII letters/digits/hyphen/underscore, starting with a letter")
    if name in ("base",) + D.ALL_ARMS:
        raise ValueError("Built-in policy ID is reserved; cannot register it as a new checkpoint")
    return name


def fingerprint_checkpoint(path, kind):
    """Read immutable weight/config hashes at registration and verify again on run."""
    root = Path(path).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Checkpoint must be a local model/adapter directory")
    if kind == "lora":
        files = ["adapter_config.json", "adapter_model.safetensors"]
        if not all((root / f).is_file() for f in files):
            raise FileNotFoundError(f"LoRA config/weights not found under {root}")
    elif kind == "full":
        files = [p.name for p in sorted(root.glob("*.safetensors"))]
        if not files or not (root / "config.json").is_file():
            raise FileNotFoundError(f"Model config/full-weight safetensors missing under {root}")
        files += ["config.json"]
        if (root / "model.safetensors.index.json").exists():
            files.append("model.safetensors.index.json")
    else:
        raise ValueError("checkpoint-kind must be lora or full")
    return dict(path=str(root), kind=kind, sha256={f: sha_file(root / f) for f in sorted(files)})


def stage_register_probe(args):
    """New GRPO/MCR adapter(s) may be registered by an explicit, separately
    dated amendment. This never updates LOCK or original main hypotheses.
    """
    lock = locked()
    name = _validate_policy_id(args.policy_id)
    if not args.approve or not args.revision:
        raise RuntimeError("Registering a future checkpoint requires --approve and --revision")
    if not args.checkpoint_paths or len(args.checkpoint_paths) not in (1, 3):
        raise ValueError("Give either one checkpoint or three training-seed checkpoints")
    if args.checkpoint_kind == "full" and len(args.checkpoint_paths) != 1:
        raise ValueError("Full checkpoints must be registered individually: one full model per policy ID")
    path = PROBE_REGISTRY / f"{name}.json"
    ensure_absent(path)
    files = [fingerprint_checkpoint(d, args.checkpoint_kind) for d in args.checkpoint_paths]
    rec = dict(protocol="credit-grok-probe-registry-v1", policy_id=name, revision=args.revision,
               lock_sha256=sha_file(LOCK), probe_ids_sha256=D.digest(lock["probe_ids"]),
               probe_reps=D.PROBE_REPS, checkpoint_kind=args.checkpoint_kind,
               checkpoints=files, seed_assignment="probe_seed_for(pid,u,n_checkpoints)",
               note="Separate post-E4d amendment. Original main contrast and reading unchanged.")
    save_json(path, rec)
    print("REGISTERED", name, str(path), "SHA256", sha_file(path))


def probe_spec(policy_id, lock):
    if policy_id == "base":
        return dict(id="base", kind="base", paths=[], registration_sha256=None, n_checkpoints=1)
    if policy_id in D.ALL_ARMS:
        if policy_id not in D.PRIMARY_ARMS:
            raise RuntimeError("Only base/Azero/Dzero are built-in probe arms; register others as separate variants")
        paths = [str((ADAPTER_DIR / f"{policy_id}_s{seed}").resolve()) for seed in D.SEEDS]
        return dict(id=policy_id, kind="lora", paths=paths,
                    n_checkpoints=3, registration_sha256=None)
    name = _validate_policy_id(policy_id)
    reg = PROBE_REGISTRY / f"{name}.json"
    if not reg.exists():
        raise FileNotFoundError(f"No frozen checkpoint registration: {reg}")
    r = load(reg)
    if (r["policy_id"] != name or r["lock_sha256"] != sha_file(LOCK) or
        r["probe_ids_sha256"] != D.digest(lock["probe_ids"]) or
        r["probe_reps"] != D.PROBE_REPS):
        raise RuntimeError("Checkpoint registration mismatches E4d lock or probe cohort")
    for c in r["checkpoints"]:
        if fingerprint_checkpoint(c["path"], c["kind"]) != c:
            raise RuntimeError(f"Registered checkpoint changed after immutable registration: {c['path']}")
    return dict(id=name, kind=r["checkpoint_kind"], paths=[c["path"] for c in r["checkpoints"]],
                registration_sha256=sha_file(reg), n_checkpoints=len(r["checkpoints"]))


def stage_armprobe(args):
    """72 (or pre-frozen 96) points, 5 fixed arms, 8 continuations per policy.
    Built-in models cycle across three E4b seeds, exactly 8 TOTAL per model,
    not 8 per seed. Registered RL/MCR checkpoints share cohort, actions and
    continuation random seeds with base/Azero/Dzero.
    """
    lock = locked()
    frozen_first_manifest(lock)
    if args.policy_id and args.arm and args.policy_id != args.arm:
        raise ValueError("Conflicting --arm and --policy-id")
    policy_id = args.policy_id or args.arm
    if not policy_id:
        raise ValueError("Provide --policy-id base|Azero|Dzero|registered_name")
    spec = probe_spec(policy_id, lock)
    ps = {p["point_id"]: p for p in zpoints(lock)}
    subset = [ps[pid] for pid in lock["probe_ids"]]
    if len(subset) != len(lock["probe_ids"]):
        raise RuntimeError("Probe cohort differs from frozen lock")
    subset = shard_points(subset, args)
    from envpool import EnvPool
    from e4_policy import LoraPolicy
    pool = EnvPool(args.workers)
    model_kw = dict(model_path=spec["paths"][0]) if spec["kind"] == "full" else {}
    pol = LoraPolicy(**model_kw)
    summaries = []
    for checkpoint_index in range(spec["n_checkpoints"]):
        if spec["kind"] == "lora":
            pol.set_adapter(spec["paths"][checkpoint_index], 1+checkpoint_index)
            assert pol.lora is not None
        else:
            pol.set_adapter(None, 0)
        jobs = []
        for p in subset:
            forced = [p["factual"]] + p["alternatives"]
            for j, action in enumerate(forced):
                for u in range(D.PROBE_REPS):
                    if D.probe_seed_for(p["point_id"], u, spec["n_checkpoints"]) != checkpoint_index:
                        continue
                    path = probe_path(policy_id, p["point_id"], j, u)
                    if path.exists():
                        rec = load(path)
                        verify_target_log(rec)
                        meta = rec.get("provenance", {})
                        if (meta.get("lock_sha256") != sha_file(LOCK) or
                            meta.get("probe_registration_sha256") != spec["registration_sha256"] or
                            meta.get("policy_id") != policy_id or
                            meta.get("checkpoint_index") != checkpoint_index or
                            rec.get("factual_action") != action):
                            raise RuntimeError(f"Stale armprobe result: {path}")
                        continue
                    jobs.append((p, action, p["factual_valid"] if j == 0 else True, path, u))
        prov = dict(protocol="credit-grok-armprobe-v2", policy_id=policy_id,
                    arm=policy_id, checkpoint_index=checkpoint_index, post_policy=policy_id,
                    lock_sha256=sha_file(LOCK), probe_sha256=D.digest(lock["probe_ids"]),
                    probe_registration_sha256=spec["registration_sha256"],
                    n_reps_per_policy=D.PROBE_REPS, seed_offset=D.PROBE_OFFSET,
                    commit=io()[0].git_commit(), lo_ra_explicit=spec["kind"] == "lora",
                    full_checkpoint=spec["kind"] == "full")
        result = replay_run(jobs, pol, pool, D.PROBE_OFFSET, args, prov)
        summaries.append(dict(checkpoint_index=checkpoint_index, **result))
        print("PROBE", policy_id, checkpoint_index, result, flush=True)
    pool.shutdown(); pol.close()
    print("ARM PROBE COMPLETE", policy_id, summaries)


def read_probe_policy(policy_id, lock):
    """Only return complete 5x8 state arrays with exact frozen continuation IDs."""
    spec = probe_spec(policy_id, lock)
    ps = {p["point_id"]: p for p in zpoints(lock)}
    out = {}
    for pid in lock["probe_ids"]:
        p = ps[pid]
        outcomes = []
        target = []
        for j, action in enumerate([p["factual"]] + p["alternatives"]):
            rewards, found = [], []
            for u in range(D.PROBE_REPS):
                path = probe_path(policy_id, pid, j, u)
                if not path.exists():
                    raise FileNotFoundError(path)
                r = load(path)
                verify_target_log(r)
                prov = r["provenance"]
                idx = D.probe_seed_for(pid, u, spec["n_checkpoints"])
                if (prov.get("lock_sha256") != sha_file(LOCK) or
                    prov.get("probe_sha256") != D.digest(lock["probe_ids"]) or
                    prov.get("policy_id") != policy_id or
                    prov.get("checkpoint_index") != idx or
                    prov.get("probe_registration_sha256") != spec["registration_sha256"] or
                    r["factual_action"] != action or len(r["factual_outcomes"]) != 1 or
                    r["continuations"][0]["rep"] != u):
                    raise RuntimeError(f"Invalid or stale probe output: {path}")
                if any(c["refeed_div"] for c in r["continuations"]):
                    raise RuntimeError(f"Divergent refeed: {path}")
                rewards.append(int(r["factual_outcomes"][0]))
                found.append(int(r["target_observation"]["ever_seen"]))
            outcomes.append(rewards)
            target.append(found)
        seed_ids = [D.probe_seed_for(pid, u, spec["n_checkpoints"]) for u in range(D.PROBE_REPS)]
        gap = D.crossfit_range(outcomes, seed_ids)
        discovered = D.seed_balanced_arm_means(target, seed_ids)
        out[pid] = dict(point_id=pid, game=p["game"], remaining=p["remaining"],
                        O_h=int(lock["peek_signal_O"][pid]), **gap,
                        discovery=float(discovered.mean()), positive_heldout=bool(gap["heldout_gap"] > 0))
    return out


def stage_probe_analyze(args):
    """Secondary triage only. No change to registered E4d primary reading."""
    lock = locked()
    frozen_first_manifest(lock)
    policy_ids = ["base"] + list(D.PRIMARY_ARMS)
    if args.policy_id:
        if args.policy_id not in policy_ids:
            policy_ids.append(args.policy_id)
    allp = {a: read_probe_policy(a, lock) for a in policy_ids}
    points = {p["point_id"]: p for p in zpoints(lock)}
    per_policy = {}
    for a in policy_ids:
        per = list(allp[a].values())
        grouped = collections.defaultdict(list)
        for row in per:
            grouped[row["game"]].append(row)
        games = [dict(game=g, **{key: float(np.mean([z[key] for z in rr]))
                                   for key in ("mean", "range_naive", "heldout_gap", "discovery", "positive_heldout")})
                 for g, rr in sorted(grouped.items())]
        per_policy[a] = dict(n_states=len(per), n_games=len(games),
                             summaries={k: D.bootstrap_game(games, k)
                                        for k in ("mean", "range_naive", "heldout_gap", "discovery", "positive_heldout")})

    comparison = probe_comparison(lock, allp, D.PRIMARY_ARMS)
    # Registered RL checkpoints receive exactly the same analysis, using the
    # base armprobe as the reference; do not re-use the Azero/Dzero contrasts.
    extra = {}
    if args.policy_id and args.policy_id not in ("base",) + D.PRIMARY_ARMS:
        extra[args.policy_id] = probe_comparison(lock, allp, (args.policy_id,))
    output = dict(protocol="credit-grok-armprobe-v2", n_states=len(lock["probe_ids"]),
                  lock_sha256=sha_file(LOCK), requested_probe_n=lock["probe_requested"],
                  reps_per_policy=D.PROBE_REPS, reading_secondary=comparison["reading_secondary"],
                  arms=per_policy, trained_minus_base=comparison["trained_minus_base"],
                  O_groups=comparison["O_groups"], O_prediction=comparison["O_prediction"],
                  remaining_bins=comparison["remaining_bins"], pointwise=comparison["pointwise"],
                  checkpoint_comparisons=extra,
                  caution=("Positive 4+4 heldout scores are noisy; they are not per-point "
                           "significance detections. No observed new fork is not a refutation "
                           "of the policy-dependent-credit hypothesis. Repeated checkpoint "
                           "comparisons are exploratory, not independent confirmations. "
                           "Primary E4d reading is unchanged."))
    path = (DATA / "probe_reports" / f"{args.policy_id}.json") if extra else (HERE / "e4d_armprobe.json")
    save_json(path, output)
    print("PROBE ANALYSIS", path,
          json.dumps({"pooled_reading": comparison["reading_secondary"],
                      "checkpoint_readings": {k:v["reading_secondary"] for k,v in extra.items()},
                      "pooled_O_prediction": comparison["O_prediction"]},
                     indent=2, ensure_ascii=False))


def probe_comparison(lock, allp, trained_arms):
    """One frozen cohort, mean-gain and crossfit-gap comparison with Base.
    `trained_arms` may be (Azero,Dzero) or a registered RL policy alone.
    """
    diffs = []
    for pid in lock["probe_ids"]:
        base = allp["base"][pid]
        trained = {k: float(np.mean([allp[a][pid][k] for a in trained_arms]))
                   for k in ("mean", "heldout_gap", "discovery", "positive_heldout")}
        diffs.append(dict(point_id=pid, game=base["game"], O_h=base["O_h"], remaining=base["remaining"],
                          mean_gain=trained["mean"] - base["mean"],
                          gap_gain=trained["heldout_gap"] - base["heldout_gap"],
                          discovery_gain=trained["discovery"] - base["discovery"],
                          positive_gap_trained=trained["positive_heldout"],
                          positive_gap_base=base["positive_heldout"]))
    by_game = collections.defaultdict(list)
    for row in diffs:
        by_game[row["game"]].append(row)
    game_rows = [dict(game=g, **{k: float(np.mean([x[k] for x in rows]))
                                  for k in ("mean_gain", "gap_gain", "discovery_gain")})
                 for g, rows in sorted(by_game.items())]
    overall = {k: D.bootstrap_game(game_rows, k)
               for k in ("mean_gain", "gap_gain", "discovery_gain")}
    O_groups = {}
    for bit in (0, 1):
        xx = [r for r in diffs if r["O_h"] == bit]
        gr = collections.defaultdict(list)
        for row in xx:
            gr[row["game"]].append(row)
        gg = [dict(game=g, **{key: float(np.mean([x[key] for x in rows]))
                                    for key in ("mean_gain", "gap_gain", "discovery_gain")})
              for g, rows in sorted(gr.items())]
        O_groups[str(bit)] = dict(n_states=len(xx), n_games=len(gg),
                                  rate_positive_gap_trained=float(np.mean([r["positive_gap_trained"] for r in xx])) if xx else None,
                                  rate_positive_gap_base=float(np.mean([r["positive_gap_base"] for r in xx])) if xx else None,
                                  stats={k: D.bootstrap_game(gg, k) for k in ("mean_gain", "gap_gain", "discovery_gain")} if gg else None)
    O_test = D.bootstrap_O_contrast(diffs, field="gap_gain")
    remaining = {}
    for b in ("10-19", "20-29", "30+"):
        rows = [r for r in diffs if D.remaining_bin(r["remaining"]) == b]
        remaining[b] = dict(n=len(rows), O_counts={str(i): sum(r["O_h"] == i for r in rows) for i in (0,1)},
                             predicted_gap_difference=D.bootstrap_O_contrast(rows, field="gap_gain"))
    triage = D.probe_triage(overall["mean_gain"]["ci95_pp"], overall["gap_gain"]["ci95_pp"],
                            O_test["ci95_pp"] if O_test else None,
                            O_groups["1"]["n_states"], O_groups["0"]["n_states"],
                            O_groups["1"]["n_games"], O_groups["0"]["n_games"],
                            O_groups["1"]["stats"]["gap_gain"]["ci95_pp"] if O_groups["1"]["stats"] else None)
    return dict(n_states=len(diffs), n_games=len(game_rows), trained_arms=list(trained_arms),
                reading_secondary=triage, trained_minus_base=overall, O_groups=O_groups,
                O_prediction=O_test, remaining_bins=remaining, pointwise=differences_summary(diffs))


def differences_summary(diffs):
    return dict(n_positive_gap_gain=sum(r["gap_gain"] > 0 for r in diffs),
                frac_positive_gap_gain=float(np.mean([r["gap_gain"] > 0 for r in diffs])),
                note="A positive pointwise heldout score is not an individual significance test")

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["candidates", "screen", "lock", "first", "run", "analyze", "register-probe", "armprobe", "probe-analyze"])
    ap.add_argument("--arm", choices=["base"]+list(D.ALL_ARMS), default=None)
    ap.add_argument("--seed", type=int, choices=D.SEEDS, default=None)
    ap.add_argument("--cond", choices=["00", "01", "10", "11"], default=None)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--max_concurrent", type=int, default=400)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, choices=(1, 2), default=1)
    ap.add_argument("--limit", type=int, default=0, help="screen only: first N selected points (pi0 pilot)")
    ap.add_argument("--approve", action="store_true", help="lock / register-probe: explicit approval")
    ap.add_argument("--probe-n", type=int, choices=D.PROBE_N_OPTIONS, default=None,
                    help="lock only: choose 72 (default) or 96 before any pi1 observations")
    ap.add_argument("--policy-id", default=None, help="armprobe / probe-analyze / register-probe checkpoint identifier")
    ap.add_argument("--checkpoint-paths", nargs="+", default=None,
                    help="register-probe only: one or three local checkpoint directories")
    ap.add_argument("--checkpoint-kind", choices=("lora", "full"), default="lora")
    ap.add_argument("--revision", default=None, help="register-probe: human-readable revision/amendment ID")
    args = ap.parse_args()
    if args.shard_count != 1 and args.stage not in ('screen', 'run', 'armprobe'):
        ap.error('Only replay stages support point sharding')
    if args.probe_n is not None and args.stage != "lock":
        ap.error("--probe-n may only be supplied to lock; changing it after lock is forbidden")
    f = dict(candidates=stage_candidates, screen=stage_screen, lock=stage_lock,
             first=stage_first, run=stage_run, analyze=stage_analyze,
             armprobe=stage_armprobe, **{"probe-analyze": stage_probe_analyze,
                                        "register-probe": stage_register_probe})[args.stage]
    f(args)


if __name__ == "__main__":
    main()
