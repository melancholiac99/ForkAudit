"""E5 (PREREG_E5.md): the M0 measurement and the P4 E2 judge experiment, replicated on WebShop.

Stages (run in order; each is resumable or idempotent):
  freeze            150 measurement sessions from the WebShop training goal range -> tasks.json (commit before rollout)
  rollout           one trajectory per session (base 7B, T=0.7, 15 steps, history 2)
  bank --b 1..6     executed replay: b=1 draws drafts and the 4 alternatives; b>=3 only on Live (banks 1+2 pool not constant)
  peek              one-step responses of the five arms
  labels            Live / Fork (five-arm G test, 12 runs per arm from banks 3-6) / Credit-pivotal / Masked, null expectations
  strata            I / U / N from the goal ASIN (decisive fact) in prompt, history and arm responses
  signals           within-Live B2 AUROC of entropy_sampled and of the W1 goal-mention bit, with progress-tercile shuffle
  judge-prepare     S / L / F / Fx contexts for Live I and U points
  judge-score       --cond C --model PATH --name {7b,32b} --revision REV   (P4 backend and numerical gate)
  analyze           predictions W-1..W-5 and the reading -> e5.json, E5_REPORT.txt
Add --dry (random policy; --out_root /tmp/x) for plumbing tests; WebShop itself is required for every stage but analyze.
"""
import argparse
import glob
import gzip
import hashlib
import importlib
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "exploratory_E4"))
import common  # noqa: E402
import ws_common as W  # noqa: E402

N_TASKS, TASK_SEED, FIRST_TRAIN_GOAL = 150, 20261031, 1500
K_ALT, DRAFT_BATCH, MAX_DRAWS, N_REP = 4, 32, 300, 3
SEED_BASE, OFFSET = 5000, 9 * 10 ** 12
sha = lambda b: hashlib.sha256(b).hexdigest()


def paths(root):
    return types.SimpleNamespace(root=root, tasks=os.path.join(HERE, "tasks.json"), rollout=os.path.join(root, "rollout"),
                                 bank=lambda b: os.path.join(root, f"replay_b{b}"), peek=os.path.join(root, "peek"),
                                 labels=os.path.join(root, "labels.json"), strata=os.path.join(root, "strata.json"),
                                 signals=os.path.join(root, "signals.json"), judge=os.path.join(root, "judge"))


# ----------------------------------------------------------------------------- pure functions (test_e5.py)

def stratum(prompt, history_text, arm_obs, asin):
    """P4 definitions with the goal ASIN as the decisive fact (case-insensitive substring)."""
    a = asin.lower()
    pt = a in prompt.lower()
    mention = [a in o.lower() for o in arm_obs]
    new = any(mention) and a not in history_text.lower() and not pt
    if not pt and new:
        return "I"
    if pt and any(mention) and not new:
        return "U"
    if not pt and not any(mention):
        return "N"
    return "other"


def point_labels(f_banks, a_banks, rng_seed, n_perm=10_000, alpha=0.10):
    """f_banks: {b: [3 outcomes]}, a_banks: {b: [[3 outcomes] x 4]} for b in 1..6 (3-6 may be missing if not Live).
    -> dict(live, het, credit_pivotal, masked, p_het, q12, q3456, E_het)."""
    H = importlib.import_module("18_het_fork")
    pool12 = list(f_banks[1]) + list(f_banks[2]) + [o for b in (1, 2) for arm in a_banks[b] for o in arm]
    live = 0 < sum(pool12) < len(pool12)
    q12 = [float(np.mean(f_banks[1] + f_banks[2]))] + [float(np.mean(a_banks[1][j] + a_banks[2][j])) for j in range(4)]
    out = dict(live=bool(live), q12=q12, het=False, credit_pivotal=False, masked=False, p_het=None, q3456=None, E_het=None)
    if not live or not all(b in f_banks and b in a_banks for b in (3, 4, 5, 6)):
        return out
    f = np.concatenate([f_banks[b] for b in (3, 4, 5, 6)])
    a = np.stack([np.concatenate([a_banks[b][j] for b in (3, 4, 5, 6)]) for j in range(4)])
    c = np.concatenate([[f.sum()], a.sum(1)])[None, :]
    G = H.g_stat(c)[0]
    o = np.concatenate([f, a.ravel()])
    perm = np.random.default_rng(rng_seed).permuted(np.tile(o, (n_perm, 1)), axis=1)
    Gp = H.g_stat(perm.reshape(n_perm, 5, 12).sum(2))
    p = (1 + np.sum(Gp >= G - 1e-9)) / (n_perm + 1)
    srt = np.sort(Gp)
    p_perm = (1 + (n_perm - np.searchsorted(srt, Gp - 1e-9, side="left"))) / (n_perm + 1)
    het, agg = bool(p <= alpha), bool(H.agg_flag(c)[0])
    out.update(het=het, credit_pivotal=agg, masked=het and not agg, p_het=float(p), E_het=float((p_perm <= alpha).mean()),
               q3456=[float(f.mean())] + [float(x.mean()) for x in a])
    return out


def task_bootstrap(values, clusters, B=10000, seed=31102026):
    v = np.asarray(values, float)
    cl, inv = np.unique(np.asarray(clusters), return_inverse=True)
    Wt = np.random.default_rng(seed).multinomial(len(cl), np.full(len(cl), 1 / len(cl)), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (Wt @ np.bincount(inv, v, len(cl))) / (Wt @ np.bincount(inv, None, len(cl)))
    d = d[np.isfinite(d)]
    return dict(estimate=float(100 * v.mean()), ci95=[float(100 * np.percentile(d, 2.5)), float(100 * np.percentile(d, 97.5))],
                n=int(len(v)), n_tasks=int(len(cl)))


def predictions(c, n_fork):
    """c: contrast dict (pp, CI) keyed as in analyze(); n_fork: {'I': n, 'U': n}. -> (preds, reading)."""
    lo = lambda k: c[k]["ci95"][0]
    hi = lambda k: c[k]["ci95"][1]
    powered = n_fork["I"] >= 40 and n_fork["U"] >= 40
    w3 = lo("32b-7b/S/U") > 0 and lo("32b-7b/S/I") <= 0 <= hi("32b-7b/S/I") and lo("interaction/S") > 0
    w4 = lo("7b/F-S/I") > 0 and lo("32b/F-S/I") > 0 and lo("32b/F-Fx/I") > 0
    w5 = lo("7b/F-L/I") > 0 and lo("32b/F-L/I") > 0
    preds = {"W-3": w3, "W-4": w4, "W-5": w5, "powered": powered}
    if not powered:
        return preds, "R-E5-UNDERPOWERED"
    return preds, ("R-E5-REPLICATE" if (w3 and w4 and w5) else "R-E5-PARTIAL")


# ----------------------------------------------------------------------------- environment stages

def get_pool(n):
    import ws_env
    return ws_env.WSPool(n)


def get_policy(dry):
    if dry:
        return W.DryPolicy()
    from policy import Policy
    return Policy()


def stage_freeze(P, args):
    import ws_env
    assert not os.path.exists(P.tasks), "tasks.json already frozen"
    s = ws_env.Session()
    n_goals = len(s.env.server.goals)
    rng = np.random.default_rng(TASK_SEED)
    sessions = sorted(rng.choice(np.arange(FIRST_TRAIN_GOAL, n_goals), size=N_TASKS, replace=False).tolist())
    tasks = []
    for k, x in enumerate(sessions):
        s._reset(x)
        s.current = None
        tasks.append(dict(seed=SEED_BASE + k, session=int(x), goal_asin=s.goal_asin()))
    json.dump(dict(n_goals=n_goals, first_train_goal=FIRST_TRAIN_GOAL, task_seed=TASK_SEED, tasks=tasks), open(P.tasks, "w"), indent=1)
    print(len(tasks), "tasks frozen; goals in env:", n_goals)


def stage_rollout(P, args):
    tasks = json.load(open(P.tasks))["tasks"][:args.limit or None]
    todo = [t for t in tasks if not os.path.exists(os.path.join(P.rollout, f"w{t['seed']}.json"))]
    if not todo:
        return
    pool = get_pool(args.workers)
    pol = get_policy(args.dry)
    keys = [str(t["seed"]) for t in todo]
    init = pool.open([(k, str(t["session"]), [], None) for k, t in zip(keys, todo)])
    st = {}
    for k, t in zip(keys, todo):
        obs, adm, _, _ = init[k]
        st[k] = dict(traj_id=f"w{t['seed']}", seed=t["seed"], session=t["session"], game_file=str(t["session"]),
                     goal_asin=t["goal_asin"], task=W.extract_task(obs), initial_obs=obs, obs=obs, adm=adm, mem=[], steps=[],
                     won=False, done=False)
    for turn in range(1, W.MAX_STEPS + 1):
        act = [k for k in keys if not st[k]["done"]]
        if not act:
            break
        prompts = [W.build_prompt(st[k]["task"], st[k]["mem"], st[k]["obs"], st[k]["adm"]) for k in act]
        seeds = [1_000_000 * st[k]["seed"] + turn + OFFSET for k in act]
        outs = pol.sample(prompts, seeds)
        acts = {}
        for k, p, sd, o in zip(act, prompts, seeds, outs):
            a, valid, fmt = common.parse(o[0]["text"])
            acts[k] = a
            st[k]["steps"].append(dict(turn=turn, prompt=p, obs_before=st[k]["obs"], admissible=st[k]["adm"], response=o[0]["text"],
                                       action=a, valid=valid, fmt=fmt, sample_seed=sd))
        res = pool.step(acts)
        for k in act:
            obs, adm, won, done = res[k]
            x = st[k]
            x["mem"].append({"text_obs": x["obs"], "action": acts[k]})
            x["steps"][-1].update(obs_after=obs, won_after=won)
            x["obs"], x["adm"] = obs, adm
            if won or done:
                x["won"], x["done"] = won, True
        print(f"turn {turn} active {len(act)}", flush=True)
    for k in keys:
        x = st[k]
        rec = {kk: v for kk, v in x.items() if kk not in ("obs", "adm", "mem")}
        rec.update(n_turns=len(x["steps"]), provenance=dict(commit=common.git_commit(), argv=sys.argv, vllm=pol.vllm_version))
        common.write_json(os.path.join(P.rollout, f"{x['traj_id']}.json"), rec)
    pool.shutdown()


def load_trajs(P, limit=0):
    fs = sorted(glob.glob(os.path.join(P.rollout, "w*.json")))
    return [json.load(open(f)) for f in fs[:limit or None]]


def stage_bank(P, args):
    import ws_env
    R = importlib.import_module("11_replay")
    W.patch_common()
    b = args.b
    trajs = load_trajs(P, args.limit)
    pool = get_pool(args.workers)
    pol = get_policy(args.dry)
    dps = []
    for tr in trajs:
        trd = dict(traj_id=tr["traj_id"], task_id=tr["game_file"], task_type="webshop", seed=tr["seed"], game_file=tr["game_file"],
                   task=tr["task"], steps=[dict(obs_before=s["obs_before"], action=s["action"], valid=s["valid"]) for s in tr["steps"]])
        for i, s in enumerate(tr["steps"]):
            path = os.path.join(P.bank(b), tr["traj_id"], f"t{s['turn']:02d}.json")
            if os.path.exists(path):
                continue
            d = dict(tr=trd, i=i, turn=s["turn"], path=path, prefix=[x["action"] for x in tr["steps"][:i]],
                     expected=[x["obs_after"] for x in tr["steps"][:i]], factual=s["action"], drafts=[], alts=[], draws=0,
                     prompt=s["prompt"], adm=s["admissible"])
            if b >= 2:
                b1 = json.load(open(os.path.join(P.bank(1), tr["traj_id"], f"t{s['turn']:02d}.json")))
                if not b1["cf_defined"]:
                    continue
                if b >= 3:
                    pool12 = []
                    for bb in (1, 2):
                        r = json.load(open(os.path.join(P.bank(bb), tr["traj_id"], f"t{s['turn']:02d}.json")))
                        pool12 += r["factual_outcomes"] + sum(r["alt_outcomes"], [])
                    if not 0 < sum(pool12) < len(pool12):
                        continue
                d["alts"] = list(b1["alternatives"])
            dps.append(d)
    if b == 1:  # drafts -> alternatives (M0 phase A with WebShop admissibility)
        pending, batch = list(range(len(dps))), 0
        while pending:
            outs = pol.sample([dps[j]["prompt"] for j in pending],
                              [R.seed_of(dps[j]["tr"]["seed"], dps[j]["turn"], 0, batch) + OFFSET for j in pending], n=DRAFT_BATCH)
            still = []
            for j, o in zip(pending, outs):
                d = dps[j]
                for c in o:
                    a, valid, fmt = common.parse(c["text"])
                    ok = ws_env.is_admissible(a, d["adm"])
                    d["drafts"].append(dict(batch=batch, action=a, valid=valid, admissible=ok, text=c["text"]))
                    if valid and ok and a != d["factual"] and a not in d["alts"] and len(d["alts"]) < K_ALT:
                        d["alts"].append(a)
                d["draws"] += len(o)
                if len(d["alts"]) < K_ALT and d["draws"] < MAX_DRAWS:
                    still.append(j)
            pending, batch = still, batch + 1
    conts = []
    for j, d in enumerate(dps):
        d["cf_defined"] = len(d["alts"]) == K_ALT
        d["n_open"] = 0
        roles = [(1, d["factual"])] + ([(2 + k, a) for k, a in enumerate(d["alts"])] if d["cf_defined"] else [])
        for role, a in roles:
            for r in range(N_REP):
                conts.append(R.Cont(f"{j}:{role}:{r}", j, role, r, a))
                d["n_open"] += 1
    print(f"bank {b}: {len(dps)} points, {len(conts)} continuations", flush=True)
    stream = R.DryStream(pol) if args.dry else R.VllmStream(pol)
    fake_alf = types.SimpleNamespace(ALFWORLD_DATA="")
    prov = dict(script="exploratory_E5/e5_run.py bank", b=b, commit=common.git_commit(), argv=sys.argv, seed_offset=OFFSET + (b - 1) * 10 ** 11)
    t0 = time.time()
    a_ = types.SimpleNamespace(max_concurrent=args.max_concurrent)
    n_gen, div = R.run_async(dps, conts, pool, stream, a_, OFFSET + (b - 1) * 10 ** 11, prov, fake_alf, t0) if dps else (0, 0)
    pool.shutdown()
    print(f"bank {b} done: {n_gen} generated steps, refeed divergences {div}, {time.time() - t0:.0f}s")


def stage_peek(P, args):
    trajs = load_trajs(P, args.limit)
    pool = get_pool(args.workers)
    jobs, meta = [], {}
    for tr in trajs:
        for i, s in enumerate(tr["steps"]):
            out = os.path.join(P.peek, tr["traj_id"], f"t{s['turn']:02d}.json")
            b1p = os.path.join(P.bank(1), tr["traj_id"], f"t{s['turn']:02d}.json")
            if os.path.exists(out) or not os.path.exists(b1p):
                continue
            b1 = json.load(open(b1p))
            if not b1["cf_defined"]:
                continue
            acts = [s["action"]] + b1["alternatives"]
            for j, a in enumerate(acts):
                key = f"{tr['traj_id']}:{s['turn']}:{j}"
                jobs.append((key, tr["game_file"], [x["action"] for x in tr["steps"][:i]], [x["obs_after"] for x in tr["steps"][:i]], a))
            meta[(tr["traj_id"], s["turn"])] = (out, acts, s["obs_after"])
    for c0 in range(0, len(jobs), 500):
        chunk = jobs[c0:c0 + 500]
        pool.submit("open_step", chunk, key_of=lambda it: it[0])
        res = {}
        while len(res) < len(chunk):
            for cmd, r in pool.poll(timeout=0.05):
                res.update(r)
        pool.submit("close", [k for k, *_ in chunk], key_of=lambda k: k)
        while not pool.idle_async():
            pool.poll(timeout=0.05)
        for (tid, turn), (out, acts, fac_obs) in list(meta.items()):
            keys = [f"{tid}:{turn}:{j}" for j in range(5)]
            if all(k in res for k in keys):
                peeks = [dict(action=a, obs=res[k][1][0], won=bool(res[k][1][2]), done=bool(res[k][1][3]), refeed_div=res[k][0][3])
                         for a, k in zip(acts, keys)]
                common.write_json(out, dict(traj_id=tid, turn=turn, peeks=peeks, factual_obs_match=peeks[0]["obs"] == fac_obs))
                meta.pop((tid, turn))
    pool.shutdown()


# ----------------------------------------------------------------------------- label and analysis stages (no environment)

def stage_labels(P, args):
    rows, tasks = [], {t["seed"]: t for t in json.load(open(P.tasks))["tasks"]}
    for tr in load_trajs(P):
        for s in tr["steps"]:
            f, a = {}, {}
            for b in range(1, 7):
                p = os.path.join(P.bank(b), tr["traj_id"], f"t{s['turn']:02d}.json")
                if os.path.exists(p):
                    r = json.load(open(p))
                    f[b], a[b] = r["factual_outcomes"], r["alt_outcomes"]
                    if b == 1 and not r["cf_defined"]:
                        break
            if 1 not in f or len(a[1]) != 4 or 2 not in f:
                continue
            lab = point_labels(f, a, 1_000_000 + 1000 * (tr["seed"] - SEED_BASE) + s["turn"])
            lab.update(traj_id=tr["traj_id"], turn=s["turn"], won=bool(tr["won"]))
            rows.append(lab)
    live = [r for r in rows if r["live"]]
    summ = dict(n_cf_defined=len(rows), n_live=len(live), n_het=sum(r["het"] for r in live),
                E_het=float(sum(r["E_het"] or 0 for r in live)), n_credit_pivotal=sum(r["credit_pivotal"] for r in live),
                n_masked=sum(r["masked"] for r in live), dead=len(rows) - len(live))
    json.dump(dict(summary=summ, points=rows), open(P.labels, "w"), indent=1)
    print(summ)


def history_text(tr, i):
    s = tr["steps"]
    return "\n".join([tr["initial_obs"]] + [x["obs_before"] + "\n" + "\n".join(x["admissible"]) for x in s[:i + 1]])


def stage_strata(P, args):
    lab = {(r["traj_id"], r["turn"]): r for r in json.load(open(P.labels))["points"]}
    out = []
    for tr in load_trajs(P):
        for i, s in enumerate(tr["steps"]):
            k = (tr["traj_id"], s["turn"])
            pk = os.path.join(P.peek, tr["traj_id"], f"t{s['turn']:02d}.json")
            if k not in lab or not os.path.exists(pk):
                continue
            peeks = json.load(open(pk))["peeks"]
            z = stratum(s["prompt"], history_text(tr, i), [p["obs"] for p in peeks], tr["goal_asin"])
            out.append(dict(traj_id=k[0], turn=k[1], stratum=z, live=lab[k]["live"], het=lab[k]["het"],
                            goal_mention=[tr["goal_asin"].lower() in p["obs"].lower() for p in peeks]))
    json.dump(out, open(P.strata, "w"), indent=1)
    from collections import Counter
    print(Counter((r["stratum"], r["live"], r["het"]) for r in out))


def stage_signals(P, args):
    ev = importlib.import_module("evalcore")
    lab = {(r["traj_id"], r["turn"]): r for r in json.load(open(P.labels))["points"] if r["live"]}
    st = {(r["traj_id"], r["turn"]): r for r in json.load(open(P.strata))}
    rows = []
    for tr in load_trajs(P):
        n = len(tr["steps"])
        for s in tr["steps"]:
            k = (tr["traj_id"], s["turn"])
            if k not in lab or k not in st:
                continue
            b1 = json.load(open(os.path.join(P.bank(1), tr["traj_id"], f"t{s['turn']:02d}.json")))
            first = [d["action"] for d in b1["drafts"] if d["batch"] == 0 and d["valid"] and d["admissible"]]
            ent = float("nan")
            if first:
                _, c = np.unique(first, return_counts=True)
                ent = float(-(c / c.sum() * np.log(c / c.sum())).sum())
            rows.append(dict(traj=tr["traj_id"], y=lab[k]["het"], ent=ent, w1=float(any(st[k]["goal_mention"][1:])),
                             prog3=min(int(3 * (s["turn"] - 1) / n), 2)))
    y = np.array([r["y"] for r in rows]); g = [r["traj"] for r in rows]; p3 = [r["prog3"] for r in rows]
    out = {}
    for name in ["ent", "w1"]:
        s_ = np.array([r[name] for r in rows])
        ok = np.isfinite(s_)
        sh, _ = ev.shuffle_null(y[ok], s_[ok], np.array(g)[ok], np.array(p3)[ok], n=1000, seed=7)
        out[name] = dict(auroc=float(ev.auroc(y[ok], s_[ok])), shuffle_p975=float(np.nanpercentile(sh, 97.5)), n=int(ok.sum()))
    json.dump(out, open(P.signals, "w"), indent=1)
    print(out)


def display_order(pid):
    seed = int(sha(f"E5-v1:31102026:{pid}".encode())[:16], 16)
    return np.random.default_rng(seed).permutation(5).tolist()


def stage_judge_prepare(P, args):
    tmpl = {k: open(os.path.join(HERE, "prompts", f"judge_{k}.txt")).read() for k in ["base", "log", "feedback"]}
    st = {(r["traj_id"], r["turn"]): r for r in json.load(open(P.strata))}
    rows = []
    for tr in load_trajs(P):
        for s in tr["steps"]:
            z = st.get((tr["traj_id"], s["turn"]))
            if not z or not z["live"] or z["stratum"] not in ("I", "U"):
                continue
            pid = f"{tr['traj_id']}_t{s['turn']:02d}"
            peeks = json.load(open(os.path.join(P.peek, tr["traj_id"], f"t{s['turn']:02d}.json")))["peeks"]
            actions = [p["action"] for p in peeks]
            order = display_order(pid)
            k = int(sha(f"E5-v1:{pid}".encode())[:16], 16)
            shift = 1 + k % 3
            mapping = [0] + [1 + (i - 1 + shift) % 4 for i in range(1, 5)]
            base = (tmpl["base"] + f"\nMarked decision: turn {s['turn']}.\n<decision_time_prompt>\n" + s["prompt"]
                    + "\n</decision_time_prompt>\nCandidates:\n" + json.dumps([actions[j] for j in order]))
            fb = lambda m: [dict(action=actions[i], response=peeks[m[i]]["obs"]) for i in order]
            episode = dict(task=tr["task"], initial_observation=tr["initial_obs"],
                           steps=[dict(turn=x["turn"], response=x["response"], executed_action=x["action"], observation_after=x["obs_after"])
                                  for x in tr["steps"]], final_reward=int(tr["won"]))
            texts = {"S": base, "L": base + "\n" + tmpl["log"] + json.dumps(episode, ensure_ascii=False),
                     "F": base + "\n" + tmpl["feedback"] + json.dumps(fb(list(range(5))), ensure_ascii=False),
                     "Fx": base + "\n" + tmpl["feedback"] + json.dumps(fb(mapping), ensure_ascii=False)}
            for cond, text in texts.items():
                row = dict(request_id=f"{pid}__{cond}", point_id=pid, turn=s["turn"], condition=cond, text=text, actions=actions)
                row["request_sha256"] = sha(json.dumps(row, sort_keys=True, ensure_ascii=False).encode())
                rows.append(row)
    body = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode()
    os.makedirs(P.judge, exist_ok=True)
    open(os.path.join(P.judge, "requests.jsonl.gz"), "wb").write(gzip.compress(body, mtime=0))
    common.write_json(os.path.join(P.judge, "preparation.json"), dict(contexts=len(rows), requests_sha256=sha(body)))
    print(len(rows), "contexts")


def stage_judge_score(P, args):
    J = importlib.import_module("e4_03_judge")
    J.OUTD = P.judge
    J.score(args)


def stage_analyze(P, args):
    lab = {f"{r['traj_id']}_t{r['turn']:02d}": r for r in json.load(open(P.labels))["points"]}
    st = {f"{r['traj_id']}_t{r['turn']:02d}": r for r in json.load(open(P.strata))}
    probs = {}
    for m in ["7b", "32b"]:
        for c in ["S", "L", "F", "Fx"]:
            run = json.load(open(os.path.join(P.judge, f"{m}_{c}", "run.json")))
            assert run["complete"], (m, c)
            for line in open(os.path.join(P.judge, f"{m}_{c}", "scores.jsonl")):
                r = json.loads(line)
                probs[m, c, r["point_id"]] = np.array(r["candidate_prob"])
    pts = {z: sorted(p for p, r in st.items() if r["live"] and r["het"] and r["stratum"] == z and (("7b", "S", p) in probs))
           for z in ("I", "U")}
    gu = lambda m, c, p: float(probs[m, c, p] @ np.array(lab[p]["q12"]) - np.mean(lab[p]["q12"]))
    tr = lambda ps: [p.split("_t")[0] for p in ps]
    table, con = {}, {}
    for z, ps in pts.items():
        for m in ["7b", "32b"]:
            for c in ["S", "L", "F", "Fx"]:
                table[f"{m}/{c}/{z}"] = task_bootstrap([gu(m, c, p) for p in ps], tr(ps))
            for l, r in [("F", "S"), ("F", "Fx"), ("F", "L"), ("L", "S")]:
                con[f"{m}/{l}-{r}/{z}"] = task_bootstrap([gu(m, l, p) - gu(m, r, p) for p in ps], tr(ps))
        con[f"32b-7b/S/{z}"] = task_bootstrap([gu("32b", "S", p) - gu("7b", "S", p) for p in ps], tr(ps))
    allp = pts["I"] + pts["U"]
    sign = np.array([1.0 if p in set(pts["U"]) else -1.0 for p in allp])
    # interaction (U minus I) of the 32B-7B difference: bootstrap over tasks jointly
    d = np.array([gu("32b", "S", p) - gu("7b", "S", p) for p in allp])
    cl, inv = np.unique(tr(allp), return_inverse=True)
    Wt = np.random.default_rng(31102026).multinomial(len(cl), np.full(len(cl), 1 / len(cl)), size=10000).astype(float)
    isU = sign > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = (Wt @ np.bincount(inv, d * isU, len(cl))) / (Wt @ np.bincount(inv, isU.astype(float), len(cl)))
        mi = (Wt @ np.bincount(inv, d * ~isU, len(cl))) / (Wt @ np.bincount(inv, (~isU).astype(float), len(cl)))
    dist = 100 * (mu - mi)
    dist = dist[np.isfinite(dist)]
    con["interaction/S"] = dict(estimate=float(100 * (d[isU].mean() - d[~isU].mean())),
                                ci95=[float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))])
    labsum = json.load(open(P.labels))["summary"]
    sig = json.load(open(P.signals))
    preds, rd = predictions(con, {z: len(v) for z, v in pts.items()})
    preds["W-1"] = bool(labsum["n_het"] > labsum["E_het"] + 2 * np.sqrt(max(labsum["E_het"], 1)))
    preds["W-2"] = bool(sig["ent"]["auroc"] <= sig["ent"]["shuffle_p975"] and sig["w1"]["auroc"] > sig["w1"]["shuffle_p975"])
    out = dict(labels=labsum, signals=sig, n_fork={z: len(v) for z, v in pts.items()}, table=table, contrasts=con,
               predictions=preds, reading=rd)
    json.dump(out, open(os.path.join(HERE, "e5.json"), "w"), indent=1)
    lines = [f"E5 reading: {rd}", f"predictions: {preds}", f"labels: {labsum}", f"signals: {sig}", ""]
    lines += [f"{k:14s} {v['estimate']:+7.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]" for k, v in {**table, **con}.items()]
    open(os.path.join(HERE, "E5_REPORT.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["freeze", "rollout", "bank", "peek", "labels", "strata", "signals", "judge-prepare",
                                      "judge-score", "analyze"])
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_concurrent", type=int, default=400)
    ap.add_argument("--out_root", default=os.path.join(HERE, "data"))
    ap.add_argument("--cond", choices=["S", "L", "F", "Fx"])
    ap.add_argument("--model")
    ap.add_argument("--name", default="32b")
    ap.add_argument("--revision")
    args = ap.parse_args()
    P = paths(args.out_root)
    {"freeze": stage_freeze, "rollout": stage_rollout, "bank": stage_bank, "peek": stage_peek, "labels": stage_labels,
     "strata": stage_strata, "signals": stage_signals, "judge-prepare": stage_judge_prepare, "judge-score": stage_judge_score,
     "analyze": stage_analyze}[args.stage](P, args)


if __name__ == "__main__":
    main()
