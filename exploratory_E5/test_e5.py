"""CPU tests for E5 rules that do not need WebShop or a GPU. Run: python test_e5.py"""
import gzip
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5_run as X  # noqa: E402
import ws_common as W  # noqa: E402
import ws_env  # noqa: E402


def test_stratum():
    a = "B07ABC1234"
    assert X.stratum("results: click[b07abc1234]", "hist", ["x", "item B07ABC1234"], a) == "U"   # in prompt, an arm mentions
    assert X.stratum("search page", "hist", ["x", "page with B07ABC1234", "z"], a) == "I"  # new via an arm
    assert X.stratum("search page", "earlier saw b07abc1234", ["x", "b07abc1234"], a) == "other"
    assert X.stratum("search page", "hist", ["x", "y"], a) == "N"
    assert X.stratum("click[b07abc1234]", "hist", ["x", "y"], a) == "other"


def test_admissible_and_prompt():
    adm = [ws_env.SEARCH_SLOT, "click[back to search]"]
    assert ws_env.is_admissible("search[red shoes]", adm) and ws_env.is_admissible("click[back to search]", adm)
    assert not ws_env.is_admissible("search[]", adm) and not ws_env.is_admissible("click[buy now]", adm)
    assert not ws_env.is_admissible("search[red]", ["click[buy now]"])
    t = "Task {task_description} step {current_step} hist {history_length}:{action_history} obs {current_observation} acts [{available_actions}]"
    p = W.build_prompt("buy shoes", [{"text_obs": "o1", "action": "search[shoes]"}], "o2", adm, template=t)
    assert "step 2" in p and "'search[<your query>]',\n'click[back to search]'" in p and "Action 1: 'search[shoes]'" in p
    out = W.DryPolicy().sample([p], [3], n=5)[0]
    assert all(("search[test query]" in o["text"]) or ("click[back to search]" in o["text"]) for o in out)
    assert W.extract_task("WebShop [SEP] Instruction: [SEP] i need red shoes [SEP] Search") == "i need red shoes"


def test_point_labels():
    dead = {b: [0, 0, 0] for b in range(1, 7)}
    deada = {b: [[0, 0, 0]] * 4 for b in range(1, 7)}
    r = X.point_labels(dead, deada, 1, n_perm=500)
    assert not r["live"] and not r["het"]
    f = {b: [1, 1, 1] for b in range(1, 7)}
    a = {b: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] for b in range(1, 7)}
    r = X.point_labels(f, a, 2, n_perm=2000)
    assert r["live"] and r["het"] and r["credit_pivotal"] and not r["masked"] and r["q12"][0] == 1.0
    a2 = {b: [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]] for b in range(1, 7)}
    f2 = {b: [1, 0, 1] for b in range(1, 7)}
    r = X.point_labels(f2, a2, 3, n_perm=2000)
    assert r["het"] and r["masked"]
    r = X.point_labels(f2, {b: a2[b] for b in (1, 2)}, 3)
    assert r["live"] and r["p_het"] is None                # banks 3-6 missing -> no Fork label


def test_bootstrap_and_predictions():
    b = X.task_bootstrap([.1] * 6, ["a", "a", "b", "b", "c", "c"])
    assert abs(b["estimate"] - 10) < 1e-9 and abs(b["ci95"][0] - 10) < 1e-9
    ok = lambda lo, hi: dict(estimate=(lo + hi) / 2, ci95=[lo, hi])
    c = {"32b-7b/S/U": ok(5, 15), "32b-7b/S/I": ok(-3, 4), "interaction/S": ok(2, 12),
         "7b/F-S/I": ok(10, 30), "32b/F-S/I": ok(30, 50), "32b/F-Fx/I": ok(20, 40), "7b/F-L/I": ok(5, 25), "32b/F-L/I": ok(20, 45)}
    assert X.predictions(c, {"I": 60, "U": 70}) == ({"W-3": True, "W-4": True, "W-5": True, "powered": True}, "R-E5-REPLICATE")
    assert X.predictions(c, {"I": 30, "U": 70})[1] == "R-E5-UNDERPOWERED"
    c["32b/F-L/I"] = ok(-5, 10)
    assert X.predictions(c, {"I": 60, "U": 70})[1] == "R-E5-PARTIAL"


def _synthetic(root):
    """Two trajectories, three steps each, all banks and peeks present; returns the asin."""
    rng = np.random.default_rng(0)
    asin = "B0TESTASIN"
    tasks = []
    for k in range(2):
        seed = X.SEED_BASE + k
        steps = []
        for t in range(1, 4):
            steps.append(dict(turn=t, prompt=f"prompt {k} {t}" + (" click[b0testasin]" if t == 3 else ""), obs_before=f"ob{t}",
                              admissible=[ws_env.SEARCH_SLOT], response=f"<think>r</think><action>search[q{t}]</action>",
                              action=f"search[q{t}]", valid=True, obs_after=f"oa{t}"))
        tr = dict(traj_id=f"w{seed}", seed=seed, session=100 + k, game_file=str(100 + k), goal_asin=asin, task="buy x",
                  initial_obs="WebShop [SEP] Instruction: [SEP] buy x [SEP] Search", steps=steps, won=k == 0)
        tasks.append(dict(seed=seed, session=100 + k, goal_asin=asin))
        os.makedirs(os.path.join(root, "rollout"), exist_ok=True)
        json.dump(tr, open(os.path.join(root, "rollout", f"w{seed}.json"), "w"))
        for t in range(1, 4):
            alts = [f"search[a{t}{j}]" for j in range(4)]
            for b in range(1, 7):
                if t == 1 and b >= 3:
                    continue                                      # step 1 treated as not Live: no banks 3-6
                good = 1 + (t % 4)
                rec = dict(factual_action=f"search[q{t}]", alternatives=alts, cf_defined=True,
                           factual_outcomes=[0, 0, 0] if t > 1 else [0, 0, 0],
                           alt_outcomes=[[1, 1, 1] if (j + 1 == good and t > 1) else [0, 0, 0] for j in range(4)],
                           drafts=[dict(batch=0, action=a, valid=True, admissible=True) for a in alts * 3])
                d = os.path.join(root, f"replay_b{b}", f"w{seed}")
                os.makedirs(d, exist_ok=True)
                json.dump(rec, open(os.path.join(d, f"t{t:02d}.json"), "w"))
            peeks = [dict(action=a, obs=("found " + asin) if (j == 2 and t == 2) else f"page {j}")
                     for j, a in enumerate([f"search[q{t}]"] + alts)]
            os.makedirs(os.path.join(root, "peek", f"w{seed}"), exist_ok=True)
            json.dump(dict(peeks=peeks), open(os.path.join(root, "peek", f"w{seed}", f"t{t:02d}.json"), "w"))
    return asin, tasks


def test_pipeline_offline():
    root = tempfile.mkdtemp()
    try:
        asin, tasks = _synthetic(root)
        P = X.paths(root)
        P.tasks = os.path.join(root, "tasks.json")
        json.dump(dict(tasks=tasks), open(P.tasks, "w"))
        X.stage_labels(P, None)
        lab = json.load(open(P.labels))
        assert lab["summary"]["n_cf_defined"] == 6 and lab["summary"]["n_live"] == 4 and lab["summary"]["n_het"] == 4
        X.stage_strata(P, None)
        st = {(r["traj_id"], r["turn"]): r["stratum"] for r in json.load(open(P.strata))}
        assert st[("w5000", 2)] == "I" and st[("w5000", 3)] == "other"
        X.stage_judge_prepare(P, None)
        rows = [json.loads(l) for l in gzip.open(os.path.join(root, "judge", "requests.jsonl.gz"), "rt")]
        by = {}
        for r in rows:
            by.setdefault(r["point_id"], {})[r["condition"]] = r
        assert set(by) == {"w5000_t02", "w5001_t02"} and all(set(v) == {"S", "L", "F", "Fx"} for v in by.values())
        S, F, Fx = (by["w5000_t02"][c]["text"] for c in ("S", "F", "Fx"))
        assert F.startswith(S) and Fx.startswith(S) and "WebShop task" in S and "15-step" in S
        fa, fx = json.loads(F[F.rindex("\n[") + 1:]), json.loads(Fx[Fx.rindex("\n[") + 1:])
        assert sorted(x["response"] for x in fa) == sorted(x["response"] for x in fx) and fa != fx
        fac = by["w5000_t02"]["F"]["actions"][0]
        assert [x["response"] for x in fa if x["action"] == fac] == [x["response"] for x in fx if x["action"] == fac]
    finally:
        shutil.rmtree(root)


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
    print("all E5 tests passed")


def test_analyze_offline():
    root = tempfile.mkdtemp()
    here_json = os.path.join(X.HERE, "e5.json")
    keep = [p for p in (here_json, os.path.join(X.HERE, "E5_REPORT.txt")) if os.path.exists(p)]
    assert not keep, "refuse to overwrite real E5 outputs"
    try:
        P = X.paths(root)
        rng = np.random.default_rng(5)
        labels, strata = [], []
        os.makedirs(P.judge, exist_ok=True)
        files = {(m, c): open(os.path.join(P.judge, f"{m}_{c}", "scores.jsonl"), "w") if os.makedirs(os.path.join(P.judge, f"{m}_{c}"), exist_ok=True) is None else None
                 for m in ("7b", "32b") for c in ("S", "L", "F", "Fx")}
        for k in range(100):
            tid, turn = f"w{5000 + k // 4}", 1 + k % 4
            z = "I" if k % 2 == 0 else "U"
            best = int(rng.integers(0, 5))
            q = [0.2] * 5
            q[best] = 0.9
            labels.append(dict(traj_id=tid, turn=turn, live=True, het=True, q12=q, E_het=0.1, credit_pivotal=False, masked=True))
            strata.append(dict(traj_id=tid, turn=turn, stratum=z, live=True, het=True, goal_mention=[False] * 5))
            onehot = np.eye(5)[best].tolist()
            unif = [0.2] * 5
            for (m, c), f in files.items():
                p = onehot if (c == "F" and (m == "32b" or z == "I")) or (m == "32b" and c == "S" and z == "U") else unif
                f.write(json.dumps(dict(point_id=f"{tid}_t{turn:02d}", condition=c, candidate_prob=p)) + "\n")
        for (m, c), f in files.items():
            f.close()
            json.dump(dict(complete=True), open(os.path.join(P.judge, f"{m}_{c}", "run.json"), "w"))
        json.dump(dict(summary=dict(n_het=100, E_het=10.0), points=labels), open(P.labels, "w"))
        json.dump(strata, open(P.strata, "w"))
        json.dump(dict(ent=dict(auroc=.5, shuffle_p975=.6), w1=dict(auroc=.7, shuffle_p975=.6)), open(P.signals, "w"))
        X.stage_analyze(P, None)
        out = json.load(open(here_json))
        assert out["reading"] == "R-E5-REPLICATE", out["predictions"]
        assert out["predictions"]["W-1"] and out["predictions"]["W-2"]
        assert abs(out["contrasts"]["32b/F-S/I"]["estimate"] - 56.0) < 1e-6
    finally:
        shutil.rmtree(root)
        for p in (here_json, os.path.join(X.HERE, "E5_REPORT.txt")):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    test_analyze_offline()
    print("ok test_analyze_offline")
