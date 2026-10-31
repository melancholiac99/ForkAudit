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
    assert r["q34"] == [1.0, 0, 0, 0, 0] and r["q56"] == [1.0, 0, 0, 0, 0]
    assert 0 <= r["E_masked"] <= r["E_het"] <= 1 and 0 <= r["E_agg"] <= 1
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


def test_amendment_pure():
    r = X.ratio_bootstrap([1, 0, 1, 1], [1, 1, 1, 1], ["a", "a", "b", "b"])
    assert r["estimate"] == 0.75 and 0.5 <= r["ci95"][0] <= r["ci95"][1] <= 1.0
    assert X.split_gain([.5, .2, .9, .9, 0], [.4, .1, .7, .2, .3]) == .7 - .4          # tie on banks 3-4 -> lowest index
    p = X.policy_marginal(["a", "a", "b", "z"], ["a", "b", "c", "d", "e"])
    assert np.allclose(p, np.array([2.5, 1.5, .5, .5, .5]) / 5.5)
    assert np.allclose(X.rule_dist([0, 1, 0, 1, 0]), [0, .5, 0, .5, 0]) and np.allclose(X.rule_dist([0] * 5), [1, 0, 0, 0, 0])
    assert abs(X.gu(np.eye(5)[1], [0, 1, 0, 0, 0]) - .8) < 1e-12
    assert X.structure_reading(True, True, True, True, False, True) == "R-E5-STRUCT-CONTROL-FAILED"
    assert X.structure_reading(True, True, True, True, True, True) == "R-E5-STRUCT-REPLICATE"
    assert X.structure_reading(True, False, True, True, True, True) == "R-E5-STRUCT-PARTIAL"


def _write_bank1(root, tid, turn, arms, drafts):
    d = os.path.join(root, "replay_b1", tid)
    os.makedirs(d, exist_ok=True)
    json.dump(dict(factual_action=arms[0], alternatives=arms[1:], cf_defined=True, factual_outcomes=[0, 0, 0],
                   alt_outcomes=[[0, 0, 0]] * 4, drafts=[dict(batch=0, action=a, valid=True, admissible=True) for a in drafts]),
              open(os.path.join(d, f"t{turn:02d}.json"), "w"))


def test_signals_offline():
    """Planted signal: hidden feature 0 tracks Live (B1) but not Fork within Live (B2); W1 bit tracks Fork."""
    root = tempfile.mkdtemp()
    try:
        P = X.paths(root)
        rng = np.random.default_rng(11)
        labels, strata, ip = [], [], []
        H = {L: [] for L in X.LAYERS}
        os.makedirs(P.rollout, exist_ok=True)
        for k in range(40):
            tid = f"w{5000 + k}"
            steps = []
            for t in range(1, 9):
                live = bool(rng.random() < .5)
                het = bool(live and rng.random() < .5)
                arms = [f"click[a{t}{j}]" for j in range(5)]
                steps.append(dict(turn=t, prompt="p", action=arms[0]))
                _write_bank1(root, tid, t, arms, list(rng.choice(arms, 32)))
                labels.append(dict(traj_id=tid, turn=t, live=live, het=het))
                strata.append(dict(traj_id=tid, turn=t, stratum="N", live=live, het=het,
                                   goal_mention=[False] + [bool(het and j == 1) or bool(rng.random() < .1) for j in range(1, 5)]))
                ip.append(dict(traj_id=tid, turn=t, conf=float(rng.normal()), n_tokens=3, row=len(ip)))
                for L in X.LAYERS:
                    h = rng.normal(size=16)
                    h[0] += 2.5 * live
                    H[L].append(h.astype(np.float16))
            json.dump(dict(traj_id=tid, steps=steps), open(os.path.join(P.rollout, f"{tid}.json"), "w"))
        json.dump(dict(points=labels), open(P.labels, "w"))
        json.dump(strata, open(P.strata, "w"))
        os.makedirs(os.path.join(root, "introspect"))
        json.dump(dict(points=ip), open(os.path.join(root, "introspect", "points.json"), "w"))
        for L in X.LAYERS:
            np.save(os.path.join(root, "introspect", f"hidden_L{L}.npy"), np.stack(H[L]))
        X.stage_signals(P, None)
        sig = json.load(open(P.signals))
        assert sig["B1"]["C_L21"]["above_shuffle"] and sig["B1"]["C_L21"]["auroc"] > .8
        assert not sig["B2"]["C_L21"]["above_shuffle"] and not sig["B2"]["neg_conf"]["above_shuffle"]
        assert sig["B2"]["w1_goal_mention"]["above_shuffle"] and "w1_goal_mention" not in sig["B1"]
        assert sig["B2"]["C_L14"]["role"] == "descriptive" and sig["B2"]["C_L21"]["role"] == "registered"
    finally:
        shutil.rmtree(root)


def test_analyze_offline():
    root = tempfile.mkdtemp()
    here_json = os.path.join(X.HERE, "e5.json")
    assert not [p for p in (here_json, os.path.join(X.HERE, "E5_REPORT.txt")) if os.path.exists(p)], "refuse to overwrite real E5 outputs"
    try:
        P = X.paths(root)
        rng = np.random.default_rng(5)
        labels, strata = [], []
        for m in ("7b", "32b"):
            for c in ("S", "L", "F", "Fx"):
                os.makedirs(os.path.join(P.judge, f"{m}_{c}"), exist_ok=True)
        files = {(m, c): open(os.path.join(P.judge, f"{m}_{c}", "scores.jsonl"), "w") for m in ("7b", "32b") for c in ("S", "L", "F", "Fx")}
        for k in range(120):
            tid, turn = f"w{5000 + k // 4}", 1 + k % 4
            z = "I" if k % 2 == 0 else "U"
            best = int(rng.integers(1, 5))
            q = [0.2] * 5
            q[best] = 0.9
            masked = k % 3 != 0
            labels.append(dict(traj_id=tid, turn=turn, live=True, het=True, q12=q, q34=q, q56=q, p_het=0.01, E_het=0.1,
                               E_agg=0.1, E_masked=0.05, credit_pivotal=not masked, masked=masked))
            gm = [j == best for j in range(5)]
            strata.append(dict(traj_id=tid, turn=turn, stratum=z, live=True, het=True, goal_mention=gm))
            arms = [f"click[a{k}{j}]" for j in range(5)]
            _write_bank1(root, tid, turn, arms, [arms[0]] * 30 + [arms[best]] * 2)
            onehot, unif = np.eye(5)[best].tolist(), [0.2] * 5
            for (m, c), f in files.items():
                pr = onehot if (c == "F" and (m == "32b" or z == "I")) or (m == "32b" and c == "S" and z == "U") else unif
                f.write(json.dumps(dict(point_id=f"{tid}_t{turn:02d}", condition=c, candidate_prob=pr)) + "\n")
        for (m, c), f in files.items():
            f.close()
            json.dump(dict(complete=True), open(os.path.join(P.judge, f"{m}_{c}", "run.json"), "w"))
        for k in range(30):                                  # dead points
            labels.append(dict(traj_id=f"w{5000 + k}", turn=9, live=False, het=False, q12=[0] * 5, q34=None, q56=None, p_het=None,
                               E_het=None, E_agg=None, E_masked=None, credit_pivotal=False, masked=False))
        summ = dict(n_het=120, E_het=12.0, E_agg=12.0, E_masked=6.0)
        json.dump(dict(summary=summ, points=labels), open(P.labels, "w"))
        json.dump(strata, open(P.strata, "w"))
        e = lambda a, p, above: dict(auroc=a, ci95=[a - .05, a + .05], shuffle_p975=p, above_shuffle=above, n=100, n_traj=30, role="registered")
        json.dump(dict(B1=dict(neg_conf=e(.55, .6, False), entropy_sampled=e(.5, .6, False), C_L21=e(.75, .6, True)),
                       B2=dict(neg_conf=e(.52, .6, False), entropy_sampled=e(.5, .6, False), C_L21=e(.55, .6, False),
                               w1_goal_mention=e(.7, .6, True))), open(P.signals, "w"))
        X.stage_analyze(P, None)
        out = json.load(open(here_json))
        assert out["reading"] == "R-E5-REPLICATE", out["predictions"]
        assert out["structure_reading"] == "R-E5-STRUCT-REPLICATE", out["predictions"]
        pr = out["predictions"]
        assert pr["W-6a"] and pr["W-6b"] and pr["W-6c"] and pr["W-0b"] and pr["W-0c"]
        assert abs(out["contrasts"]["32b/F-S/I"]["estimate"] - 56.0) < 1e-6
        assert abs(out["structure"]["masked_share_of_forks"]["estimate"] - 80 / 120) < 1e-9
        assert abs(out["structure"]["proportions_pct"]["dead"]["estimate"] - 100 * 30 / 150) < 1e-9
    finally:
        shutil.rmtree(root)
        for p in (here_json, os.path.join(X.HERE, "E5_REPORT.txt")):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
    print("all E5 tests passed")
