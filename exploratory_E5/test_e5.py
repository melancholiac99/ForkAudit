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
import search_env as SE  # noqa: E402
import sokoban_env as SK  # noqa: E402
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
                  meta=dict(seed=seed, session=100 + k, goal_asin=asin),
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
    X.set_env("webshop")
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
                   alt_outcomes=[[0, 0, 0]] * (len(arms) - 1), drafts=[dict(batch=0, action=a, valid=True, admissible=True) for a in drafts]),
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
    X.set_env("webshop")
    root = tempfile.mkdtemp()
    here_json, here_rep = os.path.join(X.HERE, "e5_webshop.json"), os.path.join(X.HERE, "E5_webshop_REPORT.txt")
    assert not [p for p in (here_json, here_rep) if os.path.exists(p)], "refuse to overwrite real E5 outputs"
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
            strata.append(dict(traj_id=tid, turn=turn, stratum=z, live=True, het=True, goal_mention=gm, reveal=gm, rule=gm))
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
        for p in (here_json, here_rep):
            if os.path.exists(p):
                os.remove(p)




# ----------------------------------------------------------------------------- amendment 2: K arms, absence, cross, pilot

def test_point_labels_k4():
    f = {b: [1, 1, 1] for b in range(1, 7)}
    a = {b: [[0, 0, 0], [0, 0, 0], [0, 0, 0]] for b in range(1, 7)}
    r = X.point_labels(f, a, 2, n_perm=2000)
    assert r["n_arms"] == 4 and r["live"] and r["het"] and r["credit_pivotal"] and len(r["q12"]) == 4
    # agg_flag with K = 5 equals the amendment-1 rule (factual 12 vs pooled 48)
    R = __import__("importlib").import_module("15_regimes")
    c = np.array([[9, 2, 3, 1, 4], [6, 6, 6, 6, 6]])
    lo, hi = R.newcombe(c[:, 0], 12, c[:, 1:].sum(1), 48)
    assert (X.agg_flag(c) == ((lo > 0) | (hi < 0))).all()


def test_absence_cross_pilot_pure():
    ok = lambda lo, hi: dict(estimate=(lo + hi) / 2, ci95=[lo, hi])
    c = {"32b/F-S/U": ok(-3, 6), "32b-7b/S/U": ok(2, 9)}
    assert X.absence_predictions(c, 50) == ({"A-1": True, "A-2": True, "powered": True}, "R-ABS-CONFIRMED")
    assert X.absence_predictions(c, 30)[1] == "R-ABS-UNDERPOWERED"
    c["32b/F-S/U"] = ok(2, 14)
    assert X.absence_predictions(c, 50)[1] == "R-ABS-PARTIAL"
    rng = np.random.default_rng(1)
    qa = [f"q{i % 40}" for i in range(80)]
    d = X.diff_bootstrap(rng.normal(.3, .1, 80), qa, rng.normal(0, .1, 60), qa[:60], paired_clusters=True)
    assert 20 < d["ci95"][0] < d["estimate"] < d["ci95"][1] < 40
    d2 = X.diff_bootstrap(rng.normal(.3, .1, 80), qa, rng.normal(0, .1, 60), [f"r{i % 30}" for i in range(60)])
    assert 20 < d2["ci95"][0] < d2["ci95"][1] < 40
    summ = dict(success=.4, live_frac=.3, n_live_I=10, n_live_U=13, n_live=40, refeed_div=0, factual_match_frac=1.0)  # 200 / 260 projected
    ch, passed = X.pilot_gate(summ, True, 1000, 50)
    assert passed and set(ch) == {"success", "live_frac", "determinism", "projected_I", "projected_U"}
    ch, passed = X.pilot_gate(dict(summ, n_live_I=5), True, 1000, 50)
    assert not passed and not ch["projected_I"]
    assert not X.pilot_gate(dict(summ, success=.9), False, 150, 50)[1]
    assert X.pilot_gate(dict(summ, n_live=100), False, 150, 50)[1] and not X.pilot_gate(summ, False, 150, 50)[1]
    assert X.derangement("p", 4)[0] == 0 and sorted(X.derangement("p", 4)) == [0, 1, 2, 3]
    assert all(X.derangement(f"p{i}", 4)[j] != j for i in range(20) for j in (1, 2, 3))
    assert X.display_order("w5000_t02", 5) == X.display_order("w5000_t02")       # amendment-1 order unchanged


# ----------------------------------------------------------------------------- search environment

def _fake_retriever(query):
    n = sum(map(ord, query)) % 7
    return [dict(contents=f'"Title {query}"\nabout {query}'), dict(contents=f'"Other {n}"\nfiller {n}'),
            dict(contents=f'"Other {n + 1}"\nfiller')]


def test_search_projection_and_game():
    P = SE.parse
    assert P("<think>t</think><search> who is x </search>") == ("<search>who is x</search>", True, "search")
    assert P("<think>t</think><answer>Paris</answer>") == ("<answer>Paris</answer>", True, "answer")
    assert P("<search>a</search><answer>b</answer>") == ("<search>a</search>", False, "search")     # both tags: invalid
    assert P("<answer>b</answer> then <search>a</search>") == ("<search>a</search>", False, "search")
    assert P("<search>a</search><search>b</search>") == ("<search>a</search>", False, "search")
    assert P("nothing here") == ("", False, "none")
    assert P("<ANSWER>Rome</ANSWER>")[0] == "<answer>Rome</answer>"
    assert SE.display("<search>q</search>") == "search[q]" and SE.display("<answer>a b</answer>") == "answer[a b]"
    SE._RETRIEVE = lambda q: [dict(contents=f'"Title {q}"\ntext about {q}'), dict(contents='"Paris"\nCapital.')]
    cache0, SE.CACHE_PATH = SE.CACHE_PATH, None
    try:
        q = dict(qid="x1", question="Where?", answers=["The Paris"], context=[["Paris", [" Capital", " of France."]]])
        g = SE.Game(q, "open")
        obs, adm, won, done = g.step("<search>q1</search>", {})
        assert obs.startswith("<information>{") and obs.endswith("}</information>") and not done and adm == SE.ADM
        assert json.loads(obs[len("<information>"):-len("</information>")])["result"] == 'Doc 1: "Title q1"\ntext about q1\nDoc 2: "Paris"\nCapital.\n'
        assert SE.titles_in(obs) == {"title q1", "paris"}
        assert g.step("", {}) == ("", SE.ADM, False, False)                      # no block: wasted turn, empty observation
        assert g.step("<answer>paris</answer>", {}) == ("", [], True, True)       # EM after normalisation
        g = SE.Game(q, "open")
        for k in range(3):
            assert not g.step(f"<search>s{k}</search>", {})[3]
        assert g.step("<search>s3</search>", {}) == ("", [], False, True)          # 4th action ends the episode
        c = SE.Game(q, "ctx").initial_obs()
        assert c.startswith("Where?\nRetrieved documents: <information>") and SE.titles_in(c) == {"paris"}
        p1 = SE.build_prompt("Where?", [], "Where?", SE.ADM)
        assert "Your question: Where?" in p1 and "Prior to this step" not in p1 and not p1.startswith("\n")
        mem = [{"text_obs": "Where?", "action": "<search>q1</search>"}, {"text_obs": "INFO1", "action": ""}]
        p3 = SE.build_prompt("Where?", mem, "", SE.ADM)
        assert "already taken 2 step(s)" in p3 and "Step 1:<search>q1</search> INFO1\n\nStep 2: \n" in p3
        z, rev = SE.stratum("Your question: x", "x", [obs, "", obs.replace("Paris", "Rome")], ["Paris", "Lyon"])
        assert z == "I" and rev == [True, False, False]
        assert SE.stratum(obs, obs, [obs], ["Paris"]) == ("U", [False])
        assert SE.stratum("x", "x", ["", ""], ["Paris"]) == ("N", [False, False])
        assert SE.stratum("x", obs, [obs], ["Paris"])[0] == "other"
    finally:
        SE._RETRIEVE, SE.CACHE_PATH = None, cache0


def _search_fixture(tmp, n=70):
    dev, wiki = [], []
    for i in range(n):
        ans = f"a{i % 3}" if i % 5 else "yes"
        dev.append(dict(_id=f"h{i}", question=f"question {i}?", answer=ans, type="bridge", level="hard",
                        supporting_facts=[["Title q1", 0], ["Title q2", 0]],
                        context=[["Title q1", [" s1."]], ["Title q2", [" s2."]], [f"Distractor {i}", [" d."]]]))
    for t in ["Title q1", "Title q2"] + [f"Other {k}" for k in range(9)]:
        wiki.append(json.dumps(dict(id=t, contents=f'"{t}"\ntext')))
    json.dump(dev, open(os.path.join(tmp, "dev.json"), "w"))
    open(os.path.join(tmp, "wiki.jsonl"), "w").write("\n".join(wiki) + "\n")
    os.environ["HOTPOT_DEV"], os.environ["WIKI18"] = os.path.join(tmp, "dev.json"), os.path.join(tmp, "wiki.jsonl")


def _run(env, tmp, root, stages, **kw):
    E = X.set_env(env)
    E.tasks_path = lambda pilot=False, level=None: os.path.join(tmp, f"tasks_{E.task_set}{'_pilot' if pilot else ''}.json")
    if env == "sokoban":
        E.tasks_path = lambda pilot=False, level=None: os.path.join(tmp, f"tasks_sokoban{'_pilot_' + level if pilot else ''}.json")
    if env.startswith("search"):
        E.N_MAIN = 20
    args = types.SimpleNamespace(limit=0, workers=2, dry=True, max_concurrent=64, b=1, level=None, pilot=True, n_main=0,
                                 tasks=None, out_root=root)
    for k, v in kw.items():
        setattr(args, k, v)
    P = X.paths(root, E.tasks_path(pilot=args.pilot, level=args.level))
    for st in stages:
        if st == "freeze":
            E.freeze(args)
        elif st.startswith("bank"):
            args.b = int(st[4:])
            X.stage_bank(P, args)
        else:
            getattr(X, "stage_" + st.replace("-", "_"))(P, args)
    return P


def test_search_pipeline_dry():
    """freeze -> rollout -> banks 1-6 -> peek -> labels -> strata -> pilot -> judge contexts, for search and search_ctx,
    with a fake retriever and the dry policy (exact replay: zero refeed divergences, all factual peeks match)."""
    tmp = tempfile.mkdtemp()
    cache0 = SE.CACHE_PATH
    SE._RETRIEVE, SE.CACHE_PATH = _fake_retriever, os.path.join(tmp, "cache.sqlite")
    try:
        _search_fixture(tmp)
        stages = ["freeze", "rollout", "bank1", "bank2", "peek", "labels", "strata", "pilot", "bank3", "bank4", "bank5", "bank6",
                  "labels", "judge-prepare"]
        P = _run("search", tmp, os.path.join(tmp, "s"), stages)
        frozen = json.load(open(os.path.join(tmp, "tasks_search.json")))
        assert frozen["n_eligible"] == 56 and len(frozen["tasks"]) == 20 and frozen["tasks"][0]["seed"] == 6000
        assert all(t["answers"][0] != "yes" for t in frozen["tasks"])
        pilot = json.load(open(P.pilot))
        assert pilot["summary"]["refeed_div"] == 0 and pilot["summary"]["factual_match_frac"] == 1.0 and pilot["n_main"] == 20
        lab = json.load(open(P.labels))
        assert lab["summary"]["n_cf_defined"] > 0 and lab["summary"]["n_live_without_banks3to6"] == 0
        st = json.load(open(P.strata))
        assert {"I", "U"} & {r["stratum"] for r in st} and all(len(r["rule"]) == 5 for r in st)
        tr = json.load(open(sorted(glob_(P.rollout))[0]))
        assert "context" not in tr["meta"] and tr["task"] == tr["initial_obs"] and tr["meta"]["supporting_titles"]
        rows = [json.loads(l) for l in gzip.open(os.path.join(P.judge, "requests.jsonl.gz"), "rt")]
        assert rows and all(len(r["actions"]) == 5 and all(a.startswith(("search[", "answer[")) for a in r["actions"]) for r in rows)
        assert "answering its question correctly" in rows[0]["text"]
        F = [r for r in rows if r["condition"] == "F"]
        assert any("(empty observation)" in r["text"] for r in F) and any("five labeled" in r["text"] for r in F)
        # analysis path on uniform judges and placeholder signals (exercises W-6 with the reveal rule and the arm mix)
        here = [os.path.join(X.HERE, f) for f in ("e5_search.json", "E5_search_REPORT.txt")]
        assert not [h for h in here if os.path.exists(h)], "refuse to overwrite real E5 outputs"
        for m in ("7b", "32b"):
            for c in ("S", "L", "F", "Fx"):
                d = os.path.join(P.judge, f"{m}_{c}")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "scores.jsonl"), "w") as f:
                    for r in rows:
                        if r["condition"] == c:
                            f.write(json.dumps(dict(point_id=r["point_id"], condition=c, candidate_prob=[.2] * 5)) + "\n")
                json.dump(dict(complete=True), open(os.path.join(d, "run.json"), "w"))
        e = dict(auroc=.5, ci95=[.45, .55], shuffle_p975=.6, above_shuffle=False, n=10, n_traj=5, role="registered")
        json.dump(dict(B1=dict(neg_conf=e, entropy_sampled=e, C_L21=e), B2=dict(neg_conf=e, entropy_sampled=e, C_L21=e)),
                  open(P.signals, "w"))
        try:
            X.stage_analyze(P, types.SimpleNamespace())
            out = json.load(open(here[0]))
            assert out["reading"] == "R-E5-UNDERPOWERED" and out["structure_reading"] == "R-E5-STRUCT-CONTROL-FAILED"
            assert out["w6"] is None or abs(out["w6"]["policy"]["estimate"]) < 100
            assert all(0 <= v["answer_arm_share"] <= 1 for v in out["arms_at_forks"].values())
        finally:
            for h in here:
                if os.path.exists(h):
                    os.remove(h)
        P = _run("search_ctx", tmp, os.path.join(tmp, "c"), ["rollout", "bank1", "bank2", "peek", "labels", "strata", "pilot"])
        st = json.load(open(P.strata))
        assert st and all(r["stratum"] == "U" for r in st)                    # C-0 holds by construction
        assert json.load(open(P.pilot))["summary"]["refeed_div"] == 0
        tr = json.load(open(sorted(glob_(P.rollout))[0]))
        assert "Retrieved documents: <information>" in tr["steps"][0]["prompt"]
        import sqlite3
        n = sqlite3.connect(SE.CACHE_PATH).execute("SELECT COUNT(*) FROM c").fetchone()[0]
        assert 0 < n <= 6 + 1                                   # dry queries 'q0'..'q5' (one cache for both variants)
    finally:
        SE._RETRIEVE, SE.CACHE_PATH = None, cache0
        shutil.rmtree(tmp)


def glob_(d):
    import glob
    return glob.glob(os.path.join(d, "*.json"))


# ----------------------------------------------------------------------------- Sokoban

def _corridor():
    # row 1: # P X O #   (player, box, target in a row)
    fixed = [[0] * 6, [0, 1, 1, 2, 1, 0], [0, 1, 1, 1, 1, 0], [0] * 6]
    state = [[0] * 6, [0, 5, 4, 2, 1, 0], [0, 1, 1, 1, 1, 0], [0] * 6]
    return fixed, state


def test_sokoban_rules():
    fixed, state = _corridor()
    r = SK.Room(fixed, state)
    assert r.render().split("\n")[1] == " # \t P \t X \t O \t _ \t # \t" and not r.solved()
    assert r.act(1) is False and r.player.tolist() == [1, 1]                  # up: wall, no move
    assert r.act(4) and r.solved() and r.render().split("\n")[1] == " # \t _ \t P \t √ \t _ \t # \t"
    assert r.act(4) and not r.solved()                                        # pushes the box off its target
    assert r.act(4) is False and r.player.tolist() == [1, 3]                  # box against the wall: no push, no move
    r = SK.Room(fixed, state)
    assert SK.solve_distances(r, 15) == 1 and SK.optimal_arms(r, ["Right", "Up", "Down", "Left"], 15) == [True, False, False, False]
    assert SK.optimal_arms(r, ["Right"], 0) == [False]
    r.act(2)                                                                    # down
    assert r.render().split("\n")[2] == " # \t P \t _ \t _ \t _ \t # \t" and SK.solve_distances(r, 15) == 2
    g = SK.Game(dict(fixed=fixed, state=state), dict(max_steps=2))
    assert g.step("Still")[3] is False and g.step("Right")[2:] == (True, True)
    g = SK.Game(dict(fixed=fixed, state=state), dict(max_steps=2))
    g.step("Up")
    assert g.step("Left")[2:] == (False, True)                                 # budget exhausted
    assert SK.parse("<think>x</think><action>Up</action>") == ("Up", True, "xml")
    assert SK.parse("<action>move LEFT</action>") == ("Left", False, "unparsed")         # no think tags: invalid, executed
    assert SK.parse("<think></think><action>still</action>") == ("Still", True, "xml")
    assert SK.parse("<think></think><action>left or right</action>")[0] == "Left"
    assert SK.parse("I go up")[:2] == ("Still", False)
    p1 = SK.build_prompt("", [], "GRID", SK.ACTIONS)
    p2 = SK.build_prompt("", [{"text_obs": "G1", "action": "Up"}, {"text_obs": "G2", "action": "Still"}, {"text_obs": "G3", "action": "Left"}],
                         "GRID", SK.ACTIONS)
    assert "Your current observation is:\nGRID" in p1 and "Prior to this step" not in p1
    assert "already taken 3 step(s). Below are the most recent 2 observations" in p2
    assert "[Observation 2: 'G2', Action 2: 'Still']\n[Observation 3: 'G3', Action 3: 'Left']" in p2 and "step 4" in p2
    lv = SK.LADDER[0]
    f1, s1, g1 = SK.generate(lv, 7000)
    assert (f1, s1, g1) == SK.generate(lv, 7000) and len(f1) == 6 and sum(v == 5 for row in s1 for v in row) == 1
    import hashlib                                                             # PREREG_E5 §13.8 gate S0'
    h = hashlib.sha256(json.dumps([SK.generate(lv, 7500 + n) for n in range(5)]).encode()).hexdigest()
    assert h == "71c6148a8ef71c0bd95588376000b467417627b053c0a9bc78c075ebc12649b7", h


def test_sokoban_pipeline_dry():
    tmp = tempfile.mkdtemp()
    try:
        P = _run("sokoban", tmp, os.path.join(tmp, "k"), ["freeze", "rollout", "bank1", "bank2", "peek", "labels", "strata", "pilot"],
                 level="L3", limit=6)
        frozen = json.load(open(os.path.join(tmp, "tasks_sokoban_pilot_L3.json")))
        assert len(frozen["tasks"]) == 50 and frozen["tasks"][0]["seed"] == 7700 and frozen["level"]["name"] == "L3"
        pilot = json.load(open(P.pilot))
        assert pilot["summary"]["refeed_div"] == 0 and pilot["summary"]["factual_match_frac"] == 1.0 and pilot["n_main"] == 150
        b1 = [json.load(open(f)) for f in __import__("glob").glob(os.path.join(P.bank(1), "*", "*.json"))]
        assert b1 and all(len(r["alternatives"]) == 3 and r["factual_action"] not in r["alternatives"] for r in b1 if r["cf_defined"])
        assert all(len(r["drafts"]) == X.DRAFT_BATCH for r in b1)
        st = json.load(open(P.strata))
        assert st and all(r["stratum"] == "U" and len(r["rule"]) == 4 for r in st)
        assert any(any(r["rule"]) for r in st)
    finally:
        shutil.rmtree(tmp)


def test_analyze_absence_offline():
    """Sokoban: 32B picks the best arm with or without feedback (A-1, A-2 hold); the planner rule picks it (K-0)."""
    X.set_env("sokoban")
    root = tempfile.mkdtemp()
    here_json, here_rep = os.path.join(X.HERE, "e5_sokoban.json"), os.path.join(X.HERE, "E5_sokoban_REPORT.txt")
    assert not [p for p in (here_json, here_rep) if os.path.exists(p)], "refuse to overwrite real E5 outputs"
    try:
        P = X.paths(root, os.path.join(root, "tasks.json"))
        json.dump(dict(level=SK.LADDER[0], tasks=[]), open(P.tasks, "w"))
        rng = np.random.default_rng(6)
        labels, strata = [], []
        files = {}
        for m in ("7b", "32b"):
            for c in ("S", "L", "F", "Fx"):
                os.makedirs(os.path.join(P.judge, f"{m}_{c}"), exist_ok=True)
                files[m, c] = open(os.path.join(P.judge, f"{m}_{c}", "scores.jsonl"), "w")
        for k in range(80):
            tid, turn = f"k{7000 + k // 4}", 1 + k % 4
            best = int(rng.integers(1, 4))
            q = [0.2] * 4
            q[best] = 0.9
            labels.append(dict(traj_id=tid, turn=turn, live=True, het=True, q12=q, q34=q, q56=q, p_het=0.01, E_het=0.1, E_agg=0.1,
                               E_masked=0.05, credit_pivotal=True, masked=False))
            rule = [j == best for j in range(4)]
            strata.append(dict(traj_id=tid, turn=turn, stratum="U", live=True, het=True, reveal=[False] * 4, rule=rule))
            arms = ["Up", "Down", "Left", "Right"]
            _write_bank1(root, tid, turn, arms, [arms[0]] * 32)
            for (m, c), f in files.items():
                pr = np.eye(4)[best].tolist() if m == "32b" else [0.25] * 4
                f.write(json.dumps(dict(point_id=f"{tid}_t{turn:02d}", condition=c, candidate_prob=pr)) + "\n")
        for (m, c), f in files.items():
            f.close()
            json.dump(dict(complete=True), open(os.path.join(P.judge, f"{m}_{c}", "run.json"), "w"))
        json.dump(dict(summary=dict(n_het=80, E_het=8.0, E_agg=8.0, E_masked=1.0), points=labels), open(P.labels, "w"))
        json.dump(strata, open(P.strata, "w"))
        e = lambda a, above: dict(auroc=a, ci95=[a - .05, a + .05], shuffle_p975=.6, above_shuffle=above, n=80, n_traj=20, role="registered")
        json.dump(dict(B1=dict(neg_conf=e(.5, False), entropy_sampled=e(.5, False), C_L21=e(.7, True)),
                       B2=dict(neg_conf=e(.5, False), entropy_sampled=e(.5, False), C_L21=e(.5, False))), open(P.signals, "w"))
        X.stage_analyze(P, None)
        out = json.load(open(here_json))
        assert out["reading"] == "R-ABS-CONFIRMED" and out["checks"]["K-0"]["holds"], out["predictions"]
        assert out["structure_reading"] is None and out["structure_descriptive"]["W-2c"]
        for s in strata:
            s["rule"] = [False] * 4
        json.dump(strata, open(P.strata, "w"))
        X.stage_analyze(P, None)
        assert json.load(open(here_json))["reading"] == "R-ABS-CONTROL-FAILED"
    finally:
        shutil.rmtree(root)
        for p in (here_json, here_rep):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
    print("all E5 tests passed")
