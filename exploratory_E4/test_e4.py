"""CPU unit tests for the frozen E4 rules (no GPU, ALFWorld or torch needed). Run: python test_e4.py"""
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4_common as E  # noqa: E402


def test_target_and_search_phase():
    assert E.target_word("clean some knife and put it in diningtable.") == "knife"
    assert E.target_word("put a cool winebottle in garbagecan.") == "winebottle"
    assert E.target_word("examine the box with the desklamp.") == "box"
    assert E.search_phase("you see a fork 1 and a spoon 2", "knife")
    assert not E.search_phase("On the countertop 1, you see a knife 3.", "knife")
    assert not E.search_phase("anything", None)


def test_pair_rules():
    assert E.pair_judge([.5, .1, .1, .2, .1]) is None
    assert E.pair_judge([.1, .1, .6, .1, .1]) == (2, 0)
    obs = ["nothing", "you see a knife 1", "nothing", "a knife 2 here", "x"]
    assert E.pair_rule(obs, "knife") == (1, 0)
    assert E.pair_rule(["a knife 1"] + obs[1:], "knife") is None
    assert E.pair_rule(["n"] * 5, "knife") is None
    r1, r2 = E.pair_random("s1_t03", True), E.pair_random("s1_t03", True)
    assert r1 == r2 and 1 <= r1[0] <= 4 and E.pair_random("s1_t03", False) is None
    assert E.pair_explore("p", ["E_new"] * 5) is None
    assert E.pair_explore("p", ["R_seen", "Other", "E_new", "Other", "Other"]) == (2, 0)
    assert E.pair_explore("p", ["R_seen", "Other", "Other", "Other", "Other"]) is None
    assert E.pair_replay([0, 0, 1 / 3, 0, 0]) == (2, 0)
    assert E.pair_replay([1 / 3, 1 / 3, 1 / 3, 1 / 3, 0]) == (0, 4)
    assert E.pair_replay([.5, .5, .5, .5, .5]) is None
    assert E.pair_replay([1 / 3, 0, 0, 0, 1 / 3 - 1e-9]) == (0, 1)


def test_cost_and_subsets():
    assert E.matched_count(1000.0, 10000.0, 500) == 50
    assert E.matched_count(1e9, 10.0, 500) == 500
    ids = [f"p{i}" for i in range(100)]
    ent = {p: float(i % 7) for i, p in enumerate(ids)}
    ent["p3"] = float("nan")
    b, c = E.matched_subsets(ids, ent, 10)
    assert len(b) == len(c) == 10 and b == E.matched_subsets(ids, ent, 10)[0]
    assert all(ent[p] == 6.0 for p in c) and "p3" not in c


def test_stats_and_reading():
    a = np.random.default_rng(0).integers(0, 2, size=(134, 9))
    r = E.game_bootstrap_diff(a, a)
    assert r["diff"] == 0 and r["ci95"] == [0.0, 0.0]
    b = np.zeros((134, 3))
    r = E.game_bootstrap_diff(np.ones((134, 9)), b)
    assert r["diff"] == 100 and r["ci95"][0] == 100
    c = E.cluster_bootstrap_mean([.1, .1, .1, .1], ["a", "a", "b", "b"])
    assert abs(c["mean"] - 10) < 1e-9 and abs(c["ci95"][0] - 10) < 1e-9
    assert E.reading(False, True, True) == "R-L-NULL"
    assert E.reading(True, True, False) == "R-L-HEURISTIC"
    assert E.reading(True, False, True) == "R-L-COSTLY"
    assert E.reading(True, True, True) == "R-L-WIN"


def test_draw_cap_lr():
    assert E.quotas(400, 6) == [67, 67, 67, 67, 66, 66] and E.quotas(40, 6) == [7, 7, 7, 7, 6, 6]
    games = sorted([(f"g{t}_{i}", f"type{t}") for t in range(6) for i in range(100)])
    excl = {g for g, _ in games[::10]}
    d = E.stratified_draw(games, 400, excl, 1)
    assert len(d) == 400 and len({g for g, _ in d}) == 400 and not ({g for g, _ in d} & excl)
    assert sorted(sum(1 for _, t in d if t == f"type{k}") for k in range(6)) == [66, 66, 67, 67, 67, 67]
    assert d == E.stratified_draw(games, 400, excl, 1)
    rows = list(range(2500))
    cr = E.cap_rows("A", rows)
    assert len(cr) == E.CAP and cr == sorted(cr) and cr == E.cap_rows("A", rows)
    assert E.select_lr({5e-6: .40, 2e-5: .45, 5e-5: .45}) == 2e-5


def test_pairs_build():
    import e4_05_pairs as P
    pts = []
    rng = np.random.default_rng(1)
    for i in range(60):
        hist = [dict(executed_action="go to drawer 1", observation_after="You arrive at drawer 1. On the drawer 1, you see nothing.")]
        pts.append(dict(point_id=f"s{1000 + i}_t05", prompt=f"prompt {i}", factual="go to drawer 1",
                        alternatives=["go to shelf 1", "go to drawer 2", "examine drawer 1", "inventory"],
                        peeks=[dict(obs="x")] * 2 + [dict(obs="a knife 1")] + [dict(obs="y")] * 2, target_word="knife",
                        history=hist, entropy_sampled=float(i % 5), arm_responses=[f"r{i}_{j}" for j in range(5)]))
    jF = {p["point_id"]: np.eye(5)[rng.integers(0, 5)] for p in pts}
    jL = {p["point_id"]: np.eye(5)[0] for p in pts}
    rep = {p["point_id"]: (rng.integers(0, 4, 5) / 3 if k < 40 else None) for k, p in enumerate(pts)}
    arms, bsub, csub = P.build(pts, jF, jL, rep, 10)
    assert len(arms["H"]) == 0                                    # judge picks factual -> no pairs
    assert len(arms["Aprime"]) == 60 and all(r["preferred"] == 2 for r in arms["Aprime"])
    assert {r["point_id"] for r in arms["Rnd"]} == {r["point_id"] for r in arms["A"]}
    assert all(r["preferred"] in (1, 2) for r in arms["D"]) and len(arms["D"]) == 60   # revisit of drawer 1 -> E_new arms 1, 2
    assert set(bsub) <= {p for p, q in rep.items() if q is not None} and len(bsub) == len(csub) == 10
    assert {r["point_id"] for r in arms["Bmatch"]} <= set(bsub) and {r["point_id"] for r in arms["Cmatch"]} <= set(csub)
    for r in arms["A"]:
        assert r["chosen"].endswith(f"_{r['preferred']}") and r["rejected"].endswith("_0")


def test_judge_requests_match_p4():
    """Rebuild real P4 E2 F and L requests from their components and require byte-identical text."""
    import e4_03_judge as J
    p4 = os.path.join(E.ROOT, "exploratory_P4")
    assert E.display_order("s000_t02") == json.load(open(os.path.join(p4, "e2_point_audit.json")))[0]["display_order"]
    templates = {k: open(os.path.join(p4, "prompts", f"{k}.txt")).read() for k in ["base", "log", "feedback"]}
    by = {}
    for line in gzip.open(os.path.join(p4, "e2_requests.jsonl.gz"), "rt"):
        r = json.loads(line)
        if r["point_id"] in ("s000_t02", "s013_t01", "s050_t03"):
            by.setdefault(r["point_id"], {})[r["condition"]] = r
    assert len(by) == 3
    dec = json.JSONDecoder()
    for pid, rr in by.items():
        S, F, L = rr["S"]["text"], rr["F"]["text"], rr["L"]["text"]
        prompt = S.split("<decision_time_prompt>\n", 1)[1].rsplit("\n</decision_time_prompt>", 1)[0]
        acts = rr["S"]["actions"]
        aligned = json.loads(F[len(S) + 1 + len(templates["feedback"]):])
        resp = {x["action"]: x["response"] for x in aligned}
        ep = dec.raw_decode(L[L.find('{"task":'):])[0]
        rec = dict(point_id=pid, turn=rr["S"]["turn"], factual=acts[0], alternatives=acts[1:], prompt=prompt,
                   peeks=[dict(obs=resp[a]) for a in acts], task=ep["task"], initial_obs=ep["initial_observation"],
                   episode=ep["steps"], final_reward=ep["final_reward"])
        rows = {x["condition"]: x for x in J.build_requests(rec, templates)}
        assert rows["F"]["text"] == F, pid
        assert rows["L"]["text"] == L, pid


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
    print("all E4 tests passed")
