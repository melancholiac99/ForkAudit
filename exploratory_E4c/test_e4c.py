"""CPU tests for E4c (no GPU, no ALFWorld). Run: python test_e4c.py"""
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e4c_common as C  # noqa: E402
from e4c_common import common  # noqa: E402
import e4c_eval  # noqa: E402
import e4c_analyze  # noqa: E402


def test_template():
    t = C.hinted_template()
    assert t.count(C.HINT) == 1 and t.replace(C.HINT + "\n", "") == common.TEMPLATE
    assert C.template_for("base") is common.TEMPLATE and C.template_for("R") is common.TEMPLATE
    assert C.HINT in C.template_for("P") and C.HINT in C.template_for("PR")
    mem = [{"text_obs": "o1", "action": "go to drawer 1"}]
    p = common.build_prompt("put a cup in sink", mem, "o2", ["go to sink 1", "look"], template=t)
    p0 = common.build_prompt("put a cup in sink", mem, "o2", ["go to sink 1", "look"])
    assert p.replace(C.HINT + "\n", "") == p0
    i, j = p.index("Your admissible actions"), p.index(C.HINT)
    assert i < j < p.index("Now it's your turn")


def test_blocking_and_choice():
    assert not C.is_blocked("base", "examine drawer 1", "examine drawer 1") and not C.is_blocked("P", "look", "look")
    assert C.is_blocked("R", "go to drawer 1", "Go to  drawer 1") and not C.is_blocked("R", "examine drawer 1", "go to drawer 1")
    assert not C.is_blocked("R", "go to drawer 1", None)
    assert C.is_blocked("X", "look", None) and C.is_blocked("X", "inventory", "go to x") and C.is_blocked("X", "examine sink 1", None)
    assert C.is_blocked("X", "go to sink 1", "go to sink 1") and not C.is_blocked("X", "go to sink 1", "look")
    assert not C.INSPECT.match("lookout")                                # word boundary: 'lookout' is not 'look'
    assert C.INSPECT.match("look") and C.INSPECT.match("examine desklamp 1") and C.INSPECT.match("inventory")
    tx = lambda a: f"<think>r</think><action>{a}</action>"
    assert C.choose("X", [tx("look"), tx("examine sink 1"), tx("go to sink 1")], None)[:2] == (2, "go to sink 1")
    assert C.choose("X", [tx("look")] * 9, None)[:2] == (8, "look")                 # all blocked -> last draw kept
    assert C.choose("base", [tx("look"), tx("go to x")], None)[:2] == (0, "look")
    roll_seed = 1_000_000 * 3133 + 50 + 10 ** 11 * 3
    assert C.resample_seed(roll_seed, 1) > 10 ** 12 > roll_seed and len({C.resample_seed(roll_seed, k) for k in range(1, 9)}) == 8


class FakePool:
    """One room: 'go to Y' -> 'You arrive at Y.'; 'take apple 1 from sink 1' wins; everything else -> 'Nothing happens.'"""

    def __init__(self):
        self.k = {}

    def open(self, jobs):
        out = {}
        for key, *_ in jobs:
            self.k[key] = 0
            out[key] = ("-= Welcome =- You are in a room. Your task is to: put a apple in fridge.", ["go to sink 1", "look"], False, 0)
        return out

    def step(self, acts):
        out = {}
        for key, a in acts.items():
            self.k[key] += 1
            won = a == "take apple 1 from sink 1"
            obs = f"You arrive at {a[6:]}. On it you see a apple 1." if a.startswith("go to") else "Nothing happens."
            out[key] = (obs, ["go to sink 1", "take apple 1 from sink 1", "look"], won, won)
        return out

    def close(self, keys):
        pass


class FakePolicy:
    """First draws: turn 1 'look', later 'go to sink 1' (a repeat from turn 3 on). Redraw k: 'examine sink 1' for k = 1,
    'take apple 1 from sink 1' for k >= 2."""
    vllm_version = "fake"

    def __init__(self):
        self.calls = []

    def sample(self, prompts, seeds, n=1, logprobs=False):
        out = []
        for p, s in zip(prompts, seeds):
            self.calls.append((p, s))
            k = s // C.RESAMPLE_OFFSET
            turn = (s % C.RESAMPLE_OFFSET) % 10 ** 6 % 1000
            a = ("look" if turn == 1 else "go to sink 1") if k == 0 else ("examine sink 1" if k == 1 else "take apple 1 from sink 1")
            out.append([{"text": f"<think>t</think><action>{a}</action>"}])
        return out


def _run(cond, root):
    pool, pol = FakePool(), FakePolicy()
    games = [dict(seed=3000, game_file="g", task_type="pick_and_place_simple")]
    e4c_eval.run(games, 2, pol, pool, os.path.join(root, cond), types.SimpleNamespace(ALFWORLD_DATA=""), {}, cond)
    eps = [json.load(open(os.path.join(root, cond, f"s3000_r{r}.json"))) for r in range(2)]
    return eps, pol


def test_run_conditions():
    root = tempfile.mkdtemp()
    try:
        roll = __import__("e4_01_rollout")
        eps, pol = _run("base", root)
        assert all(s["draw_index"] == 0 for e in eps for s in e["steps"]) and not eps[0]["won"] and len(eps[0]["steps"]) == common.MAX_STEPS
        assert eps[0]["steps"][4]["sample_seed"] == roll.seed_for(3000, 5, 0) and C.HINT not in pol.calls[0][0]
        eps, pol = _run("P", root)
        assert all(C.HINT in p for p, _ in pol.calls) and [s["action"] for s in eps[0]["steps"][:3]] == ["look", "go to sink 1", "go to sink 1"]
        eps, _ = _run("R", root)                                  # turn 3 repeats 'go to sink 1' -> redraw 1 'examine', not a repeat
        s3 = eps[0]["steps"][2]
        assert s3["action"] == "examine sink 1" and s3["draw_index"] == 1 and s3["blocked"] == ["go to sink 1"] and not s3["kept_blocked"]
        assert s3["sample_seed"] == C.resample_seed(roll.seed_for(3000, 3, 0), 1)
        eps, _ = _run("X", root)                                  # turn 1 'look' blocked, redraw 1 'examine' blocked, redraw 2 wins
        s1 = eps[0]["steps"][0]
        assert eps[0]["won"] and s1["action"] == "take apple 1 from sink 1" and s1["blocked"] == ["look", "examine sink 1"]
        assert s1["draw_index"] == 2 and s1["n_draws"] == 3
        b = C.behaviour(eps)
        assert b["success"] == 1.0 and b["inspect_share"] == 0.0 and b["steps"] == 2
    finally:
        shutil.rmtree(root)


def test_behaviour_and_reading():
    ep = lambda acts, won, obs="nothing": dict(task="put a apple in fridge.", initial_obs="room", won=won,
                                               steps=[dict(action=a, obs_after=obs) for a in acts])
    eps = [ep(["look", "look", "go to sink 1"], True, "you see a apple 1"), ep(["examine x"] * 50, False)]
    b = C.behaviour(eps)
    assert abs(b["inspect_share"] - 52 / 53) < 1e-12 and abs(b["repeat_share"] - 50 / 53) < 1e-12
    assert b["target_seen"] == 0.5 and b["success_given_seen"] == 1.0 and b["timeout_share_of_failures"] == 1.0
    assert C.in_trained_range(dict(inspect_share=0.2, repeat_share=0.03)) and not C.in_trained_range(dict(inspect_share=0.3, repeat_share=0.01))
    lo = lambda *v: {c: x for c, x in zip(C.CONTROLS, v)}
    ok = {c: True for c in C.CONTROLS}
    assert C.reading(lo((-9, -1), (-9, -2), (-3, 2), (-9, -1)), lo((-9, -1), (-9, -1), (-9, -1), (-9, -1)), ok) == "E4c-SUBSTITUTE"
    vsA = lo((-12, -6), (-12, -6), (-12, -6), (-12, -6))
    assert C.reading(vsA, lo((-5, 1), (-9, -3), (-3.5, 1), (-9, -3)), ok) == "E4c-HEURISTIC-ONLY"
    vsD = lo((-9, -2), (-9, -2), (-9, -2), (-9, -2))
    assert C.reading(vsA, vsD, {c: False for c in C.CONTROLS}) == "E4c-NOMANIP"
    assert C.reading(vsA, vsD, ok) == "E4c-LEARNED"
    assert C.reading(vsA, lo((-9, -2), (-9, 0.5), (-9, -2), (-9, -2)), ok) == "E4c-MIXED"


def test_analyze_offline():
    tmp = tempfile.mkdtemp()
    here_json = os.path.join(tmp, "e4c.json")
    old = C.HERE, C.DATA, C.E4_DATA, sys.argv
    try:
        C.HERE = tmp
        C.DATA, C.E4_DATA = os.path.join(tmp, "e4c"), os.path.join(tmp, "e4")
        games = json.load(open(C.E.GAMES_PATH))["valid_unseen"]
        rng = np.random.default_rng(0)
        rates = dict(base=.37, P=.40, R=.38, PR=.41, X=.43, A=.48, D=.45, base_E4=.37)

        def write(root, tag, p, reps=3):
            d = os.path.join(root, "rollout", "valid_unseen", tag)
            os.makedirs(d, exist_ok=True)
            per = {str(g["seed"]): rng.binomial(1, p, reps).tolist() for g in games}
            json.dump(dict(per_game=per), open(os.path.join(d, "summary.json"), "w"))
            act = "go to x" if tag in ("X", "A_s0", "A_s1", "A_s2", "D_s0", "D_s1", "D_s2") else "look"
            for g in games[:3]:
                for r in range(reps):
                    json.dump(dict(task="put a apple in fridge.", initial_obs="room", won=bool(per[str(g["seed"])][r]),
                                   steps=[dict(action=act, obs_after="o", draw_index=0, n_draws=1, blocked=[], kept_blocked=False)]),
                              open(os.path.join(d, f"s{g['seed']}_r{r}.json"), "w"))
        for c in C.CONDITIONS:
            write(C.DATA, c, rates[c])
        for ref in ("A", "D"):
            for s in (0, 1, 2):
                write(C.E4_DATA, f"{ref}_s{s}", rates[ref])
        write(C.E4_DATA, "base", rates["base_E4"])
        sys.argv = ["e4c_analyze.py", "--splits", "valid_unseen"]
        e4c_analyze.main()
        out = json.load(open(here_json))
        u = out["splits"]["valid_unseen"]
        assert out["reading"] in ("E4c-SUBSTITUTE", "E4c-HEURISTIC-ONLY", "E4c-LEARNED", "E4c-MIXED", "E4c-NOMANIP")
        assert {"P-A", "X-D", "A-base", "base-base_E4"} <= set(u["diff"]) and u["behaviour"]["X"]["inspect_share"] == 0
        assert u["behaviour"]["base"]["inspect_share"] == 1.0 and out["behaviour_in_trained_range"]["X"] is True
    finally:
        C.HERE, C.DATA, C.E4_DATA, sys.argv = old
        shutil.rmtree(tmp)


def test_analyze_preserves_existing_outputs():
    from contextlib import redirect_stdout
    from io import StringIO
    old_here, old_argv = C.HERE, sys.argv
    with tempfile.TemporaryDirectory() as root:
        sentinels = {"e4c.json": b'{"existing_formal_result": true}\n', "E4c_REPORT.txt": b"existing formal report\n"}
        for name, content in sentinels.items():
            with open(os.path.join(root, name), "wb") as f:
                f.write(content)
        try:
            C.HERE = root
            with redirect_stdout(StringIO()):
                test_analyze_offline()
            assert C.HERE == root and sys.argv is old_argv
            for name, content in sentinels.items():
                with open(os.path.join(root, name), "rb") as f:
                    assert f.read() == content
        finally:
            C.HERE, sys.argv = old_here, old_argv


def test_analyze_restores_state_on_failure():
    from unittest.mock import patch
    old = C.HERE, C.DATA, C.E4_DATA, sys.argv
    with patch.object(e4c_analyze, "main", side_effect=RuntimeError("test analysis failure")):
        try:
            test_analyze_offline()
        except RuntimeError as exc:
            assert str(exc) == "test analysis failure"
        else:
            raise AssertionError("expected analysis failure")
    assert (C.HERE, C.DATA, C.E4_DATA) == old[:3] and sys.argv is old[3]


if __name__ == "__main__":
    for name, f in sorted(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
    print("all E4c tests passed")
