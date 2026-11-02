"""CPU tests for E4b (no GPU, no ALFWorld). Optional real-data check: set E4B_E4DATA to an extracted E4 data/ tree.
  python test_e4b.py
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

import e4b_common as B
from e4b_common import E

HERE = os.path.dirname(os.path.abspath(__file__))


def test_classify_and_bfac():
    z = [[0, 0, 0]] * 5
    assert B.classify(z)[0] == "Z3"
    assert B.classify([[1, 0, 0]] * 5)[0] == "E3"
    cls, q = B.classify([[0, 0, 1], [0, 0, 0], [1, 1, 0], [1, 1, 0], [0, 0, 0]])
    assert cls == "V3" and np.allclose(q, [1 / 3, 0, 2 / 3, 2 / 3, 0])
    assert B.pair_bfac(q) == (2, 0)                                   # tie between arms 2 and 3 -> lowest index
    assert B.pair_bfac(np.array([2 / 3, 1 / 3, 0, 2 / 3, 0])) is None  # no alternative strictly above factual
    assert B.pair_bfac(np.array([0, 0, 0, 0, 1 / 3])) == (4, 0)


def test_arm_routing():
    pA, pD = (3, 0), (1, 0)
    q = np.zeros(5)
    assert B.arm_pair("Azero", "Z3", q, pA, pD) == pA and B.arm_pair("Agap", "Z3", q, pA, pD) is None
    assert B.arm_pair("Dzero", "Z3", q, pA, pD) == pD and B.arm_pair("Dgap", "V3", q, pA, None) is None
    assert B.arm_pair("Bfacgap", "Z3", q, pA, pD) is None
    assert B.arm_pair("Bfacgap", "V3", np.array([0, 0, 1 / 3, 0, 0]), pA, pD) == (2, 0)
    assert B.arm_pair("Azero", "E3", q, pA, pD) is None


def test_quotas_and_common_n():
    shares = {"a": 0.5, "b": 0.3, "c": 0.2}
    for n in (1, 7, 100, 960):
        q = B.task_quotas(shares, n)
        assert sum(q.values()) == n
    avail = {"x": {"a": 1000, "b": 1000, "c": 1000}, "y": {"a": 1000, "b": 1000, "c": 101}}
    n = B.common_n(avail, shares)
    assert n % B.N_MULT == 0 and B.task_quotas(shares, n)["c"] <= 101 < B.task_quotas(shares, n + B.N_MULT)["c"]
    assert B.common_n({"x": {"a": 5}}, {"a": 1.0}) == 0


def test_draw():
    rows = [dict(point_id=f"s{i:04d}_t01", task_type="ab"[i % 2]) for i in range(100)]
    d1, d2 = B.draw("Azero", rows, {"a": 10, "b": 20}), B.draw("Azero", rows, {"a": 10, "b": 20})
    assert d1 == d2 and len(d1) == 30 and len({r["point_id"] for r in d1}) == 30
    assert sum(r["task_type"] == "a" for r in d1) == 10
    assert B.draw("Agap", rows, {"a": 10, "b": 20}) != d1               # arm name enters the seed
    try:
        B.draw("Azero", rows, {"a": 51})
        raise AssertionError("shortfall must fail")
    except AssertionError as e:
        assert "shortfall" not in str(e)


def test_reading():
    assert B.reading(-0.1, 5, [0, 1]) == "E4b-NULL"
    assert B.reading(1, 1, [-3.9, 2]) == "E4b-SUPPORT"
    assert B.reading(1, 1, [-3.95, -0.5]) == "E4b-SUPPORT"               # within delta even if gap is better
    assert B.reading(1, 1, [-9, -4.5]) == "E4b-COUNTER"
    assert B.reading(1, 1, [-6, 1]) == "E4b-PARTIAL"


def test_pooled_contrast_is_average():
    rng = np.random.default_rng(0)
    m = {a: rng.integers(0, 2, size=(30, 9)).astype(float) for a in B.PRIMARY}
    Z, G = np.concatenate([m["Azero"], m["Dzero"]], 1), np.concatenate([m["Agap"], m["Dgap"]], 1)
    want = 100 * ((m["Azero"].mean(1) + m["Dzero"].mean(1)) / 2 - (m["Agap"].mean(1) + m["Dgap"].mean(1)) / 2).mean()
    assert abs(E.game_bootstrap_diff(Z, G)["diff"] - want) < 1e-9


def test_analyze_end_to_end():
    import e4b_analyze as A
    games = json.load(open(E.GAMES_PATH))["valid_unseen"]
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(1)
        for tag in ["base"] + [f"{a}_s{s}" for a in B.ARMS for s in B.SEEDS]:
            per = {str(g["seed"]): rng.integers(0, 2, 3).tolist() for g in games}
            E.write_json(os.path.join(tmp, "rollout", "valid_unseen", tag, "summary.json"), dict(per_game=per))
        M = {a: A.matrix("valid_unseen", a, games, tmp) for a in ["base"] + B.ARMS}
        assert M["base"].shape == (len(games), 3) and M["Azero"].shape == (len(games), 9)


def test_wrapper_redirects_train():
    out = subprocess.run([sys.executable, os.path.join(HERE, "e4b_run.py"), "train", "--arm", "Azero", "--seed", "0",
                          "--lr", "5e-06", "--help"], capture_output=True, text=True)
    assert "Azero" in out.stdout and "Bfacgap" in out.stdout, out.stdout + out.stderr
    bad = subprocess.run([sys.executable, os.path.join(HERE, "e4b_run.py"), "train", "--arm", "Azero", "--seed", "0",
                          "--lr", "2e-05"], capture_output=True, text=True)
    assert bad.returncode != 0 and "learning rate" in bad.stderr


def test_real_data_gate():
    d = os.environ.get("E4B_E4DATA")
    if not d:
        print("  (skipped: set E4B_E4DATA to run the real-data gate check)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "g.json")
        subprocess.run([sys.executable, os.path.join(HERE, "e4b_pairs.py"), "--e4data", d, "--gate-only", "--out", out],
                       check=True, capture_output=True)
        g, ref = json.load(open(out)), json.load(open(os.path.join(HERE, "e4b_gate.json")))
        for k in ("classes", "N", "quotas", "V3_with_fac_alternative", "selected"):
            assert g[k] == ref[k], k


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("all E4b CPU tests passed")
