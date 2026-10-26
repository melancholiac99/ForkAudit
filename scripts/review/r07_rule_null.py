"""Null false-positive calibration of the two Pivotal rules (label-only, §9.2 safe; supports D-042).

For every live point, permute its 60 bank 3-6 outcomes across the five arms (conditional on margins, keeping the
bank-position layout: positions 0-5 of each arm = banks 3+4, 6-11 = banks 5+6) and apply
  main rule        : Newcombe 90% (factual 12 vs pool 48) excludes 0
  sensitivity rule : |A_3456| >= 0.25 and A_34, A_56 same sign, both nonzero
Sum of per-point flag probabilities = expected flags if no live point had any arm difference (an upper bound on the
false-positive count). Output eval/rule_null.json.
"""
import importlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

R = importlib.import_module("15_regimes")
N_PERM = 2000


def rules(f, a):
    """f (..., 12), a (..., 4, 12) -> (main, sensitivity) flags."""
    lo, hi = R.newcombe(f.sum(-1), 12, a.sum((-1, -2)), 48)
    A = f.mean(-1) - a.mean((-1, -2))
    A34 = f[..., :6].mean(-1) - a[..., :6].mean((-1, -2))
    A56 = f[..., 6:].mean(-1) - a[..., 6:].mean((-1, -2))
    sens = (np.abs(A) >= 0.25) & (A34 != 0) & (A56 != 0) & (np.sign(A34) == np.sign(A56))
    return (lo > 0) | (hi < 0), sens


def main():
    df = R.load()
    fs, fn, as_, an = R.contrast(df, [1, 2])
    L = df[(fs + as_ > 0) & (fs + as_ < fn + an)].reset_index(drop=True)
    f = np.stack([np.concatenate([L.loc[i, f"f{b}"] for b in (3, 4, 5, 6)]) for i in range(len(L))])
    a = np.stack([np.concatenate([L.loc[i, f"a{b}"] for b in (3, 4, 5, 6)], axis=1) for i in range(len(L))])
    obs_main, obs_sens = rules(f, a)
    e_main = e_sens = e_both = 0.0
    for i in range(len(L)):
        rng = np.random.default_rng(3_000_000 + 1000 * int(L.traj_id[i][1:]) + int(L.turn[i]))
        o = np.concatenate([f[i], a[i].ravel()])
        p = rng.permuted(np.tile(o, (N_PERM, 1)), axis=1).reshape(N_PERM, 5, 12)
        m, s = rules(p[:, 0], p[:, 1:])
        e_main += m.mean()
        e_sens += s.mean()
        e_both += (m & s).mean()
    out = {"n_live": int(len(L)), "n_perm": N_PERM,
           "observed": {"main": int(obs_main.sum()), "sensitivity": int(obs_sens.sum()), "both": int((obs_main & obs_sens).sum())},
           "expected_under_global_null": {"main": float(e_main), "sensitivity": float(e_sens), "both": float(e_both)},
           "naive_precision_floor": {"main": float(1 - e_main / obs_main.sum()), "sensitivity": float(1 - e_sens / obs_sens.sum())},
           "note": "expected counts assume every live point is null, so they bound the false-positive count from above; "
                   "1 - E/observed is therefore a conservative (low) precision estimate",
           "provenance": common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}}
    common.write_json(os.path.join(common.ROOT, "eval", "rule_null.json"), out)
    print({k: v for k, v in out.items() if k != "provenance"})


if __name__ == "__main__":
    main()
