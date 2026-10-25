"""a'_u sensitivity analysis (D-037; RULING 2026-10-25 §7 row 3). Label-only.

On the 100 Indifferent points:
  pivotal_vs_factual : 90% Newcombe interval for p(factual, banks 3-6, n=12) - p(a'_u, n=6) excludes 0
  differs_from_alts  : 90% Newcombe interval for p(bank-1 alternatives, banks 3-6, n=48) - p(a'_u, n=6) excludes 0
Fractions with trajectory cluster bootstrap CI; mean success of factual / policy alternatives / a'_u.
Reading (ruling): if most points turn pivotal under a'_u, "Indifferent" is policy-relative (an artefact of the policy's
own alternatives being similar), not a property of the state.
"""
import json
import os

import numpy as np
import pandas as pd

import common
import importlib.util

spec = importlib.util.spec_from_file_location("reg", os.path.join(os.path.dirname(os.path.abspath(__file__)), "15_regimes.py"))
reg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reg)


def main():
    pts = json.load(open(os.path.join(common.ROOT, "eval", "au_points.json")))["points"]
    rows = []
    for t, u, alt in pts:
        f = a = 0
        fn = an = 0
        for b in ("replay_b2", "replay_b3", "replay_b4", "replay_b5"):
            r = json.load(open(os.path.join(common.ROOT, "data", b, t, f"t{u:02d}.json")))
            f += sum(r["factual_outcomes"]); fn += len(r["factual_outcomes"])
            a += sum(map(sum, r["alt_outcomes"])); an += sum(map(len, r["alt_outcomes"]))
        ra = json.load(open(os.path.join(common.ROOT, "data", "replay_au", t, f"t{u:02d}.json")))
        assert ra["alternatives"] == [alt]
        x = sum(ra["alt_outcomes"][0]); xn = len(ra["alt_outcomes"][0])
        rows.append(dict(traj_id=t, turn=u, alt=alt, atype=common.action_type(alt), f=f, fn=fn, a=a, an=an, x=x, xn=xn))
    df = pd.DataFrame(rows)
    lo, hi = reg.newcombe(df.f.values, df.fn.values, df.x.values, df.xn.values)
    df["pivotal_vs_factual"] = (lo > 0) | (hi < 0)
    lo2, hi2 = reg.newcombe(df.a.values, df.an.values, df.x.values, df.xn.values)
    df["differs_from_alts"] = (lo2 > 0) | (hi2 < 0)
    df["au_worse_than_factual"] = df.pivotal_vs_factual & (df.f / df.fn > df.x / df.xn)
    rep = {"provenance": common.provenance(__file__), "n_points": len(df), "n_traj": int(df.traj_id.nunique())}
    for c in ("pivotal_vs_factual", "differs_from_alts", "au_worse_than_factual"):
        rep[c] = {"n": int(df[c].sum()), "frac": float(df[c].mean()),
                  "ci95_cluster": reg.cluster_ci(df, lambda d, c=c: float(d[c].mean()))}
    rep["mean_success"] = {"factual_b3_6": float(df.f.sum() / df.fn.sum()), "policy_alts_b3_6": float(df.a.sum() / df.an.sum()),
                           "a_u": float(df.x.sum() / df.xn.sum())}
    rep["by_au_type"] = {k: {"n": len(g), "pivotal_frac": float(g.pivotal_vs_factual.mean()),
                             "a_u_success": float(g.x.sum() / g.xn.sum())} for k, g in df.groupby("atype")}
    rep["reading"] = ("majority pivotal => Indifferent is policy-relative" if rep["pivotal_vs_factual"]["frac"] > 0.5
                      else "Indifferent largely survives a uniform-random admissible alternative")
    common.write_json(os.path.join(common.ROOT, "eval", "au_sensitivity.json"), rep)
    df.to_parquet(os.path.join(common.ROOT, "eval", "au_sensitivity_points.parquet"))
    print(json.dumps({k: v for k, v in rep.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    main()
