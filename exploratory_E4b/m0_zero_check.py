"""Exploratory (post hoc, zero GPU): how often is a 15/15-constant point under 3 runs x 5 arms a fork at deeper replay?

Uses M0's frozen labels (same policy, template and 5-arm x 3-run protocol as E4): bank 1 = the 3-run bank; banks 1+2 =
30 runs; banks 3-6 = 12 runs per arm on points live in banks 1+2 (Pivotal_het, alpha = .10, permutation null).
This stands in for PREREG_E4b's optional deeper-replay check; population differs from E4 (M0 = all phases, 100 games).
Output: m0_zero_check.json
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "exploratory_P4", "post_e2"))
try:
    import pyarrow.parquet as pq
    read = lambda p: pq.read_table(p).to_pydict()       # noqa: E731
except ImportError:                                     # no pyarrow -> P4's pure-Python fallback reader
    import minipq
    read = minipq.read


def main():
    L = read(os.path.join(ROOT, "data", "labels_L.parquet"))
    H = read(os.path.join(ROOT, "eval", "het_points.parquet"))
    het = {(t, u): (bool(h), bool(a)) for t, u, h, a in zip(H["traj_id"], H["turn"], H["het"], H["piv_agg"])}
    c = collections.Counter()
    for t, u, l1, l12 in zip(L["traj_id"], L["turn"], L["live_b1"], L["live_b12"]):
        if l1:
            continue
        c["bank1_constant"] += 1
        if not l12:
            c["constant_30_runs"] += 1
            continue
        h, a = het[(t, u)]
        c["live_after_bank2"] += 1
        c["het"] += h
        c["het_and_credit_pivotal"] += h and a
    n_live, e_het = 1478, 103.8735                     # eval/het_fork.json: live points and null-expected het count
    out = dict(counts=dict(c),
               frac_het_given_bank1_constant=c["het"] / c["bank1_constant"],
               frac_het_among_those_turned_live=c["het"] / c["live_after_bank2"],
               null_het_rate_on_live=e_het / n_live,
               note="bank-1 constant includes a few all-success points (M0 dead pools: 1516 all-fail, 44 all-success)")
    json.dump(out, open(os.path.join(HERE, "m0_zero_check.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
