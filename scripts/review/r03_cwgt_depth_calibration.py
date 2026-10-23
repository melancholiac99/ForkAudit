"""Phase 1C.0 (RULING 2026-10-23 §7.5): depth calibration on CWGT ICLR-2027 evidence.json.

For each turn (172 = 43 trajectories x 4 positions) CWGT stores, per bank (A, B) and per arm (factual + 4 alts),
the number of successes out of n=16 continuations. We subsample d of the 16 without replacement
(hypergeometric) independently per bank/arm, 2,000 times, and report for d in {3, 6, 9, 12, 16}:
  (i)   live fraction: pool (5 arms x d) not constant, per bank;
  (ii)  sign agreement of A_d vs B_d on turns where both contrasts are nonzero;
  (iii) Pearson / pooled Spearman of A_d vs B_d over all turns;
  (iv)  d=16 is deterministic and must reproduce the paper (Spearman 0.330, Pearson 0.736).
A = mean(factual) - mean(alternatives). Reads the downloaded file only; nothing from the download is executed.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

EV = os.path.join(common.ROOT, "downloads", "openreview_supp", "supplementry", "submission_source",
                  "scientific", "inputs", "evidence.json")
DEPTHS = [3, 6, 9, 12, 16]
N_REP = 2000


def load():
    e = json.load(open(EV))
    turns = {}
    for bank in ("A", "B"):
        for r in e["arms"][bank]:
            key = (r["trajectory"], r["turn"])
            col = 0 if r["kind"] == "factual" else 1 + r["k"]
            assert r["n"] == 16
            turns.setdefault(key, {}).setdefault(bank, [None] * 5)[col] = r["successes"]
    keys = sorted(k for k, v in turns.items() if all(b in v and None not in v[b] for b in ("A", "B")))
    S = {b: np.array([turns[k][b] for k in keys]) for b in ("A", "B")}  # (n_turns, 5)
    return keys, S


def contrast(s, d):
    return s[:, 0] / d - s[:, 1:].sum(1) / (4 * d)


def main():
    keys, S = load()
    print(f"turns {len(keys)}; trajectories {len({k[0] for k in keys})}")
    rng = np.random.default_rng(0)
    out = {"provenance": common.provenance(__file__, [EV]), "n_turns": len(keys), "depths": {}}
    for d in DEPTHS:
        rows = []
        reps = 1 if d == 16 else N_REP
        for _ in range(reps):
            sub = {b: (S[b] if d == 16 else rng.hypergeometric(S[b], 16 - S[b], d)) for b in ("A", "B")}
            a, b = contrast(sub["A"], d), contrast(sub["B"], d)
            liveA = (sub["A"].sum(1) > 0) & (sub["A"].sum(1) < 5 * d)
            liveB = (sub["B"].sum(1) > 0) & (sub["B"].sum(1) < 5 * d)
            both_nz = (a != 0) & (b != 0)
            agree = np.mean(np.sign(a[both_nz]) == np.sign(b[both_nz])) if both_nz.any() else np.nan
            live_both = liveA & liveB
            nz_live = both_nz & live_both
            agree_live = np.mean(np.sign(a[nz_live]) == np.sign(b[nz_live])) if nz_live.any() else np.nan
            pr = pearsonr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan
            sr = spearmanr(a, b)[0] if a.std() > 0 and b.std() > 0 else np.nan
            rows.append([liveA.mean(), liveB.mean(), live_both.mean(), both_nz.mean(), agree, agree_live, pr, sr])
        R = np.array(rows, dtype=float)
        names = ["live_frac_A", "live_frac_B", "live_frac_both", "both_nonzero_frac", "sign_agree_both_nonzero",
                 "sign_agree_both_live_nonzero", "pearson", "spearman_pooled"]
        res = {}
        for j, n in enumerate(names):
            col = R[:, j][~np.isnan(R[:, j])]
            res[n] = {"mean": float(col.mean()), "p2.5": float(np.percentile(col, 2.5)),
                      "p97.5": float(np.percentile(col, 97.5))}
        out["depths"][d] = res
        print(f"d={d:2d}  live(A) {res['live_frac_A']['mean']:.3f}  both_nonzero {res['both_nonzero_frac']['mean']:.3f}  "
              f"sign_agree {res['sign_agree_both_nonzero']['mean']:.3f} "
              f"[{res['sign_agree_both_nonzero']['p2.5']:.3f},{res['sign_agree_both_nonzero']['p97.5']:.3f}]  "
              f"pearson {res['pearson']['mean']:.3f}  spearman {res['spearman_pooled']['mean']:.3f}")
    d16 = out["depths"][16]
    out["check_d16_vs_paper"] = {"spearman": [d16["spearman_pooled"]["mean"], 0.330],
                                 "pearson": [d16["pearson"]["mean"], 0.736]}
    d12 = out["depths"][12]["sign_agree_both_nonzero"]["mean"]
    out["rule_7_5"] = {"d12_sign_agreement": d12, "threshold": 0.65,
                       "expand_to_banks_3_10": bool(d12 < 0.65)}
    print("check d=16 vs paper:", out["check_d16_vs_paper"])
    print("rule 7.5: d=12 sign agreement", round(d12, 3), "-> expand to banks 3-10:", d12 < 0.65)
    common.write_json(os.path.join(common.ROOT, "eval", "phase1c0_depth_calibration.json"), out)


if __name__ == "__main__":
    main()
