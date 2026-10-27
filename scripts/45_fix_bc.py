"""D-051: recompute the B+C rows after the bc_stack fix (D-050), in eval/main{,_vllm}.json and eval/scores_*.parquet.

Everything else in main.json is untouched. The superseded rows are kept under "B+C (D-050 superseded)". The
R-MODEL/R-LADDER paired comparison (B+C minus W1-pure on B2), the verdict and the predictions are recomputed with the
same functions as 40_eval.py. Bootstrap weights are the same (same groups, same seed), so CIs stay comparable.
"""
import argparse
import importlib
import json
import os
import time

import numpy as np
import pandas as pd

import common
import evalcore as E

M = importlib.import_module("40_eval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    t0 = time.time()
    mp = os.path.join(common.ROOT, "eval", f"main{args.tag}.json")
    res = json.load(open(mp))
    df, pinfo = M.load(args.tag)
    Bf = list(dict.fromkeys(M.F1 + M.F2 + M.F3 + M.F4 + M.F5 + ["entropy_sampled"])) + M.B_CTRL
    for bname, (S, y, strat) in M.sets(df, False).items():
        tb = time.time()
        S = S.reset_index(drop=True)
        g = S.traj_id.values
        strata = S[strat].values if strat else np.zeros(len(S), int)
        fold = E.outer_folds(g)
        boot = E.Boot(g)
        cn = {"B2": "B2het"}.get(bname, bname)
        sp = os.path.join(common.ROOT, "eval", f"scores_{bname}{args.tag}.parquet")
        sc = pd.read_parquet(sp)
        assert (sc.traj_id.values == S.traj_id.values).all() and (sc.turn.values == S.turn.values).all()
        new, meta = M.bc_stack(S, y, g, fold, Bf, cn, pinfo, 1)
        B = res["boundaries"][bname]
        ref = {r: (sc[r].values, boot.auroc_dist(y, sc[r].values)) for r in ("V", "A(-conf)")}
        sm = E.summarize(y, new, boot, g, strata, ceiling=B["ceiling"], ref=ref)
        d = sm.pop("_dist")
        sm["REV"] = next((int(m) for m, v in sorted(B["ceiling_curve"].items(), key=lambda x: int(x[0])) if v >= sm["auroc"]), ">6")
        sm["passes"] = bool(sm["ci95"][0] > sm["shuffle_p97.5"] and sm["paired"]["V"]["ci95"][0] > 0
                            and sm["paired"]["A(-conf)"]["ci95"][0] > 0)
        B["scores"]["B+C (D-050 superseded)"] = B["scores"].pop("B+C")
        B["scores"]["B+C"] = sm | meta | {"fixed_by": "D-051"}
        if "W1pure_multi" in sc:
            dw = boot.auroc_dist(y, sc["W1pure_multi"].values)
            if "paired_model_minus_w1pure" in B:
                B["paired_model_minus_w1pure (D-050 superseded)"] = B.pop("paired_model_minus_w1pure")
            B["paired_model_minus_w1pure"] = {"diff": sm["auroc"] - E.auroc(y, sc["W1pure_multi"].values),
                                              "ci95": E.Boot.ci(d - dw), "fixed_by": "D-051"}
        sc["B+C (D-050 superseded)"] = sc["B+C"]
        sc["B+C"] = new
        sc.to_parquet(sp)
        print(bname, f"B+C {res['boundaries'][bname]['scores']['B+C (D-050 superseded)']['auroc']:.3f} -> {sm['auroc']:.3f}",
              f"passes {sm['passes']}", meta["layer_probeC_stackC_per_fold"], f"{time.time() - tb:.0f}s", flush=True)
    old = res["verdict"]
    res["verdict"] = M.verdict(res, False)
    res["verdict"]["before_D051"] = old
    res["predictions"] = M.predictions(res)
    res["provenance_D051"] = common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}
    common.write_json(mp, res)
    print("verdict", json.dumps({k: v for k, v in res["verdict"].items() if k != "before_D051"}, default=str),
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
