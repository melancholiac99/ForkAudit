"""Check of the full fp32 (HF + shared-prefix KV cache) scoring pass against the vLLM bf16 pass (D-036).
Scores only; reads `cf_defined` for the conf-shift sanity row (§9.2 safe: no A_replay, no outcomes).

  1. integrity      : same keys / rows, finite values, identical n_tokens (same tokenization), model hash
  2. validation     : on the 20 alignment points, hfprefix fp32 (GPU, 7B) vs full-sequence HF fp32 (eval/prereq)
  3. agreement      : fp32 vs vLLM bf16 per candidate and per point (Spearman, top-1, chosen_rank, margin sign)
  4. noise floor    : share of points whose top-1/top-2 gap is below the bf16 noise (0.3 nats) in fp32
  5. chosen variants: conf (bare) included-minus-excluded on all 3,818 points, both backends
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

D = os.path.join(common.ROOT, "data")
K = ["traj_id", "turn", "action"]
v = pd.read_parquet(os.path.join(D, "scores.parquet"))
f = pd.read_parquet(os.path.join(D, "scores_fp32.parquet"))
vc = pd.read_parquet(os.path.join(D, "scores_chosen.parquet"))
fc = pd.read_parquet(os.path.join(D, "scores_chosen_fp32.parquet"))
rep = {"provenance": common.provenance(__file__)}

# 1 integrity
m = v.merge(f, on=K, suffixes=("_v", "_f"), how="outer", indicator=True)
num = ["logp_sum", "logp_mean", "logp_sum_notask", "logp_mean_notask"]
rep["integrity"] = {
    "rows_vllm": len(v), "rows_fp32": len(f), "only_in_one": int((m._merge != "both").sum()),
    "nonfinite_fp32": int((~np.isfinite(f[num].values)).sum()), "nonfinite_vllm": int((~np.isfinite(v[num].values)).sum()),
    "n_tokens_mismatch": int((m.n_tokens_v != m.n_tokens_f).sum()),
    "is_chosen_mismatch": int((m.is_chosen_v != m.is_chosen_f).sum()),
    "scorer": [v.scorer.unique().tolist(), f.scorer.unique().tolist()],
    "model_hash": [v.model_hash.unique().tolist(), f.model_hash.unique().tolist()],
    "chosen_rows": [len(vc), len(fc)],
    "chosen_available_fp32": fc.groupby("variant").available.mean().round(4).to_dict(),
}
m = m[m._merge == "both"]

# 2 validation vs full-sequence fp32 reference (prereq, 20 points)
ref = pd.read_parquet(os.path.join(common.ROOT, "eval", "prereq", "scores_hf32.parquet"))
r = ref.merge(f, on=K, suffixes=("_ref", "_f"))
e = (r.logp_sum_f - r.logp_sum_ref).abs()
en = (r.logp_sum_notask_f - r.logp_sum_notask_ref).abs()
rep["prefix_vs_fullseq_fp32_20pts"] = {"pairs": len(r), "max_abs_diff_bare": float(e.max()), "p99_bare": float(e.quantile(.99)),
                                       "max_abs_diff_notask": float(en.max())}

# 3 agreement
d = m.logp_sum_v - m.logp_sum_f
per = []
for (t, u), g in m.groupby(["traj_id", "turn"]):
    gv = g.sort_values("logp_sum_v", ascending=False)
    gf = g.sort_values("logp_sum_f", ascending=False)
    rk_v = 1 + (g.logp_sum_v > g[g.is_chosen_v].logp_sum_v.iloc[0]).sum()
    rk_f = 1 + (g.logp_sum_f > g[g.is_chosen_f].logp_sum_f.iloc[0]).sum()
    gap_f = gf.logp_sum_f.iloc[0] - gf.logp_sum_f.iloc[1] if len(g) > 1 else np.inf
    per.append(dict(sp=spearmanr(g.logp_sum_v, g.logp_sum_f)[0] if len(g) > 2 else np.nan,
                    sp_n=spearmanr(g.logp_sum_notask_v, g.logp_sum_notask_f)[0] if len(g) > 2 else np.nan,
                    top1=gv.action.iloc[0] == gf.action.iloc[0], top2=list(gv.action.iloc[:2]) == list(gf.action.iloc[:2]),
                    rank_same=rk_v == rk_f, gap_f=gap_f, shift=float((g.logp_sum_v - g.logp_sum_f).mean()),
                    resid=float(((g.logp_sum_v - g.logp_sum_f) - (g.logp_sum_v - g.logp_sum_f).mean()).abs().median())))
P = pd.DataFrame(per)
rep["agreement"] = {
    "delta_logp_sum_vllm_minus_fp32": {"mean": float(d.mean()), "abs_p50": float(d.abs().median()),
                                      "abs_p90": float(d.abs().quantile(.9)), "abs_max": float(d.abs().max())},
    "per_point_mean_shift": {"mean": float(P["shift"].mean()), "sd": float(P["shift"].std())},
    "within_point_residual_abs_p50": float(P.resid.median()),
    "spearman_bare": {"median": float(P.sp.median()), "p5": float(P.sp.quantile(.05)), "min": float(P.sp.min())},
    "spearman_notask": {"median": float(P.sp_n.median()), "p5": float(P.sp_n.quantile(.05))},
    "top1_agreement": float(P.top1.mean()), "top2_ordered_agreement": float(P.top2.mean()),
    "chosen_rank_agreement": float(P.rank_same.mean()),
}
# 4 noise floor
rep["noise_floor"] = {"frac_points_fp32_top_gap_lt_0.3": float((P.gap_f < 0.3).mean()),
                      "frac_points_fp32_top_gap_lt_0.1": float((P.gap_f < 0.1).mean()),
                      "top1_agreement_when_gap_ge_0.3": float(P[P.gap_f >= 0.3].top1.mean()),
                      "top1_agreement_when_gap_lt_0.3": float(P[P.gap_f < 0.3].top1.mean())}

# 5 conf shift (M5 sanity row; label = cf_defined only)
lab = pd.read_parquet(os.path.join(D, "labels_L.parquet"))[["traj_id", "turn"]]  # D of the 100-trajectory set
dd = set(map(tuple, lab.values.tolist()))
rng = np.random.default_rng(0)
out = {}
for name, s in (("vllm_bf16", v), ("fp32", f)):
    c = s[s.is_chosen][["traj_id", "turn", "logp_mean"]].copy()
    c["inc"] = [(t, u) in dd for t, u in zip(c.traj_id, c.turn)]
    g = {t: x for t, x in c.groupby("traj_id")}
    T = list(g)
    bs = []
    for _ in range(4000):
        b = pd.concat([g[t] for t in rng.choice(T, len(T))])
        bs.append(b[b.inc].logp_mean.mean() - b[~b.inc].logp_mean.mean())
    out[name] = {"included_minus_excluded": float(c[c.inc].logp_mean.mean() - c[~c.inc].logp_mean.mean()),
                 "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                 "n_inc": int(c.inc.sum()), "n_exc": int((~c.inc).sum()), "median_conf": float(c.logp_mean.median())}
rep["conf_shift_all_points"] = out | {"cwgt": [-0.70, -0.81, -0.59]}
common.write_json(os.path.join(common.ROOT, "eval", "phase2_fp32_check.json"), rep)
print(json.dumps({k: v for k, v in rep.items() if k != "provenance"}, indent=1, default=str))
