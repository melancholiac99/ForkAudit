"""Phase 2 prereq follow-up: size and structure of HF vs vLLM candidate log-prob differences (20 alignment points)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402

P = os.path.join(common.ROOT, "eval", "prereq")
h = pd.read_parquet(os.path.join(P, "scores_hf.parquet"))
v = pd.read_parquet(os.path.join(P, "scores_vllm.parquet"))
m = v.merge(h, on=["traj_id", "turn", "action"], suffixes=("_v", "_h"))
d = m.logp_sum_v - m.logp_sum_h
print("pairs", len(m))
print("delta logp_sum (vllm - hf): mean %.4f | |delta| p50 %.4f p90 %.4f p99 %.4f max %.3f"
      % (d.mean(), d.abs().median(), d.abs().quantile(.9), d.abs().quantile(.99), d.abs().max()))
print("corr(delta, logp_sum_hf) = %.3f" % np.corrcoef(d, m.logp_sum_h)[0, 1])
print("largest deviations:")
print(m.loc[d.abs().sort_values().index[-4:], ["traj_id", "turn", "action", "logp_sum_h", "logp_sum_v"]].to_string())
for (t, u), g in m.groupby(["traj_id", "turn"]):
    a = g.sort_values("logp_sum_h", ascending=False)
    b = g.sort_values("logp_sum_v", ascending=False)
    if a.action.iloc[0] != b.action.iloc[0]:
        gap = a.logp_sum_h.iloc[0] - a.logp_sum_h.iloc[1]
        print("top1 differs %s t%d: HF top1-top2 gap %.3f nats" % (t, u, gap))

# ---- fp32 reference (HF, same token ids): which bf16 backend is closer?
r = pd.read_parquet(os.path.join(P, "scores_hf32.parquet"))
from scipy.stats import spearmanr  # noqa: E402
mm = m.merge(r[["traj_id", "turn", "action", "logp_sum", "logp_sum_notask"]], on=["traj_id", "turn", "action"])
for name, col in (("vllm-bf16", "logp_sum_v"), ("hf-bf16", "logp_sum_h")):
    e = mm[col] - mm.logp_sum
    rho = mm.groupby(["traj_id", "turn"]).apply(lambda g: spearmanr(g[col], g.logp_sum)[0])
    print("%-9s vs fp32: mean %.4f | |err| p50 %.4f p90 %.4f max %.3f | per-point Spearman median %.4f min %.4f"
          % (name, e.mean(), e.abs().median(), e.abs().quantile(.9), e.abs().max(), rho.median(), rho.min()))
