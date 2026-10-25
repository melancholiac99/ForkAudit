"""a'_u calibration (label-only): flag rate of the same 90% Newcombe rule when there is no real difference,
and when the alternative is one of the policy's own alternatives (same point set as a'_u)."""
import importlib.util, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common  # noqa: E402
spec = importlib.util.spec_from_file_location("reg", os.path.join(common.ROOT, "scripts", "15_regimes.py"))
reg = importlib.util.module_from_spec(spec); spec.loader.exec_module(reg)
pts = json.load(open(os.path.join(common.ROOT, "eval", "au_points.json")))["points"]
R = lambda b, t, u: json.load(open(os.path.join(common.ROOT, "data", b, t, f"t{u:02d}.json")))  # noqa: E731
null66, pol, n = [], [], 0
for t, u, _ in pts:
    rs = [R(b, t, u) for b in ("replay_b2", "replay_b3", "replay_b4", "replay_b5")]
    f = [sum(r["factual_outcomes"]) for r in rs]                     # 3 runs per bank
    lo, hi = reg.newcombe(np.array([f[0] + f[1]]), np.array([6]), np.array([f[2] + f[3]]), np.array([6]))
    null66.append(bool(lo[0] > 0 or hi[0] < 0))                       # factual vs factual, 6 vs 6
    for j in range(4):                                                # policy alternative j: banks 3+4 (6) vs factual 12
        x = sum(rs[0]["alt_outcomes"][j]) + sum(rs[1]["alt_outcomes"][j])
        lo, hi = reg.newcombe(np.array([sum(f)]), np.array([12]), np.array([x]), np.array([6]))
        pol.append(bool(lo[0] > 0 or hi[0] < 0))
print(f"same rule, factual(6) vs factual(6) [true diff = 0]: flag rate {np.mean(null66):.3f} (n={len(null66)})")
print(f"same rule, factual(12) vs ONE policy alternative(6): flag rate {np.mean(pol):.3f} (n={len(pol)} pairs)")
print("a'_u (uniform admissible, 6) vs factual(12): flag rate 0.110 (eval/au_sensitivity.json)")
