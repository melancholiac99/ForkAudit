"""Phase 2 (FEATURES_FROZEN §2-§3): arm-B features 1-19 + control columns, arm A conf variants, arm H1 rho_t,
feature 20 budget_left. One row per rollout step (all 3,818; neighbours need t-1 / t+1..t+5).

Inputs: candidate scores (default data/scores_fp32.parquet, the primary backend, D-041; --tag _vllm uses
data/scores.parquet for the numerical-robustness rows), chosen variants (scores_chosen*.parquet), rollouts, bank-1
drafts (first batch of 32, D-008), frozen sentence encoder BAAI/bge-small-en-v1.5 (normalised embeddings).
Deterministic choices not spelled out in CLAUDE.md §4 (D-046): a1/a2 = top-2 by logp_sum (ties: candidate order);
margin_mean ranks by logp_mean; JS in nats over the union of candidate sets; sem_dispersion =
sum_{i!=j} q_i q_j d_ij / sum_{i!=j} q_i q_j; entropy_sampled NaN if no admissible draft; chosen_freq counts all 32
first-batch drafts whose parsed action equals chosen.
Output data/features_B{tag}.parquet (+ .provenance.json). No label is read.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import common

TYPES = common.ACTION_TYPES


def softmax(x):
    z = np.exp(x - x.max())
    return z / z.sum()


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def js(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        k = a > 0
        return float((a[k] * np.log(a[k] / b[k])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="'' = fp32 primary; '_vllm' = vLLM bf16 sensitivity")
    args = ap.parse_args()
    sp = {"": "scores_fp32.parquet", "_vllm": "scores.parquet"}[args.tag]
    cp = {"": "scores_chosen_fp32.parquet", "_vllm": "scores_chosen.parquet"}[args.tag]
    S = pd.read_parquet(os.path.join(common.ROOT, "data", sp))
    X = pd.read_parquet(os.path.join(common.ROOT, "data", cp))
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(os.path.join(os.path.dirname(common.MODEL_PATH), "bge-small-en-v1.5"), device="cpu")
    emb_cache = {}

    def emb(texts):
        new = [t for t in set(texts) if t not in emb_cache]
        if new:
            for t, e in zip(new, enc.encode(new, normalize_embeddings=True, batch_size=256)):
                emb_cache[t] = e
        return np.stack([emb_cache[t] for t in texts])

    t0 = time.time()
    rows = []
    groups = {k: g for k, g in S.groupby(["traj_id", "turn"], sort=False)}
    xv = {(r.traj_id, r.turn, r.variant): r for r in X.itertuples()}
    for p in sorted(os.listdir(os.path.join(common.ROOT, "data", "rollout"))):
        tr = json.load(open(os.path.join(common.ROOT, "data", "rollout", p)))
        chosen_seq = [s["action"] for s in tr["steps"]]
        prev_q = None
        per_traj = []
        for i, stp in enumerate(tr["steps"]):
            t = stp["turn"]
            g = groups[(tr["traj_id"], t)]
            acts = g.action.tolist()
            lp, lm, lpn = g.logp_sum.values, g.logp_mean.values, g.logp_sum_notask.values
            q = softmax(lp)
            order = np.argsort(-lp, kind="stable")
            a1, a2 = order[0], (order[1] if len(order) > 1 else order[0])
            om = np.argsort(-lm, kind="stable")
            chosen = stp["action"]
            ci = acts.index(chosen)
            types = [common.action_type(a) for a in acts]
            E = emb(acts)
            D = 1 - E @ E.T
            W = np.outer(q, q)
            np.fill_diagonal(W, 0)
            sem_disp = float((W * D).sum() / W.sum()) if W.sum() > 0 else 0.0
            if prev_q is None:
                jsd = np.nan
            else:
                u = sorted(set(acts) | set(prev_q))
                pa = np.array([dict(zip(acts, q)).get(a, 0.0) for a in u])
                pb = np.array([prev_q.get(a, 0.0) for a in u])
                jsd = js(pa, pb)
            prev_q = dict(zip(acts, q))
            b1 = json.load(open(os.path.join(common.ROOT, "data", "replay", tr["traj_id"], f"t{t:02d}.json")))
            first = [d for d in b1["drafts"] if d["batch"] == 0]
            assert len(first) == 32, (tr["traj_id"], t, len(first))
            adm_dr = [d["action"] for d in first if d["admissible"]]
            if adm_dr:
                _, cnt = np.unique(adm_dr, return_counts=True)
                ent_s = entropy(cnt / cnt.sum())
            else:
                ent_s = np.nan
            c = sum(d["action"] == chosen for d in first)
            n_adm = len([a for a in stp["admissible"] if a != "help"])
            row = {"traj_id": tr["traj_id"], "turn": t, "seed": tr["seed"],
                   "margin_raw": float(lp[a1] - lp[a2]), "margin_mean": float(lm[om[0]] - lm[om[1]] if len(om) > 1 else 0.0),
                   "margin_detask": float((lp[a1] - lpn[a1]) - (lp[a2] - lpn[a2])),
                   "entropy_q": entropy(q), "eff_size": float(np.exp(entropy(q))), "chosen_prob": float(q[ci]),
                   "chosen_rank": int(np.where(order == ci)[0][0]) + 1,
                   "runner_same_type": int(types[a2] == types[ci]),
                   "top3_n_types": len({types[j] for j in order[:3]}),
                   "loop_chosen": int(chosen in chosen_seq[max(0, i - 5):i]),
                   "loop_runner": int(acts[a2] in chosen_seq[max(0, i - 5):i]),
                   "alt_revisit": int(acts[a2] in chosen_seq[i + 1:i + 6]),
                   "js_prev": jsd, "sem_dispersion": sem_disp, "sem_dist_chosen_runner": float(D[ci, a2]),
                   "turn_index": t, "n_admissible": n_adm, "chosen_n_tokens": int(g.n_tokens.values[ci]),
                   "chosen_type": types[ci], "entropy_sampled": ent_s,
                   "chosen_conf": float(lm[ci]),
                   "chosen_freq": float(np.log((c + 0.5) / (32 + 0.5 * len(acts)))),
                   "chosen_admissible": int(chosen in stp["admissible"]),
                   "budget_left": common.MAX_STEPS - t,
                   "n_candidates": len(acts), "top1_action": acts[a1], "runner_action": acts[a2]}
            for v in ("emptythink", "realized", "hindsight"):
                r = xv.get((tr["traj_id"], t, v))
                ok = r is not None and bool(r.available)
                row[f"conf_{v}"] = float(r.logp_mean) if ok else np.nan
                if v == "hindsight":
                    row["_h_sum"] = float(r.logp_sum) if ok else np.nan
                    row["_h_n"] = int(r.n_tokens) if ok else 0
            per_traj.append(row)
        qh = np.array([np.exp(r["_h_sum"] / (5.0 * r["_h_n"])) if r["_h_n"] else np.nan for r in per_traj])
        mq = np.nanmean(qh)
        for r, v in zip(per_traj, qh):
            r["rho_h1"] = float(np.clip(v / mq, 0.8, 1.2)) if np.isfinite(v) else np.nan
            r["q_h1"] = float(v)
            del r["_h_sum"], r["_h_n"]
        rows.extend(per_traj)
    df = pd.DataFrame(rows)
    for tname in TYPES:
        df[f"type_{tname}"] = (df.chosen_type == tname).astype(int)
    df["arm_A_score"] = -df.chosen_conf
    out = os.path.join(common.ROOT, "data", f"features_B{args.tag}.parquet")
    df.to_parquet(out)
    nan = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}
    common.write_json(out.replace(".parquet", ".provenance.json"),
                      {"provenance": common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))},
                       "scores": sp, "chosen_scores": cp, "n_rows": len(df), "nan_counts": nan,
                       "chosen_rank1_frac": float((df.chosen_rank == 1).mean()), "seconds": time.time() - t0})
    print(f"wrote {out}: {len(df)} rows, {time.time() - t0:.0f}s; NaN {nan}; chosen_rank==1 {(df.chosen_rank == 1).mean():.3f}")


if __name__ == "__main__":
    main()
