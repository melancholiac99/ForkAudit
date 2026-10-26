"""Phase 2 (FEATURES_FROZEN §3): W1 features from the five peeks (factual + 4 bank-1 alternatives) of each point in D.

--part pure  : 21-28 (world text only) + interactions 34-36 (x budget_left)            -> data/features_W1.parquet
--part model : 29-33 (V probe on peek-state hidden states, policy entropy at peek states, W1-J labels), merged into
               the same file (needs hidden_peek/, data/scores_peek_fp32.parquet, data/w1_judge.parquet and the V
               out-of-fold models from 31_probes.py).
Deterministic choices (D-046): standard deviations use ddof = 0; "Nothing happens." compared after strip();
embeddings = bge-small-en-v1.5, normalised; target word per FEATURES_FROZEN §3 (four regexes, \\bX \\d+\\b match on the
lower-cased observation). No label is read.
"""
import argparse
import itertools
import json
import os
import re
import time

import numpy as np
import pandas as pd

import common

D_ = r"(?:a |an |some |two |the )?"
TARGET_PATS = [rf"^(?:clean|heat|cool) {D_}(\w+) and put", rf"^find {D_}(\w+) and put",
               rf"^put {D_}(?:clean |hot |cool |cold )?(\w+) (?:in|on|into)\b",
               rf"^(?:look at|examine) {D_}(\w+) (?:under|with)\b"]


def target_word(task):
    t = task.lower().strip().rstrip(".")
    for p in TARGET_PATS:
        m = re.search(p, t)
        if m:
            return m.group(1)
    return None


def iter_points():
    for p in sorted(os.listdir(os.path.join(common.ROOT, "data", "rollout"))):
        tr = json.load(open(os.path.join(common.ROOT, "data", "rollout", p)))
        for stp in tr["steps"]:
            b1 = json.load(open(os.path.join(common.ROOT, "data", "replay", tr["traj_id"], f"t{stp['turn']:02d}.json")))
            if not b1["cf_defined"]:
                continue
            pk = json.load(open(os.path.join(common.ROOT, "data", "peek_alts", tr["traj_id"], f"t{stp['turn']:02d}.json")))
            assert pk["checks"]["factual_obs_match"] and pk["checks"]["refeed_div_total"] == 0
            peeks = sorted(pk["peeks"], key=lambda q: (q["role"] != "factual", -1 if q["alt_index"] is None else q["alt_index"]))
            assert len(peeks) == 5 and peeks[0]["role"] == "factual"
            yield tr, stp, peeks


def part_pure():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(os.path.join(os.path.dirname(common.MODEL_PATH), "bge-small-en-v1.5"), device="cpu")
    pts = list(iter_points())
    texts = sorted({q["obs"] for _, _, pk in pts for q in pk} | {tr["task"] for tr, _, _ in pts})
    E = dict(zip(texts, enc.encode(texts, normalize_embeddings=True, batch_size=256)))
    rows, n_nan_target = [], 0
    for tr, stp, pk in pts:
        obs = [q["obs"] for q in pk]
        e = np.stack([E[o] for o in obs])
        sims = e @ E[tr["task"]]
        n_adm = len([a for a in stp["admissible"] if a != "help"])
        X = target_word(tr["task"])
        if X is None:
            n_nan_target += 1
            rev = np.nan
        else:
            rx = re.compile(rf"\b{re.escape(X)} \d+\b")
            rev = float(any(rx.search(o.lower()) for o in obs))
        div = float(np.mean([1 - e[i] @ e[j] for i, j in itertools.combinations(range(5), 2)]))
        b = common.MAX_STEPS - stp["turn"]
        rows.append({"traj_id": tr["traj_id"], "turn": stp["turn"],
                     "w1_nothing_frac": float(np.mean([o.strip() == "Nothing happens." for o in obs[1:]])),
                     "w1_obs_len_sd": float(np.std([len(o) for o in obs])),
                     "w1_adm_delta_sd": float(np.std([len(q["admissible_after"]) - n_adm for q in pk])),
                     "w1_task_sim_max": float(sims.max()), "w1_task_sim_mean": float(sims.mean()),
                     "target_revealed": rev, "w1_div_emb": div, "w1_div_n_distinct": len(set(obs)),
                     "budget_left": b, "w1_div_emb_x_budget": div * b, "w1_task_sim_max_x_budget": float(sims.max()) * b,
                     "target_revealed_x_budget": rev * b if rev == rev else np.nan,
                     "target_word": X})
    df = pd.DataFrame(rows)
    return df, {"n_points": len(df), "target_word_nan_points": n_nan_target,
                "target_revealed_rate": float(df.target_revealed.mean()), "n_unique_texts": len(texts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["pure", "model"], required=True)
    args = ap.parse_args()
    t0 = time.time()
    out = os.path.join(common.ROOT, "data", "features_W1.parquet")
    prov = common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}
    if args.part == "pure":
        df, info = part_pure()
        df.to_parquet(out)
        common.write_json(out.replace(".parquet", ".provenance.json"), {"pure": {"provenance": prov, **info,
                                                                              "seconds": time.time() - t0}})
        print(info, f"{time.time() - t0:.0f}s")
    else:
        df, info = part_model()
        df.to_parquet(out)
        pj = json.load(open(out.replace(".parquet", ".provenance.json")))
        pj["model"] = {"provenance": prov, **info, "seconds": time.time() - t0}
        common.write_json(out.replace(".parquet", ".provenance.json"), pj)
        print(info, f"{time.time() - t0:.0f}s")


def part_model():
    """29-30: V (the point's own out-of-fold models, mean of the bank-1 and bank-2 fits) on the five peek states;
    peek-state controls = (turn + 1, |peek admissible set|). 31: mean over the peeks of the entropy of the softmax of
    peek-state candidate logp_sum. 32-33: W1-J label fractions over parsed peeks (NaN if none parsed)."""
    df = pd.read_parquet(os.path.join(common.ROOT, "data", "features_W1.parquet"))
    pr = os.path.join(common.ROOT, "data", "probes")
    fold = pd.read_parquet(os.path.join(pr, "V_oof.parquet")).set_index(["traj_id", "turn"]).fold
    M = np.load(os.path.join(pr, "V_models.npz"))
    idx = pd.read_parquet(os.path.join(common.ROOT, "hidden_peek", "index.parquet"))
    H = {L: np.load(os.path.join(common.ROOT, "hidden_peek", f"prompt_end_{L}.npy"), mmap_mode="r") for L in (14, 21, 26)}
    sc = pd.read_parquet(os.path.join(common.ROOT, "data", "scores_peek_fp32.parquet"))
    ent = sc.groupby(["traj_id", "turn", "role", "alt_index"]).logp_sum.apply(
        lambda x: float(-(np.exp(x - x.max()) / np.exp(x - x.max()).sum() * np.log(np.exp(x - x.max()) / np.exp(x - x.max()).sum() + 1e-300)).sum()))
    nadm = sc.groupby(["traj_id", "turn", "role", "alt_index"]).size()
    J = pd.read_parquet(os.path.join(common.ROOT, "data", "w1_judge.parquet"))
    vpk = np.full(len(idx), np.nan)
    for k in range(5):
        sel = np.array([fold.get((t, u), -1) == k for t, u in zip(idx.traj_id, idx.turn)])
        if not sel.any():
            continue
        ctrl = np.column_stack([idx.turn.values[sel] + 1,
                                [nadm.get((t, u, r, a), 0) for t, u, r, a in
                                 zip(idx.traj_id[sel], idx.turn[sel], idx.role[sel], idx.alt_index[sel])]]).astype(float)
        v = []
        for tgt in ("pool_b1", "pool_b2"):
            key = f"{tgt}_k{k}"
            L = int(M[f"{key}__layer"])
            X = np.hstack([ctrl, np.asarray(H[L][idx.row.values[sel]], dtype=np.float64)])
            Z = ((X - M[f"{key}__mu"]) / np.where(M[f"{key}__sd"] > 1e-12, M[f"{key}__sd"], 1.0))[:, M[f"{key}__keep"].astype(bool)]
            v.append(Z @ M[f"{key}__w"] + float(M[f"{key}__ym"]))
        vpk[sel] = 0.5 * (v[0] + v[1])
    idx["V_peek"] = vpk
    idx["P_ent"] = [ent.get((t, u, r, a), np.nan) for t, u, r, a in zip(idx.traj_id, idx.turn, idx.role, idx.alt_index)]
    g = idx.groupby(["traj_id", "turn"])
    vsd = g.V_peek.std(ddof=0)
    vfac = idx[idx.role == "factual"].set_index(["traj_id", "turn"]).V_peek
    valt = idx[idx.role != "factual"].groupby(["traj_id", "turn"]).V_peek.mean()
    pent = g.P_ent.mean()
    Jp = J[J.label != "unparsed"].groupby(["traj_id", "turn"]).label
    prog = Jp.apply(lambda x: float((x == "progress").mean()))
    regr = Jp.apply(lambda x: float((x == "regress").mean()))
    key = list(zip(df.traj_id, df.turn))
    df["w1_V_sd"] = [vsd.get(k, np.nan) for k in key]
    df["w1_V_chosen_minus_mean"] = [vfac.get(k, np.nan) - valt.get(k, np.nan) for k in key]
    df["w1_P_entropy_mean"] = [pent.get(k, np.nan) for k in key]
    df["w1_J_progress_frac"] = [prog.get(k, np.nan) for k in key]
    df["w1_J_regress_frac"] = [regr.get(k, np.nan) for k in key]
    info = {"n_points": len(df), "peek_states": int(len(idx)), "V_peek_nan": int(np.isnan(vpk).sum()),
            "judge_label_counts": J.label.value_counts().to_dict(),
            "nan": {c: int(df[c].isna().sum()) for c in ("w1_V_sd", "w1_V_chosen_minus_mean", "w1_P_entropy_mean",
                                                         "w1_J_progress_frac", "w1_J_regress_frac")}}
    return df, info


if __name__ == "__main__":
    main()
