"""Phase 3 main table (FEATURES_FROZEN §4-§5; CLAUDE.md §6.3, §8): every arm x boundary.

Boundaries: B1 (D, Live), B2 (𝓛, Fork = het; main), B2cp (𝓛, Credit-pivotal; report), B3 (Credit-pivotal AND het,
sign; unless switched to report-only by D-045), L2sign (𝓛 \\ {A=0}; report).
Per score: AUROC (pre-registered direction for single features), trajectory-cluster bootstrap CI (10,000, common
weights per set), stratified within-trajectory shuffle (1,000; B2/B3/L2sign by progress tercile, B1 unstratified),
paired differences vs V and vs arm A, normalised effect, REV. "Passes" := CI low > shuffle p97.5 AND paired-vs-V
CI low > 0 AND paired-vs-A CI low > 0 (D-048: a difference must favour the arm).
Learned scores are out-of-fold logistic (evalcore) on the frozen feature sets; missing arms (not yet computed) are
reported as missing and the verdict is not issued until every arm is present.
Output eval/main.json (+ eval/main_scores.parquet with every score per point).
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import common
import evalcore as E

PRIOR = {  # FEATURES_FROZEN §3: (B1, B2, B3); B2 also used for B2cp, B3 also for L2sign
    "margin_raw": "--+", "margin_mean": "--+", "margin_detask": "0+0", "entropy_q": "++0", "eff_size": "++0",
    "chosen_prob": "--+", "runner_same_type": "0-0", "top3_n_types": "++0", "loop_chosen": "-0-", "loop_runner": "-0+",
    "alt_revisit": "000", "js_prev": "++0", "sem_dispersion": "++0", "sem_dist_chosen_runner": "0+0",
    "entropy_sampled": "++0", "chosen_conf": "--+", "chosen_freq": "--+",
    "w1_nothing_frac": "0++", "w1_obs_len_sd": "0+0", "w1_adm_delta_sd": "0+0", "w1_task_sim_max": "++0",
    "w1_task_sim_mean": "++0", "target_revealed": "++0", "w1_div_emb": "0+0", "w1_div_n_distinct": "0+0",
    "w1_V_sd": "++0", "w1_V_chosen_minus_mean": "00+", "w1_P_entropy_mean": "+00", "w1_J_progress_frac": "++0",
    "w1_J_regress_frac": "0+0", "w1_div_emb_x_budget": "000", "w1_task_sim_max_x_budget": "000",
    "target_revealed_x_budget": "000", "g_cover": "000", "g_succ_state": "000", "g_succ_alt_sd": "0+0",
    "g_singleton": "000"}
CTRL = ["turn_index", "n_admissible"]
TYPES = [f"type_{t}" for t in common.ACTION_TYPES]
F1 = ["margin_raw", "margin_mean", "chosen_prob", "chosen_rank", "chosen_freq"]
F2 = ["entropy_q", "eff_size", "entropy_sampled"]
F3 = ["runner_same_type", "top3_n_types", "sem_dispersion", "sem_dist_chosen_runner"]
F4 = ["loop_chosen", "loop_runner", "js_prev"]
F5 = ["margin_detask"]
B_CTRL = CTRL + ["chosen_n_tokens", "chosen_conf", "chosen_admissible"] + TYPES
W1P = ["budget_left", "w1_nothing_frac", "w1_obs_len_sd", "w1_adm_delta_sd", "w1_task_sim_max", "w1_task_sim_mean",
       "target_revealed", "w1_div_emb", "w1_div_n_distinct"]
W1X = ["w1_div_emb_x_budget", "w1_task_sim_max_x_budget", "target_revealed_x_budget"]
W1M = ["budget_left", "w1_V_sd", "w1_V_chosen_minus_mean", "w1_P_entropy_mean", "w1_J_progress_frac", "w1_J_regress_frac"]
G = ["g_cover", "g_succ_state", "g_succ_alt_sd", "g_singleton"]
BI = {"B1": 0, "B2": 1, "B2cp": 1, "B3": 2, "L2sign": 2}


def load(tag=""):
    lab = E.labels()
    F = pd.read_parquet(os.path.join(common.ROOT, "data", f"features_B{tag}.parquet"))
    df = lab.merge(F, on=["traj_id", "turn"], how="left", suffixes=("", "_f"))
    for name in ("features_W1", "features_G"):
        p = os.path.join(common.ROOT, "data", f"{name}.parquet")
        if os.path.exists(p):
            X = pd.read_parquet(p).drop(columns=["budget_left"], errors="ignore")
            df = df.merge(X, on=["traj_id", "turn"], how="left")
    pr = os.path.join(common.ROOT, "data", "probes")
    df = df.merge(pd.read_parquet(os.path.join(pr, "V_oof.parquet"))[["traj_id", "turn", "V"]], on=["traj_id", "turn"])
    C = pd.read_parquet(os.path.join(pr, "C_oof.parquet"))
    return df.merge(C, on=["traj_id", "turn"], how="left"), json.load(open(os.path.join(pr, "probes.json")))


def sets(df, b3_report_only):
    L = df[df.live]
    return {"B1": (df, df.live.values, None),
            "B2": (L, L.het.values, "prog3"), "B2cp": (L, L.credit_pivotal.values, "prog3"),
            "B3": (df[df.cap], df[df.cap].b3_pos.values, "prog3"),
            "L2sign": (L[L.A_3456 != 0], (L[L.A_3456 != 0].A_3456 > 0).values, "prog3")}


def c_sel(S, name, info):
    """C component of B+C / model-side check: prompt_end layer chosen per outer fold by inner CV (best mean inner
    AUROC over the C grid), i.e. that layer's out-of-fold score in that fold."""
    fold = E.outer_folds(S.traj_id.values)
    out = np.full(len(S), np.nan)
    for k in range(5):
        best = max((max(info["C"][name][f"C_{name}_prompt_end_{L}"]["inner_auroc"][k]), L) for L in (14, 21, 26))[1]
        out[fold == k] = S[f"C_{name}_prompt_end_{best}"].values[fold == k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="'_vllm' = numerical-robustness rows")
    ap.add_argument("--b3_report_only", action="store_true", help="set by D-045 if the B3 switch fires")
    ap.add_argument("--n_jobs", type=int, default=12)
    args = ap.parse_args()
    t0 = time.time()
    df, pinfo = load(args.tag)
    ceil = json.load(open(os.path.join(common.ROOT, "eval", "ceilings.json")))
    fa = ceil["freeze_addendum"]
    CEIL = {"B1": {m: v["auroc_mean"] for m, v in ceil["B1_live_bank1_label_score_bank2_poolvar"].items()},
            "B2": {m: v["auroc_mean"] for m, v in fa["B2het_on_L_score_G"].items()},
            "B2cp": {m: v["auroc_mean"] for m, v in ceil["B2_pivotal_on_L_score_absA"].items()},
            "B3": {m: v["auroc_mean"] for m, v in fa["B3cap_sign_in_cap_score_A"].items()},
            "L2sign": {m: v["auroc_mean"] for m, v in ceil["L2sign_on_L_score_A"].items()}}
    CMAX = {"B1": CEIL["B1"]["3"], "B2": CEIL["B2"]["6"], "B2cp": CEIL["B2cp"]["6"], "B3": CEIL["B3"]["6"],
            "L2sign": CEIL["L2sign"]["6"]}
    have = {c for c in df.columns if df[c].notna().any()}
    result = {"missing_features": sorted(c for c in PRIOR if c not in have), "boundaries": {}}
    for bname, (S, y, strat) in sets(df, args.b3_report_only).items():
        tb = time.time()
        S = S.reset_index(drop=True)
        g = S.traj_id.values
        strata = S[strat].values if strat else np.zeros(len(S), int)
        fold = E.outer_folds(g)
        boot = E.Boot(g)
        scores, meta = {}, {}
        bi = BI[bname]
        # ---- single scores
        scores["A(-conf)"] = -S.chosen_conf.values
        scores["A(+conf)"] = S.chosen_conf.values
        scores["A_emptythink(-conf)"] = -S.conf_emptythink.values
        scores["A_realized(-conf)"] = -S.conf_realized.values
        scores["H1(rho)"] = S.rho_h1.values
        scores["V"] = S.V.values
        for f, pr in PRIOR.items():
            if f in have:
                d = pr[bi]
                scores[f"single:{f}"] = (-1 if d == "-" else 1) * S[f].values.astype(float)
                meta[f"single:{f}"] = {"prior": d}
        # ---- learned scores
        def learned(name, cols):
            cols = [c for c in cols if c in have]
            miss = [c for c in cols if S[c].isna().all()]
            if not cols or len(cols) == len(CTRL) or miss:
                return
            oof, fi = E.fit_oof_logistic(S[cols].values.astype(float), y, g, fold, n_jobs=args.n_jobs)
            scores[name] = oof
            meta[name] = {"features": cols, "C": fi["C"], "nonconverged_fits": fi["nonconverged_fits"]}
        Bf = F1 + F2 + F3 + F4 + F5 + ["entropy_sampled"]
        Bf = list(dict.fromkeys(Bf)) + B_CTRL
        learned("B_multi", Bf)
        learned("B_multi+hindsight", Bf + ["alt_revisit"])
        for fam, cols in (("F1", F1), ("F2", F2), ("F3", F3), ("F4", F4), ("F5", F5)):
            learned(f"B_family_only:{fam}", cols + B_CTRL)
            learned(f"B_leave_out:{fam}", [c for c in Bf if c not in cols])
        if all(c in have for c in W1P):
            learned("W1pure_multi", CTRL + W1P + W1X)
            learned("W1pure_multi-minus-interactions", CTRL + W1P)
        if all(c in have for c in W1M):
            learned("W1model_multi", CTRL + W1M)
        if all(c in have for c in G):
            learned("G_multi", CTRL + G)
        cn = {"B2": "B2het"}.get(bname, bname)
        for pos in ("prompt_end", "pre_action"):
            for L in (14, 21, 26):
                c = f"C_{cn}_{pos}_{L}"
                if c in S:
                    scores[f"C:{pos}_{L}"] = S[c].values
        if all(f"C_{cn}_prompt_end_{L}" in S for L in (14, 21, 26)):
            scores["C_sel"] = c_sel(S, cn, pinfo)
            # B+C stacking: the C component for training rows comes from inner out-of-fold fits (same layer and C)
            scores["B+C"], meta["B+C"] = bc_stack(S, y, g, fold, Bf, cn, pinfo, args.n_jobs)
        # ---- summaries
        ref = {}
        for r in ("V", "A(-conf)"):
            ref[r] = (scores[r], boot.auroc_dist(y, scores[r]))
        out = {}
        dists = {}
        for name, s in scores.items():
            if np.all(~np.isfinite(np.asarray(s, float))) or np.nanstd(np.asarray(s, float)) == 0:
                out[name] = {"constant_or_missing": True, **meta.get(name, {})}
                continue
            sm = E.summarize(y, s, boot, g, strata, ceiling=CMAX[bname], ref=ref)
            d = sm.pop("_dist")
            dists[name] = d
            sm["REV"] = next((int(m) for m, v in sorted(CEIL[bname].items(), key=lambda x: int(x[0])) if v >= sm["auroc"]), ">6")
            sm["passes"] = bool(sm["ci95"][0] > sm["shuffle_p97.5"]
                                and sm["paired"]["V"]["ci95"][0] > 0 and sm["paired"]["A(-conf)"]["ci95"][0] > 0)
            out[name] = sm | meta.get(name, {})
        extra = {}
        if "B+C" in dists and "W1pure_multi" in dists:  # FREEZE ADDENDUM §3 item 6: model-side (B+C) minus W1-pure
            dd = dists["B+C"] - dists["W1pure_multi"]
            extra["paired_model_minus_w1pure"] = {"diff": out["B+C"]["auroc"] - out["W1pure_multi"]["auroc"],
                                                  "ci95": E.Boot.ci(dd)}
        if "W1pure_multi" in dists and "W1pure_multi-minus-interactions" in dists:  # P4
            dd = dists["W1pure_multi"] - dists["W1pure_multi-minus-interactions"]
            extra["P4_paired_full_minus_no_interactions"] = {
                "diff": out["W1pure_multi"]["auroc"] - out["W1pure_multi-minus-interactions"]["auroc"], "ci95": E.Boot.ci(dd)}
        sc = S[["traj_id", "turn"]].copy()
        for name, s in scores.items():
            sc[name] = np.asarray(s, float)
        sc["y"] = y
        sc.to_parquet(os.path.join(common.ROOT, "eval", f"scores_{bname}{args.tag}.parquet"))
        result["boundaries"][bname] = {"n": len(S), "pos": int(np.sum(y)), "ceiling": CMAX[bname], "ceiling_curve": CEIL[bname],
                                       "scores": out, **extra, "seconds": time.time() - tb}
        print(bname, len(S), f"{time.time() - tb:.0f}s",
              {k: round(v["auroc"], 3) for k, v in out.items() if "auroc" in v and not k.startswith("single:")}, flush=True)
    result["verdict"] = verdict(result, args.b3_report_only)
    result["predictions"] = predictions(result)
    result["provenance"] = common.provenance(__file__) | {"script_sha256": common.file_sha256(os.path.abspath(__file__))}
    common.write_json(os.path.join(common.ROOT, "eval", f"main{args.tag}.json"), result)
    print(json.dumps({"verdict": result["verdict"], "predictions": result["predictions"]}, indent=1, default=str))
    print(f"done {time.time() - t0:.0f}s")


def bc_stack(S, y, g, fold, Bf, cn, pinfo, n_jobs):
    """B+C: logistic on B features + C score; for each outer fold the C score of training rows is an inner
    out-of-fold prediction of the inner-CV-selected prompt_end probe (layer, C), the test rows use that layer's outer
    out-of-fold score (D-048)."""
    from sklearn.model_selection import GroupKFold
    import importlib
    P = importlib.import_module("31_probes")
    keys = list(zip(S.traj_id, S.turn))
    oof = np.full(len(S), np.nan)
    chosen = []
    for k in range(5):
        cand = [(max(pinfo["C"][cn][f"C_{cn}_prompt_end_{L}"]["inner_auroc"][k]), L,
                 pinfo["C"][cn][f"C_{cn}_prompt_end_{L}"]["C"][k]) for L in (14, 21, 26)]
        _, L, Cc = max(cand)
        chosen.append((L, Cc))
        tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
        H = P.hidden("prompt_end", L, keys)
        ctr = np.full(len(tr), np.nan)
        for a, b in GroupKFold(5).split(tr, groups=g[tr]):
            p = E.Prep().fit(H[tr[a]])
            bas = E._Basis(p.transform(H[tr[a]]))
            ctr[b] = bas.fit(y[tr[a]], Cc).decision_function(bas.project(p.transform(H[tr[b]])))
        cte = S[f"C_{cn}_prompt_end_{L}"].values[te]
        Xtr = np.column_stack([S[Bf].values[tr].astype(float), ctr])
        Xte = np.column_stack([S[Bf].values[te].astype(float), cte])
        sub_fold = E.outer_folds(g[tr])
        # choose C for the stacked logistic by inner CV on the training rows
        _, fi = E.fit_oof_logistic(Xtr, y[tr], g[tr], sub_fold, n_jobs=n_jobs)
        Cst = max(set(fi["C"]), key=fi["C"].count)
        p = E.Prep().fit(Xtr)
        bas = E._Basis(p.transform(Xtr))
        oof[te] = bas.fit(y[tr], Cst).decision_function(bas.project(p.transform(Xte)))
    return oof, {"features": Bf + ["C_sel(prompt_end)"], "layer_C_per_fold": chosen}


def passes(res, b, name):
    s = res["boundaries"].get(b, {}).get("scores", {}).get(name)
    return bool(s and s.get("passes"))


def verdict(res, b3_report_only):
    """CLAUDE.md §8 with the FREEZE ADDENDUM priority rule; other readings in table order (D-048)."""
    need = ["B_multi", "C_sel", "B+C", "W1pure_multi", "W1model_multi", "G_multi"]
    miss = [n for n in need if n not in res["boundaries"]["B2"]["scores"]]
    if miss:
        return {"issued": False, "reason": f"arms missing: {miss}"}
    model = ("B_multi", "C_sel", "B+C")
    model_b1 = any(passes(res, "B1", n) for n in model)
    model_b2 = any(passes(res, "B2", n) for n in model)
    w1p_b2 = passes(res, "B2", "W1pure_multi")
    w1m_b2 = passes(res, "B2", "W1model_multi")
    any_b1 = any(v.get("passes") for v in res["boundaries"]["B1"]["scores"].values())
    any_b2 = any(v.get("passes") for v in res["boundaries"]["B2"]["scores"].values())
    b3_pass = [] if b3_report_only else [n for n, v in res["boundaries"]["B3"]["scores"].items() if v.get("passes")]
    b3_ok = set(b3_pass) <= {"W1pure_multi", "W1model_multi", "W1pure_multi-minus-interactions"} or \
        all(n.startswith("single:w1_") or n.startswith("W1") for n in b3_pass)
    ladder = model_b1 and w1p_b2 and b3_ok
    mdl = model_b1 and model_b2
    if ladder and mdl:
        ci = res["boundaries"]["B2"]["paired_model_minus_w1pure"]["ci95"]
        reading = "R-MODEL" if ci[0] > 0 else ("R-LADDER" if ci[1] < 0 else "R-EITHER")
    elif ladder:
        reading = "R-LADDER"
    elif mdl:
        reading = "R-MODEL"
    elif (w1p_b2 or w1m_b2) and not model_b2:
        reading = "R-WORLD" + ("" if w1p_b2 else " (model)")
    elif not any_b2:
        reading = "R-REPLAY-ONLY"
    elif not any_b1:
        reading = "R-NULL"
    else:
        reading = "none of the table rows (see flags)"
    return {"issued": True, "reading": reading,
            "flags": {"model_B1": model_b1, "model_B2": model_b2, "W1pure_B2": w1p_b2, "W1model_B2": w1m_b2,
                      "any_B1": any_b1, "any_B2": any_b2, "B3_passing": b3_pass, "B3_condition_ok": b3_ok,
                      "B3_report_only": b3_report_only,
                      "paired_model_minus_w1pure_B2": res["boundaries"]["B2"].get("paired_model_minus_w1pure")}}


def predictions(res):
    out = {}
    b = res["boundaries"]
    try:
        e1, e2 = b["B1"]["scores"]["single:entropy_q"], b["B2"]["scores"]["single:entropy_q"]
        out["P1"] = {"holds": bool(e1["passes"] and e2["norm_effect_ci95"][0] <= 0 <= e2["norm_effect_ci95"][1]),
                     "B1_passes": e1["passes"], "B2_norm_effect": e2["norm_effect"], "B2_norm_ci": e2["norm_effect_ci95"]}
    except KeyError as ex:
        out["P1"] = {"missing": str(ex)}
    for key, (a, c, thr) in {"P2": ("single:target_revealed", "W1pure_multi", 0.5),
                              "P3": ("W1pure_multi", "W1model_multi", 0.5)}.items():
        try:
            na, nc = b["B2"]["scores"][a]["norm_effect"], b["B2"]["scores"][c]["norm_effect"]
            out[key] = {"holds": bool(na >= thr * nc), "lhs": na, "rhs": nc}
        except KeyError as ex:
            out[key] = {"missing": str(ex)}
    p4 = b["B2"].get("P4_paired_full_minus_no_interactions")
    out["P4"] = {"holds": bool(p4["ci95"][0] > 0), **p4} if p4 else {"missing": "W1pure arms"}
    out["F1"] = {"masked_share_of_forks": 0.642, "ci95": [0.592, 0.690], "source": "D-042"}
    return out


if __name__ == "__main__":
    main()
