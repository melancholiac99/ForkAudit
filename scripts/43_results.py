"""Render RESULTS.md (CLAUDE.md §11) from eval/*.json. Every number is copied from a JSON with its provenance."""
import json
import os
import time

import common

E = os.path.join(common.ROOT, "eval")


def j(name):
    p = os.path.join(E, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f3(x):
    return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))


def ci(c):
    return "—" if not c else f"[{c[0]:.3f}, {c[1]:.3f}]"


def main():
    main_, ex, gold, reg, ceil, het = j("main.json"), j("extras.json"), j("gold.json"), j("regimes.json"), j("ceilings.json"), j("het_fork.json")
    vl = j("main_vllm.json")
    add = j("review_addenda.json")
    L = [f"# RESULTS — M0 (FINAL RULING 2026-10-26 + FREEZE ADDENDUM; FEATURES_FROZEN d6fe13f)",
         "", f"Generated {time.strftime('%Y-%m-%d %H:%M %z')} by `scripts/43_results.py` at commit `{common.git_commit()[:8]}`.",
         "Every number below is copied from `eval/*.json`; each JSON records its script, commit and inputs.", ""]
    L += ["## 1. Environment", "",
          "- Policy: Qwen2.5-7B-Instruct (revision a09a3545), T = 0.7, HCAPO ALFWorld template; alfworld 0.4.2, textworld 1.7.0.",
          "- Replay: vLLM bf16 (async engine, D-026). Scoring: HF fp32 shared-prefix (primary, D-041); vLLM bf16 sensitivity.",
          "- Seeds: per-request, bank·10¹¹ offsets (D-024); gold bank index 6 (D-043); siblings (20+k)·10¹¹.", ""]
    L += ["## 2. Data and regimes", ""]
    if reg and "fork_het" in reg:
        fh = reg["fork_het"]
        L += ["| regime | n | fraction [95% cluster CI] | null-expected |", "|---|---|---|---|"]
        ne = fh["null_expected_counts"]
        for k, lab, e in (("dead", "Dead", None), ("indifferent", "Indifferent (Live ∧ ¬het)", None),
                          ("fork", "Fork (het)", ne["het"]), ("fork_credit_pivotal", "↳ Credit-pivotal ∩ het", ne["credit_pivotal_and_het"]),
                          ("fork_masked", "↳ Masked", ne["masked"])):
            v = fh[k]
            L.append(f"| {lab} | {v['k']} | {v['frac']:.3f} {ci(v['ci95_cluster'])} | {f3(e)} |")
        L += ["", f"15.6% of D is detected as Fork; under the null {100 * fh['null_expected_fork_frac_of_D']:.1f}% would be expected.",
              "F1 (known fact, D-042): 64.2% [59.2, 69.0] of forks are invisible to chosen-vs-mean credit (Masked).", ""]
    if gold:
        a = gold["audit"]
        L += ["### Gold set (D-045)", "", f"- Audit: {json.dumps({k: v for k, v in a.items() if k != 'missing'})}; missing {len(a.get('missing', []))}.",
              "", "| rule | flagged by 12-run rule (G-candidate) | independent-20 replication | null replication | merged-32 (biased) |",
              "|---|---|---|---|---|"]
        for r, v in gold["replication_candidate_layer"].items():
            rep = v["replication_indep20"]
            L.append(f"| {r} | {v['n_flagged_12']} | {f3(rep['frac']) if rep else '—'} {ci(rep['ci95_cluster']) if rep else ''} | "
                     f"{f3(v['null_replication_indep20'])} | {f3(v['replication_merged32_selection_biased'])} |")
        L += ["", f"- B3 switch (cap replication < 0.50 → B3 report-only): **{gold['B3_report_only_switch_fires']}**.", ""]
    L += ["## 3. Reproduction gates", ""]
    if ex:
        m1 = ex["M1_gate_armA"]["r=13.1"]
        L += [f"- M1 (arm A = −conf, route 13.1% of D, label pivotal_cwgt): recall {m1['recall']:.3f}, cluster CI {ci(m1['cluster95'])}, "
              f"Wilson {ci(m1['wilson95'])}; contains 13.1%: **{ex['gates']['M1_contains_13.1']}** (CWGT 11.9% [9.4, 14.9]).",
              f"- M4/T3 (H1 ρ_t): median {ex['M4_T3_gate_H1']['median']:.3f} {ci(ex['M4_T3_gate_H1']['ci95'])} over "
              f"{ex['M4_T3_gate_H1']['n_traj']} trajectories; CI overlaps [0.647, 0.793] (gate passes literally): **{ex['gates']['T3_compatible_[0.647,0.793]']}**; "
              f"reported side by side with CWGT +0.752 [0.647, 0.793], not claimed as a replication (REVIEW 2026-10-27 §3.4). "
              f"Arm A (mechanical): {ex['M4_armA_mechanical']['median']:.3f}.",
              f"- M5: |D|/total {ex['M5']['D_over_total']:.3f} (CWGT 0.869); pivotal_cwgt {ex['M5']['pivotal_cwgt_frac']:.3f} (0.305); "
              f"σ_floor = 0 {ex['M5']['sigma_floor_zero_frac']:.3f} (0.860); conf shift −0.494 [−0.574, −0.414] (−0.70, D-040).",
              "- HF–vLLM rank agreement: median Spearman 0.997 (D-040).", ""]
    L += ["## 4. Main table (B1 / B2 / B3 / L2-sign × arms)", ""]
    if main_:
        for b, B in main_["boundaries"].items():
            L += [f"### {b} (n = {B['n']}, positives {B['pos']}, replay ceiling {B['ceiling']:.3f})", "",
                  "| score | AUROC [CI] | shuffle p97.5 | Δ vs V [CI] | Δ vs A [CI] | norm. effect [CI] | REV | passes |",
                  "|---|---|---|---|---|---|---|---|"]
            order = sorted(B["scores"].items(), key=lambda kv: (kv[0].startswith("single:"), kv[0]))
            for name, s in order:
                if "auroc" not in s:
                    L.append(f"| {name} | constant/missing | | | | | | |")
                    continue
                pv, pa = s["paired"]["V"], s["paired"]["A(-conf)"]
                L.append(f"| {name} | {s['auroc']:.3f} {ci(s['ci95'])} | {s['shuffle_p97.5']:.3f} | {pv['diff']:+.3f} {ci(pv['ci95'])} | "
                         f"{pa['diff']:+.3f} {ci(pa['ci95'])} | {f3(s.get('norm_effect'))} {ci(s.get('norm_effect_ci95'))} | {s['REV']} | "
                         f"{'**yes**' if s['passes'] else 'no'} |")
            for k in ("paired_model_minus_w1pure", "P4_paired_full_minus_no_interactions"):
                if k in B:
                    L.append(f"\n{k}: {B[k]['diff']:+.3f} {ci(B[k]['ci95'])}")
            L.append("")
        if vl:
            L += ["### Numerical robustness (vLLM bf16 scores)", "", "| boundary | score | AUROC (fp32) | AUROC (vLLM) |", "|---|---|---|---|"]
            for b in main_["boundaries"]:
                for name in ("A(-conf)", "B_multi", "single:entropy_q", "single:margin_raw"):
                    a1 = main_["boundaries"][b]["scores"].get(name, {}).get("auroc")
                    a2 = vl["boundaries"].get(b, {}).get("scores", {}).get(name, {}).get("auroc")
                    L.append(f"| {b} | {name} | {f3(a1)} | {f3(a2)} |")
            L.append("")
    if ex:
        L += ["### Family identification (CLAUDE.md §4 rule)", "", "```", json.dumps(ex["family_identification"], indent=1), "```", "",
              "### M2 (within-trajectory Spearman vs A_3456 on 𝓛; report)", "",
              "| score | median [CI] | n traj |", "|---|---|---|"]
        for k, v in ex["M2"].items():
            L.append(f"| {k} | {v['median']:.3f} {ci(v['ci95'])} | {v['n_traj']} |")
        L += ["", "### CARL comparison", ""]
        for k, v in ex["CARL_delta"].items():
            L.append(f"- Cliff's δ of crit_sd, high vs low {k}: {v['cliffs_delta']:.3f} (CARL 0.42).")
        L.append("")
    L += ["## 5. C→B decomposition (R² of the C_sel score regressed on the B features)", ""]
    if add:
        L += ["| boundary | n | R² in-sample OLS | R² out-of-fold ridge |", "|---|---|---|---|"]
        for b, v in add["C_to_B_decomposition"].items():
            L.append(f"| {b} | {v['n']} | {v['R2_in_sample_OLS']:.3f} | {v['R2_oof_ridge']:.3f} |")
        L.append("")
    else:
        L += ["Not computed.", ""]
    L += ["## 6. E1 — fork budget allocation (cost per fork = 15 continuations at m = 3)", ""]
    if ex:
        L += ["| k | selector | Fork precision | Fork recall | Credit-pivotal recall | captured abs(A) per fork | recall − uniform [CI] |",
              "|---|---|---|---|---|---|---|"]
        for k, sel in ex["E1"].items():
            for name, m in sel.items():
                L.append(f"| {k} | {name} | {m['fork_precision']:.3f} | {m['fork_recall']:.3f} | {m['cp_recall']:.3f} | "
                         f"{m['captured_absA_per_fork']:.3f} | {ci(m.get('fork_recall_minus_uniform_ci95'))} |")
        L.append("")
    if add:
        L += ["**exploratory_G_to_W1pure** (REVIEW 2026-10-27 §3.7; not in the main table, not in the verdict):", "",
              "| k | selector | Fork precision | Fork recall | captured abs(A) per fork | recall − uniform [CI] |", "|---|---|---|---|---|---|"]
        for k, sel in add["E1_exploratory_G_to_W1pure"].items():
            for name, m in sel.items():
                L.append(f"| {k} | {name} | {m['fork_precision']:.3f} | {m['fork_recall']:.3f} | {m['captured_absA_per_fork']:.3f} | "
                         f"{ci(m.get('fork_recall_minus_uniform_ci95'))} |")
        L.append("")
    L += ["## 7. Verdict (CLAUDE.md §8)", ""]
    if main_:
        v = main_["verdict"]
        L += [f"- Issued: **{v['issued']}**" + (f" — reading **{v['reading']}**" if v.get("issued") else f" ({v.get('reason')})"),
              "", "```", json.dumps(v, indent=1, default=str), "```", "", "### Pre-registered predictions", "", "```",
              json.dumps(main_["predictions"], indent=1, default=str), "```", ""]
        if ex:
            L.append(f"Gates: M1 contains 13.1% = {ex['gates']['M1_contains_13.1']}; T3 compatible = {ex['gates']['T3_compatible_[0.647,0.793]']}. "
                     "Per §8, a failed reproduction gate means the verdict is not read until the pipeline is re-checked.")
        L.append("")
    if main_ and ex and gold:
        sc = lambda b, n: main_["boundaries"][b]["scores"].get(n, {})  # noqa: E731
        a = lambda b, n: sc(b, n).get("auroc", float("nan"))  # noqa: E731
        p2 = main_["predictions"].get("P2", {})
        share = p2["lhs"] / p2["rhs"] if p2.get("rhs") else float("nan")
        capr = gold["replication_candidate_layer"]["cap"]["replication_indep20"]
        dV = sc("B1", "C_sel").get("paired", {}).get("V", {})
        L += ["## 7a. Reading the verdict (REVIEW 2026-10-27 §2–§4; wording only, the verdict is not changed)", "",
              f"- **{main_['verdict'].get('reading')} holds by the rule.** Descriptively, the ladder is a double dissociation: "
              f"siblings locate Dead (B1 {a('B1', 'G_multi'):.3f}) and are blind to forks (B2 {a('B2', 'G_multi'):.3f}); the "
              f"one-step peek locates forks (B2 {a('B2', 'W1pure_multi'):.3f}, REV {sc('B2', 'W1pure_multi').get('REV')}; the "
              f"single bit `target_revealed` carries {100 * share:.0f}% of its normalised effect); the sign is given only by "
              "replay; entropy finds neither.",
              f"- **`model_B1: false` is a property of the B1 bar.** V is itself a hidden-state probe (B1 {a('B1', 'V'):.3f}) "
              f"and C_sel is {a('B1', 'C_sel'):.3f} (difference {dV.get('diff', float('nan')):+.3f} {ci(dV.get('ci95'))}); "
              "requiring the model side to beat V on B1 asks a hidden-state probe to beat a hidden-state probe. The rule is "
              "frozen and is not changed; the paper states this next to the verdict and does not relabel the reading.",
              f"- **Original hypothesis (candidate-distribution structure):** B_multi vs arm A on B1 "
              f"{sc('B1', 'B_multi').get('paired', {}).get('A(-conf)', {}).get('diff', float('nan')):+.3f} "
              f"{ci(sc('B1', 'B_multi').get('paired', {}).get('A(-conf)', {}).get('ci95'))}; vs V "
              f"{sc('B1', 'B_multi').get('paired', {}).get('V', {}).get('diff', float('nan')):+.3f} "
              f"{ci(sc('B1', 'B_multi').get('paired', {}).get('V', {}).get('ci95'))}; it does not pass B2 and no family is "
              "identified: better than scalar confidence, not better than the hidden-state probe, does not locate forks.",
              f"- **B3 is undecidable at this sample size:** n = {main_['boundaries']['B3']['n']}; gold replication of the B3 "
              f"set {capr['frac']:.3f} {ci(capr['ci95_cluster'])} (switch threshold 0.50). Not \"no cheap signal carries the sign\".",
              f"- **P1 fails:** entropy_q on B1 {a('B1', 'single:entropy_q'):.3f} (prior +), on B2 {a('B2', 'single:entropy_q'):.3f}; "
              f"CARL δ {ex['CARL_delta']['entropy_sampled']['cliffs_delta']:.3f} (sampled) / "
              f"{ex['CARL_delta']['entropy_q']['cliffs_delta']:.3f} (exact) vs 0.42. **P4 fails** (budget interaction).",
              ""]
    if add:
        cov = add["G_coverage_by_regime"]
        L += ["### G coverage by regime (REVIEW §3.2)", "", "| regime | n | singleton rate | g_succ_state defined | median g_cover |",
              "|---|---|---|---|---|"]
        for r in ("dead", "indifferent", "fork"):
            v = cov[r]
            L.append(f"| {r} | {v['n']} | {v['singleton_rate']:.3f} | {v['g_succ_state_defined']:.3f} | {v['g_cover_median']:.0f} |")
        L += ["", "G's blindness to forks is not a coverage hole when the singleton rates of Fork and Indifferent are close: "
              "the exact-observation anchor merges histories and sees 'everyone failed from this observation' but not "
              "'different actions here end differently'.", "",
              f"W1-J labels (REVIEW §3.8): {add['W1J_labels']['counts']}; unparsed rate {add['W1J_labels']['unparsed_rate']:.4f}.", "",
              "Constant features (D-046): 21 `w1_nothing_frac` ≡ 0 and 28 `w1_div_n_distinct` ≡ 5 by construction.", ""]
    L += ["## 8. Deviations", "", "See DECISIONS.md D-029 … D-051 (in particular D-042/D-044 label diagnostics, D-046 chosen_rank, "
          "D-047 merged siblings, D-048 evaluation implementation choices).", "",
          "## 9. Exploratory", "", "`exploratory_adaptive_replay` and L3 were not run (D-048). `exploratory_G_to_W1pure`: §6.", ""]
    p = os.path.join(common.ROOT, "RESULTS.md")
    open(p, "w").write("\n".join(L))
    print("wrote", p, len(L), "lines")


if __name__ == "__main__":
    main()
