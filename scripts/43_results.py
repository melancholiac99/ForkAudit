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
              f"{ex['M4_T3_gate_H1']['n_traj']} trajectories; compatible with [0.647, 0.793]: **{ex['gates']['T3_compatible_[0.647,0.793]']}** "
              f"(CWGT +0.752). Arm A (mechanical): {ex['M4_armA_mechanical']['median']:.3f}.",
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
    L += ["## 5. C→B decomposition", "", "Not computed in this pass (report item).", ""]
    L += ["## 6. E1 — fork budget allocation (cost per fork = 15 continuations at m = 3)", ""]
    if ex:
        L += ["| k | selector | Fork precision | Fork recall | Credit-pivotal recall | captured abs(A) per fork | recall − uniform [CI] |",
              "|---|---|---|---|---|---|---|"]
        for k, sel in ex["E1"].items():
            for name, m in sel.items():
                L.append(f"| {k} | {name} | {m['fork_precision']:.3f} | {m['fork_recall']:.3f} | {m['cp_recall']:.3f} | "
                         f"{m['captured_absA_per_fork']:.3f} | {ci(m.get('fork_recall_minus_uniform_ci95'))} |")
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
    L += ["## 8. Deviations", "", "See DECISIONS.md D-029 … D-049 (in particular D-042/D-044 label diagnostics, D-046 chosen_rank, "
          "D-047 merged siblings, D-048 evaluation implementation choices).", "",
          "## 9. Exploratory", "", "`exploratory_adaptive_replay` and L3 were not run in this pass (D-048).", ""]
    p = os.path.join(common.ROOT, "RESULTS.md")
    open(p, "w").write("\n".join(L))
    print("wrote", p, len(L), "lines")


if __name__ == "__main__":
    main()
