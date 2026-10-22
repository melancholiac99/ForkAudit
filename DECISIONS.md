# DECISIONS.md

Every choice not fixed by CLAUDE.md is recorded here before it is acted on.
Status tags: **OPEN** (awaiting instruction), **DECIDED** (choice + reason), **INFO** (fact recorded, no choice needed).

---

## D-001 — STOP: Appendix Q contains concurrent work closer to this project (CARL) — DECIDED 2026-10-22

**Ruling (user, 2026-10-22):** CARL is not a stop reason, but it changes the entry point and the pre-registered rules.
CLAUDE.md amended (commit recorded in git log, "CLAUDE.md: CARL amendments"):
(1) §2 table gains a CARL row (search-QA + math, Qwen3-4B; criticality = unsigned outcome SD with randomness isolated to
the state's action sampling; entropy is *sampled-estimated* over free-text actions; finding is distribution-level; CWGT
states the two results "are not in conflict" — that tension is this project's entry point);
(2) §1 gains the "已知两端" sentence; (3) §4 arm B gains feature 16 `entropy_sampled` and 17 `chosen_conf`, both `forward`;
(4) §6 gains secondary target `crit_sd(t)` + Cliff's δ (high- vs low-entropy) next to CARL's 0.42;
(5) §8 R1 split into R1a ("entropy suffices") / R1b ("structure beyond entropy"); §11 main table gives `entropy_q` and
`entropy_sampled` their own rows labelled "CARL 复现".
FEATURES_FROZEN.md had not been committed, so adding features 16–17 is permitted under §9.2.

Original stop record follows.

Per CLAUDE.md §2 ("若其中有与本项目更近的工作，立即写入 DECISIONS.md 并停下等指示"), work is paused here.

**Source:** CWGT (arXiv 2608.19760) main.tex, `\section{Concurrent work, reconciled}` (= Appendix Q), plus Sec. 6 last paragraph.

**CARL — Criticality-Aware Agentic Reinforcement Learning** (Shen, Zhang, Ling, Zhao, Chua; arXiv 2512.04949, v3 2026-05).
- Abstract (fetched 2026-10-22): "only the action choices on a small fraction of states are critical in determining the final
  outcome … CARL leverages entropy as a heuristic proxy for state criticality", trains only on high-criticality states.
- Per CWGT App. Q: CARL reports that **policy entropy separates critical from non-critical states at the distribution level
  (Cliff's δ = 0.42, n = 294, Brunner–Munzel), with criticality estimated from resampled continuations**, in multi-hop search.
  CARL reports **no precision / recall / F1 / AUC** for criticality identification, and no threshold / top-k rule.
- CWGT's own reconciliation (Sec. 6): "Entropy separates critical from non-critical states at the distribution level …
  that separation and chance-level recall at a cost-matched threshold are not in conflict."

**Why this is closer than anything in CLAUDE.md §2's table:**
Arm B's core hypothesis is "decision-structure features computed from the policy's distribution over candidate actions
identify replay-pivotal steps". CARL already tests the most basic such feature (entropy ≈ our `entropy_q` / `eff_size`)
against a resampled-continuation criticality label — i.e., a pre-decision distributional feature vs a replay-style label.
EMPG (listed in §2) only uses entropy for training; CARL *validates* it against a replay-like label. So the
"entropy is associated with pivotality" half of arm B is not new.

**What CARL does NOT do (i.e., what would remain ours):**
- Not ALFWorld, not Qwen2.5-7B, not the CWGT A_replay estimand / σ_floor / K=4 policy-supported counterfactual.
- No recall@route, no AUROC, no within-trajectory shuffle control, no fluency-contamination (M4) test.
- Only entropy — no margin / de-tasked margin / candidate-type / loop / runner-up / semantic-dispersion features;
  no hidden-state probe (arm C); no B-vs-C decomposition.

**Other Appendix Q entries (judged NOT closer than the §2 table):**
- CAR (Shah, 2606.08275): formalizes replay-based causal attribution, validated on synthetic SCMs. Same ground-truth
  idea as CWGT, no decision-structure signal. Not closer.
- CSO (Li et al., 2602.03412): PRM nominates candidate critical steps, expert alternatives, verify by policy continuation;
  GAIA / XBench. Uses "high-entropy" only as motivation. Not closer.
- C3 (Chen et al., 2603.06859): multi-agent credit auditing tool, self-referential replay. Not closer.

**Options for the user (my recommendation first):**
1. *(Recommended)* Continue as specified, add CARL to §2's table as "entropy ↔ resampled criticality, distribution-level only",
   and reframe arm B's novelty as: (i) operating-point metrics (recall@route, AUROC) with shuffle controls on CWGT's estimand,
   (ii) features beyond entropy (margins, de-tasked margin, runner-up structure), (iii) the fluency-contamination test,
   (iv) arm C upper bound + C→B decomposition. `entropy_q` stays in the frozen list as the CARL-replication single feature.
2. Continue, but pre-register `entropy_q` single-feature as a named CARL replication with its own row in the main table.
3. Pause the project to read CARL v3 in full before deciding.

---

## D-002 — Template lists admissible actions in the prompt — INFO (answers Phase 0.3 / Phase 1 question)

The HCAPO ALFWorld template (verbatim, from CWGT bundle `appendix_sources/hcapo_alfworld_template_ORIGINAL_FROM_BUNDLE.txt`,
copied to `prompts/hcapo_alfworld_template.txt`) contains
`Your admissible actions of the current situation are: [{admissible_actions}].` → **Yes, admissible actions are listed.**
History length = 2 for ALFWorld.

Hash note: local copy is 1,084 B, sha256 fc6b468a…; CWGT states 1,085 B, sha256 e2f580f9…. Likely a trailing newline lost
in arXiv packaging. Content to be diffed against the HCAPO paper appendix once read.

## D-003 — Template requires `<think>…</think>` reasoning BEFORE `<action>` — OPEN (affects arms A, B, C)

The template asks the model to "first reason step-by-step … enclosed within <think> </think> tags", then emit
`<action>…</action>`. Consequences for CLAUDE.md §4 definitions:
- `conf_t = mean token logπ(a_t | s_t)`: does "s_t" include the sampled `<think>` text? CWGT's T3 definition
  (KILLSHOT_PREREG §3.5) says "mean token logπ(a_t|s_t) under the policy without s_final" and cites
  `scripts/dump_logprobs.py:41`, which is not in the arXiv bundle. Unknown whether CWGT conditioned on the think text.
- Arm B scores *every* admissible action. Conditioning candidates on the realized think is biased toward the chosen action
  (the think was generated en route to it); conditioning on the bare prompt (skipping think) is a distribution the policy
  never actually samples from.
- Arm C "last prompt token, before the first token of the chosen action": with CoT these are different positions
  (end of prompt vs. after `</think><action>`).

Candidate resolution (to decide before Phase 2, ideally after seeing CWGT's dump_logprobs.py if the archive is found):
score all candidates under prompt + `<think>`(realized)`</think><action>` as primary (matches what the policy does),
and under bare prompt + `<action>` as a sensitivity; export hidden states at both the end of the prompt and the
position just before the action's first token.

## D-004 — Arm A routing direction — OPEN

CWGT's router (Sec. 6) "routes **low**-confidence turns" and gets recall 11.9% [9.4, 14.9] at 13.1% routed (n_pivotal = 540).
CWGT also says "the stuck, pivotal steps are the high-confidence ones". CLAUDE.md M1 ranks by s **descending**.
To replicate 11.9%, arm A must use s = −conf_t. Proposal: primary arm-A score = −conf (CWGT direction), report +conf as a
pre-declared sensitivity row. For arm B single features, report both signs? (would need a rule fixed before seeing data —
e.g., sign fixed by a priori hypothesis per feature in FEATURES_FROZEN.md).

## D-005 — M4 replication target (+0.75) is a T3 number for the implicit ρ_t family, not for confidence — OPEN

KILLSHOT_PREREG §3.5 T3: per-trajectory Spearman(Δ_t, mean token logπ(a_t|s_t)), Δ_t = z(credit) − z(A_replay).
The +0.752 [0.647, 0.793] (n = 47 traj.) is for credit = HCAPO ρ_t (outcome-conditioned, clipped ratio). If arm A's
s = conf_t = mean logπ itself, then Δ_t = z(logπ) − z(A_replay) and Spearman(Δ_t, logπ) is high almost by construction,
so "arm A reproduces ≈ +0.75" is not an informative pipeline check. Options: (a) keep as-is but label it mechanical;
(b) additionally compute HCAPO ρ_t (needs the hindsight-injection prompt line, T_temp = 5.0, clip [0.8, 1.2]) as the
actual T3 replication target. Recommend (b) if compute allows; it is a single extra teacher-forced pass per decision point.

## D-006 — M2 trajectory inclusion threshold differs from CWGT — OPEN (minor)

CLAUDE.md M2: trajectories with ≥ 5 D-points and non-constant A_replay. CWGT (prereg Amendment v3.2): ≥ 4 complete turns.
Proposal: primary = CLAUDE.md's ≥ 5; also report ≥ 4 for the arm-A-vs-CWGT comparison row.

## D-007 — Hardware: no GPU currently attached — INFO / blocker for Phase 1+

2026-10-22: the instance is in AutoDL no-GPU mode (1 CPU, no /dev/nvidia*, `nvidia-smi` is a 0-byte stub, torch.cuda False).
Phase 0 (archive search, reading, env setup, random-policy ALFWorld run) is CPU-feasible. Phase 1 needs the GPU instance.

## D-008 — Draft budget for `entropy_sampled` in Branch B replay — DECIDED 2026-10-22

Problem: CWGT's rejection sampler stops as soon as 4 distinct admissible alternatives are found (often within a handful of
draws), so "use the replay drafts" would give `entropy_sampled` a variable, state-dependent sample size (n=5 at a diffuse
state, up to 300 at a peaked one) — the estimator's bias would itself correlate with entropy.
Choice: at every decision point the sampler draws a **fixed first batch of 32** samples (one vLLM call, n=32, T=0.7,
seeded); `entropy_sampled` is computed on the admissible, parsed actions of exactly those 32 (plug-in entropy over
empirical frequencies; non-admissible / unparseable drafts counted and stored, not in the entropy). Alternatives are taken
in draw order from these drafts; if fewer than 4 distinct admissible non-factual actions appear, sampling continues in
further batches up to the 300-draw cap (CWGT's rule unchanged). All drafts (raw text + parsed action + admissible flag)
are stored. This also matches the 32-sample fallback specified for Branch A, so both branches use the same estimator.
Cost: ≈0 marginal (the 32 drafts are the first draws the sampler makes anyway).
