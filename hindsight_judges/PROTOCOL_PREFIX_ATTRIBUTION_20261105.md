# Prefix Provenance × Message-Role Intervention — Hindsight/OPD Mechanism Protocol

**Date:** 2026-11-05  
**Status:** PROPOSED — protocol and execution specification only; **not run**.  
**Repository / branch:** `melancholiac99/ForkAudit` / `research/hindsight-judges` (**do not touch `main`**).  
**Scope:** P4 frozen 386 decision points / 60 original trajectories; action continuation scoring only; frozen Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct; M/F × provenance/role conditions. **No new trajectory collection, OPD training, H-LN DPO, or alteration of prior results.**

## 0. Research question and motivation

**Question:** Does Student-generated reasoning reduce a hindsight Teacher's ability to use the same action-symmetric environmental evidence because of (i) the *semantic commitment* in the reasoning, (ii) its position as the Teacher's active **assistant-generation prefix**, or (iii) an attribution cue implying that the reasoning belongs to the Teacher rather than another agent?

This is a **mechanism diagnostic**. We do **not** assume that role attribution explains the previously observed effect. A change in action score is not, by itself, proof of self-attribution or a beneficial OPD update.

Relevant prior results, scored on the **same five original candidate actions** and weighted by **independent q3456 replay value** (not measured policy success):

| 32B condition | M | F | F−M |
|---|---:|---:|---:|
| Current Think **keep** | 37.6 pp | 42.2 pp | +4.6 pp |
| Current Think **drop** | 41.2 pp | 67.8 pp | +26.6 pp |

The same qualitative interaction appears for 7B. This establishes that the current reasoning prefix strongly alters *observed action choice*, but **does not by itself distinguish** strong preexisting action prior from suppressed response to evidence. Nor does it establish a self/other authorship effect.

Sources: `hindsight_judges/results/action_diagnostic/analysis_full.json`, `hindsight_judges/results/action_diagnostic/READOUT_20261104.txt`, `hindsight_judges/exp3_action_diagnostic.py`. The historical full-response token diagnostic is a distinct experiment and is not being silently reused as an OPD training result.

## 1. Cohort, invariants, and evidence

- Use **all 386** original P4 I/U decision forks / **60** trajectory clusters, with the exact frozen current-turn Student response, factual executed action `a0=f`, four original legal alternatives `a1..a4`, action formatting, original decision-time prompt, and five candidate independent **q3456** values. Log all parse failures; no silent repair or outcome-based filtering.
- **M:** original Student-visible decision prompt, no privilege. **F:** exactly the existing, action-symmetric one-step evidence for all five candidate actions, in the same original encoding and order, appended with the existing evidence wrapper. Do not invent an alternative future log. F is privileged and **never** a Student input.
- All conditions at a point share the same task, state, candidate list, replay Q, Teacher weights, decoding-free teacher-forced candidate scorer, and (within each model) precision.
- **Only the presentation and claimed provenance of the same verbatim Student reasoning may change.** No wording such as “wrong,” “please critique,” “review for errors,” or “ignore your draft.” No gold solution or target action exposed.
- Distinguish **stated authorship** (“your draft” vs “another agent's draft”) from actual chat role (**user** vs active **assistant**). An authorship statement is an experimental cue, not proof of an internal self-recognition mechanism.

## 2. Conditions: matched 2×2 plus original anchors

A complete, interpretable factorial needs **explicit, symmetric provenance wording on both sides of the 2×2**. Retain the original `KEEP` and `DROP` as historical anchors; do **not** silently treat the unlabelled KEEP as a fully matched “self”-label condition.

| ID | Source cue | Student reasoning placement | Action-output prefix |
|---|---|---|---|
| **AS** — ASSISTANT-self | explicitly “your earlier draft” | verbatim text in the **active assistant continuation** | original action-open tag |
| **AO** — ASSISTANT-other | explicitly “another agent's draft” | **same** active assistant continuation as AS | original action-open tag |
| **QS** — QUOTE-self | explicitly “your earlier draft” | verbatim text **quoted within user message** | action-open tag, without active Think |
| **QO** — QUOTE-other | explicitly “another agent's draft” | **same** user quote location as QS | action-open tag, without active Think |
| **KEEP** (original anchor) | implicit assistant authorship; **no added cue** | original active assistant Think | original action-open tag |
| **DROP** (original anchor) | no current Think | absent from active assistant; same old non-Think prefix | original action-open tag |

**Primary factorial:** `AS/AO × QS/QO` (assistant/user placement × self/other provenance cue). **Anchors:** original `KEEP/DROP` contextualize the mechanism and permit comparison with already published diagnostics. To minimize GPU work, first complete the four factorial cells, using strictly validated historical KEEP/DROP values when possible.

### Exact rendering requirement

Use Qwen's **real chat template** via `tokenizer.apply_chat_template`; the labels `[User]` and `[Assistant]` below are explanatory, **not** literal text to inject into prompts.

Preserve the raw original reasoning **bytes** before the opening action tag, including its `<think>...</think>` (or original variant) and any other pre-action preamble. Let `R` denote that exact span; let `TAG` be the actual original action-open tag, and let `LEAD`/`TAIL` be the candidate's original whitespace and action-close serialization as recovered by `action_format`. All five candidates are scored as `TAG + LEAD + action_j + TAIL` in the appropriate active assistant prefix, using the scorer's existing token-boundary logic.

The source cue should be a predeclared, otherwise **identical** one-sentence block. For example:

- `[DRAFT PROVENANCE] The following draft was written by you in an earlier attempt. [/DRAFT PROVENANCE]`
- `[DRAFT PROVENANCE] The following draft was written by another agent in an earlier attempt. [/DRAFT PROVENANCE]`

**AS/AO:** include the matched provenance statement in the **user** message (alongside the unmodified M or F content); the currently scored assistant continuation remains exactly `R + TAG`. Thus, within AS vs AO, **chat role, target continuation, evidence and original text are identical; only the authorship cue changes.** This also means that the text is still *structurally assistant-owned* regardless of the cue.

**QS/QO:** put the **same exact R** into a delimited quotation within the **user** message, adjacent to the matched provenance statement; the active assistant continuation begins at `TAG` with **no R**. Quote boundaries and common request to output the next legal action must be *identical* across QS and QO. Never put the factual action or any token beyond `TAG` inside the quote. No request to critique or revise. Quote is context, not an environment observation.

**KEEP/DROP:** reproduce the previously released prompts and prefixes *bit-for-bit*, including the established M/F evidence wrapper and original Student action serialization. Adding the matched source cue creates AS/AO rather than modifying KEEP.

**Important confound control:** placing R in a user quote instead of the active assistant prefix also changes the point at which assistant generation starts and may change the interpretation of the task. Thus `QS−AS` is a **combined placement/restart/prompt-format intervention**, *not* a pure “within-turn coherence” causal effect. Keep this distinction in every report.

### Cheap optional robustness

After primary analysis (not as a condition selected based on favorable outcomes): one paraphrased self/other provenance pair, or a neutral-quote condition, can test sensitivity to the exact phrasing. Do not expand the main factorial unless its costs are measured and approved.

## 3. Comparisons, estimands, and what they identify

For model `m∈{7B,32B}`, evidence `e∈{M,F}`, condition `c`, and point `i`, score the full five action-string log-likelihoods `ell[m,e,c,i,j]` with the original calibrated backend. Define conditional **five-candidate** probabilities `p_j = softmax_j(ell_j)` and offline independent replay-weighted value:

`V[m,e,c,i] = sum_{j=0}^4 p_j * q3456[i,j]`.

This is a *restricted five-action diagnostic*, **not** an unrestricted policy, observed online success rate, true action Q-estimator, or an OPD loss/gradient.

**Primary estimands (paired within point, 95% trajectory-cluster bootstrap):**

1. **External quote provenance effect at F:** `P_Q = mean_i(V[F,QO,i] - V[F,QS,i])`. The same effect at M is a necessary companion.
2. **External quote evidence-responsiveness interaction:** `I_Q = mean_i([V[F,QO]-V[M,QO]] - [V[F,QS]-V[M,QS]])`. Distinguishes a main shift in baseline action preference from improved *incremental use of F*.

**Matched factorial and controls (equally reported, secondary):**

3. **Active assistant provenance:** `P_A(e) = mean_i(V[e,AO]-V[e,AS])`; evidence interaction `I_A` defined analogously.
4. **Placement/restart contrasts:** `V[e,QS]-V[e,AS]`, `V[e,QO]-V[e,AO]`, each explicitly marked *mixed interventions*, plus `V[e,AS]-V[e,KEEP]` to measure how merely adding explicit self provenance changes the baseline.
5. **Role × provenance interaction:** `(V[e,QO]-V[e,QS]) - (V[e,AO]-V[e,AS])`.
6. **Anchoring controls:** `V[e,DROP]-V[e,KEEP]`; factual candidate argmax rate and `p_f` for all conditions; differences in the fraction of bad factual choices corrected.

**Disentangle evidence assimilation from strong action priors (essential):** record all five raw `ell` values. For each alternative `j`, using the factual action `f=0` as a fixed reference, compute the **evidence-induced log-odds shift**

`E[c,i,j] = (ell[F,c,i,j] - ell[F,c,i,0]) - (ell[M,c,i,j] - ell[M,c,i,0])`.

Compare its sign/rank against the fixed independent `q3456[i,j]-q3456[i,0]` on non-tie pairs; summarize concordance or slope at point level, plus paired attribution and placement contrasts. Use unchanged action strings and model weights across M/F within each c. This answers whether F changes *relative action support* coherently with Q **even when F fails to overturn a strong prior**. A large `V` effect without a matched `E` effect should not be called recovered evidence assimilation.

**Statistical protocol:** all 386 points; unit of resampling = original trajectory (**60 clusters**); 4,000 cluster bootstrap draws, fixed seed `20261105`; pointwise paired differences first, then mean. Preserve all 386 unless a documented frozen-source parse failure; report full cohort and any first-64 pilot separately. Numerical CIs exploratory; no multiplicity correction, and CI crossing zero is not evidence of equivalence. Report absolute values, differences and CIs in **percentage points** for V, **nat** for log-odds, and 0–1 for rates.

## 4. What each possible result would / would not support

- **QO consistently beats QS** within identical user-message quote format: evidence that **stated self/other provenance cue** affects action scoring. If it also increases `I_Q` and Q-aligned `E`, stronger evidence that provenance changes *use of F*. **It is not automatic proof** of the Teacher internally believing it wrote the text.
- **AO beats AS** despite identical active assistant prefix: evidence that **explicit authorship cues** can modulate continuation *even inside the assistant role*.
- **QS≈QO≫AS≈AO≈KEEP** with DROP higher: relocation/restart is more promising than provenance wording; the action task itself changed, so do not label it a pure role-causality result.
- **QUOTE conditions≈KEEP** despite DROP≫KEEP: content-dependent action consistency is one interpretation, but **failed manipulation, prompt format, or overpowering assistant-style behavior** remain alternatives; do not declare “content-only mechanism proven.”
- **F−M value gain small, but Q-aligned E remains strong**: action prior dominance is plausible; naive raw-value gain underestimates evidence extraction. **E loses Q alignment in keep/assistant and recovers under quoted conditions:** supports a deeper prefix-dependent evidence-response effect.
- **Results weak or contradictory:** document as null/mixed. Do not invent a distillation method or selectively redefine the primary metric afterward.

For a publishable attribution mechanism, look for **consistency across both models, across M/F, in value and in Q-aligned log-odds**, and replication under a paraphrase/neutral condition if a source-label effect is large. Report exceptions.

## 5. Implementation outline and resource gates

Add a **new script** (e.g. `hindsight_judges/exp4_prefix_attribution.py`) and its CPU tests in a *separate implementation commit after protocol review*. This document authorizes **no GPU run by itself**.

1. Reuse the frozen `exp3_action_diagnostic.py` original P4 input loader, `extract_decision_prompt`, current-turn Think parser, `action_format`, F payload and original q3456 source hash checks. Never edit or overwrite `results/action_diagnostic/`, `results/token_diagnostic_*/`, `exploratory_P4/`, or any original E1c/E4 artifacts.
2. Generate deterministic `(point, M/F, condition)` request rows, including SHA256 of: source input, exact user message, exact assistant prefix, serialized candidate actions, quotation/provenance template, tokenizer chat-template version. Emit a **condition matrix audit** showing source cue, actual role tokens and text span location. Assert the student reasoning text is verbatim-identical in AS/AO/QS/QO/KEEP, and absent only from DROP active/quoted context.
3. Validate every candidate: exact factual action match, original `LEAD/TAIL`, five candidate distinct action strings, no future supervised action in quote, no hidden target leakage. For quote rows, **active assistant prefix starts at action-open tag**; no newly generated Think or accidental extra role boundary.
4. Use the existing verified **7B FP32** and **32B FP16-transformer/FP32-head** MATH/1024-chunk backends and original canonical candidate-token scoring. Numerically gate prefix-cache vs direct full continuation (TV <0.005), and run compact CPU unit tests for role/provenance construction, action serialization, counterfactual candidate identity, and raw-score log-odds calculations before a full GPU run.
5. First score **first 32–64 P4 points** in a separate exploratory directory to measure real GPU-seconds/context, VRAM and correctness; **do not** use this pilot for final confirmation or choose the best condition based on it. The earlier “1–2 GPU hours” is only a guess; benchmark before estimating a full 386-point run. Each model stays on a recorded GPU UUID with immutable precision/weight SHA fingerprints, resumable completed-run hashes, and no silent truncation.
6. For the six conditions, there are `386 × 2(M/F) × 6 = 4,632` point/context jobs **per model** (9,264 across 7B/32B), each scoring five candidate actions. Original KEEP/DROP under M/F are `386×2×2=1,544` per model and can be reused **only if their exact tokenized prompt, prefix, candidate suffixes, tokenizer, model weights and precision match**; otherwise re-score. The four new cells are 3,088 jobs per model (6,176 total). Save under `hindsight_judges/results/prefix_attribution_20261105/`; never mix pilot/full files.
7. Save per-point/per-condition five raw candidate log-likelihoods, candidate-normalized probabilities, raw context text hashes, Q values from independently frozen replay source, source/model fingerprints, numerical checks, computation timestamps, cluster IDs, analysis code version, and a verification report. A separate independent verification pass should recompute the primary paired contrasts from raw scores.

### Execution budget and authorization

**This commit is a protocol only.** It requests neither server execution nor an automatic shutdown. A server Agent can implement it after review and obtain a real timing estimate from the pilot. No H-LN/OPD training or mathematical cross-domain study is bundled into this task.

## 6. Relationship to OPD, follow-up decision, and novelty limit

This is **action-continuation diagnostic**, not a tokenwise OPD update. Its specific value is to test whether a Student-generated reasoning string changes Teacher scoring by **role placement / attribution cue** rather than only by semantic commitment. It **does not** prove that a different Teacher presentation helps Student learn.

If a provenance/role condition reliably restores evidence-sensitive action evaluation, the **next, separate** question is whether it supports an actual distillation target. A true token-level “externalized prefix” Teacher must receive **only `y_<t` at position `t`**, never the entire completed Student response when supervising early Think tokens; align vocabulary, current prediction slot and permissible output space. User-quoted prefixes change the task and can alter format; one must validate teacher and student token probabilities and real training performance rather than transferring action-scorer gains by assertion.

Related literature already investigates model self/other error attribution and critique-assisted or evidence-contrastive OPD. This experiment does not preclaim novelty. It can at most isolate a **specific OPD-relevant role/presentation mechanism** in the controlled ALFWorld action setting. Cross-domain mathematical replication and training are separate later decisions.

## 7. Mandatory report format

A final report should include:

- Population, model/precision, exact six prompts (and message-role tokenization), final counts, numerical gates, runtime/GPU usage, any exclusions and source hashes.
- **By model, condition and M/F:** expected replay-weighted five-candidate decision value, factual argmax rate, `p_f`, and mean evidence-induced Q-aligned log-odds.
- All primary `P_Q`, `I_Q` plus factorial `P_A`, `I_A`, placement/restart and role × provenance contrasts, paired 95% trajectory-cluster CIs.
- Plots/tables separating **M→F evidence effect** from **self→other cue** and **assistant→quote placement**, not only a single favorable chart.
- A short conclusion that labels findings as *supported*, *not supported*, or *unresolved* and explicitly avoids claims about OPD gradient quality, actual Student learning, internal identity attribution or universal generalization.

**Decision rule:** Run this bounded mechanism test first. Do not propose a “role-transfer distillation” algorithm as validated until the same-source paired attribution/role interactions and evidence-response metrics justify it and actual token-level/learning tests are performed.
