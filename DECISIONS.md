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
**Update 2026-10-22 (evening):** the no-GPU instance ran out of memory during the vLLM install; the user restarted with a GPU.
Now: 1× NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB, driver 580.95.05. Container cgroup limits:
**110 GiB RAM, 22 CPUs** (host shows 208 — ignore). Env-worker parallelism is therefore capped at ~20 processes.

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

## D-009 — Prompt construction details not fixed by the HCAPO appendix — DECIDED 2026-10-22

Sources: HCAPO arXiv 2603.08754 TeX `tex/app.tex` §C.2.1 (template), §C.1 ("follow all configurations in GiGPO",
50 env steps, max prompt 2048 / response 512 tokens, history length 2); verl-agent (GiGPO code, langfengQ/verl-agent master,
fetched 2026-10-22 into `downloads/verl_agent/`) for everything the appendix leaves implicit.
- **Template text:** the CWGT-bundled verbatim appendix text (`prompts/hcapo_alfworld_template.txt`, header comment lines
  stripped) — this is what CWGT says it used. It matches HCAPO TeX except ’ vs ' (ASCII kept, as in CWGT's bundle).
  It differs cosmetically from verl-agent's code string ("Embodied", missing periods); we follow the paper/CWGT text.
- **Step 1:** HCAPO's appendix gives only the with-history template; verl-agent uses a separate NO_HIS template at step 1.
  CWGT says it used the appendix template verbatim, so we use the *same* template at step 1 with step_count=0,
  history_length=0, empty action_history. (Disclosed deviation from verl-agent; unknown whether CWGT did the same.)
- **{task_description}:** text after "Your task is to: " in the initial observation, stripped (verl-agent `extract_task`).
  Note the template appends "." after it; ALFWorld task strings already end in "." → "…countertop.." as in HCAPO.
- **{current_observation}:** raw env text; at step 1 the full initial observation (verl-agent behaviour).
- **{action_history}:** verl-agent `SimpleMemory.fetch`: last ≤2 records, each
  `[Observation {k}: '{obs_k}', Action {k}: '{act_k}']` joined by "\n", where obs_k is the observation the action was taken
  in and act_k the *parsed* action string sent to the env.
- **{admissible_actions}:** `"\n ".join(f"'{s}'" for s in admissible if s != 'help')`, env order (verl-agent).
- **Chat template:** single user message = filled template, Qwen2.5 tokenizer chat template with its default system prompt,
  `add_generation_prompt=True` (verl-agent default; no custom system prompt).
- **Action parsing:** verl-agent `alfworld_projection`: lowercase; text between first `<action>` and first `</action>`;
  invalid if tags missing, `<think>` tags missing, or CJK chars. Invalid outputs are still sent to the env as in verl-agent
  (env answers "Nothing happens."); they are counted. For the alternative sampler, a draft counts only if valid *and*
  its parsed action ∈ admissible set.
- **Step cap:** 50 env steps (GiGPO/HCAPO ALFWorld). Response cap 512 tokens.
- **Reward/outcome:** binary success (env `won`), as in CWGT.

## D-010 — Task list and seeds for Branch B — DECIDED 2026-10-22 (provisional until the archive search returns)

CWGT's frozen task list is not published in the bundle. Choice: ALFWorld `train` split game files, task list frozen by
`scripts/05_freeze_tasks.py` (numpy seed 0): 50 games stratified over the 6 task types (9/9/8/8/8/8, type order fixed
alphabetically), seed i ↔ game i. Frozen list committed to `data/task_list.json` before any rollout.
Rationale: no training is done in this phase, so a held-out split buys nothing; stratification guarantees all task types
appear. If the CWGT archive turns up with its task list, it replaces this list.

## D-011 — Archive status: not found → Branch B; new related work from the search — DECIDED 2026-10-22

Full log: `scripts/00_find_archive.md`. No public archive found via arXiv, HF, alphaXiv, S2, OpenReview, GitHub, Zenodo,
OSF, 4open. **Proceeding with Branch B** (§7 0.1(e): do not wait).

Pending user actions (outward-facing / needs a browser, so not done by me):
1. Email haiyuez@usc.edu requesting the KS1 `collect_v2` archive. Draft:
   > Subject: Request for the KS1 replay archive of "Credit Without Ground Truth" (arXiv 2608.19760)
   > Dear Dr. Zhang, we are running a replay-based audit that builds directly on your ALFWorld/Qwen2.5-7B protocol
   > (K=4 policy-supported alternatives, ≥3 replicas, A_replay). The paper states the KS1 `collect_v2` replay archive,
   > per-turn summaries, logprob dumps and the generator are released through an anonymized repository, but we could
   > not find the link in the arXiv version. Would you be willing to share the link, or the archive (in particular per-turn
   > prompts, admissible sets, rejection-sampler drafts and outcomes)? We will cite the paper and the archive. Thank you!
2. Download the supplementary zip of OpenReview Me3WCeeMWW in a logged-in browser and drop it in `downloads/openreview_supp/`.
   If it contains the archive/repo link → re-evaluate Branch A.

Related work surfaced by the search (none triggers the §2 stop rule; none tests decision-structure signals):
- **CWGT ICLR-2027 revision (Me3WCeeMWW):** same author; redesign with 16 continuations/arm/bank on 172 turns.
  **Relevant caveat for us:** even at 16 continuations per arm, cross-bank replay-contrast pooled Spearman is only 0.330
  (Pearson 0.736). Our A_replay uses 3 replicas/arm, so its reliability is lower still → M2 ceilings are low and a null on
  M2 is weakly informative. M1 (pivotal = nonzero) is affected differently (lattice/divisibility structure, CWGT App. E).
  Not acted upon now (protocol frozen); flagged for the RESULTS discussion. If the user wants, a reliability sub-study
  (second independent replay bank on a subset) could be pre-registered before Phase 3.
- **RMPO (ac6BotAkVF, ICLR 2027):** replay-measured credit operators (deletion / prefix progress / same-state
  replacement) added to GRPO advantage; Qwen3-4B/8B; ALFWorld deletion credit vs planner progress ρ=0.842. Training method
  using replay as signal — not a forward decision-structure predictor. Cite as related.
- **ACRE audit (HbNTIXwoeU / plrHvqYzhK, ARR 2026):** audits pipelines' credit observables vs walkthrough progress;
  fidelity −0.037…0.789 tracks held-out gain. Not decision structure. Cite as related.

## D-012 — Model weights provenance — INFO 2026-10-22

Weights copied from the user's existing copy at `/autodl-fs/data/Agent-R1/models/Qwen2.5-7B-Instruct` (ModelScope-style
download) to `/root/autodl-tmp/models/Qwen2.5-7B-Instruct`. All 4 safetensors sha256 and all 7 config/tokenizer git-blob
hashes (`env/qwen_sha256.txt`) match HF `Qwen/Qwen2.5-7B-Instruct` revision **a09a35458c702b33eeacc393d103063234e8bc28**
(lastModified 2025-01-12) exactly. `model_hash` in scores.parquet = this revision id.

## D-013 — Sampling parameters and env execution — DECIDED 2026-10-22

- **Sampling = pure temperature 0.7**: `top_p=1.0, top_k=-1, repetition_penalty=1.0, max_tokens=512`, explicitly set in
  every vLLM `SamplingParams`. Qwen2.5's `generation_config.json` carries top_p=0.8/top_k=20/rep-penalty 1.05, which vLLM
  can pick up as defaults; we override them so the sampled distribution is exactly softmax(logits/0.7) — the object
  that teacher-forced scoring and the "policy-supported alternative" definition refer to. verl-agent/verl rollouts
  likewise use top_p=1, top_k=-1. CWGT states only T=0.7.
- **Per-request seeds** (vLLM `SamplingParams.seed`): rollout seed = 1_000_000·task_seed + turn; replay seeds derived
  from (task_seed, turn, role, replica) by a fixed formula in `11_replay.py`, all recorded per request.
- **Env:** measured 52 ms/step, 0.65 s reset, ~12 MB/game on this box → env steps run in a persistent pool of 20 worker
  processes (`scripts/envpool.py`), each owning many `AlfGame` instances. State restore = fresh game + re-feed of the
  stored action prefix, with every re-fed observation compared to the stored one; divergences are counted and reported
  (CWGT: 0 over 20,538). Smoke test: 0 divergences over 3 fresh envs × 50 steps (`logs/06_random_policy.log`).
- **Turn indexing:** turn = 1-based step number as shown in the prompt ("You are now at step {turn}").

## D-014 — Action parser: tag-variant-tolerant (supersedes the strict parser in D-009) — DECIDED 2026-10-22

Finding (`scripts/explore/fmt_probe.py`, exploratory, step-1 prompts of task seeds 0–9, 32 samples each, T=0.7):
under the HCAPO template Qwen2.5-7B-Instruct writes the action as `<action>…</action>` only 24% of the time (78/320);
63% use `[action] … [/action]`, 12.5% an unclosed `[action] …` at the end. The strict verl-agent parser would turn ~76%
of steps into invalid "Nothing happens." moves, i.e. measure the parser rather than the policy.
CWGT hit the same drift and its harness parser handled the variants (KS1_XFAM_PREREG §5 "e17 tag-format drift";
A3B_PREREG "Normalization … the `[action>` / `<action>` / `[action]` lesson … strip all tag variants").
Choice (`common.parse_action`): search only the text after the last `</think>` (whole text if absent); take the last
span opened by `<action>`/`[action]`/`[action>`/`<action]` and closed by any `/action` variant; else the last unclosed
opening tag up to end of line; lowercase + whitespace-collapse. valid = an action was extracted ∧ `<think>…</think>`
present ∧ no CJK. Unparseable → last 30 chars sent to env (verl-agent behaviour), counted. The tag format (`fmt` ∈
xml/variant/unclosed/none) is stored for every sample so strict-xml rates can be reported.
Result on the same probe: 320/320 parsed, 320/320 admissible. Template text unchanged (§9.1 respected).

## D-015 — Seeded generation is not bitwise reproducible — DECIDED 2026-10-22

vLLM 0.32 with prefix caching / batching: the same request with the same `seed` returned different samples on a second
call (`logs/07_vllm_smoke.log`, seed_reproducible False). Batch-invariant kernels would cost throughput and are not
required for validity. Choice: seeds are fixed and recorded per request (D-013), and **every generated text is stored**
(rollout responses, all replay drafts, all continuation action sequences); every downstream number is recomputed from
stored artifacts, never by regeneration. Disclosed as a reproducibility limitation.

## D-016 — Replay disk exhaustion diagnosis — INFO 2026-10-22

Replay pilot failed with ENOSPC while copying fast_downward/libdownward.so into /tmp.
TextWorld 1.7.0 PddlEnv inherits Environment.close(), a no-op. Each environment loads
a private ~33 MiB library, immediately unlinks the temporary file, but leaves the
mapping (and its disk blocks) alive until dlclose or process exit. A six-environment
create/reset/close probe retained six mappings and consumed ~198 MiB despite GC.
An explicit fast_downward.close_lib workaround was tested but REJECTED and reverted:
although retained mappings stopped growing, repeated create/reset/close eventually
segfaulted. The library exports GNU UNIQUE symbols; do not assume dlclose is safe.
No runtime code changes retained. Prefer bounded environment reuse and/or worker
recycling to release native resources at process exit; validate before replay resumes.
At 600 live environments the libraries alone require ~19.3 GiB; current free disk
is ~44 GiB, so fix the leak before considering expansion. Reducing concurrency alone
does not bound cumulative leakage. Unlinked mapped files explain why du misses the
space and why disk capacity returned when the failed workers exited.

## D-017 — Bounded reusable environment slots — DECIDED 2026-10-22

EnvPool now retains idle AlfGame instances and reuses their native library slots.
On reuse, load a different game through the existing TextWorld wrapper stack when
needed, then reset and re-feed the exact stored prefix as before. Assign new jobs
to the least-loaded worker so native slot count is bounded by peak concurrency
plus worker rounding, rather than cumulative continuation count. Do not dlclose.
Validate repeated cross-game reuse against stored observations and outcomes before
resuming seeds 0–4; preserve completed replay JSONs. Write new JSONs atomically to
avoid treating interrupted writes as completed points. Continue seeds 0–49 after
the pilot completes and summarize pilot cost and M5 before advancing phases.

## D-018 — User-authorized Phase 1 review trigger — DECIDED 2026-10-23

Monitor the existing runner once per minute without interrupting it. Queue a review
in the originating Codex thread on exit (including completion), 30 minutes without
log changes, or free disk below 5 GiB. Match PID plus process start time to avoid PID
reuse. One successful queued review stops the monitor; failures retry. A lock
prevents duplicate monitors. No automatic blind restart. Codex daemon, credentials,
and network must remain available for the queued review to execute.

## D-019 — Phase 1 completion review — INFO 2026-10-23

Runner completed normally at 06:38:58 +0800 (22:38:58 UTC on Oct 22).
Audit script scripts/12_audit_replay.py validates every rollout/replay key, source
rollout hash, alternative support/uniqueness, draw cap, replica roles/outcomes,
continuation length, prompt existence, and absence of extra/missing replay files.
All 2,029 points across 50 trajectories passed; 25,491 continuations; zero re-feed
divergences. Created data/decision_points.parquet and eval/phase1_audit.json plus
a SHA256 input manifest. A_replay uses exact integer numerator (4*sum factual -
sum alternative outcomes)/12; undefined contrasts remain null. No feature/credit
associations computed; FEATURES_FROZEN.md has not yet been written.

Defined: 1,617/2,029 (79.69%; CWGT 86.9%); excluded 412. Pivotal among defined:
718/1,617 (44.40%; CWGT 30.5%). Factual sigma_floor=0: 77.18% among defined,
77.13% overall (CWGT 86.0%). Minimum |D|>=1,200 passes, but these distribution
differences require reporting/investigation, not protocol adjustment or an
unqualified M5 pass. Chosen-action logprob exclusion shift and HF-vLLM action-only
scoring alignment remain pending. Do not substitute full-response logprobs for
action-only scores. D-003/D-004/D-005 scoring/replication ambiguities remain open.
GPU is idle after normal cleanup; disk has ~44 GiB free. The one-shot monitor has
successfully queued this review and exited; no experiment restart is needed.

## D-020 — Independent review of the Phase 1 replay — INFO 2026-10-23

Scripts: `scripts/review/r01_phase1_review.py`, `r02_phase1_review.py` (target structure only; no feature-vs-A_replay
association computed, §9.2). Raw replay JSONs are not in git (189 MB); every file's sha256 is in
`eval/phase1_input_manifest.json`.

1. **Table integrity:** A_replay, sigma_floor and crit_sd were recomputed from the raw replay JSONs: 0 mismatches in 2,029
   points. Re-feed divergences 0 over 25,491 continuations. Every lost continuation runs to the 50-step cap, none won
   beyond it, and len = (turn−1) + #actions holds for all of them.
2. **M5 vs CWGT:** |D| 1,617/2,029 = 79.7% (CWGT 86.9%); pivotal 718/1,617 = 44.4% (30.5%); sigma_floor = 0 77.2%
   (86.0%). Same order of magnitude, but the measurability map differs, and that is reported, not adjusted.
   - Mechanism (CWGT App. E divisibility classes, computed for D): constant pool 859 (53.1%; CWGT 68.1%), forced
     nonzero 718 (44.4%; 28.8%), stochastic zero 40 (2.5%; 3.1%). Pivotal therefore ≈ the fraction of non-constant
     pools: our dynamics are less absorbing than CWGT's.
   - Heterogeneity is trajectory-level and bimodal. Per-trajectory pivotal fraction has quartiles 0.05 / 0.63 / 0.78.
     It is 74.8% in the 16 won trajectories (n = 254) vs 38.7% in the 33 lost (n = 1,363); 808 of the 859 constant pools
     are all-fail. The pilot's 31.8% came from 5 trajectories and was not representative.
   - Continuation success: factual 0.191 vs alternative 0.195; rollout success 0.34. Pivotal sign: 299 positive,
     419 negative.
3. **Exclusions (412):** alternatives found = 0: 19, 1: 69, 2: 120, 3: 204. Turn 1 is excluded in 37 of 50
   trajectories: the step-1 policy collapses onto essentially one action. Among defined points, the median draws needed
   is 32 and the 90th percentile is 160. Drafts: 91.3% valid and admissible; `help` never used.
4. **Provisional M5 logprob shift — evidence for resolving D-003:** taking conf = mean logprob of the action-span tokens
   *in the stored rollout response*, i.e. conditioned on the realized `<think>`, the value is degenerate. Median is
   −0.000 and the 10th percentile is −0.03; included-minus-excluded = −0.002 [−0.008, +0.005] vs CWGT −0.70
   [−0.81, −0.59]. Once the reasoning has been written, the action tokens are near-deterministic. CWGT's −0.70 is
   therefore only possible if CWGT's confidence scores the action *without* the realized reasoning (teacher-forced
   after the bare prompt). This favours the bare-prompt reading of `conf_t = logπ(a_t | s_t)`. To be confirmed by HF
   action-only scoring before FEATURES_FROZEN.md is written.
5. **Chosen action not admissible:** 205 points (195 in D, 12.1% of D). The parsed factual action is not in the
   admissible set (e.g. "move ladle 1 to countertop 1", "examine countertop 1"). The env answers "Nothing happens.",
   which is still a legitimate factual arm. Arm-B quantities defined over 𝒜_t (chosen_prob, chosen_rank,
   runner_same_type, …) need a rule for these points → D-021 (OPEN).
6. Parquet `admissible_actions` includes `help`; the prompt shows the set without `help` (verl-agent). Arm B's candidate
   set must be the set shown to the policy, i.e. without `help` → to fix in Phase 2 scoring.

## D-021 — Arm-B features at points whose chosen action is not admissible — OPEN

195 points in D (12.1%) have chosen ∉ 𝒜_t. Options: (a) score candidate set 𝒜_t ∪ {chosen} for all points, so q_t,
chosen_prob and chosen_rank are always defined, and add an indicator control column `chosen_admissible` (recommended;
keeps all of D on the same target); (b) restrict arm B to chosen ∈ 𝒜_t and evaluate all arms on that subset.
Must be decided before FEATURES_FROZEN.md.

## D-022 — Proposal P1: L1/L2 target split and pre-registration amendment — OPEN 2026-10-23

Full text: `proposals/P1_L1L2_restructure.md`. Trigger: D-020. All 718 pivotal points lie inside the 758 non-constant pools,
so `A_replay ≠ 0` measures state criticality, not action contribution. The proposal:
- L1 = state criticality (descriptive); L2 = sign/size of A_replay inside the live set (primary).
- A second independent replay bank with cross-fitting.
- A state-value baseline.
- HCAPO ρ_t as an audited hindsight arm.
- Pre-registered feature families.
- Revised verdict rules: R0 / R1 on L2 / R1a / R1b.
- A cluster-bootstrap CI.
Not adopted: CLAUDE.md is unchanged. Nothing proposed here touches feature-vs-A_replay associations (§9.2).

## D-025 — RULING 2026-10-23 adopted — DECIDED 2026-10-23

`RULING_2026-10-23.md` (researcher) adopted into CLAUDE.md, commit ff7ce9d ("CLAUDE.md: P1 ruling 2026-10-23").
This closes D-003, D-004, D-005, D-006, D-021 and D-022 as ruled:
- **D-003:** conf = bare prompt, teacher-force `<action>{a}</action>`, command tokens only. Confirmed by CWGT
  supplementary `r18_scoring.tex` / `STAGE1_PLAN.md`; their `policy_confidence` values are non-degenerate.
- **D-004:** arm A uses s = −conf; single features use declared prior signs.
- **D-005:** arm H1 = ρ_t carries the M4 target.
- **D-006:** threshold ≥5 primary, ≥4 also reported.
- **D-021:** candidate set = 𝒜 ∪ {chosen} \ {help}, plus `chosen_admissible`.
- **P1:** adopted with the ruling's modifications. 100 trajectories; banks 2 (all D) and 3–6 (𝓛 only); new feature
  `chosen_freq`; H2 judge dropped.
CWGT ICLR-2027 supplementary moved to `downloads/openreview_supp/supplementry/` (not committed: downloaded data,
read-only, none of its scripts executed). It contains no replay archive, so Branch A remains infeasible.

## D-023 — Phase 1C.0 calibration result: the expansion rule fires; evidence that expansion buys little — INFO / FLAG 2026-10-23

Script `scripts/review/r03_cwgt_depth_calibration.py`; output `eval/phase1c0_depth_calibration.json`. 172 CWGT turns,
2,000 hypergeometric resamples. Script check passes: d=16 reproduces the paper exactly (Spearman 0.3301, Pearson 0.7362).

| d | live frac (A) | sign agree, both live & nonzero (ruling metric) | Pearson | pooled Spearman |
|---|---|---|---|---|
| 3 | 0.432 | 0.641 | 0.352 | 0.246 |
| 6 | 0.496 | 0.628 | 0.516 | 0.277 |
| 9 | 0.529 | 0.625 | 0.617 | 0.298 |
| 12 | 0.551 | **0.626** | 0.679 | 0.316 |
| 16 | 0.570 | 0.624 | 0.736 | 0.330 |

**Rule (CLAUDE.md §7 1C.0):** d=12 sign agreement 0.626 < 0.65 → banks 3–6 are expanded to banks 3–10
(24 per arm, about +22 h GPU). The rule fires and will be executed as pre-registered unless the researcher overrides it.

**Flag for the researcher.** Exploratory diagnostics on the same file, run in-session:
- Sign agreement does not rise with depth. It also fails to rise on a fixed turn set: on the 85 turns nonzero in both
  banks at d=16, agreement is 0.649 at d=3, 0.639 at d=6, 0.631 at d=12 and 0.624 at d=16.
- A proxy of our own design (𝓛 from an independent 6-per-arm subsample, then compare two disjoint subsamples) gives
  ≈0.64 at every label depth.
- Interpretation: the limit is not continuation noise. A large share of live turns has a true contrast ≈ 0, so their
  signs are coin flips at any depth. Roughly, ~26% real-effect turns plus ~74% null turns gives 0.26 + 0.74 × 0.5 ≈ 0.63.
- Consequences:
  - Expanding to 24 per arm will very likely not push agreement above 0.65.
  - The L2 sign-AUROC ceiling on CWGT's turn distribution is about 0.6, so R1's absolute floor (lower bound ≥ 0.55)
    may be nearly unattainable.
- Caveat: this is CWGT's turn distribution (four position-quartile midpoints per trajectory), not ours. Ours has a
  different measurability map (D-020).
- Options for the researcher:
  1. Keep the rule as written.
  2. Replace "sign agreement" with a depth-sensitive reliability metric, such as Pearson on a fixed set, before banks
     3+ run.
  3. Restrict L2 to turns with a resolvable contrast in banks 1+2, which changes the estimand.
- No change made. The decision is the researcher's (the rule's threshold was ruled "定死").

## D-024 — Bank seed offset: bank·10¹¹, not bank·10⁹ — DECIDED 2026-10-23 (deviation from RULING §7.5 wording)

`seed_of(task_seed, turn, role, idx) = ((task_seed·100 + turn)·10 + role)·10⁵ + idx` advances by 10⁸ per task seed and
reaches ≈ 9.95·10⁹ at task 99. An offset of bank·10⁹ therefore equals a +10 task-seed shift: bank 2 of task s would
reuse bank 1's seeds for task s+10, violating the ruling's "seeds must be distinct". The offset is bank·10¹¹, which
exceeds the maximum `seed_of` value, so seeds are distinct across banks, trajectories, turns, roles and replicas. The
ruling's intent is kept.
(Minor, pre-existing, harmless: rollout seeds 10⁶·s + turn can coincide with replay `seed_of` values for other
prompts, e.g. rollout (s=1, turn=1) = 1,000,001 = seed_of(0, 1, 0, 1). The prompts differ, so this has no
statistical consequence; recorded for completeness.)

## D-026 — Replay GPU efficiency: measurements and plan — INFO / DECIDED 2026-10-23

The researcher asked whether the Phase 1 replay ignored available speed-ups. Measurements
(`scripts/bench/b01_vllm_ceiling.py`, logs `logs/bench_b01*.log`):
- Workload per generated step: prompt 527 tokens mean (p90 708, max 987) and response 134 tokens mean (p90 180, max 489).
  The stored-prompt sample is 400 prompts.
- Phase 1 lockstep engine: 26.0 steps/s overall (860 iterations, median 20 s per iteration of ~573 requests).
- vLLM ceiling with 1,500 cold real prompts submitted at once under continuous batching: ≈30–35 steps/s. The first run
  was 30.2 steps/s seeded; 29.9 unseeded with max_num_batched_tokens=32768. The workload is prefill-compute-bound,
  ≈26k prefill tok/s ≈ 0.37 PFLOP/s.
- **Correction of an in-session claim:** an apparent 39% penalty for per-request seeds was a benchmark artifact. The
  second run reused the same prompts and hit the prefix cache (49.6 steps/s). Seeds cost ≈ nothing; per-request seeds
  stay.
- Prefill dedup across the 3 replicas of an arm would save only 5.8% of generated-step prompts, because replicas
  diverge quickly.
- 79.9% of continuations are failures and run to the 50-step cap, since ALFWorld has no failure terminal. Early
  stopping would change the estimand, so it is not done.
- fp8 weights or KV, or shortening prompts, would change the policy or the template (§3, §9.1), so neither is done.
Plan: an asynchronous continuous pipeline replaces the lockstep barrier and the serial env time. Expected gain
+15–25%, with protocol, seeds and output schema unchanged. Until it is validated, the Phase 1C chain runs on the
lockstep engine. Both engines produce identical schemas, and points are resumable, so switching mid-bank is safe; the
engine is recorded in provenance. The large lever is design-level: banks 7–10 (≈22–25 h), see D-023 (researcher's
decision).
**D-026 update (measured):** the async engine on GPU (bank 2, s000–s049, 600 live continuations) runs at **28.0
steps/s** steady state, against 26.0 for lockstep (+8%), at GPU utilization 100%. That is ≈93% of the ≈30 steps/s
cold-prompt ceiling, so the run is now compute-bound on prefill. The chain was switched to the async engine at commit
c3c71c9 with zero completed points lost.
Projected chain time:
- bank 2 (s000–049): ≈5 h
- s050–099 rollout + bank 1: ≈6 h
- bank 2 (s050–099): ≈5 h
- banks 3–6 on 𝓛: ≈24 h
- banks 7–10 (expansion, D-023): another ≈24 h
Creating `logs/STOP_EXPANSION` before banks 7–10 start makes the chain skip them.

## D-027 — User cancels banks 7–10 — DECIDED 2026-10-24

User explicitly instructed skipping banks 7–10 and creating logs/STOP_EXPANSION.
Marker created at 10:16 +0800 while bank 3 is running. The existing chain checks
this marker after bank 6 and exits without starting banks 7–10. Banks 3–6 remain
scheduled; label depth is therefore 12 continuations per arm, rather than 24.
This overrides the D-023 triggered expansion; disclose this design change in results.

## D-028 — Save, verify and shut down after current queue — DECIDED 2026-10-24

User explicitly requests /usr/bin/shutdown after tasks complete and files are
saved to disk; also shut down after unrecoverable failure. Current GPU queue is
banks 1–6; banks 7–10 remain canceled. Attach scripts/finish_phase1c.py to the
existing chain PID/start time, without restarting it. On exit, run the integrity
audit and require exact live-set coverage in banks 3–6. Archive project code,
raw results, prompts, metadata and logs under artifacts/, read back and verify
every archived file SHA256, fsync archive/manifest/status, and sync before shutdown.
On audit/archive failure, persist errors and queue recovery review in this thread;
reviewer must re-arm finalization after recovery or save errors and shut down if
unrecoverable. If queue delivery itself fails, persist that failure, sync originals
and shut down rather than leave GPU billing unattended. Shutdown was mocked in
tests; it must not run while the queue is active. This stops after data collection,
not a claim that Phase 2/3 analyses have been performed.

## D-028 — Mid-chain QC of Phase 1C (2026-10-24) — INFO

`scripts/review/r04_phase1c_qc.py`, read-only, no features. Chain state: rollout s050–099, bank 1 (s050–099), bank 2 (all D)
are DONE; bank 3 (live set, 1,478 points, 22,170 continuations) is ≈95% done; banks 4–6 follow. Banks 7–10 are skipped (D-027).
- **Invariants:** 0 violations in banks 1–3. Checked: outcome counts, len = (turn−1)+#actions, no win past the 50-step cap,
  every loss reached the cap, bank-2/3 alternatives and factual action identical to bank 1, first action correct, seeds
  inside the bank's range, 0 duplicate seeds, re-feed divergence 0 over ≈1.5M generated steps.
- **New rollouts s050–099:** success 0.42 (old 0.34), valid 0.9994, admissible-hit 0.897 (old 0.899), same format mix; all six
  task types appear, `pick_two_obj_and_place` still 0/8 won. No-cf rate 20.6% vs 20.3% on the old 50, so the 100-trajectory
  pool is homogeneous.
- **Sizes:** |D| 3,038 (1,617 + 1,421); live set (banks 1+2, 30 continuations) 1,478 = 48.7% of D; 59 of the 79 trajectories
  with a live point have ≥5 of them. This is more than the earlier 31 trajectories.
- **Cross-bank (preview, NOT a result):** on the 1,355 points already in bank 3, bank 1 vs 2 A: Pearson 0.231, sign agreement
  0.596 (n = 949 both-nonzero). The set is selected on pool non-constancy of banks 1+2, so these are biased upward for the live
  flag and are shown only as a plumbing sanity check. They are consistent with D-023 (3-per-arm labels are noisy).
- **Disk:** data 0.5 GB; free 22 GB; no growth risk.
No code or process change needed.

## D-029 — W1/W0 peek extraction as P2 groundwork (not yet a pre-registered arm) — DECIDED 2026-10-24

The user asked for groundwork for the planned P2 extension ("Where vs Which": world-side one-step peeks of non-chosen
actions as the source of credit *sign*; see the P2 proposal when written). This does not amend CLAUDE.md. W1/W0 are not
arms until a ruling adopts P2, and no association with any replay outcome is computed before `FEATURES_FROZEN.md` is
committed (§9.2). `scripts/22_peek_W1.py`:
- **What:** for every rollout step with a bank-1 file (all 3,818 rollout steps: the 3,038 points of D plus 780 cf-undefined points), reset + re-feed
  the prefix (verified against stored observations), execute ONE action, and record obs, admissible set, won/done and
  the TextWorld fact delta. No LLM is used and the policy never sees a peek. Bank-1 files are read only for their
  alternative list (sha256 recorded); outcomes are not read.
- **Sets:** `alts` = factual + the bank-1 alternatives (≤4), plus the full fact set at s_t (W0, privileged; diagnostic
  ceiling only, never a method). `admissible` = every admissible action except `help`, plus chosen; facts as deltas.
- **Determinism checks per point:** state obs = stored `obs_before`; factual peek obs = stored `obs_after`; re-feed
  divergence 0; on points with sha256(traj|turn) % 20 == 0, every peek is run twice and compared.
- **Cost:** TextWorld envs cannot be copied (ctypes pointers; deepcopy fails), so every peek is one reset (≈1 s) + re-feed
  (≈0.08 s/step). `alts` = 17,708 peeks (measured ≈0.66 points/s with 10 workers, ≈1.6 h); `admissible` ≈ 120k peeks ≈ 90 CPU-h. The container quota is 22 CPUs and
  the Phase 1C chain uses ≈8. `alts` therefore runs now with 10 workers at nice 19. `admissible` waits until the chain
  has exited, then uses 20 workers.
- **Facts dependence:** requesting `facts` does not change dynamics (s000: 19 steps, 0 divergences; peeks match stored
  observations).
- **Provenance note:** `scripts/review/r04_live_integrity.py` is untracked (pre-existing), so `scripts_dirty` is True for
  outputs produced now, including chain banks started after it appeared. The peek outputs therefore also record the
  script's own sha256.
- **Status 2026-10-24 23:48:** `alts` is done: 3,818 points, 17,708 peeks, 0 check failures, 5,031 s (`data/peek_alts/`,
  `logs/p2_peek_alts.log`). The user stopped the queued `admissible` pass before it started; it is not scheduled.

## D-030 — Proposal P2 (Where vs Which) registered — OPEN 2026-10-24

`proposals/P2_where_which.md`: merges M0 with the collaborator's information-ladder plan (2026-10-22). Model-side
structure (arms B/C, recomputable from logs + θ_old) is hypothesised to locate credit (L1, magnitude); world-side
one-step peeks (W1) are hypothesised to give its sign (L2). New rungs: siblings G, W1, W0 (diagnostic), budget replay /
REV, pair-level L2 target, a 300-point gold set at 32 runs/arm. M0 §4–§8 remain unchanged until a ruling. Five open
questions for the user are listed in §7 of the proposal. The novelty check (CRR, Causal Agent Replay, GraphHCA, ECHO,
…) is appended to the proposal when finished; a direct prior match is a stop condition (CLAUDE.md §2).
**Novelty check result (subagent, abstracts + partial full text; quotes to be verified before writing):** no stop
condition. Closest partial overlaps:
- TreeHCA 2610.12828 executes every candidate turn in the environment, but for branch selection in search QA. It has no
  replay ground truth and no ladder.
- Critical-State RL 2609.24985 locates states by executed action variance (sign-agnostic) and leaves direction to RL. It
  is also a near neighbour of M0's L1/L2 split, so M0 must cite it.
- CRR 2609.33875 picks fork points by entropy, prices fork cost, and has an immediate-invalid diagnostic.
P2 must be claimed as an empirical source decomposition (where vs which, priced in rollouts). It must not claim to be
first to execute candidates or to pick fork points by entropy. Details are in the proposal's appendix.

## D-031 — Phase 1C complete; 1C.4 labels and the L2 reliability ceiling — INFO / FLAG 2026-10-25

**Chain:** banks 1–6 complete and archived (`artifacts/phase1c_20261025_035544/`, finish audit passed on all 11,768
files, re-feed divergence 0). Banks 7–10 were skipped (D-027).

**Labels:** `scripts/14_labels_L.py` → `data/labels_L.parquet`, `eval/phase1c4_labels.json`. It reads labels only, no
features (§9.2).
- |D| 3,038. 𝓛 (banks 1+2 live) has 1,478 points over 79 trajectories; 59 have ≥5 𝓛 points and 62 have ≥4. Banks 3–6
  cover exactly 𝓛 (asserted). The bank-1-only 𝓛 (1,311 points) is a strict subset.
- L1 two-bank agreement of the live flag is 0.902 (live rate 43.2% vs 44.3%).
- L2 sign counts (12 per arm): + 575 / − 767 / 0 136. 69 𝓛 points have a constant pool in banks 3–6.

**L2 reliability ceiling (§6.0):** banks 3+4 vs 5+6 (6 per arm each), on 𝓛:
- sign agreement **0.601** (n = 1,155 both nonzero);
- Pearson 0.321; Spearman 0.240;
- Spearman–Brown estimate for the 12-per-arm label: 0.49.

The §8 flag threshold is 0.60, so L2 sits **on** the line and is not formally "标签信度不足". This matches the CWGT
calibration (D-023: ≈0.63 at every depth).

**Replay-oracle benchmark (label-only, exploratory):** the score is an independent replay of the same contrast; the
label is sign_L2; values are AUROC with trajectory cluster bootstrap CI, n = 1,342 nonzero points, 75 trajectories.

| Score | AUROC [95% CI] |
|---|---|
| Banks 1+2 (6 per arm) | 0.637 [0.615, 0.658] |
| Bank 1 (3 per arm) | 0.619 [0.591, 0.647] |
| Half-split, banks 3+4 → sign(banks 5+6) | 0.637 |
| Replay-derived pool success rate of banks 1+2 (state-value proxy, not forward) | 0.553 [0.512, 0.589] |

**Implications for the researcher (no rule changed):**
1. R1 ① requires the arm-B lower bound ≥ 0.55, while a second independent replay reaches only 0.637 (lower bound
   0.615). A forward feature would have to recover most of what an extra replay knows, so R1 ① is near-unreachable
   at this label depth. R4 or R0 is the likely outcome even if real structure exists. One option is to express effect
   size relative to the replay-oracle ceiling, e.g. (AUROC − 0.5) / (0.637 − 0.5), and keep R1 ①'s shuffle condition.
   This would be a §8 amendment and is the researcher's decision before Phase 3.
2. State value is a real competitor: the replay-derived state-value proxy alone gives 0.553. Beating V (R1 ②) is
   necessary, not a formality.
3. Most 𝓛 turns appear to have a near-zero true contrast (D-023), and more depth does not fix that. Restricting L2 to
   turns with a resolvable contrast would change the estimand.

## D-032 — Claude's assessment appended to P2; pair-level label reliability — INFO 2026-10-25

Appended "附二" to `proposals/P2_where_which.md` (an assessment, not a ruling).
New label-only measurement (§9.2-safe): pair-level contrasts C_j, banks 3+4 vs 5+6 (6 per arm each), 5,912 pairs on 𝓛.
- Both halves nonzero: 2,643 pairs (45%).
- Sign agreement 0.614.
- Pearson 0.413 (aggregate A: 0.601 / 0.321).
Recommendation: adopt P2 with three modifications.
1. H-Which is judged on reliability-corrected ΔR²; sign AUROC is secondary.
2. All L2 verdicts are reported relative to the replay ceiling, together with REV. This is a §8 amendment.
3. W1 is split into pure-world features (W1-N/T) and model-processed features (W1-V/P).
Scope: ALFWorld + E1 + WebShop is the core; search QA, multiple models and E2 are conditional.
Awaiting the researcher's ruling.

## D-033 — Phase 2 preconditions run (2.0 D-003 check, HF–vLLM alignment) — INFO / DECIDED 2026-10-25

Scripts: `scripts/13_align_check.py` (points: numpy seed 0; 140 random from D + 60 random cf-undefined; the first 20
D points are the alignment subset), `scripts/review/r05_backend_diff.py`. Outputs: `eval/phase2_prereq.json`, `eval/prereq/`.
Only `cf_defined` was used; no A_replay or replay outcome was read (§9.2). No main-experiment scoring was run.

**(a) D-003 pipeline check (HF bf16; chosen conf = mean command-token log-prob, trajectory cluster bootstrap):**

| Construction | Included − excluded [95% CI] | Median conf |
|---|---|---|
| bare (primary) | −0.388 [−0.656, −0.102] | −0.72 |
| emptythink | −0.335 [−0.620, −0.004] | −0.37 |
| realized think | −0.031 [−0.076, −0.005] | −0.0000068 (degenerate, as predicted in D-020) |
| CWGT | −0.70 [−0.81, −0.59] | — |

Verdict: **pass.** The sign is negative and the CI excludes 0. The magnitude is about half of CWGT's, but the CIs overlap
(n = 200, a different task list). Sharp contrast with the realized-think construction confirms that the bare construction
is the right reading of CWGT's confidence.

**(b) HF vs vLLM candidate scoring (20 points, 662 candidate pairs):**
- Per-point Spearman: bare median 0.994 (min 0.958); notask median 0.998 (min 0.988). **Gate (median > 0.95) passes.**
- Top-1 agreement 17/20. The three disagreements had HF top-1/top-2 gaps of 0.09, 0.67 and 0.96 nats.
- Absolute deviations are larger than expected. |Δ logp_sum| has median 0.42 nats, p90 1.06, max 2.39; mean Δ (vLLM − HF) = −0.21.
- fp32 reference (HF fp32, same token ids):

  | Backend | Mean error | \|error\| p50 | \|error\| p90 | Spearman median |
  |---|---|---|---|---|
  | vLLM bf16 | −0.217 | 0.328 | 0.839 | 0.997 |
  | HF bf16 | −0.002 | 0.308 | 0.931 | 0.996 |

  Both bf16 backends carry the same-sized per-candidate noise (≈0.3 nats median), which is bf16 accumulation over ≈600-token
  prompts. vLLM adds a ≈ −0.22 nat shift, near-uniform within a point; it cancels in q_t, margins and ranks, and shifts
  absolute conf.

**Decision:** primary scoring backend = **vLLM bf16** (CLAUDE.md §3; the same numerics the policy acted under during rollout
and replay). Every arm-A/B/H1 quantity comes from this single backend; hidden states come from HF bf16 (arm C, V).
- **Noise floor disclosed:** candidate log-prob differences below ≈0.3 nats are within numerical noise. This affects
  `margin_raw`/`margin_mean` near 0 and `chosen_rank` near-ties, and goes into FEATURES_FROZEN as a stated limitation.
- **Option not taken:** fp32 scoring (HF) for the main run, ≈3–4× the bf16 cost (≈10 h vs ≈2.7 h), only on the
  researcher's request.
- The M5 conf shift is recomputed from the full vLLM scores in Phase 3; the HF numbers above are the 2.0 precondition only.

## D-034 — RULING 2026-10-25 received; adoption pending; shared Phase 2 work started — DECIDED 2026-10-25

`RULING_2026-10-25.md` (three regimes, B1/B2/B3) has been pulled. The researcher is still deciding whether to adopt it,
so CLAUDE.md is **unchanged**. Note: the ruling refers to its own 5.1/5.2 record as "D-033", but D-033 is already the
Phase 2 precondition entry; the record is D-035.

The researcher asked to run, now, only the work that is identical under both versions. Started
`scripts/run_phase2_shared.sh` (GPU, detached):
1. vLLM bf16 raw scoring of all 3,818 points: candidates × {bare, notask}; chosen × {emptythink, realized, hindsight}.
   Output `data/scores.parquet` and `data/scores_chosen.parquet`.
2. HF bf16 two-position hidden-state export (arm C and V inputs). Output `hidden/`.

Both versions use identical arms A/B/C/V/H1 inputs. **Procedural deviation:** CLAUDE.md §7 orders 2.1 (commit
FEATURES_FROZEN) before 2.2 (scoring). Raw log-probs and hidden states are produced before the freeze, but no
feature is derived and nothing is joined with any label until FEATURES_FROZEN.md is committed. Feature definitions are
already fixed in CLAUDE.md §4. §9.2 (no association before the freeze) is untouched.

Not started, because these exist only in the new ruling:
- CLAUDE.md §1/§6/§8 edits;
- gold set (≈7 h);
- siblings G;
- a′_u sensitivity;
- W1 peek-state hidden states (W1-V/P);
- W1-D;
- E1;
- FEATURES_FROZEN content specific to B1/B2/B3.

## D-035 — RULING 2026-10-25 §5.1–5.2 computed (label-only) — INFO 2026-10-25

`scripts/15_regimes.py` → `eval/regimes.json`, `eval/pivotal_rule.json`, `eval/ceilings.json`. Labels only (§9.2-safe).
Run as decision support; it does not imply adoption.

**Three regimes (D, n = 3,038; trajectory cluster bootstrap 95% CI):**

| Regime | n | Fraction [95% CI] |
|---|---|---|
| Dead | 1,560 | 51.3% [42.8, 60.0] |
| Indifferent | 1,249 | 41.1% [33.8, 48.6] |
| Pivotal (Newcombe 90%, banks 3–6) | 229 | 7.5% [5.7, 9.4] |

- Dead pools: 1,508 all-fail, 52 all-success.
- By trajectory outcome:
  - won trajectories (n = 455): dead 10.3%, indifferent 74.1%, pivotal 15.6%;
  - lost trajectories (n = 2,583): dead 58.6%, indifferent 35.3%, pivotal 6.1%.
- By progress tercile (dead / pivotal): early 39.5% / 8.1%, middle 45.6% / 7.6%, late 68.3% / 7.0%.
- Pivotal counts: sensitivity rule 193; overlap with the main rule 187. Signs + 107 / − 122. 54 trajectories contain a
  pivotal point; 36 contain ≥3. Gate |Pivotal| ≥ 150 passes.
- Split-half sign agreement (banks 3+4 vs 5+6): 0.978 inside Pivotal (partly circular, since Pivotal is selected on the
  pooled banks 3–6) and 0.511 inside Indifferent (n = 932, coin-flip, as the regime definition predicts).

**Depth invariance on our own data (𝓛, independent banks 3–6 only):**

| Depth per arm | Sign agreement | Pearson |
|---|---|---|
| d = 3 (6 single-bank pairs) | 0.593 | 0.186 |
| d = 6 (3 half-splits) | 0.600 | 0.314 |

The sign gain is +0.007, below the ruling's 0.05 "change my mind" threshold. Sign invariance with Pearson rising
replicates the CWGT pattern (D-023). Mixture fit p (fraction with a real effect) = 0.25 / 0.22 / 0.20 for
q = 0.90 / 0.95 / 1.00.

**Replay ceilings B_m (AUROC, mean of 200 subsamples, banks 1+2):**

| Target | m=1 | m=2 | m=3 | m=6 |
|---|---|---|---|---|
| B2 pivotal on 𝓛 (score \|A\|) | 0.604 | 0.607 | 0.616 | 0.661 |
| L2-sign on 𝓛 | 0.576 | 0.593 | 0.606 | 0.637 |
| B3 sign within Pivotal | 0.706 | 0.755 | 0.790 | 0.836 |
| B1 live (label bank 1, score bank-2 pool variance) | 0.829 | 0.902 | 0.930 | — |

The B1 row deviates from RULING §2.4 because the Live label is built from banks 1+2, so the score must come from an
independent bank.

**Reading:**
- B1 is easy, with a replay ceiling ≈ 0.93.
- B2 is hard even for replay: 6 extra continuations per arm reach only 0.66. A cheap signal that passes B2 would be
  worth more than one replay; REV makes that measurable.
- B3 inside Pivotal is well defined (0.84 at 6 per arm), unlike L2-sign on the whole of 𝓛 (0.64).

## D-036 — Scoring throughput: shared-prefix KV cache (HF), fp32 second scoring pass — DECIDED 2026-10-25

**Finding.** The vLLM run sits at 100% GPU but recomputes the ≈600-token prompt for every candidate:
- ≈38 sequences/s ≈ 23k prefill tokens/s, which is the prefill ceiling (D-026);
- requests that ask for `prompt_logprobs` do not reuse the prefix cache;
- the ≈43 candidates of a point share the prompt and differ only in 5–10 command tokens, so ≈95% of the compute is repeated.

**Change.** `20_score_candidates.py --backend hfprefix` (`HfPrefixScorer`):
- one prefix forward per distinct context;
- the KV cache is replicated over the candidates (`DynamicCache.batch_repeat_interleave`);
- only the command tokens are fed.

**Validation** (CPU, Qwen2.5-0.5B, fp32, 2 points, 184 sequences): max |Δ| = 3.2e-5 and mean 3.2e-6 vs full-sequence scoring,
8.6× faster.

**Plan.** The running vLLM bf16 job (≈1 h left at 400/3,818 points) is not interrupted. After the shared chain
(scoring + hidden states) completes, `scripts/run_fp32_after_chain.sh` runs a second full scoring pass:
- HF fp32 + prefix cache → `data/scores_fp32.parquet`;
- estimated ≈30–40 min GPU.

**Effect.** Removes the ≈0.3-nat bf16 noise floor of D-033. **Which pass is primary is an open choice for the researcher:**
- fp32: clean numerics;
- vLLM bf16: same numerics as the acting policy.

Recommended: fp32 primary, vLLM bf16 as a numerical-robustness sensitivity row.

## D-037 — a′_u sensitivity scheduled before the fp32 pass (user instruction) — DECIDED 2026-10-25

The researcher asked to run the a′_u check (RULING 2026-10-25 §7 row 3) ahead of the fp32 scoring pass. It is a
label-only experiment and does not imply adopting the ruling.
- **Points:** `scripts/16_select_au.py` → `eval/au_points.json`. 100 Indifferent points, drawn with numpy seed 0 from
  the 1,249 Indifferent points (regime rule of D-035), covering 41 trajectories. None were skipped.
- **Alternative a′_u:** uniform over admissible \ {help, chosen, the 4 bank-1 alternatives}, with
  rng = default_rng(1,000,003·task_seed + turn). Median pool size is 27.
- **Run:** `11_replay.py --engine async --bank 10 --points D --points_file eval/au_points.json --n_rep 6 --no_factual`
  → `data/replay_au/`. The bank index is 10 (seed offset 10¹², D-024 format) and the factual arm is not re-run.
- **Comparison:** factual from banks 3–6 (12 runs) vs a′_u (6 runs), using the same Newcombe 90% rule. "Pivotal under
  a′_u" means the interval excludes 0. Report the fraction with a cluster CI, and also against the bank-1 alternatives'
  mean on the same points. The ruling's reading applies: if most points turn pivotal, Indifferent is policy-relative.
  This is a sensitivity analysis and enters no verdict.
- **Dry run** (random policy, 100 points): 600 continuations, schema correct, re-feed divergence 0.

GPU queue: shared chain (vLLM scoring → hidden states) → a′_u (≈20–40 min) → fp32 hfprefix pass.

## D-038 — Auto-shutdown after a'_u and fp32 jobs — DECIDED 2026-10-25

User explicitly requests automatic shutdown after these two queued jobs, error-triggered
Codex review/recovery, and saved error logs if recovery fails (prior shutdown instruction
still applies to unrecoverable failure). Attach finish_after_chain.py to the existing
run_after_chain.sh PID/start time. Validate all selected a'_u points and six replicas,
and fp32 score row coverage, finite values and provenance; archive and read-back hash
check code/data/logs via the existing finalizer, fsync/sync, then /usr/bin/shutdown.
On failure, persist logs and queue this conversation to repair/resume only these jobs;
reviewer must re-arm finalization or save failure and shut down. If queue unavailable,
save that error and sync before shutdown. Tests mocked queue and shutdown; neither
was actually invoked. Banks 7–10 remain canceled; a'_u uses offset bank index 10 only.

## D-039 — a′_u sensitivity result: Indifferent is not an artefact of the policy's alternatives — INFO 2026-10-25

Run: `logs/au_replay.log`. 100 points, 600 continuations, re-feed divergence 0. Analysis: `scripts/17_au_analysis.py` →
`eval/au_sensitivity.json`; calibration: `scripts/review/r06_au_null.py`. Label-only.

| Comparison (same 90% Newcombe rule, same 100 Indifferent points) | Flag rate |
|---|---|
| factual (6) vs factual (6): true difference 0, the rule's own false-positive rate | 0.020 |
| factual (12) vs ONE policy alternative (6), 400 pairs | 0.098 |
| **factual (12) vs a′_u, uniform admissible alternative (6)** | **0.110** [cluster 95% CI 0.055, 0.173] |

- a′_u differs from the pooled policy alternatives (48 runs) at 16% of points.
- At 3 points a′_u is worse than factual and at 8 it is better.
- Mean success: factual 0.268, policy alternatives 0.285, a′_u 0.258.

**Reading:** a uniform-random admissible alternative makes an Indifferent point "pivotal" no more often than one of the
policy's own alternatives does (0.110 vs 0.098). So Indifferent is **not** explained by the policy sampling similar
alternatives. RULING 2026-10-25 §7 row 3 does not fire; the ruling's 70% confidence item is supported.
- **Nuance:** single alternatives flag at ≈10% vs the 2% exact null. Regimes are defined on the aggregate of 4
  alternatives, and some individual alternatives within Indifferent points do differ from factual. Pair-level effects
  exist but average out.
- **Caveat:** 82/100 a′_u are `go to` actions, because the admissible set is dominated by navigation. This tests a random
  admissible alternative, not a semantically extreme one. It is a sensitivity analysis and enters no verdict.

## D-040 — fp32 scoring pass checked; correction to D-033 — INFO 2026-10-26

The a′_u + fp32 queue completed on 2026-10-25 at 18:56. The finalization was recovered manually on 2026-10-26 at 10:29 and
archived in `artifacts/phase1c_20261026_102929`, result `both_tasks_complete_and_verified`.
Check script: `scripts/review/r07_fp32_check.py` → `eval/phase2_fp32_check.json`. It reads scores plus `cf_defined` only.

1. **Integrity:** fp32 and vLLM have the same 116,960 candidate rows and 11,454 chosen-variant rows. No key is missing
   on either side, there are 0 non-finite values, 0 n_tokens mismatches (identical tokenization) and 0 is_chosen
   mismatches, and the model hash is a09a3545 in both. `realized` is available at 3,816 of 3,818 points.
2. **Prefix-cache scorer validated on the real 7B on GPU:** against full-sequence HF fp32 (D-033 reference, 20 points,
   662 pairs), max |Δ| = 1.1e-3 (bare) and 0.9e-3 (notask).
3. **fp32 vs vLLM bf16, all 3,818 points:**

   | Quantity | Value |
   |---|---|
   | Per-candidate \|Δ logp_sum\| | median 0.28, p90 0.86, max 6.7 |
   | Mean Δ | −0.03 |
   | Per-point mean shift (cancels in q_t / margins / ranks) | mean −0.04, SD 0.39; \|shift\| > 1 nat at 91 points (2.4%), > 2 at 5 |
   | Within-point residual \|Δ\| | median 0.17 |
   | Per-point Spearman | median 0.997 (p5 0.989, min 0.93); notask median 0.998 |
   | Agreement | top-1 94.1%, ordered top-2 84.9%, `chosen_rank` 81.0% |

   **Correction to D-033:** the "≈ −0.22 nat systematic vLLM shift" came from 20 points. At full scale the mean shift is
   −0.03, with a point-to-point SD of 0.39, so the shift is point-specific and not systematic.
4. **Noise floor:** at 9.3% of points the fp32 top-1/top-2 gap is < 0.3 nats (3.4% are < 0.1). Top-1 agreement across
   backends is 62% at those points and 97% elsewhere. Rank and margin features near ties are backend-dependent,
   which supports fp32 as the primary pass (recommendation of D-036; the decision is still the researcher's).
5. **Conf shift, M5 sanity row, all 3,818 points:**
   - included − excluded = −0.494 [−0.574, −0.414] (fp32) and −0.497 [−0.579, −0.412] (vLLM bf16);
   - n = 3,038 / 780; median conf −0.75;
   - CWGT reference: −0.70 [−0.81, −0.59].

   The sign replicates, the magnitude is ≈0.7× CWGT, and the intervals are adjacent without overlapping. Per the
   D-003 rule this is recorded and we continue. Possible reasons: a different task list, and a less absorbing
   measurability map (D-020).

## D-041 — Primary scoring backend = HF fp32 (shared-prefix KV cache) — DECIDED 2026-10-26 (researcher)

The researcher chose fp32. Every arm-A/B/H1 quantity (candidate log-probs, q_t, margins, ranks, conf, ρ_t) is computed
from `data/scores_fp32.parquet` and `data/scores_chosen_fp32.parquet` (scorer "hfprefix", D-036, validated in D-040).
The vLLM bf16 pass (`data/scores.parquet`, `data/scores_chosen.parquet`) is kept as a numerical-robustness
sensitivity row and is not used for any primary number. File names are left unchanged (archive manifests reference
them). CLAUDE.md §3 and the §5 `scorer` field are updated accordingly; this is an environment-lock change recorded
here as §3 requires. Hidden states remain HF bf16 (`hidden/`).
Reasons: fp32 removes the ≈0.3-nat per-candidate bf16 noise (D-033/D-040). Rank and margin features otherwise flip at
the 9.3% of points with near-ties (cross-backend `chosen_rank` agreement 81%). Neither bf16 pass reproduces the
acting policy's exact sampling-time numerics anyway (D-015).

## D-042 — FINAL RULING §6.2 computed (label-only): heterogeneity forks, pair-level contrasts, T-ref, cross-table — INFO / FLAG 2026-10-26

`scripts/18_het_fork.py` produces `eval/het_fork.json`, `eval/pairs.json`, `eval/het_points.parquet` and
`eval/pairs_points.parquet`. `scripts/review/r07_rule_null.py` produces `eval/rule_null.json`. Labels only (§9.2-safe);
commit 64d325d, clean provenance. Pivotal_agg was recomputed as 229 (asserted equal to D-035).

**Pivotal_het** (5-arm G test, 10,000 permutations, α = 0.10): 475 / 1,478 = 32.1% [27.3, 37.1].

| 2×2 on 𝓛 | agg | ¬agg |
|---|---|---|
| het | 170 | **305** (masked) |
| ¬het | 59 | 944 |

- Masked forks (het ∧ ¬agg) = 20.6% of 𝓛 [17.2, 24.3].
- **P5 statistic** = masked / |het| = 0.642 [0.592, 0.690].
- **Null calibration** (same permutation null applied to every live point; an upper bound on false flags):
  E[het] = 103.9, E[agg] = 118.3, E[masked] = 68.9, against observed 475 / 229 / 305.
  - FP-adjusted masked share ≈ (305 − 69) / (475 − 104) ≈ 0.64 (rough).

**Pair-level C_j** (factual 12 vs alternative j 12, Newcombe 90%), 5,912 pairs:
- sig 1,016 = 17.2% [15.1, 19.2]; null expectation 516 (8.7%).
- Inside Pivotal_agg: 51.3% sig (null 10.7%). Inside Indifferent: 10.9% sig (null 8.4%), an excess of ≈ 2.5 pp,
  consistent with the D-039 nuance.
- Signs of sig pairs: factual better 442, worse 574.
- Number of sig pairs per point: 0 → 808, 1 → 443, 2 → 140, 3 → 55, 4 → 32.

**T-ref:** 227 live points have ≥2 sig pairs. Of these, 14.1% [8.3, 20.9] have sig pairs of both signs (the
contribution's sign depends on the reference alternative). a′_u flip rate = 0.110 (D-039, copied).

**Three regimes, outcome × progress tercile** (fractions, cluster 95% CI):

| cell | n | dead | indifferent | pivotal_agg | pivotal_het | pivotal_agg among live |
|---|---|---|---|---|---|---|
| lost, early | 783 | 0.460 | 0.470 | 0.070 [0.048, 0.093] | 0.153 | 0.130 |
| lost, middle | 919 | 0.517 | 0.415 | 0.069 [0.042, 0.097] | 0.149 | 0.142 |
| lost, late | 881 | 0.770 | 0.185 | 0.045 [0.023, 0.075] | 0.098 | 0.197 |
| won, early | 148 | 0.054 | 0.811 | 0.135 [0.085, 0.189] | 0.297 | 0.143 |
| won, middle | 169 | 0.124 | 0.757 | 0.118 [0.068, 0.174] | 0.249 | 0.135 |
| won, late | 138 | 0.130 | 0.645 | 0.225 [0.165, 0.277] | 0.333 | 0.258 |

Pivotal_agg among live, by progress tercile: 13.3% [10.9, 15.7] → 14.0% [10.6, 17.6] → 22.0% [16.3, 28.2].

**Flags for the researcher (no design change made):**
1. **The Pivotal rules have a high false-flag rate.** Under the conditional permutation null, the main rule
   (12 vs 48, 90% Newcombe) flags ≈ 8% of null live points, so E[flags | all null] = 118 of the 229 observed. For
   the sensitivity rule it is 86 of 193 (r07). The conservative precision floors are 0.49 and 0.56. If 75–80% of
   live points are truly null (mixture fit, D-035), roughly 90 of the 229 main-rule points may be false, giving
   precision ≈ 0.6. The §6.3a mechanical switch (precision < 0.70) is therefore likely to fire. The sensitivity
   rule may also fall below 0.70, and the ruling has no further fallback. The headline "Pivotal 7.5%" includes these
   false flags; gold will measure them directly.
2. **Gold precision definition.** "Interval still excludes 0 at 32/arm" pools the 12 runs that selected the point
   into the 32, which biases precision upward. Recommendation: also report precision on the 20 new independent runs
   alone. The pre-registered switch stays as written unless the researcher rules otherwise.
3. **P5 is already settled by this label-only step.** The P5 statistic is 0.642 ≥ 0.15, and this is known before
   FEATURES_FROZEN.md is committed. P5 is therefore a reported fact, not a forward prediction. The frozen file
   should say so.

## D-043 — FINAL RULING execution choices (CLAUDE.md edit, gold, siblings, peek states, W1-J, GPU queue) — DECIDED 2026-10-26

**CLAUDE.md edit (commit 602b23f):**
- §1, §6.0, §6.3, §8 replaced by the ruling text; §4.1–4.3 added (features 20–40, F6–F9, P1–P5, cost vector);
  §6.3a gold; §6.6 E1; Phase 1D execution order. Phase 2/3/4 got pointer notes.
- §4's identification rule and prior-sign rule now refer to B1/B2/B3 instead of L1/L2.
- Interpretation choices:
  - M2 / M3 / M4 / V partial regression are kept as report items under §6.3. M3 is the stratified shuffle used in
    "passes the bar"; M4 is needed for the ρ_t T3 gate. M7 is dropped (replaced by B2/B3/L2-sign).
  - The CWGT label is named `pivotal_cwgt` (bank-1 A ≠ 0), to keep it apart from Pivotal_agg.
- §9.7 now reads "results are only the metrics defined in §6".

**Gold selection** (`scripts/19_select_gold.py` → `eval/gold_points.json`, seed 20261026):
- G-unbiased: 200 points, proportional allocation over 51 strata (48 get ≥1 point), 58 trajectories.
  - n_draws terciles are **value-based** (cut points = 1/3 and 2/3 quantiles: {32}, {64}, {≥96}; bin sizes
    902 / 250 / 326). n_draws comes in multiples of the 32-draft batch and is heavily tied, so rank terciles would
    only split identical values at random.
- G-candidate: 100 points drawn from the 250 𝓛 points with |A_3456| ≥ 0.2 that are not in G-unbiased. All 300
  points are distinct; 44 trajectories.

**Gold replay seeds:**
- One bank index, 6 (seed offset 6·10¹¹), with `--n_rep 20`, instead of bank indices 7–13.
  - 20 extra runs per arm is not a multiple of 3.
  - Bank index 10 is already used by a′_u (D-037).
  - The replica index (0–19, in `seed_of`'s idx = replica·100 + step) keeps every request seed distinct.
- New `11_replay.py --subset_file` option: runs the listed points, keeps the factual and the bank-1 alternatives.
  Output `data/replay_gold/`.
- Dry run with the random policy (2 points, 2 reps): 20 continuations, alternatives equal to bank 1, seed offset
  correct, re-feed divergence 0.
- The D-028-style finish audit is done at step 6.4.

**Siblings:**
- New `10_rollout.py --siblings k-k` option, run for k = 1…7 separately (100 envs per run, as in the original).
- Sample seeds = 10⁶·seed + turn + (20+k)·10¹¹; traj_id `s###_k#`; output `data/siblings/`, `prompts/siblings/`.
- The default path (k = 0) is unchanged. This could not be tested here: 10_rollout has no dry policy and the
  instance has no GPU.

**Peek states** (`scripts/24_peek_states.py`):
- D only: 3,038 points × 5 = 15,190 states. Prompt = rollout memory + (obs_t, a), current obs = peek obs,
  admissible = peek set.
- Check: the factual-peek prompt equals the stored prompt of turn t+1 (2/2 in the CPU test; counted in the run).
- `--part hidden`: bf16, the same pipeline as `hidden/`, so the V probe transfers. Output `hidden_peek/`.
- `--part score`: fp32 hfprefix over the peek admissible set. Output `data/scores_peek_fp32.parquet`.
  - W1-P needs the policy's candidate distribution at the peek state. The ruling's "≈1 h" did not include this
    scoring; estimate +1–1.5 h.
- CPU test with Qwen2.5-0.5B passed. The no-GPU instance has a 2 GB memory limit, so the test used
  `--cand_batch 4`.

**W1-J (DRAFT, awaiting researcher confirmation):**
- Template `prompts/w1_judge_template.txt`, sha256 `9804c516…e233b1`. Chat template, temperature 0,
  max_tokens 4. Label = first word if in {progress, neutral, regress}, else `unparsed`.
- `scripts/25_w1_judge.py` refuses to run unless `logs/W1J_APPROVED` contains the template's sha256.
- Prompt text:
  > You are evaluating one step of an agent that is carrying out a text-based task. / Task: {task} / Observation
  > before the step: {obs_before} / Action taken: {action} / Environment response: {peek_obs} / Judge only this
  > step. Compared with the situation before the action, does the environment response show that the agent got
  > closer to completing the task (progress), stayed about as far from completing it as before (neutral), or got
  > farther from completing it (regress)? Answer with exactly one word: progress, neutral, or regress.

**GPU queue:**
- `scripts/run_phase1d_gpu.sh`: gold → siblings k=1…7 → peek hidden → peek scoring → W1-J (only if approved).
- It aborts without a GPU and has no auto-shutdown.
- **Not started:** the instance is in no-GPU mode (0.5 CPU, 2 GB RAM).

## D-044 — FREEZE ADDENDUM 2026-10-26 applied; B2-het and B3(∩) ceilings; regimes under the new names — DECIDED / INFO 2026-10-26

`RULING_2026-10-26_FREEZE_ADDENDUM.md` was pulled (commit 63cf1c9).

**CLAUDE.md (commit 702f89b):**
- §1 gets an addendum note.
- §4.2: P5 becomes known fact F1; P1–P4's "B2" means het.
- §6.0 relabelled:
  - Credit-pivotal = the old Pivotal_agg;
  - Fork = Pivotal_het, the B2 main label;
  - Indifferent := Live ∧ ¬het;
  - B3 set = Credit-pivotal ∩ het.
- §6.3 table updated.
- §6.3a gold: independent-20 replication rate; the new switch (B3 → report-only if the ∩ replication < 0.50).
- §6.6: uniform / 2k-prefilter definitions and `exploratory_adaptive_replay`.
- §8: label gate updated; priority rule with R-EITHER.
- Phase 1D gets the addendum's execution order.

**Interpretation choices (recorded per §9.10):**
1. The ruling writes Indifferent = 33.1%. The exact value is 1,003 / 3,038 = 33.0%, and CLAUDE.md uses 33.0%.
2. The old "(het)" suffix in §8 no longer applies, because het is now the main B2 label. If a reading holds on het but
   not on Credit-pivotal, the paper explains the difference; no suffix is added. The addendum did not address this.
3. **Procedural note:** the GPU queue (`run_phase1d_gpu.sh`) was started at 12:49, before this addendum arrived and
   before the freeze (addendum §5 places it at step 5). It produces raw data only (gold continuations, sibling
   rollouts, peek-state hidden states and scores) and computes no features or associations. The addendum does not
   change any of these jobs. Precedent: D-034.

**Ceilings** (`scripts/26_addendum_labels.py` → `eval/ceilings.json` field `freeze_addendum`; commit f88680f, clean).
Banks 1+2 subsample, m runs per arm, 200 repeats:

| target | score | m=1 | m=2 | m=3 | m=6 |
|---|---|---|---|---|---|
| B2 = het on 𝓛 (475 / 1,478) | five-arm G of the subsample | 0.656 | 0.720 | 0.745 | **0.763** |
| B2 = het on 𝓛 | \|A\| of the subsample (comparison) | 0.632 | 0.616 | 0.612 | 0.626 |
| B3 = sign(A_3456) on Credit-pivotal ∩ het (170; 72 positive) | A of the subsample | 0.743 | 0.801 | 0.834 | **0.883** |

The B2 normalised-effect denominators are therefore 0.763 (het, m=6) and 0.883 (B3, m=6), with B1 still 0.93 (m=3).
At m=6 the subsample is all six runs, so its band has zero width.

**Regimes** (`eval/regimes.json` field `fork_het`, cluster 95% CI):

| regime | n | fraction |
|---|---|---|
| Dead | 1,560 | 51.3% [42.7, 60.0] |
| Indifferent | 1,003 | 33.0% [27.1, 39.0] |
| Fork | 475 | 15.6% [11.8, 19.8] |
| ↳ Fork: Credit-pivotal ∩ het | 170 | 5.6% [4.1, 7.3] |
| ↳ Fork: Masked | 305 | 10.0% [7.5, 12.9] |

- Null-expected counts: het 103.9, Credit-pivotal 118.3, ∩ 35.0, Masked 68.9. The null-expected Fork share of D is
  3.4%.
- Fork among live, by progress tercile: 29.1% → 30.2% → 40.9%.
- Fork among live, by trajectory outcome: lost 32.1%, won 32.4%. Unlike Dead, it does not depend on the outcome.

**Freeze record (FREEZE ADDENDUM §5 step 3):** `FEATURES_FROZEN.md` committed alone at **d6fe13f468a8086abc300d3a542c03dd17fb9f30**, built from `proposals/FEATURES_FROZEN_DRAFT.md` with the addendum's §3 confirmations and §1 label rewrite. From this commit on, targets, the feature list and the verdict rules are frozen.

Implementation details added while finalising are deterministic choices that involve no outcome:
- §3 "feature computation details" (exact formulas for 21–39);
- §4 solver (liblinear) and the default GroupKFold split;
- §5: "model-side multi-feature" for the R-MODEL / R-LADDER / R-EITHER comparison = the B+C joint model, a single
  pre-specified score, so that the better of B and C is not picked after seeing results;
- §6: the k-shortfall rule and the `exploratory_adaptive_replay` allocation rule.

`logs/W1J_APPROVED` was written with template sha256 9804c516…e233b1 (addendum §3 item 10). The running GPU queue
will run W1-J as its last step.

## D-046 — Phase 2 (6.6) part 1: arm-B / A / H1 features and W1-pure features; stop/go checks — INFO / DECIDED 2026-10-26

These are computed after the freeze (d6fe13f). No label was read.

**`data/features_B.parquet`** (`scripts/21_features_B.py`, fp32 primary; `--tag _vllm` gives the bf16 sensitivity
version):
- 3,818 rows. NaN: `js_prev` 100 (t = 1, by definition), `conf_realized` 2 (realized prefix not locatable, as in D-040).
- Deterministic choices not spelled out in CLAUDE.md §4 or FEATURES_FROZEN:
  - a1 / a2 = top-2 by logp_sum (ties: candidate order); `margin_mean` ranks by logp_mean (literal reading of
    "同上用 logp_mean");
  - JS divergence in nats over the union of candidate sets;
  - `sem_dispersion` = Σ_{i≠j} q_i q_j d_ij / Σ_{i≠j} q_i q_j;
  - `sem_dist_chosen_runner` uses a2 literally (0 when chosen = a2);
  - `entropy_sampled` = plug-in entropy of the admissible first-batch drafts (D-008); never NaN here;
  - `chosen_freq` counts all 32 first-batch drafts parsed as chosen;
  - `rho_h1`: the trajectory mean uses every scored turn.
- **Stop/go "chosen_rank == 1 should be the majority": NOT met, and not a pipeline error.**
  - chosen_rank == 1 at 28.2% of points (vLLM bf16: 28.1%; ranking by logp_mean: 28.5%).
  - The bare arm-A context (CWGT's definition, D-003) scores actions with no reasoning, while the chosen action was
    sampled after reasoning. Evidence:
    - the chosen action is 41% of the 32 first-batch drafts (mean);
    - teacher-forced after the realized think, chosen conf has mean −0.02 (bare: −1.05);
    - the bare q_t is peaked (median entropy 0.55 nats).
  - This is the contrast CLAUDE.md §4 builds in (`chosen_conf` without reasoning vs `chosen_freq` with reasoning).
    It is reported, not adjusted.
- **Stop/go "arm-A conf agrees with the Phase-1 log-probs"** (`scripts/review/r08_conf_phase1.py` →
  `eval/phase2_conf_phase1.json`, 2,438 xml-format points): passes for the comparable realized-think conf.
  - conf_realized vs mean Phase-1 sampling log-prob of the action tokens: Spearman 0.994, Pearson 0.925, median
    |Δ| 6e-7.
  - The bare conf is not comparable by construction (Spearman 0.29, mean shift −1.05).

**`data/features_W1.parquet`, pure part** (`scripts/27_features_W1.py --part pure`):
- 3,038 D points, 5 peeks each; the peek checks (factual obs match, 0 re-feed divergence) are asserted per point.
- Target word extracted for 100% of points; `target_revealed` = 1 at 55.0%.
- Choices: SD with ddof = 0; "Nothing happens." compared after strip().
- **Two frozen features are constant by construction:**
  - 21 `w1_nothing_frac` ≡ 0: the alternatives are sampled from the admissible set, so no peek returns
    "Nothing happens.";
  - 28 `w1_div_n_distinct` ≡ 5: the five peek observations are always distinct strings.
  - They carry no information. Their single-feature rows are reported as constant (no AUROC), and the multi-feature
    models drop zero-variance columns automatically. This is not a change to the frozen list.

GPU queue status at writing: gold running since 12:50 (30,000 continuations, ETA ≈ 19:00 at 0.75 s/continuation),
then siblings, peek states and W1-J.

## D-047 — Phase 1D queue: siblings merged into one 700-trajectory run (speed-up) — DECIDED 2026-10-26

Researcher-suggested speed-up. The original queue (`run_phase1d_gpu.sh`) ran the 7 sibling groups as 7 runs of 100
lockstep trajectories, which leaves the GPU under-fed. `10_rollout.py --siblings 1-7` instead advances all 700 in one
batch.
- **Design unchanged:** per-request sample seeds = 10⁶·seed + turn + (20+k)·10¹¹ depend only on (task, turn, k).
  Seeded generation is batch-dependent at the bit level anyway (D-015).
- **Memory:** env workers hold ≈20 MB per game (measured 650 MB per worker at ≈30 games), so 700 games ≈ 14 GB, well
  inside the 118 GB cgroup.
- **Mechanics:**
  - A running bash script must not be edited in place, so a new script `scripts/run_phase1d_gpu2.sh` holds the
    post-gold steps (siblings 1–7 in one batch → peek hidden → peek scoring → W1-J).
  - The old queue's bash shell (PID 1770) was terminated; its gold replay child (PID 1814) keeps running.
  - The new script waits for that PID, checks `DONE 300 points` in `logs/1d_gold.log`, then continues.
- **Expected saving:** ≈ 15–20 min.

## D-048 — Phase 3 evaluation implementation choices (written before any feature–label association was viewed) — DECIDED 2026-10-26

These are deterministic choices that FEATURES_FROZEN does not spell out. Scripts: `scripts/evalcore.py`,
`31_probes.py`, `40_eval.py`, `41_eval_extras.py`, `42_gold.py`, `27_features_W1.py --part model`,
`28_features_G.py`.

**Probe fitting and V:**
- **Logistic solver.** liblinear with sklearn defaults (max_iter = 100, tol = 1e-4), as FEATURES_FROZEN §4 states.
  - When n < d, the fit runs on the thin SVD basis of [Z, 1]. liblinear penalises the intercept as an extra column,
    so this is the same L2 problem and has the same optimum. Verified: max |Δ decision| ≈ 1e-4 against the full fit.
  - The basis is computed once per training set and shared across the C grid.
  - Non-converged fits are counted and reported. The dual solver was tried and was slower.
- **V.** Kernel ridge, fitted separately on the bank-1 and bank-2 pool means; V = their mean (the "cross-fit" of
  CLAUDE.md §4).
  - Layer and λ are chosen per outer fold by inner MSE.
  - λ = 10⁴, the upper edge of the frozen grid, was chosen in every fold (layer 26 in 9 of 10 fits). Reported as a
    grid-edge hit; the grid is not changed.
  - V predictions on the two banks correlate 0.986.
- **C_sel and B+C.**
  - C_sel: per outer fold, the prompt_end layer with the best mean inner AUROC, using that layer's out-of-fold score.
  - B+C: logistic on the B features + C score. For training rows the C score is an inner out-of-fold prediction of
    the same (layer, C), so test labels do not leak.

**Reading the verdict table (CLAUDE.md §8):**
- **Model-side** = B_multi, C_sel or B+C. The R-MODEL / R-LADDER paired comparison uses B+C (FEATURES_FROZEN §5).
- **"Passes" requires the paired differences to favour the arm:** the CI lower bound vs V > 0 and vs A > 0. A CI
  entirely below 0 does not pass.
- **Readings** follow table order after the priority rule. R-REPLAY-ONLY = no score passes B2. R-NULL = no score
  passes B1 (checked after R-REPLAY-ONLY, per table order). The B3 condition "no cheap arm passes, or only W1" is
  checked over every score on B3.
- **Direction conventions:**
  - Single features use the frozen prior directions; "0" keeps the raw score.
  - H1 (ρ_t) and V are evaluated raw, as no prior was declared for them.
  - `chosen_rank` enters only the multi-feature models; it has no single-feature row in FEATURES_FROZEN.
- **Identification of F6–F9:** family-only vs F2 and vs V; leave-one-out not computed (no "full" cheap model is
  defined for them in the frozen file). F1–F5 use the B full model, as in CLAUDE.md §4.

**Gates and report items:**
- M1 routes the top 13.1% (and the 20.4% no-cf rate) of D by −conf; label pivotal_cwgt; cluster and Wilson CIs.
- M4 / T3: trajectories with ≥ 5 D points and non-constant score and label. Δ = z(ρ_t) − z(A_b1);
  Spearman(Δ, bare conf); median with a trajectory bootstrap. "Compatible" = CI overlaps [0.647, 0.793].
- CARL δ: median split of `entropy_sampled` (and `entropy_q`) on D; Cliff's δ of `crit_sd`.

**E1:**
- "Coverage" is reported both as precision (forks among selected) and recall (selected forks / all forks).
- Captured |A_3456| counts Dead points as 0. The Pearson uses selected live points.
- The B→W1-pure score for all of D comes from a W1-pure model fitted on live training rows.
- `exploratory_adaptive_replay` is not implemented in this pass. Its "1 run per arm at every point" first round
  exceeds the k-fork budget (15k continuations per trajectory) for small k, so the budget equivalence needs a
  definition first. It is an optional exploratory row.

**W1-model peek-state controls** for V: turn + 1 and the size of the peek admissible set.

**Gold (`42_gold.py`):**
- Independent-20 rules on the 20 new runs per arm.
- Null replication from one 10,000-permutation set per point (het p of a permuted data set = its rank).
- Merged-32 reported as selection-biased.
- Switch per CLAUDE.md §6.3a.

**L3 (optional):** not run in this pass.

## D-050 — Flag for review: B+C stacking underperforms its components on B2 — OPEN 2026-10-26

Recorded by Claude while `finish_phase1d.py` was running, from `eval/main.json`.
- On B2 (Fork / het), the B+C stacked score has AUROC 0.514. Its components are higher: B_multi 0.567, C_sel 0.594.
- On B1, B+C = C_sel = 0.748. On B3, B+C (0.576) is slightly below both components.
- A stacked model falling below both inputs suggests a defect in `bc_stack` (`40_eval.py`). Candidates: the scale or
  distribution of inner-OOF vs outer-OOF C scores, or the choice of the stacker's C.
- **It does not change the verdict:** B_multi and C_sel fail B2 on their own (C_sel CI low 0.552 < shuffle p97.5
  0.581), so model_B2 is False with or without B+C, and the reading stays R-WORLD. The R-MODEL / R-LADDER paired
  comparison was not triggered, because R-MODEL is not satisfied.
- To check on the next session (CPU): rerun `bc_stack` on B2, inspect the stacker's coefficient on the C column and
  the inner-OOF C AUROC on training rows. Any fix is a bug fix to the pre-specified B+C definition, not a new
  feature. Record it here.

## D-045 — Gold set: audit and independent-20 replication; B3 switch — INFO (auto, finish_phase1d.py) 2026-10-26

`scripts/42_gold.py` → `eval/gold.json`. Audit: `{"missing": [], "bad_counts": 0, "alt_mismatch": 0, "seed_out_of_range": 0, "dup_seeds": 0, "refeed_div": 0, "outcome_won_mismatch": 0, "n_points": 300, "n_continuations": 30000}`.
- cp: flagged by the 12-run rule 75; independent-20 replication 0.5333333333333333 (CI [0.4084507042253521, 0.6626506024096386]); null replication 0.09893199999999999; merged-32 (selection-biased) 0.7066666666666667.
- het: flagged by the 12-run rule 79; independent-20 replication 0.7341772151898734 (CI [0.6144578313253012, 0.8421052631578947]); null replication 0.09458481012658229; merged-32 (selection-biased) 0.8734177215189873.
- cap: flagged by the 12-run rule 62; independent-20 replication 0.5161290322580645 (CI [0.37681159420289856, 0.6461538461538462]); null replication 0.03435806451612903; merged-32 (selection-biased) 0.7419354838709677.
- **B3 switch (cap replication < 0.50 → B3 report-only): False.** Applied mechanically per CLAUDE.md §6.3a.

## D-049 — Phase 3 run and end-of-run shutdown — INFO (auto, finish_phase1d.py) 2026-10-26

The user asked for saving code, results and logs, a verified sync, and then `/usr/bin/shutdown`, also on unrecoverable failure. The run was driven by the detached `scripts/finish_phase1d.py`.
- Steps: `{"features_G": true, "features_W1_model": true, "features_B_vllm": true, "gold": true, "eval_main": true, "eval_main_vllm": true, "eval_extras": true, "results": true}`; errors: `[]`.
- Verdict (eval/main.json): `{"issued": true, "reading": "R-WORLD", "flags": {"model_B1": false, "model_B2": false, "W1pure_B2": true, "W1model_B2": false, "any_B1": true, "any_B2": true, "B3_passing": [], "B3_condition_ok": true, "B3_report_only": false, "paired_model_minus_w1pure_B2": {"diff": -0.1566290601878575, "ci95": [-0.22821331024169922, -0.08247429877519608]}}}`.
- Predictions: `{"P1": {"holds": false, "B1_passes": false, "B2_norm_effect": -0.05022634532007152, "B2_norm_ci": [-0.19121618568897247, 0.08808599412441254]}, "P2": {"holds": true, "lhs": 0.5823012305534305, "rhs": 0.6470567138485084}, "P3": {"holds": true, "lhs": 0.6470567138485084, "rhs": 0.335106637337414}, "P4": {"holds": false, "diff": -0.0010851655559637186, "ci95": [-0.006521346978843212, 0.004244827665388584]}, "F1": {"masked_share_of_forks": 0.642, "ci95": [0.592, 0.69], "source": "D-042"}}`.
- Gates: `{"M1_contains_13.1": true, "T3_compatible_[0.647,0.793]": true}`.
- RESULTS.md was rendered by `scripts/43_results.py`; this entry has not been reviewed by a person.

## D-051 — Proposal P4 (Agents cannot see their own forks) registered — OPEN 2026-10-30

`proposals/P4_self_blind_forks.md`. Written by Claude after reviewing `main` and `research/deep-discovery-2026-10-27`
(`c6d03ee`). Nothing in M0 changes: `CLAUDE.md`, `FEATURES_FROZEN.md`, labels, verdict rules and `RESULTS.md` are
untouched, and no new feature–outcome association was computed.
- Thesis: introspective signals (−conf, entropy, decision structure, hidden states, V) locate outcomes (Dead/Live) but
  not forks, because forks sit where the deciding world fact is absent from the policy's context. Prediction: a stronger
  policy removes utilization-limited forks, so introspective credit gets worse, not better.
- Plan: E0 novelty check, strata script and pre-registration commit; E1 (CPU) introspective signals split by stratum
  (information-limited I / utilization-limited U / no-target N, from deep's forward-only target flags); E2 (GPU, scoring
  only) 7B vs Qwen2.5-32B judges under context / full log / aligned feedback / permuted feedback; E3 (conditional,
  ≈ 3–4 GPU days) 32B policy replication on s000–s049 in a separate data root.
- Predictions P4-1…P4-10 and readings R-NULL / R-PRIOR / R-BLIND / R-BLIND-7B / R-MIXED are in §3–§4 of the proposal
  and are to be frozen in a separate `P4: pre-registration` commit before E1 runs. Data already seen is listed in §5.
- Status: awaiting the researcher's ruling (proposal §8: adoption as paper mainline, judge/policy model size, E3 budget,
  whether to fix D-050 first).

## D-052 — P4 execution authorized; E0 implementation choices — 2026-10-30

The researcher requested completing the latest proposal experiments and then said to continue.
This authorizes P4 E0/E1/E2 under the proposed default Qwen2.5-7B/32B design, with E3 conditional
on §4; it does not preemptively adopt a paper mainline. D-050 is independent and excluded as proposed.
M0 frozen definitions, results and data remain unchanged. New artifacts are in `exploratory_P4/`.
E0 text-only strata reconstruction passed all registered counts and all pointwise comparisons.
Before E1: P4-1's "five" is interpreted as all SIX explicit signals in §2.2 (both entropy variants).
Full implementation conventions are recorded in `exploratory_P4/protocol.json`, including shared
trajectory bootstrap, existing progress terciles, fixed judge response prefix without realized thought,
and deterministic feedback derangement. These conventions are frozen before association analysis.
Hardware check: no GPU detected (`nvidia-smi`: No devices were found); container limits are 2 GiB
RAM and 0.5 CPU, despite host-level free output. E2 cannot run on this allocation; CPU work proceeds.

P4 E0 pre-registration commit: `20f0976606ca01929bce67d3b850df1045fe8c4e` (`P4: pre-registration`).
Targeted novelty review found no exact stopping-criterion match; source-level scope and incomplete
retrievals are disclosed in `exploratory_P4/novelty_check.json`. E1 starts only after this commit.

## D-053 — P4 E1 complete; positive control fails; GPU stage deferred — 2026-10-30

E1 executed from commit `102d051da0389823d5fd79dddba5384e9aa72f4c`; full input SHA256,
10,000 trajectory-cluster bootstrap / 1,000 stratified shuffles and results in `exploratory_P4/e1.json`.
- P4-1 meets its literal threshold for all six signals in I, but the positive control fails:
  W1pure_multi AUROC 0.5962, shuffle p97.5 0.6120. The proposal explicitly prohibits interpreting
  E1's negative introspection result when this control fails. E1 reading: R-MIXED; no E3.
- P4-2 false: U−I AUROC C_sel −0.1026 [−0.2298, 0.0393], B_multi −0.1344 [−0.2732, 0.0136].
- P4-3 false: policy G_u on I Fork −11.13 pp [−13.85, −7.66], on U +0.48 pp [−3.15, 4.34].
  I Gold confirms negative direction: −12.01 pp [−16.89, −7.11].
- World target-mention arm rule on I Fork: G_u +46.88 pp [39.94, 51.99], Gold +47.20 pp
  [37.90, 54.44]. This is action selection, not a replacement for the state-level positive control.
- P4-4 point directions hold; positive-sign proportion U−I CI positive, chosen share on Live CI includes 0.
- Post-result interpretation (not a changed rule): target_revealed is constant within each registered stratum;
  strata condition away its between-stratum predictive power. B_multi/C_sel AUROC in I is around 0.68,
  not chance, but does not exceed the within-trajectory/progress shuffle baseline.
Independent verification passed against sklearn and pre-existing deep extraction; all 10,656 E1 input hashes
checked. E2 inputs prepared: 873 points / 3,492 contexts / 17,460 candidate sequences per model. GPU scoring
and complete-run analysis scripts provided; real GPU numerical validation has NOT run, no E2 scores exist.
After disclosure of missing GPU the researcher said “先做能做的”. E2 is deferred on this allocation;
P4-5/6/7 remain untested. E1+E2 table-order verdict is not claimed complete. No paper mainline change.

## D-054 — P4-X robustness pre-registration and proposal P5 (closed loop) registered — 2026-10-30

Written by Claude after reviewing E1 (D-053). Nothing in M0 or in P4's pre-registered predictions, readings or
E1 results changes; E2 runs as pre-registered; E3 stays off.
- **P4-X (`exploratory_P4/PREREG_X.md`, `exploratory_P4/p4_x_robust.py`).** E1's I-layer policy G_u = −11.13 pp was
  not predicted. Before interpreting it, X1–X7 check the scoring convention (logp_mean), the actual sampling
  marginal (bank-1 drafts with reasoning), the chosen action alone, dilution versus under-exploration (exploratory
  class E = `go to` an unvisited receptacle or `open` an unopened one, from factual actions before t), loops,
  over-representation of long searches (first I point per trajectory), and leave-one-trajectory-out / task type.
  Readings R-X-ARTIFACT / MISDIRECT / UNDEREXPLORE / DILUTION / OTHER. The script recomputes E1's policy G_u from
  `logp_sum` and stops unless it matches `e1.json` to 1e-9. Its `--selftest` (synthetic) shows that probability
  dilution among many unvisited options alone yields a significantly negative G_u with zero within-class gain,
  which is why the class decomposition is required. The main path was smoke-tested end to end on a synthetic
  repository with the same file layout (parquet reads mocked); it has not run on real data.
  To be committed alone as `P4-X: pre-registration` before the main run; record that hash here.
  P4-X pre-registration commit: `dd8b07973ca0d48607c8322b0c65f785352d4a8f` (`P4-X: pre-registration`); no real-data run has been performed.
- **P5 (`proposals/P5_closed_loop.md`, PROPOSED).** Paper plan toward a top venue: thesis that step-level credit is
  first an information problem; evidence inventory; E2 and P4-X as above; new E4 closed-loop learning experiment
  (feedback-grounded step preferences vs budget-matched replay credit, entropy-selected replay, an exploration
  heuristic, outcome-only RFT, base; LoRA-DPO on ALFWorld train games excluding M0's 100 tasks; valid_seen/unseen);
  E5 WebShop confirmatory replication with definitions frozen before any outcome; gates G1–G3; paper skeleton.
  Awaiting the researcher's ruling (P5 §9: venue/deadline, adoption, E4 framework, compute).
- D-050 remains OPEN: the fix code is in `c2b7ba1` but the recomputed B+C results are not in the repository.

## D-055 — Resume P4 E2 on RTX 6000D; pinned model and memory implementation — 2026-10-30

The researcher enabled an RTX 6000D and asked to finish P4. E2 is authorized; E3 remains gated.
GPU reports 85,651 MiB, container RAM 92 GiB. Retain all frozen prompts, requests, predictions, and E1/X results.
- 7B remains Qwen/Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`, verified against env/qwen_sha256.txt.
- 32B is Qwen/Qwen2.5-32B-Instruct revision `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd`, pinned via official Hugging Face API before any E2 score.
  Source metadata stored in exploratory_P4/e2_32b_model_source.json. Weights total 65,527,841,856 bytes;
  split between /root/autodl-tmp/models and /autodl-fs/data/forkaudit_models, with local symlinks and upstream LFS SHA256 checks.
- To bound memory on this GPU, p4_e2_backend.py processes the five candidate suffixes sequentially with a shared
  cached prefix, rather than expanding the prefix cache to five batch entries. Both models use this implementation.
  BF16 transformer, FP32 vocabulary projection and normalization are unchanged. The predeclared direct-scoring
  checks and total-variation <0.005 gate remain unchanged; no scores are accepted if the gate fails.
- No new trajectories, policy training, or altered analysis definitions. No shutdown is authorized in this request.

D-055 download transport note: the HF large-file route is slow. Weight bytes may be transported from the
ModelScope Qwen/Qwen2.5-32B-Instruct mirror; acceptance still requires each exact SHA256 from the pinned
HF revision above. Non-weight configuration/tokenizer files remain pinned HF downloads. This does not
change the model revision. Download concurrency is 8, with resumable partial files.

D-055 transport/resource follow-up: final download concurrency is 17 (one worker per weight shard), using
HF for the 9 system-disk shards and the hash-verified ModelScope mirror for the 8 data-disk shards.
The system/data disks have projected post-download free space about 2.02/1.16 GiB respectively;
partial files are resumed in place and renamed after verification, with no duplicate weight cache.
7B completed all 3,492 contexts successfully; 32B remains queued behind the complete verified download.

D-055 download diagnosis/update (at the researcher's request): sustained full-file download was about
9.5 MiB/s in aggregate with reconnects. A common 8 MiB byte range was tested across HF, hf-mirror,
ModelScope resolve/API, proxy/direct paths; completed range bytes matched across sources (see
`e2_download_benchmark.json`). ModelScope direct was fastest in that test; hf-mirror was slower.
Remaining data now uses `download_e2_32b_chunks.py`: 32 workers, 16 MiB bounded ranges, ModelScope direct
with pinned-HF fallback. Existing contiguous prefixes are preserved in `e2_32b_chunks_state.json`
before sparse writes. Resume ONLY with the chunked downloader; the old downloader now refuses to run
when that state exists. Every completed weight must still match the pinned full-file HF SHA256 before
any model loading. The small-file preparation and scientific scoring/analysis rules are unchanged.

## D-056 — Authorized E2 completion, GitHub publication and shutdown hook — 2026-10-30

The researcher explicitly requested a hook to analyze completed runs and upload to GitHub,
then explicitly requested shutdown last. `publish_e2_when_done.py` waits for the existing
`finish_e2.py` process to finish scoring, analysis and independent verification. It requires
both complete 3,492-context runs, a passing verification report and the driver's successful
verification event, then renders the report and creates a separate E2 artifact manifest.
It commits only explicit E2 artifacts on the existing `main` branch, pushes to the confirmed
`melancholiac99/ForkAudit` origin without force, verifies the remote branch SHA matches the
local commit, syncs files, and finally invokes the server's `/usr/bin/shutdown` script.
Unrelated untracked data, frozen preregistration and E1/P4-X conclusions are preserved.
No model weights are uploaded. Any failed gate, unexpected E3 requirement, branch change,
staged unrelated changes, or failed publication stops the hook without shutdown and records
`exploratory_P4/e2_publish_status.json`. This explicit shutdown authorization supersedes the
earlier D-055 note that shutdown had not yet been authorized.

## D-057 — P4 E2/P4-X verified; post-hoc mechanism analysis; E2-M pre-registered — 2026-10-31

Written by Claude. No frozen M0/P4 item, E1/E2/P4-X result or reading changes.
- **Verification.** `exploratory_P4/post_e2/post_e2_mechanism.py` recomputes every E2 judge-condition G_u from the
  same arms file (SHA256 equal to `e2.json`'s input) with an independent implementation; max diff 2.8e-17.
  P4-X and E2 reports agree with `x.json` / `e2.json`. Two wording constraints recorded in `post_e2/FINDINGS.md`:
  uniform-relative G_u depends on candidate construction; Fx is leaky for strong judges (observations name places).
- **Post-hoc (not pre-registered), `post_e2/FINDINGS.md`.** On I-layer Forks: without world feedback the policy and both
  judges put ~24–25% mass on unvisited places (value 65.6% vs 23–27%); the 32B judge over-weights returning to
  already-searched places outside the 2-turn window by +19.5 pp [9.4, 27.8]; 33% of executed actions at these points are
  such returns (14% at I non-Fork). Capability helps only with information present (pre-registered P4-5 confirms the
  interaction). Full hindsight logs raise judge mass on the executed action by +12 to +31 pp regardless of outcome.
  Parquet is read with a pure-Python fallback (`post_e2/minipq.py`) where pyarrow is unavailable.
- **E2-M pre-registration** (`exploratory_P4/PREREG_E2M.md`, commit `6eaf07d`, message `P4-E2M: pre-registration`):
  memory (M) and memory+feedback (MF) conditions, 1,746 contexts prepared deterministically from committed E2 requests;
  predictions M-1..M-4; readings R-M-NULL / R-M-TWO-GAPS / R-M-PARTIAL. Scoring: `p4_e2m_score.py --model PATH --name
  {7b,32b} --revision <same pinned revisions as D-055>`; analysis: `p4_e2m_analyze.py`. The analysis was tested on
  synthetic score fixtures (null and positive); no model scores exist yet. ≈ 1–1.5 GPU hours.
- **P5 §10** appended: thesis revision (credit at information-limited forks is set by information, not judge
  capability), E4 arm changes (32B+F judge; hindsight-log judge arm H; conditional memory arm R), E5 additions.

## D-058 — Execute merged post_e2 and preregistered E2M on server — 2026-10-31

User authorized merging latest origin/main on the existing server branch, preserving local work,
then post_e2 verification and E2M M/MF scoring/analysis, and pushing completed results. P5 training
and new trajectories are excluded. Merge was a conflict-free fast-forward from 8e2b692 to 27fcbe6;
server branch stays main. The three received commits are d7fd0b4, cefd377 and 27fcbe6.
D-057's abbreviated preregistration hash 6eaf07d is stale: actual received preregistration commit
is cefd377. Frozen document, requests, predictions and thresholds are unchanged.
Post_e2 selftest passes; server reproduction matches every scientific output in the patch exactly
(max E2 G_u difference 2.7755575615628914e-17). Original post_e2 JSON is preserved; reproduction is
in e2m_run/post_e2_server.json. Rebuilding M/MF in memory matches all 1,746 frozen requests byte-for-byte.
Before scoring, add explicit equality checks against original E2 run manifests for BOTH checkpoint
hashes, revisions, backend hash, tokenizer and configuration hashes. This closes the supplied scorer's
missing 32B checkpoint equality guard without changing scoring or numerical gates. Original E2,
E1, P4-X, preregistration and P5 documents remain unchanged. This new task has no shutdown step;
the preceding authorized E2 shutdown was already completed.

## D-059 — E4 and E5 pre-registered; paper story v3 — 2026-10-31

Written by Claude at the researcher's request ("把E4 E5做完，论文故事怎么讲"). This session has no GPU, ALFWorld, WebShop
or torch, so nothing was run; both experiments are prepared to run on the server. No frozen M0/P4 item changes.
- **E4** (`exploratory_E4/PREREG_E4.md`, commit `37f1873` on origin/main, applied from patch babafdd): closed loop on ALFWorld. 400 train games (M0 tasks excluded), 40 tune,
  valid_seen/unseen eval. Arms A (32B judge, F), A′ (rule), H (32B judge, L), Rnd (same points as A), D (exploration
  heuristic), Bfull / Bmatch (cost-matched by measured GPU seconds) / Cmatch (top sampled entropy), Esft, base; cap 1,000
  examples per arm; LoRA-DPO (r 16, β 0.1, 2 epochs), lr chosen once on tune with A′; 3 training seeds; 3 rollouts per game.
  Predictions PL-1..PL-5, readings R-L-NULL / HEURISTIC / COSTLY / WIN. `test_e4.py` passes on CPU, including a byte-identical
  rebuild of real P4 E2 F and L requests. Judge scoring asserts the same revision, checkpoint SHA256 and backend as P4 E2.
  Smoke gates S1–S7 precede any outcome. games.json is frozen and committed separately by `e4_00_freeze.py`.
- **E5** (`exploratory_E5/PREREG_E5.md`, commit `2d341bd` on origin/main, applied from patch bebe6d3): WebShop, 150 sessions from the training
  goal range, M0 protocol (banks 1–6, five-arm G test), strata from the goal ASIN, judges 7B/32B × S/L/F/Fx. Predictions
  W-1..W-5 with a power gate (≥ 40 Fork points in each of I and U). The WebShop adapter (`ws_env.py`) speaks the envpool
  protocol so `scripts/11_replay.run_async` runs unchanged; API details marked VERIFY and the verl-agent template are fixed
  at gates S0–S2 before any rollout. `test_e5.py` passes offline. A WebShop learning arm is deferred to a separate
  pre-registration, conditional on R-E5-REPLICATE and E4 R-L-WIN.
- **Paper story v3** (`papers/STORY_v3.md`) replaces the deep-branch draft v0.1 mainline: step-level credit at
  information-limited forks is an information problem; evidence chain M0 → P4 E2/E2-M (pre-registered) → post-hoc
  diagnostics → E4 → E5; versions of the paper under each E4/E5 reading.
