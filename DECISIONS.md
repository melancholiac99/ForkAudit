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
