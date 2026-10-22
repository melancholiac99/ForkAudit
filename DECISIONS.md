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
