Prefix provenance × message role implementation (2026-11-05)
==========================================================
Protocol: PROTOCOL_PREFIX_ATTRIBUTION_20261105.md (unchanged).
Branch: research/hindsight-judges. Implementation only: no automatic GPU run,
training, trajectory collection, publication or shutdown is launched by these scripts.

New entry point: exp4_prefix_attribution.py
Analysis: prefix_attribution_analysis.py
CPU tests: test_prefix_attribution.py
Requires the existing experiment Python environment: numpy, transformers, torch,
local frozen Qwen checkpoints, and original P4 inputs. No new dependency is added.
On this server use /root/miniconda3/envs/m0/bin/python.

Preparation / CPU audit
-----------------------
From repository root, with your Python environment activated:

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
python -m unittest discover -s hindsight_judges -p test_prefix_attribution.py -v
python hindsight_judges/exp4_prefix_attribution.py prepare \
  --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
  --arms /root/autodl-tmp/idea/exploratory_deep_discovery/exploratory_arms.parquet

--arms is explicit here because this worktree does not contain the parquet file;
the original file in the existing idea worktree is read-only and SHA-checked against
P4 e2.json. A copy with the same SHA can be supplied on another machine. Missing or
changed inputs stop execution. There are no synthetic replacements or exclusions.

Default new output root: hindsight_judges/results/prefix_attribution_20261105/
  prepare.json                  source hashes, protocol/code/tokenizer identity
  requests.jsonl.gz             4,632 requests, 386 points / 60 clusters
  condition_matrix_audit.json   every cue, reasoning location, role tokens and hashes
  exact_prompt_examples.json    all twelve exact M/F × condition prompts at first point
  prepare_errors.json           present only when a preparation attempt fails
The 32B tokenizer must have exactly the same vocabulary, special tokens and chat
template as preparation; every actual request token hash is checked before scoring.
Preparation is deterministic and restartable only when its contents are unchanged.

Presentation decisions fixed before scoring
------------------------------------------
The self/other cue is the exact example pair in the protocol. Both cues are appended
to the original user message after all original M/F evidence. AS/AO continue the exact
original pre-action span R + original action-open TAG. QS/QO quote the exact UTF-8 R,
including all whitespace and preamble, between:
  [QUOTED DRAFT — context, not an environment observation]
  [/QUOTED DRAFT]
and append the common request: "Output the next legal action."
Their active assistant prefix is TAG only. The quote ends strictly before TAG;
there is no added factual action label. Naturally occurring action names in R and
in the original five-arm F evidence remain untouched. Real Qwen chat templates are
used, with exact boundary token IDs audited. Literal role-boundary injection fails.

KEEP/DROP reproduce old prompt, prefix, leading/trailing action formatting, EOS
and canonical token-boundary scoring exactly. DROP preserves the old non-Think
preamble; it is not silently replaced with the quote prefix. All conditions retain
the same five distinct candidate actions and independent q3456 replay values.

Historical KEEP/DROP score import is intentionally disabled: the old score runs do
not record the complete immutable prompt-token, tokenizer and hardware identity
needed to certify reuse. Both anchors are re-scored on the selected hardware. The
four factorial conditions run first, then KEEP and DROP; no outcome-based selection.
9,264 contexts across two models, five full action continuations each.

Future GPU execution (not started by this implementation task)
-------------------------------------------------------------
First obtain a pilot timing for EACH model on a recorded GPU UUID. Never substitute
a numeric GPU index in the commands: UUID pinning is required and part of the resume
fingerprint. The pilot and full run for a model must match in hardware and execution
configuration. Set the UUID below to an actual available card.

CUDA_VISIBLE_DEVICES=GPU-actual-uuid python hindsight_judges/exp4_prefix_attribution.py score \
  --name 7b --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct --limit 32
python hindsight_judges/exp4_prefix_attribution.py analyze --name 7b --limit 32
python hindsight_judges/exp4_prefix_attribution.py verify --name 7b --limit 32

Repeat with --name 32b --model /root/autodl-tmp/models/Qwen2.5-32B-Instruct.
Use --limit 32 through 64 consistently. Inspect per-row seconds, progress.json,
run.json total_scoring_seconds, runtime and peak_vram_bytes. Context length varies;
a pilot extrapolation is not a guaranteed full-run budget. Full-run score refuses
an absent, unverified or configuration-incompatible pilot. A 64-point pilot needs
--pilot-size 64 on the later full command (default 32).

After pilot review and explicit execution request, full runs use:
CUDA_VISIBLE_DEVICES=GPU-actual-uuid python hindsight_judges/exp4_prefix_attribution.py score \
  --name 7b --model /root/autodl-tmp/models/Qwen2.5-7B-Instruct
python hindsight_judges/exp4_prefix_attribution.py analyze --name 7b
python hindsight_judges/exp4_prefix_attribution.py verify --name 7b

Repeat for 32b. The scripts do not schedule either model automatically.
CPU tests must pass before GPU execution. 7B uses verified FP32; 32B uses the existing
FP16-transformer / FP32-head policy preserving RoPE buffers. Both use SDPA MATH,
1024-token prefill chunks and TF32 disabled. Frozen weight SHA fingerprints are
verified against original P4 runs. The gate compares cached versus direct full
continuation on global shortest/longest contexts plus shortest F context for every
condition, deduplicated. Every gate must satisfy candidate-distribution TV < 0.005.
A gate failure blocks scoring; no threshold relaxation or automatic precision change.

Outputs / resume / validation
-----------------------------
<root>/<7b|32b>/<first32|first64|full>/:
  scores.jsonl          raw five log-likelihoods, probabilities, original request/Q,
                        per-row duration/timestamp and config/record hashes
  run.json              weight/precision/GPU/config identity and completion hashes
  numerical_gate.json   cached/direct raw scores and strict TV checks
  progress.json         live completed counts, measured throughput and peak VRAM
  analysis.json         all absolute/paired statistics, sources, runtime and caveats
  point_metrics.jsonl.gz four alternative log-odds shifts and Q gaps per point
  REPORT.txt            all primary and secondary tables, cautious interpretations
  verification.json    independent raw-score primary-estimate/bootstrap recomputation

Score rows are fsynced individually. Resumes reject changed code, source, tokenizer,
model, precision, GPU, runtime or token sequence, duplicates and corrupted/truncated
rows. Torn writes are preserved for explicit recovery, not silently dropped. Pilot
scores never enter a full run. No automatic overwrite or merge of old experiments.

Statistics / interpretation
---------------------------
All 386 points enter V statistics; 60 original trajectory clusters, 4,000 bootstrap
draws, seed 20261105. Compute paired point differences first, then a point-weighted
mean. V and V differences are reported in percentage points, rates in 0–1, raw
log-odds shifts in nats. Primary: P_Q at F, companion P_Q at M, interaction I_Q.
Secondary tables include P_A / I_A, both mixed placement/restart contrasts,
AS−KEEP, role × provenance, DROP−KEEP, and F−M under each condition.

Evidence alignment uses every non-tie alternative Q_j−Q_f, with no clipping: E_j is
the F−M change in raw log-likelihood difference relative to factual. First aggregate
within point: sign concordance (zero E is 0.5), sign(Qgap)*E mean, and through-origin
slope sum(Qgap*E)/sum(Qgap^2). All-Q-tied points remain in V; their alignment is
undefined and its count is reported. A bad factual choice means Q_f < max_j Q_j;
correction means the score-maximizing candidate has Q > Q_f (score ties averaged).
Factual argmax uses first-index argmax as in the original anchor; p_f is also shown.

No CI is a corrected confirmatory significance claim. CI crossing zero is unresolved,
not equivalence. Quote authorship effects do not demonstrate internal identity.
Placement contrasts combine role, restart and prompt/task changes. A large V effect
without improved Q-aligned E is not evidence of recovered evidence assimilation.
No result establishes an effective OPD gradient, Student learning or generalization.
