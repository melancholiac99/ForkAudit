E4 training speed study — 2026-11-01, researcher requested acceleration research

Formal scripts, hyperparameters, examples, optimizer, RNG seeds, scores and
research decisions remain unchanged. This folder is an isolated engineering
study. No optimizer step or new formal checkpoint is produced by the benchmark.

Candidates:
1. response_only: use the installed Qwen2 logits_to_keep tensor to project only
   positions predicting response tokens; keep gradient checkpointing enabled.
2. response_only_no_checkpoint: same projection reduction, disable gradient
   checkpointing and explicitly disable unused inference KV cache.

Baseline follows e4_06_train.py: BF16 base, FP32 LoRA, dropout 0.05,
response-token log-probabilities, DPO beta .1 / SFT mean NLL, loss divided by 16.
Each measured forward/backward resets the same dropout seed. We compare loss
and every trainable gradient. Engineering tolerances, frozen before GPU tests:
absolute loss error <= 1e-4, relative gradient L2 <= 1e-3, cosine >= .99999;
finite checks and exact gradient equality also recorded. These are engineering
checks, not replacements for research thresholds or proof of full-run equality.

Cases are selected before seeing losses: token-length quantiles 0/.5/.9/1
across all DPO pairs, plus the longest SFT example. Input hashes are recorded.
Real GPU measurements use the completed first learning-rate trial's adapter
(Aprime, seed 0, LR 5e-6) to exercise nonzero LoRA matrices. Weights are read
only; no adapter or optimizer state is saved. Two timed repetitions alternate
variant order. Timing covers forward/backward, not reference preprocessing,
optimizer steps or CPU gradient comparison; no full-training speedup claim.

hook.py holds only the orchestration parent while the current LR trial
continues. After that trial exits, the GPU benchmark has a 15-minute timeout.
On completion or failure the original next trial resumes. The already tested
independent guardian from exploratory_E4_extra_scoring_speed kills orphan GPU
benchmark work before resuming the queue if the hook itself is killed.

Commands (m0 Python):
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmark.py selftest
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmark.py prepare
  python benchmark.py benchmark  # only in the GPU-free inter-trial gap

Artifacts: cpu_selftest.json, inputs.json, benchmark_progress.json,
benchmark.json, REPORT.txt, hook_status.json, and execution logs.
No automatic deployment, push or shutdown is performed.

Follow-up after initial numerical checks failed:
- numerical_control.py uses deterministic CUDA algorithms and FP64 diagnostic
  reductions. It adds no_checkpoint (full vocabulary projection preserved) to
  isolate checkpointing from response-only projection. It only compares values;
  shared-GPU times are not used.
- checkpoint_speed.py measures no_checkpoint alone in a bounded exclusive
  interval, preserving the running formal trainer in memory. An independent
  guardian resumes it on failure. Exact interruption accounting is saved in
  training_pause.json. The measured pause was 33.299 seconds.
- REPORT.txt is the final interpretation; REPORT_initial.txt and the hash-checked
  benchmark_initial.py preserve the initial report and executable source.
