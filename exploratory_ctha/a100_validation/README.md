# CTHA v3 A100 engineering acceptance

This separate entry point implements the A100 engineering scope authorized on 2026-11-13. It calls the unchanged implementations in `v3_fp32_fast/` with explicit local paths and records the A100 worker configuration. The old controller, previous PID takeover and old shutdown instructions are not executed.

## Scope and evidence

The sequence is CPU regression and verified historical cold-start feasibility; numeric/performance candidates; a frozen configuration; two independent 4-task × 2-episode collections on different GPUs; comparison with the corresponding episodes from a fresh 16-task × 4-episode collection; eight balanced responses and four optimizer steps; actual checkpoint restore; updated merged/unmerged probability agreement; independent evidence verification.

There is no Tune evaluation, four-round learning pilot, full P1/P2, or formal CTHA training. Reports must explicitly distinguish `Engineering: PASS/FAIL/INCOMPLETE`, `Learning: NOT_RUN`, `Generalization: NOT_RUN`, and `CTHA online efficacy: NOT_TESTED`. CPU next-step equality uses fixed engineering gradients and is not a second online RL round.

The sixteen numeric reference prompts come from verified historical E4 training episodes. Only prompt text is reused. Candidate responses and logprobs are newly sampled on A100; candidate samples never become engineering training data. The original P0/P1 records are used only for their original offline feasibility gate.

## Current configuration and paths

The upstream baseline is `51aaa3e89`. Both GPUs are A100-SXM4 80GB, using torch 2.13.0/cu130, vLLM 0.32.0 and Triton 3.7.1. The vLLM worker checks the expected internal launch wrapper before making a process-local change.

BF16 reproduces across devices but fails the original real-sample probability gates. FP32 candidates preserve TF32x3 and K=32 while comparing native, stage-1 M16/32, and stage-2 M32/64 launches. These are FP32 weights/HF forwards with explicit multi-product TF32x3 Triton operations; they are not claimed to be bitwise IEEE FP32 matmul.

The measured selection, input hashes and frozen configuration are under `.runtime/runs/ctha_a100_eng_20261113_r2/`. The selected stage-2 M32/64, 4-warp configuration is about 168 generated tokens/s on the fixed sixteen-prompt timed batch, compared with about 43 for native FP32 and 138 for stage-1. These numbers measure this numeric benchmark, not training throughput or policy improvement.

```bash
source env/a100.sh
export CTHA_A100_ROOT="$PWD/.runtime/runs/ctha_a100_eng_20261113_r2"
export TRITON_F32_DEFAULT=tf32x3
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

`benchmark.py` prepares real reference prompts, samples each independent candidate and scores the actual returned tokens with HF. `cold_start.py` verifies imported original data and repeats the original feasibility calculation. `run.py` consumes completed candidates and the recorded preparation CPU evidence, freezes sources/configuration, then runs the engineering queue. It refuses an already registered queue. A fresh complete rerun requires a new output revision and newly registered evidence, rather than silently restarting or overwriting the current directory.

`backend.py` is the path, worker and metrics wrapper. It preserves the original trainer loss, reference, optimizer, clipping, stability and direction gates. `verify.py` reconstructs trajectory comparisons and probability errors from saved per-token arrays, checks the actual checkpoint chain and rejects a learning claim.

## TensorBoard and tables

TensorBoard is the active viewer on port 6006:

```bash
python scripts/a100_tensorboard.py --data "$CTHA_A100_ROOT" \
  --logdir .runtime/tensorboard/ctha-a100-r2
tensorboard --logdir .runtime/tensorboard/ctha-a100-r2 \
  --host 0.0.0.0 --port 6006 --reload_interval 2
```

The observer reads completed JSON/JSONL evidence without modifying frozen training code. Candidate HParams/scalars show precision, speed, memory and probability errors. Rollout panels show actual horizon progress. The engineering probe adds loss, gradient norm, k3, clip fraction and direction curves only when real updates occur. Its four-step table is saved in `training_telemetry/training_steps.csv`. W&B offline records are optional and remain locally available for later sync.

## Publication and poweroff

Only verified reports, source/protocol changes and necessary reproducible data are published to `credit-grok`. Model and optimizer binaries stay on the server with paths and SHA256 values. Publication must be independently confirmed on the remote branch and followed by `sync`. Poweroff is a separate final operation after all work, verification and upload complete; this host uses systemd and `/usr/sbin/shutdown`, not the old server's shell-script invocation.
