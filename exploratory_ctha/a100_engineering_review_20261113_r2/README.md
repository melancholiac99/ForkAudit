# A100 CTHA v3 R2 review — FAILED engineering acceptance

**Engineering: FAIL. P1: NOT_STARTED. R3: PROPOSED, AWAITING USER CONFIRMATION.**

The 2026-11-13 A100 run used two A100-SXM4 80GB GPUs, torch 2.13.0/cu130, vLLM 0.32.0 and Triton 3.7.1. BF16 failed real-sample probability parity. The selected FP32 TF32x3 kernel used stage 2, M32/64, K32 and four warps; its fixed-prompt throughput was about 168 tokens/s.

42 CPU tests, historical P0 feasibility, 8 repeated trajectories across GPUs, matching against 64 fresh trajectories, base sampled-logprob parity and stability passed. Four optimizer updates executed. Full-response mean logprob moved +0.060733 for positive samples and -0.044474 for negative samples. Action-span means moved -1.917e-7 and +5.716e-5, violating the stricter trainer direction gate. The run stopped. Accepted checkpoint restore and updated-adapter parity did not run.

See [failure review](failure_review.json), [four-step table](training_steps.csv), [direction result](direction_check.json), [base parity](base_parity.json), and [R3 proposal](R3_PROPOSAL_PENDING.json). The original PLAN says action-span changes are reported separately, while trainer.py requires their signs too; this run followed the stricter code and remains FAIL.

R3 proposes a new, explicitly approved engineering-only sample eligibility rule based solely on pre-update action NLL >=0.01, preserving all pass thresholds, rewards, optimizer settings, eight responses and four steps. The proposal has not executed and may still fail. No P1 training is allowed before independent engineering acceptance. The sibling a100_p1 directory is preparatory draft code/protocol, not a completed or running experiment.

**Publication status:** the result reports and manifest are published first. Raw archive parts are still local and pending upload; the commands below apply after all parts are published.

Reproduce/examine raw records by joining the numbered archive parts in lexical order, verifying evidence_manifest.json and extracting:

```bash
cat evidence_parts/a100_r2_evidence.tar.gz.part* > a100_r2_evidence.tar.gz
sha256sum a100_r2_evidence.tar.gz
tar -xzf a100_r2_evidence.tar.gz -C /path/to/fresh/review-directory
```

The archive includes actual sampled token IDs, behavior probabilities, eight HF base scoring arrays, full rollout records, configuration/source fingerprints and logs. Historical BF16 numeric candidates are labeled separately. No model or optimizer binary is uploaded. TensorBoard reads these records with scripts/a100_tensorboard.py; training_steps.csv contains the four actual update records.

Old experiments and the remote a100_handoff_20261113 directory are preserved. This directory makes no learning, generalization or CTHA efficacy claim.
