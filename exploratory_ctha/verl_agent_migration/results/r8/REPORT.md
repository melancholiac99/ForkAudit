# Native verl-agent one-iteration engineering smoke

**Native pipeline: PASS. Engineering: INCOMPLETE. Learning: NOT_RUN.
Generalization: NOT_RUN. CTHA online efficacy: NOT_TESTED.**

This is one native GiGPO rollout/update iteration, not the four-round learning
pilot, paper reproduction, or evidence that CTHA works. Before/final evaluation
was omitted in this engineering smoke. No checkpoint was saved.

## Frozen configuration

Upstream `langfengQ/verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8`;
run configuration SHA256
`e68463d2a05f3099fc9e09b863767835c746134c2060cdfddd2ac529a870134d`.
Original Qwen2.5-7B-Instruct, two A100-SXM4-80GB GPUs over PHB, native FSDP,
BF16 computation / FP32 master parameters, vLLM 0.11.0 / Torch 2.8.0,
FlashAttention backend, TP2, direct safetensors preload, non-layered LoRA gather.
LoRA rank/alpha64, lr3e-6, gamma0.95, reference KL off, entropy0.001,
success reward10, invalid-action penalty0.1, XML prompt/history2.

Native PPO uses its recomputed HF denominator in this smoke. **GPU bypass PPO
has not yet been validated here**; the optional compatibility hook has CPU tests.
No second importance weight or rejection mask is applied.

## Measured results

| Check | Result |
| --- | --- |
| Actual environment batch | 4 fixed train tasks ×4 trajectories; 782 real response steps |
| Native padding | 50 repeated rows; 832 processed rows, not 832 independent samples |
| Scored response tokens | 114,878 including repeated padding rows |
| Pre-update behavior k3 | 0.00207589; gate ≤0.02 passed |
| Pre-update ratio clip fraction | 1.9664%; gate ≤30% passed |
| Token / response-product ESS | 99.605% /62.340% |
| Geometric, length-normalized response ESS | 99.9969%; distinct from product ESS |
| Token ratio min /max | 0.12456 /3.38449 |
| Long replies ≥256 tokens | 4; longest289; long-token ESS99.691% |
| Raw absolute logprob error mean /P99 /max | 0.02893 /0.24067 /2.08299; diagnostic, fails the old absolute-error rule |
| Independent NumPy recomputation | Matches saved probability diagnostics |
| Native actor gradient norm | 0.258289, finite |
| Observed sampled success | 2/16=12.5%; no before/after learning comparison |
| Native valid-action ratio | 45.192%; native tag validity, not admissibility checking |

The inherited k3 is a sampled diagnostic, not exact distribution KL. The
independent audit also records signed k1 and the opposite k3 orientation.
Neither the previous eight-response ESS nor this batch proves future stability.

## Cost and evidence limits

Rollout **429.04s**, HF scoring **29.75s**, actor update **82.59s**, full iteration
**542.48s**, excluding cold initialization. A 30-second GPU0 process stack sample
spent about52.4% in weight gather/wake/sleep,46.4% in generation. This is sampled
host stack occupancy, not a whole-run CUDA-kernel timing decomposition.

Framework allocator peak counters exceeded physical card capacity; their
accounting must be clarified. Discrete NVML observations during training were
74,572/55,728MiB, which are **not continuously measured physical peaks**.

Full-model cross-GPU/batch reproduction, post-update HF/vLLM alignment and actual
native checkpoint recovery remain unverified. Native generation dumps contain
decoded display text; exact full DataProto/token IDs were not exported. The
evidence archive supports direct diagnostic recomputation, not exact token replay.
Its source snapshot matches all registered source hashes. All prior startup
failures and the r7 cost stop remain separate records.

Native prompt, reward, advantages, optimizer and batch size differ from old R3;
do not attribute differences solely to precision. The two-card candidate is
**not adopted for the four-round pilot**. Next measure a native single-card
LoRA configuration before spending the two-arm learning budget.

Local results: `.runtime/runs/verl_agent_gigpo_smoke_r8/`.
TensorBoard6006: `verl_agent_gigpo_smoke_r8`. `metrics.csv` provides the full table.
`evidence_index.json` describes reconstruction and checksums; no model or
optimizer binary is committed.
