# BF16 statistical admission on A100

**Engineering: PASS within the new native BF16 statistical-admission protocol.**
Learning: NOT_RUN. Generalization: NOT_RUN. CTHA online efficacy: NOT_TESTED.
This does not complete the original CTHA engineering checklist or reclassify
any historical failure. Exact greedy trajectory reproduction remains
**NOT_REPRODUCED**, despite exact checkpoint state recovery.

The user explicitly accepted statistical consistency with unchanged k3≤0.02
and clip≤0.30. NumPy recomputation independently confirms the saved arrays.
No learning/optimizer updates occurred in this R10 probe.

| Check | Result |
| --- | --- |
| Same161 real responses /20,130 tokens, GPU0 versus GPU1 HF, batch32 | Exact score equality; k3=0, clip=0, ESS=1 |
| Same tokens, HF request batches16/8/1 versus GPU0 batch32 | k3≤0.000689; clip≤0.338%; token ESS≥99.856% |
| Same tokens, HF versus original vLLM raw scores across all batches | k3≤0.000803; clip≤0.393%; token ESS≥99.852% |
| Eight fresh eager-vLLM replies /793 tokens | k3=0.002669; clip=3.405%; token ESS99.325%, response-product ESS50.807% |
| Fresh token ratio range | 0.634–2.277; tails retained, not exact agreement |
| Actual native checkpoint restore | All parameter, AdamW and scheduler hashes match;392 states, step4 |
| Native backward on32 longest real records | Finite gradient norm0.053456;13.93s; AdamW disabled, weight/moment hashes unchanged |
| Physical NVML memory,0.2s samples | Overall peak68,312.94MiB; backward peak50,664.94MiB |
| CPU adapter/state regression | 7 tests PASS |

Fixed-token checks on greedy-generated text compare **raw model scores**, not
importance sampling of a degenerate greedy behavior distribution. Fresh eager
sampling uses actual behavior probabilities. Response-product ESS remains
materially lower than token ESS; there is no invented ESS acceptance threshold.
The fixed validation packet contains no responses≥256 tokens. Longest-record
backward uses separate real R9 PPO data; this is not an additional online round.

Candidate fingerprint:
`16037972e22e4485e53c9c93efdc43b93247a19b6952b4921fbbc689283a3bc2`.
GPU1 UUID: `GPU-de4c1820-8c98-2448-4687-ae7c5bd4037a`.
Software: Torch2.8.0+cu128, vLLM0.11.0, Triton3.4.0,
Transformers4.57.3, FlashAttention2.8.3, pinned verl-agent20bd331b.
Sources at execution are retained under the run's `registered_source/`.

## Frozen next experiment

`BASELINE_PLAN.json` freezes independent native GRPO/GiGPO arms, each four
rounds of four fixed tasks×four trajectories, fresh original model and explicit
initialization seed0. Use TP1 eager FLASH_ATTN, BF16 frozen base, FP32 LoRA/AdamW,
actor/logprob micro16, logical PPO mini256 and maxseq16. Compare initialization
hashes; retain every real batch and pre/post gates. This differs from R9's
near-capacity CUDA-graph configuration, whose scores are not learning data for
this baseline. Before/final greedy evaluation uses the same four training tasks;
these small descriptive comparisons cannot establish generalization or CTHA
benefit. Two-card overlap is conditional on measured62GiB host-memory headroom.

Recheck locally:

```bash
source env/verl_agent.sh
python exploratory_ctha/verl_agent_migration/audit_statistical.py \
  --run .runtime/runs/verl_agent_statistical_eager_micro16_r10
```

Model/LoRA/optimizer binaries remain on the server at the R9 checkpoint paths
and hashes in `../r9/checkpoint_files.json`; they are excluded from Git.
