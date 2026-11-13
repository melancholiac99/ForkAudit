# Native single-A100 BF16 bypass engineering probe

**Full-batch pre/post stability: PASS. Actual cross-GPU checkpoint state
recovery: PASS. Exact greedy trajectory reproduction: NOT_REPRODUCED.
Statistical cross-GPU/batch scoring: PENDING. Learning: NOT_RUN.
Generalization: NOT_RUN. CTHA online efficacy: NOT_TESTED.**

One engineering rollout/update iteration from the original model, not the
four-round baseline. Native actor loss/rewards are unchanged; its PPO
denominator is the actual sampler log probability, without a second IS weight.
The frozen base is BF16; all161,480,704 trainable LoRA parameters and AdamW
moments retain FP32. Checkpoint states show **four actual AdamW steps**.

| Evidence | Result |
| --- | --- |
| Actual batch | 4 fixed tasks ×4 trajectories;771 real response rows |
| Native padding | 29 repeated rows;800 processed rows |
| Pre-update k3 /clip | 0.002155 /1.719% |
| Post-update k3 /clip | 0.002007 /1.714% |
| Pre/post token ESS | 99.611% /99.608% |
| Pre/post response-product ESS | 52.403% /60.671% |
| Unique real rows, pre k3 /clip | 0.002073 /1.722% |
| Pre token ratio extremes | 0.07962 to4.12828 |
| Post token ratio extremes | 0.04887 to4.85169 |
| Full-response positive/negative mean logprob change | +0.001022 /−0.000392 |
| Native gradient norm | 0.284706, finite |
| Updated HF/vLLM raw-score k3 /clip | 0.000694 /0.328% on20,130 generated tokens |
| Restored HF/vLLM raw-score k3 /clip | 0.001010 /0.493% on24,931 generated tokens |
| Actual restore on GPU1 | All parameter, optimizer and scheduler hashes match GPU0;392 optimizer states at step4 |

All original0.02 k3 /0.30 clip gates passed. Raw token errors remain diagnostic
under the authorized BF16 protocol. Updated/restored validation score checks
compare raw model probabilities on greedy-generated text; they are not IS of a
degenerate greedy behavior distribution. The full-batch direction diagnostic
does not claim completion of the original eight-response/four-step probe.

Sampled training success was1/16; final training-task greedy success was1/4.
These are different sampling protocols, not a before/after learning comparison.
After restoration, greedy success was0/4 despite identical live state hashes.
Exact input/response comparisons locate the first differences at real token
prefixes; retain them in `exact_trace_divergences.json`. Do not infer corrupt
weights or a numerical root cause solely from trajectory differences.

The user explicitly authorized a **new statistical consistency criterion** after
reviewing this difference. `STATISTICAL_PLAN.json` preregisters fixed-token
cross-GPU/batch checks; it does not retrospectively make exact reproduction pass.

## Measured cost and memory

Rollout318.90s, pre-score40.73s, actor124.98s, final evaluation267.03s,
checkpoint55.88s, whole iteration848.17s excluding cold startup. Post-update
scoring is included in the whole-iteration time. This has extra checks compared
with R8; do not compare total runtimes as precision-only effects.

NVML observed **80,781MiB** on GPU0 at0.2-second sampling. This is too close to
card capacity to adopt unchanged for the four-round pilot. The next candidate
uses a smaller physical microbatch and eager rollout; logical PPO mini256 and
the task/trajectory budget remain fixed. Verify memory rather than assuming
that two complete native agents fit this62GiB host.

Full model, LoRA, optimizer and extra-state binaries remain on the server under
`.runtime/runs/verl_agent_bypass_single_gpu_r9/checkpoints/global_step_1/actor/`.
`checkpoint_files.json` records exact paths, sizes and SHA256 values.
Live tensor-state fingerprints differ from file-byte checksums; both are kept.
Exact raw/PPO DataProto and token arrays are retained locally. No model or
optimizer binary is committed. TensorBoard6006 run: `verl_agent_bypass_single_gpu_r9`.
