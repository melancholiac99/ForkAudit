# Bounded CUDA Graph / micro16 probe

**Statistical engineering admission: PASS. Optimizer updates:0.
Learning: NOT_RUN. Generalization: NOT_RUN. CTHA online efficacy: NOT_TESTED.**
This is a separate performance/numerical candidate, not a four-round report.

Resume the actual R9 engineering checkpoint on GPU1, retain native full-response
loss/rewards, FP32 LoRA/AdamW, BF16 frozen base and unchanged k3≤.02 /clip≤.30.
Use actor/logprob micro16,logical PPO mini256,TP1,maxseq16 and CUDA Graph capture
sizes1/2/4/8/16. Candidate scores are not training data for another configuration.

Fixed real tokens at request batches32/16/8/1 pass independent NumPy checks.
Actual native weights/moments/scheduler match the saved checkpoint. A native
backward benchmark on32 longest real records disables AdamW.step and verifies
exact unchanged weights/moments, then reloads the checkpoint to reset scheduler.

| Check | Measured |
| --- | --- |
| Physical NVML observed peak |61,853MiB overall;50,777MiB during backward |
| Nonupdating longest-record backward |12.565s |
| Fresh8-context generation |710 valid response tokens in5.342s,including native synchronization/generation/sleep call |
| Fresh behavior k3 /clip |.00263736 /3.099% |
| Fresh token ESS /response-product ESS |99.457% /50.606% |

These are limited-context timings, not end-to-end ALFWorld rollout throughput
or a paired speedup against R10. The existing per-step synchronization remains;
Rollout Session optimization is not implemented or performance-validated here.
The probe peak does not establish worst-case training VRAM for unseen longer
batches. Prior exact-trajectory nonreproduction and R11 operational failure
remain unchanged.

Configuration SHA256:
`f58a12c7b44680a6df7fa85ed99a906d44636f3dc7bcbce54d9e28cdf64705aa`.
Local numerical arrays/source snapshots remain under
`.runtime/runs/verl_agent_graph_micro16_r13/`. Restore uses the unchanged R9
model/optimizer binaries and hashes already listed in `../r9/checkpoint_files.json`.
