# CTHA v3 numerical repair, 2026-11-13

The r2 direction probe stopped before any optimizer update: sampled logprob mean/p99/max errors were 0.02746/0.35453/1.25877. Frozen BF16 repeated and cross-batch trajectories passed, so this is distinct from rollout reproducibility. Old P1 and all failed diagnostic records are retained.

FP16 still failed the original parity gate (0.00604/0.08971/0.26838). FP32 with default Triton dot passed narrowly; explicit tf32x3 with one pipeline stage passed at 0.00000882/0.00008797/0.00019789 over 868 fresh tokens. TF32x3 uses multiple products to improve precision; it is not claimed to be bitwise IEEE FP32. On SM120, three pipeline stages exceeded shared memory capacity. The registered custom worker changes buffering stages only, without editing installed vLLM files.

This separate implementation uses FP32 weights and HF forwards, disables torch TF32/autocast, requires TRITON_F32_DEFAULT=tf32x3, and preserves every scientific/optimizer/parity/stability threshold. It collects fresh on-policy data in /root/autodl-tmp/ctha_v3_fp32_20261113. BF16 records are not mixed into FP32 training.

vLLM 0.32 native LoRA kernels reject FP32. Sampling therefore exports an ephemeral merged FP32 model on CPU, hashes it, samples without vLLM LoRA, then removes only that temporary export. Training checkpoints remain LoRA plus persistent AdamW state. An updated-adapter cross-engine parity check is required after the direction probe and before the baseline pilots. Each training round also retains its pre-update parity check.

Validation order: CPU checks; fresh repeated/cross-batch rollout; original eight-response direction probe; updated-adapter merge parity; Step-0 Tune; four rounds each of GRPO and GiGPO with fixed evaluation seeds. Engineering, learning and generalization are reported separately. No CTHA efficacy claim, full P1/P2 expansion, or automatic shutdown.
