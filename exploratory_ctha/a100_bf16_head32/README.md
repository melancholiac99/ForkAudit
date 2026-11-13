# Bounded BF16 backbone / FP32 output-head candidate

User authorized trying standard RL numerical-mismatch remedies. This candidate
keeps Full-response PPO with actual vLLM behavior probabilities as denominator
(bypass), original-base KL0.01, rewards, optimizer and clipping unchanged.
It adds no second importance weight, TIS/MIS or rejection mask.

vLLM0.32.0 natively supports `hf_overrides={"head_dtype":"float32"}`. The HF
counterpart keeps the backbone BF16, output-head weights/projection FP32,
and temperature/log-softmax FP32; global autocast does not round the head.
AdamW and trainable LoRA parameters remain FP32.

Numeric stage: sixteen fixed real prompt texts, freshly generated responses,
original mean/P99/max sampled-logprob gates0.01/0.05/0.25 plus unchanged sampled
behavior k3<=0.02 and clip<=0.30. Old BF16 failures remain failures. Any failed
gate stops before a new optimizer update.

Conditional stage only after numeric success: fresh ALFWorld trajectories on
the approved four training tasks, actual GRPO advantages, first four positive
and negative action-mappable responses, four updates. Full-response signs,
finite gradients, stability, live/disk LoRA+AdamW equality, CPU actual restore
and native-BF16-LoRA updated HF/vLLM parity must pass. Action-span signs remain
non-blocking under R3. This bounded probe is not a four-round learning run.

Outputs: `.runtime/runs/a100_bf16_head32_r1`. Isolated source/config hashes and
GPU UUIDs are recorded. No automatic adoption or replacement of the active
FP32 four-task experiment. Cross-GPU/cross-batch reproducibility for this new
precision is not established by this small candidate test. Checkpoint binaries
stay local; publish only code and reviewable evidence to `credit-grok`.
