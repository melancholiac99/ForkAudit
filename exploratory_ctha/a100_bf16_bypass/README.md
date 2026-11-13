# Explicitly authorized BF16 bypass-PPO four-step probe

This is a NEW protocol approved explicitly by the user. Raw sampled-logprob
absolute error is diagnostic; the old strict BF16 and head32 FAIL results remain
FAIL. No retrospective reclassification or silent threshold change.

Use BF16 backbone, FP32 output head/logits/softmax, FP32 trainable LoRA and
AdamW. Preserve the Full-response PPO ratio against actual vLLM behavior
logprobs, original-base KL0.01, rewards and all-row denominator. No additional
importance weights, TIS/MIS, rejection masks or loss change.

Collect fresh16 ALFWorld episodes on the approved four tasks. Select the first
four positive and four negative action-mappable actual-advantage responses.
Four updates only; finite gradients, signed full-response direction, sampled
behavior k3<=0.02 and clip<=0.30 remain blocking before/after updates. Action
span signs are diagnostic under R3. Save actual LoRA and AdamW, compare live
state to disk, restore on CPU and test fixed-gradient next-step equality.
Fresh updated-native-LoRA samples must pass the same ratio-stability limits;
retain raw probability errors and sequence-ratio diagnostics.

Outputs: `.runtime/runs/a100_bf16_bypass_r2`. Pause the existing GiGPO worker
and controller; GPU0 GRPO continues. Resume in `finally`; ten-minute ceiling.
No new cards or automatic deployment to the four-round learning experiment.
Even a successful limited probe leaves full cross-GPU/cross-batch engineering
acceptance incomplete, with Learning NOT_RUN and CTHA efficacy NOT_TESTED.
