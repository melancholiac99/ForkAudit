# BF16 bypass-PPO limited probe: PASS under the NEW protocol

Original strict BF16/head32 results remain FAIL. Raw absolute logprob errors
still exceed the old limits and are diagnostics only under the explicitly
approved new protocol. This does not establish strict training/inference equality.

Fresh16 episodes on four selected train tasks produced580 responses. The
first four positive/four negative mappable actual-GRPO-advantage responses
completed four updates. Full-response PPO+KL0.01 and rewards unchanged; no
extra IS weights, TIS/MIS, rejection or synthetic reward labels.

Signed full-response means: positive+0.010677, negative-0.003067. Action-span
positive mean remains negative and is non-blocking under R3. Post-update
sampled behavior k3=0.001288, clip=0.01549. Actual saved LoRA/AdamW live/disk
and CPU restore/fixed-gradient next-step checks passed. Native BF16 LoRA
sampling uses BF16 adapter compute from the saved FP32 LoRA; its mismatch is
measured, not described as bitwise identity. Fresh updated-model mismatch:
k3=0.001247, clip=0.01482, raw mean/P99/max=0.01893/0.23076/0.43132;
sequence ESS fraction0.8401 across eight replies.

Training portion16.18s for eight responses/four updates. Earlier FP32 probe
was70.53s with a different set of eight real responses; this is indicative,
not a controlled end-to-end speedup. Whole bounded validation took about7.4min
including fresh environment generation, model loading and checks.

BF16 limited probe: PASS. Full Engineering: INCOMPLETE (new-precision
cross-GPU/cross-batch repeat and wider validation not run). Learning: NOT_RUN.
CTHA online efficacy: NOT_TESTED. Not adopted into the live FP32 learning arms;
paused GiGPO/controller resumed. No new cards. Original FP32 results preserved.

All16 trajectories are included in compact textual transport. Decode:

```bash
cat rollout_parts/*.b64 | base64 -d > rollouts.jsonl.gz
sha256sum rollouts.jsonl.gz  # compare rollout_manifest.json
gzip -dc rollouts.jsonl.gz > rollouts.jsonl
```

Model/optimizer binaries stay under the recorded server directory; hashes are
in verification.json and train_log.json. CPU next-step equality uses fixed
engineering gradients, not a second online RL round. Four steps do not show
success-rate improvement, generalization or CTHA effectiveness.
