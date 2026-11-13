# Rollout Session paired throughput check

**PASS within the frozen real-request benchmark. Learning: NOT_RUN.
Generalization: NOT_RUN. CTHA online efficacy: NOT_TESTED.**

An opt-in TP1 session retains native vLLM residency and one LoRA adapter
through a rollout batch. Native synchronization and RNG/mode restoration happen
at begin/end, with finally cleanup and open-session guards before scoring,
updates or checkpoint operations. The default execution path is unchanged.

Same A100 GPU0, same R9 native checkpoint, same six requests of16 real saved
ALFWorld contexts per block; bounded CUDA Graphs and micro16 in both paths.
Common warmup precedes legacy/session/session/legacy ordering. Timings include
session-control RPC overhead; validation scoring and state hashing are outside
these generation timings. No optimizer or environment step occurs in this tape.

| Execution | Block seconds | Sync/wake count | Sleep count |
| --- | --- | --- | --- |
| Legacy1 |35.510 |6 |6 |
| Session1 |23.035 |1 |1 |
| Session2 |22.209 |1 |1 |
| Legacy2 |34.983 |6 |6 |

Median wall speedup is **1.558x**, about35.8% less time for the same prompt tape.
Generated response counts match but token counts differ slightly:12343/12411
legacy and12558/12502 session. Exact greedy response equality isfalse, including
variation between legacy repeats; this is statistical rather than exact
trajectory admission. Report both token throughput and wall time, not a claim
of bitwise equivalence. Native sync/wake+sleep host times identify removed
transition overhead; they are not CUDA kernel timings.

All raw-score and fresh sampled-behavior k3≤.02 /clip≤.30 gates pass independent
NumPy recomputation, with ESS/tails retained. Fixed16-row actual full-response
PPO plus entropy loss is exactly **−0.49719682335853577** before/after. Actual
weights, AdamW moments/steps and scheduler match before/after and the native
source checkpoint. Observed NVML peak59,672.94MiB.13 CPU tests pass, including
collector exception cleanup and rejection of scoring/updates in an open session.

This does not measure a complete ALFWorld rollout or training iteration, and
establishes no learning benefit. The separate R15 real-environment val-only
integration check follows. No active environment filtering, LM-head changes,
reward/loss edits or reduced logical training batch are included.

Recompute locally:

```bash
source env/verl_agent.sh
python exploratory_ctha/verl_agent_migration/audit_session.py \
  .runtime/runs/verl_agent_session_pair_r14
```

Original arrays, exact generated packets and registered source snapshot remain
in that run directory. Model/optimizer binaries remain at the existing R9
checkpoint paths. `request_tape.json` indexes the previously published R9
validation packet, so the real input contexts are reconstructible.
