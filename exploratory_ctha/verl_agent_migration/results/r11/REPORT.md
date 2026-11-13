# R11 incomplete baseline and operational failure

**Status: FAIL_OPERATIONAL. Learning: INCOMPLETE. Generalization: NOT_RUN.
CTHA online efficacy: NOT_TESTED.** No four-round completion is claimed.

The eager configuration completed the initial four-task training-set greedy
evaluation(1/4) and first16-episode training rollout. Actual data contains757
response rows /104,818 valid response tokens, then11 native padding copies.
The first pre-update gate passed: k3=.002176255,clip=.016283795.

During cost diagnosis the agent SIGSTOPped the Ray TaskRunner while the actor
update had already started. The keepalive watchdog timed out, destroying the
owning job and actors before checkpoint save. This was an agent operational
error. The actual completed AdamW step count is unverified and this update
cannot be resumed. Do not relabel this run zero-update or use its data as a
new configuration's on-policy samples. Pending second-arm execution was stopped.

Initial evaluation took about9m25s after model ready; first training rollout
about11m+ based on artifact wall boundaries. No trustworthy full actor/iteration
runtime is reported after the control process was paused. Saved probabilities,
raw/PPO DataProto and NPZ exports, memory traces, source snapshots and logs are
retained under `.runtime/runs/verl_agent_grpo_four_rounds_r11/`.
`operational_failure.json` records the source evidence hashes.

Thirty-second Python stack occupancy(563 samples): Triton launch124,
other generation364,wake45,sleep30. This is not CUDA kernel timing and does
not establish the exact speedup achievable by a Rollout Session.

Use cooperative checkpoint boundaries for subsequent interruption. Preserve
this failure and its wasted cost separately from any completed learning pilot.
