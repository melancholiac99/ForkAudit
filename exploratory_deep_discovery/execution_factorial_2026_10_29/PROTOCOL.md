# Information × one-time execution repair

Exploratory mechanism experiment, 2026-10-29. Fixed before new outcomes.

Six previously specified points: s005_t31, s006_t24, s021_t15, s078_t36
(old Dead), s010_t29, s073_t10 (old Fork). These are purposefully selected
mechanism cases, not a population sample. Original five actions, 8 repeats/cell.

Four conditions: native, completed (location clue), repair, repair_completed.
The first two reuse the first 8 repeats of the completed five-arm experiment,
with source hashes. The last two generate 480 new continuations, including
branches where repair is a no-op. No expansion based on results.

Repair targets only the unrelated object held immediately after the original
forced action. No target-category object is eligible. Before each subsequent
LLM decision, if the originally flagged object can legally be placed, execute
the lexicographically first legal `move OBJECT to ...` command exactly once.
If unavailable immediately, let the unmodified LLM continue until it is legal.
Do not navigate, open containers, repeatedly repair, or repair later unrelated
objects on the model's behalf. A repair uses one real step of the original
50-step budget and appears normally in the two-transition memory.

This clarification of the earlier draft is fixed from legal-action metadata:
several original branches have no immediate legal placement. It is a one-time
continuation controller, not a claim to improve the model's learned ability.

The truthful location clue is appended once immediately after the original
forced action in both completed conditions, BEFORE any repair. It then obeys
the original two-transition memory, including repair steps. Thus repair has
its real state, time and visible-history consequences; it is not a pure hidden
physical intervention. No clue is repeated and no location is invented.

Qwen2.5-7B-Instruct, original prompt/parser, T=0.7, 512 response tokens,
50 total steps, 2-transition history. New seeds use namespace 93e12, shared
between the two repair conditions at matching absolute step. Controls retain
their original seeds; no paired-seed identity is claimed across repair/no-repair.

Primary readout: all six complete five-action success vectors; mean success
mu; action variance C using distinct-repeat cross products without clipping;
information × repair interaction in mu and C. Also measure actual repair
activation and actual target pickup from environment responses. Report all
points. Do not use finite-sample label flips as primary mechanism evidence.

Predictions: a common execution obstacle can suppress upstream action effects;
repair may expose information-driven differences, and shared information may
reduce those differences. For branch-specific obstacles, shared information
may expose the blocked branch; repairing it should fill that remaining gap.
Improved mu alone is weaker evidence than the predicted action-vector change.
If target pickup improves but terminal success does not, report the remaining
execution boundary without extending to arbitrary multi-step repairs.

CPU preflight checks all 30 original branches, clue truth, original held
objects, and immediate legal releases. A separate random-policy dry run checks
repair integration and budget accounting; never mix dry data into results.
Inputs and code are hashed. Frozen experiment files remain unchanged.

User requested shutdown after completion. An independent two-hour deadline
watchdog is armed as a fallback; normal completion saves, uploads, syncs and
requests AutoDL shutdown immediately, without waiting for the deadline.
