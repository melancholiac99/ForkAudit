# Accelerated engineering-only acceptance

User requested acceleration after removing all four-round learning pilots. The original FP32 repeat_a completed all eight trajectories (400 steps). The second repetition was stopped early, with its logs retained, to test a faster launch configuration. No policy updates had occurred.

Only the FP32 matmul row tile and warp count change: 16 rows for M<=128, otherwise 32 rows, four warps. The K32 reduction, tf32x3 dot precision, one pipeline stage, sampling/model settings, parity gates, stability gates and engineering checks are unchanged. Original installed vLLM is not edited. Microbenchmarks and a real-response exact bridge are saved before launching this registered queue.

The completed original repeat_a is a comparison reference. Accelerated repeat_b and the 64-episode collection must reproduce its complete responses/actions/environment outcomes. All direction-probe training inputs are freshly collected using the accelerated configuration. No mixed-config training scores.

Only engineering acceptance runs: repeat/cross-batch, probability parity, eight-response four-step direction probe, actual CPU checkpoint restore and updated-adapter GPU parity. No Step-0 Tune, learning pilots, full P1/P2, or A100 experiments. Do not claim algorithm effectiveness.

Output: /root/autodl-tmp/ctha_v3_engineering_fast_20261113. On completion the assistant is awakened for independent review, publication only to credit-grok with publish.py, remote verification, sync, then /bin/bash /usr/bin/shutdown per user instruction. Failures wake for repair, never threshold relaxation. Keep all prior results.
