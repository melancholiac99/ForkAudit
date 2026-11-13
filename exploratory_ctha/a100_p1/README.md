# CTHA v3 P1 on A100

**Preparation draft: not launched. Engineering R2 failed. Main P1 orchestration is not complete; the gate must not be bypassed.**

The user authorized this phase after requesting immediate P1 training when the engineering acceptance passes. This supersedes the earlier instruction to power off immediately after engineering. P1 must not start before independent `Engineering: PASS`.

This is the existing five-arm, twelve-round, seed-zero P1 budget: GRPO, GiGPO-style base, fixed bonus, CTHA habit bonus and progress-only bonus. Each round uses sixteen fixed-order train games with four episodes each. Tune evaluation uses forty games with two repetitions every two rounds. P2 and `valid_unseen` are outside this scope.

It is a new v3 registration rather than a reproduction of old P1 numbers. It uniformly uses the A100 sampling configuration selected by engineering, FP32 HF scoring, persistent AdamW, an original-base reference penalty and 64-row gradient accumulation. Zero-advantage rows remain in the logical batch. Probability and stability gates remain unchanged.

The verified frozen-base engineering round-zero rollouts can be reused identically across arms as a documented computational cache; the engineering LoRA checkpoint cannot initialize P1. Historical P0 records serve only the auxiliary predictor. Every actor update uses the current arm's on-policy tokens and behavior probabilities.

TensorBoard remains the viewing interface. Besides loss and tune curves, the run must report gate reasons and actual auxiliary-gradient activation. A zero-activation CTHA run cannot establish the method's value. One seed and a development tune split do not establish generalization.
