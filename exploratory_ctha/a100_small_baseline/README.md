# Four-task baseline RL diagnostic

User approved four fixed tasks x four trajectories, both GRPO and GiGPO-style
arms x four rounds. This replaces the cost-paused sixteen-task pilot, which
had zero optimizer updates. Its logs/results remain intact.

Tasks are seeds1004/1028/1150 (the three original-base mixed-success tasks) and
1065 (an all-success control). This selection uses observed base outcomes and
deliberately supplies reward contrast. Results describe this enriched subset;
they do not establish generalization or CTHA efficacy.

Reuse exact original-base A100 round-zero episodes for these tasks (582 response
steps), and the already completed independent fixed-seed base evaluation.
Episode bytes/provenance are retained; subset summaries and source hashes are
recorded explicitly. No new base sampling is necessary.

The R3 full-response PPO trainer, KL0.01, all rows including zero advantage,
accumulation64, LoRA and persistent AdamW remain unchanged. HF uses strict FP32;
the ordinary TF32 candidate is not adopted. One arm per existing A100; no DDP
or additional cards. Start from the original base, never the engineering adapter.

Run all four rounds unless a numerical/stability gate fails. No extension,
automatic CTHA introduction, Tune/unseen evaluation or shutdown is added.
Baseline/final success uses the same four TRAIN tasks x four fixed eval seeds.
Four-task paired bootstrap intervals are descriptive and have limited precision.

```bash
source env/a100.sh
python exploratory_ctha/a100_small_baseline/run.py
```

Outputs: `.runtime/runs/ctha_a100_small_baseline_r1`. TensorBoard6006:
`four_task_baseline/` and `four_task_baseline_live/`. Checkpoints stay local;
code/protocol/results publish only to `credit-grok`.
