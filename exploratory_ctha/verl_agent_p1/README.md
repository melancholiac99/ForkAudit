# Full native P1

This is the newly authorized full P1: **GRPO, GiGPO, GiGPO+CTHA**, each12 rounds
of16 scheduled TRAIN tasks x4 trajectories. Each arm starts independently from
the original Qwen2.5-7B-Instruct with matching seed0 LoRA initialization.
Before training and every2 rounds, evaluate40 disjoint development Tune tasks
x2 stochastic samples. This is not the earlier four-task pilot or heldout P2.

All arms use the same pinned verl-agent, BF16 base/compute with FP32 LoRA and
AdamW masters, native full-response PPO and entropy, Bypass PPO, and the TP1
rollout-session adapter. Temperature0.7 follows full P1. vLLM must explicitly
return `processed_logprobs` so temperature-adjusted behavior probabilities are
the PPO denominator. Native episode/step advantages stay unchanged; CTHA adds
only its gated auxiliary after the native GiGPO estimator.

CTHA retains its original feature window, leave-task-type-out fitting, shuffle
gate, sign stability, legal-action centering, all-failed-group activation and
annealing. Verified P0 is predictor-only cold start. Current TRAIN episodes and
this arm's latest8 TRAIN rounds may enter fitting; Tune and other arms cannot.
If effective auxiliary values remain zero, the experiment has not tested CTHA.

```bash
source env/verl_agent.sh
python -m unittest discover -s exploratory_ctha/verl_agent_p1 -p 'test_*.py' -v
python exploratory_ctha/verl_agent_p1/check_environment.py
python exploratory_ctha/verl_agent_p1/prepare_campaign.py \
  --out .runtime/runs/verl_p1_full_r1
# Publish the registered source/protocol and write verified publication.json first.
python exploratory_ctha/verl_agent_p1/queue.py \
  --campaign .runtime/runs/verl_p1_full_r1
```

The queue serializes arms because this host has62.8GiB RAM. Generation concurrency
16 does not change64 logical training trajectories. Checkpoints are saved every2
rounds; only the latest complete checkpoint per arm is retained to fit disk.
Prior checkpoint inventories and live fingerprints survive retention. Final
native restore and independent NumPy probability/outcome audits are queued.
Any gate/error stops the queue without an automatic restart or protocol change.

TensorBoard uses the existing6006 server, under `verl_p1_full_r1_*`; primary Tune
success is `p1/tune_success_episode_mean`, not native response-weighted values.
`episode_metrics.jsonl`, `applied_auxiliary_step_*.json`, `ctha_*.json`, saved real
probability arrays and per-episode records support independent review. Native
initial validation logs global_step1 although no update occurred; exported
`tune_evaluation_00.json` is explicitly baseline, and final tables use after_round0.

Old custom P1 differs in trainer, prompts and hyperparameters. Results are a new
one-seed development comparison, not a reproduction or generalization claim.
