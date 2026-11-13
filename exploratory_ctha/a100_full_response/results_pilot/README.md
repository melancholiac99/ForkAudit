# Baseline RL pilot — initial snapshot

This is a launch snapshot, not a completed result. The controller first evaluates the original base on sixteen fixed TRAIN tasks x four independent fixed seeds. It then runs exactly four rounds each of GRPO/GiGPO-style on two independent A100s and evaluates the final policies on those same fixed eval seeds.

No pilot update or learning improvement is claimed by this initial snapshot. No CTHA, Tune or unseen evaluation. Live server status and metrics remain at the registered output path and TensorBoard port6006. A numerical/stability failure stops the affected arm without threshold changes.
