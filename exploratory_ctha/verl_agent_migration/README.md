# Pinned native verl-agent baseline

This migration uses official ALFWorld rollout, GiGPO/GRPO advantages, FSDP,
vLLM and PPO. CTHA is **not enabled**. Previous custom-engine experiments remain
separate and their controller is paused.

## Reproducible environment

Upstream: `langfengQ/verl-agent@20bd331bdbc9026a5668e11362178e10ab7400c8`.
The source archive is unpacked under `.runtime/external/verl-agent-20bd331b`;
Python 3.12 lives under `.runtime/venvs/verl-agent-20bd331b`.
`env/requirements-verl-agent.txt` records resolved dependencies.
Source `env/verl_agent.sh`; do not combine it with `env/a100.sh`.

The pinned checkout requires vLLM 0.11.0, rather than the previous 0.32.0.
Use its `setup.py` constraints: its older `requirements.txt` conflicts with
vLLM's Transformers requirement. The matching prebuilt FlashAttention 2.8.3
wheel is an explicit revision from the README's 2.7.4.post1. No CUDA 13
compilation against CUDA 12.8 Torch is performed.

## Bounded execution

```bash
source env/verl_agent.sh
python exploratory_ctha/verl_agent_migration/prepare_assets.py
python -m unittest discover -s exploratory_ctha/verl_agent_migration -v
python exploratory_ctha/verl_agent_migration/run_native.py --name UNIQUE_RUN
```

Each run creates a new directory and command registration. Start with **one
update**, four real training tasks × four trajectories; never launch the
upstream 150-epoch script. Evaluation is greedy on those same four training
tasks, with no heldout or paper reproduction claim. The upstream preprocessing
downloads Geometry3K only to create modality/index rows; its content does not
provide RL observations, rewards or synthesized trajectories.

## External adapters and evidence

`fixed_tasks.py` restricts native workers to the previously selected tasks.
`correction.py` instruments the pinned trainer without modifying upstream files
or its actor loss. Native mode retains the official recomputed PPO anchor;
`bypass_ppo_clip` uses actual sampler log probabilities exactly once. No extra
importance weight, rejection mask, TIS or Score Centering is added.

Record full-batch token/response ESS, k3, clip fraction, ratio tails, long
responses and native padding copies. Behavior k3 ≤ 0.02 and clip ≤ 0.30 remain
blocking before updates. Raw probability arrays are retained locally. A CPU
hook test and a matrix collective do **not** establish full model reproducibility,
checkpoint recovery or learning; those require separate measured results.

Official reward (10 × success), invalid-action penalty, gamma, XML prompt,
entropy and rank differ from the former engine. Do not attribute comparisons
solely to precision. TensorBoard writes under
`.runtime/tensorboard/ctha-a100-r2/<run-name>` (existing port 6006).
