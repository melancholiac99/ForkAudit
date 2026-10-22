#!/bin/bash
# Run a project script inside the m0 env with the fixed runtime environment.
# Usage: scripts/run.sh scripts/10_rollout.py --seeds 0-4
export PATH=/root/miniconda3/envs/m0/bin:$PATH
export ALFWORLD_DATA=/root/autodl-tmp/alfworld_data
export OMP_NUM_THREADS=8          # system default 0 is rejected by libgomp
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT=/root/autodl-tmp/.cache/vllm
cd "$(dirname "$0")"
exec python "$@"
