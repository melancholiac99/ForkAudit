#!/usr/bin/env bash
# Preparation only: no optimizer updates, rollout benchmark, or validation queue.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source env/a100.sh
FORKAUDIT_CHECK_ROOT="$FORKAUDIT_ROOT/.runtime/preflight/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$FORKAUDIT_CHECK_ROOT"
UV_CACHE_DIR="$FORKAUDIT_ROOT/.runtime/cache/uv" "$FORKAUDIT_ROOT/.runtime/tools/uv" pip check --python "$FORKAUDIT_ROOT/.runtime/conda/bin/python" > "$FORKAUDIT_CHECK_ROOT/dependency-check.log" 2>&1
python -m pip freeze > "$FORKAUDIT_CHECK_ROOT/requirements-lock.txt"
python -m py_compile exploratory_ctha/*.py exploratory_ctha/v3/*.py scripts/*a100*.py
python -m unittest discover -s exploratory_ctha -p 'test_ctha*.py' -v > "$FORKAUDIT_CHECK_ROOT/ctha-tests.log" 2>&1
CTHA_RUN_ROOT="$FORKAUDIT_CHECK_ROOT/cpu" python exploratory_ctha/v3/check_cpu.py --synthetic-only > "$FORKAUDIT_CHECK_ROOT/v3-cpu.log" 2>&1
python scripts/check_a100_runtime.py hardware --out "$FORKAUDIT_CHECK_ROOT/hardware.json" > "$FORKAUDIT_CHECK_ROOT/hardware.log" 2>&1
python scripts/check_a100_runtime.py environment --out "$FORKAUDIT_CHECK_ROOT/environment.json" > "$FORKAUDIT_CHECK_ROOT/environment.log" 2>&1
for FORKAUDIT_GPU in 0 1; do
    CUDA_VISIBLE_DEVICES="$FORKAUDIT_GPU" python scripts/check_a100_runtime.py vllm --out "$FORKAUDIT_CHECK_ROOT/vllm-$FORKAUDIT_GPU.json" > "$FORKAUDIT_CHECK_ROOT/vllm-$FORKAUDIT_GPU.log" 2>&1
    CUDA_VISIBLE_DEVICES="$FORKAUDIT_GPU" python scripts/check_a100_runtime.py hf --sample "$FORKAUDIT_CHECK_ROOT/vllm-$FORKAUDIT_GPU.json" --out "$FORKAUDIT_CHECK_ROOT/hf-$FORKAUDIT_GPU.json" > "$FORKAUDIT_CHECK_ROOT/hf-$FORKAUDIT_GPU.log" 2>&1
done
printf 'Preparation checks passed: %s\n' "$FORKAUDIT_CHECK_ROOT"
printf 'Original P0/P1 import and full v3 cold-start validation remain separate requirements.\n'
