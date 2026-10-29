#!/usr/bin/env bash
set -euo pipefail
FACTORIAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FACTORIAL_PY=/root/miniconda3/envs/m0/bin/python
export PATH=/root/miniconda3/envs/m0/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$FACTORIAL_PY" -u "$FACTORIAL_DIR/run.py" --run-name main_6x8 --reps 8 --workers 8 --concurrency 256
"$FACTORIAL_PY" "$FACTORIAL_DIR/analyze.py"
