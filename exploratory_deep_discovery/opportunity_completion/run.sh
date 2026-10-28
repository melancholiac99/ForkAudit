#!/usr/bin/env bash
set -euo pipefail
EXPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPT_PYTHON="${OPPORTUNITY_PYTHON:-/root/miniconda3/envs/m0/bin/python}"
# An absolute Python path does not activate its conda bin directory. FlashInfer's
# runtime compiler also needs ninja and nvcc on PATH in the GPU container.
EXPT_PYTHON="$(command -v "$EXPT_PYTHON")"
export PATH="$(dirname -- "$EXPT_PYTHON"):$PATH"
if [[ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export PATH="$CUDA_HOME/bin:$PATH"
fi
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
MODE="${1:-help}"
case "$MODE" in
  smoke)
    RUN_NAME=gpu_smoke
    RUN_ARGS=(--run-name "$RUN_NAME" --limit-points 2 --reps 2)
    OUTPUT_NAME="$RUN_NAME"
    ;;
  full)
    RUN_NAME=main_32x16
    RUN_ARGS=(--run-name "$RUN_NAME" --reps 16)
    OUTPUT_NAME="$RUN_NAME"
    ;;
  dry)
    RUN_NAME=cpu_smoke
    RUN_ARGS=(--dry --run-name "$RUN_NAME" --point-id s077_t42 --reps 1)
    OUTPUT_NAME="dry_$RUN_NAME"
    ;;
  check)
    exec "$EXPT_PYTHON" "$EXPT_DIR/run.py" --check-only
    ;;
  analyze)
    exec "$EXPT_PYTHON" "$EXPT_DIR/analyze.py" "$EXPT_DIR/runs/main_32x16"
    ;;
  *)
    echo "Usage: bash $EXPT_DIR/run.sh {smoke|full|dry|check|analyze}"
    echo "smoke: 16 GPU continuations; full: 2048 GPU continuations; same command resumes."
    exit 0
    ;;
esac
if [[ "$MODE" == dry ]]; then
  WORKERS=1
  CONCURRENCY=4
else
  WORKERS="${OPPORTUNITY_WORKERS:-4}"
  CONCURRENCY="${OPPORTUNITY_CONCURRENCY:-64}"
fi
mkdir -p "$EXPT_DIR/runs/$OUTPUT_NAME"
"$EXPT_PYTHON" -u "$EXPT_DIR/run.py" "${RUN_ARGS[@]}" --workers "$WORKERS" \
  --concurrency "$CONCURRENCY" 2>&1 | tee -a "$EXPT_DIR/runs/$OUTPUT_NAME/console.log"
"$EXPT_PYTHON" "$EXPT_DIR/analyze.py" "$EXPT_DIR/runs/$OUTPUT_NAME"
