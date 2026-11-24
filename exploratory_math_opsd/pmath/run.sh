#!/bin/bash
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=/mnt/data/ForkAudit/math-opsd
work="$root/run/P-projector"
mkdir -p "$work"
exec 9>"$work/run.lock"
flock -n 9
export PYTHONPATH="$here"
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH"
export TMPDIR=/mnt/data2/ForkAudit/math-opsd/tmp
export CUDA_HOME="$root/cuda-12.8" HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TORCH_EXTENSIONS_DIR="$root/torch-extensions" TRITON_CACHE_DIR="$root/triton-cache"
export OMP_NUM_THREADS=2
while systemctl is-active --quiet forkaudit-math-endpoints; do sleep 2; done
test -s "$root/run/evaluation/B/aime25/1001.receipt.json"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
four() {
    local stage="$1"
    local pids=()
    for lane in 0 1 2 3; do
        if [[ "$stage" == fit ]]; then
            CUDA_VISIBLE_DEVICES=$lane python "$here/fit.py" "$lane" > "$work/$stage-$lane.log" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES=$lane python "$here/data.py" "$lane" "$stage" > "$work/$stage-$lane.log" 2>&1 &
        fi
        pids+=("$!")
    done
    local status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    return "$status"
}
python "$here/install_teacher.py"
four sample
four score
four fit
CUDA_VISIBLE_DEVICES=0 python "$here/fit.py" finish
CUDA_VISIBLE_DEVICES=0 python "$here/check.py"
export CUDA_VISIBLE_DEVICES=0,1,2,3
python "$here/stages.py" train
python "$here/stages.py" evaluate
