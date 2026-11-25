#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=/mnt/data/ForkAudit/math-opsd
export PATH="$root/conda-env/bin:$root/cuda-12.8/bin:$PATH" HF_HOME="$root/hf-cache" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONNOUSERSITE=1
out="$root/run/internalization-v1"
mkdir -p "$out"
exec 9>"$out/runner.lock"
flock -n 9
while systemctl is-active --quiet forkaudit-long-math; do sleep 5; done
for model in B R@25 R@50 R@75 R@100 T0@100 P@100 Rres@100; do
    test ! -s "../evidence/internalization/$model.json" || continue
    for stage in features factors calibrate measure; do
        pids=()
        for lane in {0..3}; do
            CUDA_VISIBLE_DEVICES="$lane" python internalization.py "$stage" "$model" "$lane" > "$root/logs/internal-$model-$stage-$lane.log" 2>&1 &
            pids+=("$!")
        done
        rc=0
        for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
        test "$rc" = 0
    done
    python internalization.py report "$model"
    # Keep per-question statistics, factors and teacher cache. Release only reproducible scratch features.
    find "$out/$model" -maxdepth 1 -name 'features-*.pt' -delete
done
