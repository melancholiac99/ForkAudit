#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
probe_group() {
    local group=$1 rc=0 pid gpu
    local -a workers=()
    for gpu in {0..3}; do CUDA_VISIBLE_DEVICES=$gpu python dose15.py probe "$group" "$gpu" & workers+=($!); done
    for pid in "${workers[@]}"; do wait "$pid" || rc=1; done
    return "$rc"
}
python dose15.py training
probe_group new
python dose15.py collect new
python dose15.py aime
python dose15.py math R-lambda15@100
python dose15.py openings
python endpoint200.py aime
probe_group r
python dose15.py collect r
python dose15.py math R-extension@200
probe_group weak
python endpoint200.py collect
python evaluate.py
