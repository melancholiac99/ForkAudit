#!/usr/bin/env bash
# Prepare a private checkout and a dedicated Python environment. No model generation or training.
set -euo pipefail
base=${1:-/mnt/data/ForkAudit/math-opsd}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$base"/{run,logs}
if [[ -f "$base/conda-env.sqfs" && ! -x "$base/conda-env/bin/python" ]]; then
    OPSD_ROOT="$base" bash "$here/run_four.sh" env-check
fi
if [[ ! -d "$base/source/.git" ]]; then
    git init -q "$base/source"
    git -C "$base/source" remote add origin https://github.com/siyan-zhao/OPSD.git
    git -C "$base/source" fetch -q --depth=1 origin ae7d2519e94920c4eb6206c0c26de46d9c50abae
    git -C "$base/source" checkout -q --detach FETCH_HEAD
fi
[[ $(git -C "$base/source" rev-parse HEAD) == ae7d2519e94920c4eb6206c0c26de46d9c50abae ]]
if git -C "$base/source" diff --quiet; then
    git -C "$base/source" apply --unidiff-zero "$here/upstream.patch"
else
    git -C "$base/source" diff --binary --unified=0 | cmp "$here/upstream.patch" -
fi
if [[ ! -x "$base/conda-env/bin/python" ]]; then
    CONDA_PKGS_DIRS="$base/conda-pkgs" conda create -y --override-channels -c conda-forge \
        --prefix "$base/conda-env" python=3.10 pip
fi
python_bin="$base/conda-env/bin/python"
sed -n 's/^    - \([^ ]*==[^ ]*\)$/\1/p' "$base/source/environment.yml" > "$base/requirements-upstream.txt"
CUDA_VISIBLE_DEVICES='' DS_BUILD_OPS=0 "$python_bin" -m pip install -r "$base/requirements-upstream.txt"
CUDA_VISIBLE_DEVICES='' "$python_bin" -m pip install flash-attn==2.8.3 --no-build-isolation
if [[ ! -x "$base/cuda-12.8/bin/nvcc" ]]; then
    "$python_bin" "$here/prepare_toolkit.py" "$base"
fi
"$python_bin" "$here/prepare_assets.py" "$base"
CUDA_VISIBLE_DEVICES='' "$python_bin" "$here/preflight.py" "$base" --output "$base/run/preflight.json"
CUDA_VISIBLE_DEVICES='' "$python_bin" "$here/verify_environment.py" "$base"
CUDA_VISIBLE_DEVICES='' "$python_bin" "$here/check_optimizer.py" "$base"
