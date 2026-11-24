#!/usr/bin/env bash
# Mount the retained environment after a machine change, then enter the registered launcher.
set -euo pipefail
action=${1:-all}
base=${OPSD_ROOT:-/mnt/data/ForkAudit/math-opsd}
here=$(cd "$(dirname "$0")" && pwd)
if [[ -f "$base/conda-env.sqfs" && ! -x "$base/conda-env/bin/python" ]]; then
    (
        flock -x 9
        if [[ ! -x "$base/conda-env/bin/python" ]]; then
            fs="$base/.conda-fs"
            mkdir -p "$fs"/{lower,upper,work} "$base/conda-env"
            if ! mountpoint -q "$fs/lower"; then
                mount -t squashfs -o loop,ro "$base/conda-env.sqfs" "$fs/lower"
            fi
            if ! mountpoint -q "$base/conda-env"; then
                mount -t overlay overlay -o "lowerdir=$fs/lower,upperdir=$fs/upper,workdir=$fs/work" "$base/conda-env"
            fi
        fi
    ) 9>"$base/.mount.lock"
fi
if [[ "$action" == env-check ]]; then
    exec env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 "$base/conda-env/bin/python" "$here/verify_environment.py" "$base"
fi
exec "$base/conda-env/bin/python" "$here/launch.py" "$action" --root "$base"
