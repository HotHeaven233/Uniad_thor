#!/usr/bin/env bash
set -euo pipefail

cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate

export PYTHONPATH=/home/uestc/uniad:${PYTHONPATH:-}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

P=tools/test/cudagraph/collect_backbone_shapes.py
CONFIG=${CONFIG:-projects/configs/stage2_e2e/base_e2e_tf32.py}
CHECKPOINT=${CHECKPOINT:-ckpts/uniad_base_e2e.pth}
STAGE=${1:-smoke}
OUT=${2:-work_dirs/cudagraph_shape_census_${STAGE}}
shift || true
if (($#)); then shift; fi

case "$STAGE" in
  smoke)
    echo "[CENSUS] smoke run: first 24 dataset frames only (not a complete manifest)"
    python -u "$P" "$CONFIG" "$CHECKPOINT" \
      --out-dir "$OUT" --max-frames 24 --log-every 4 "$@"
    ;;
  full)
    echo "[CENSUS] full run: every frame in cfg.data.test"
    python -u "$P" "$CONFIG" "$CHECKPOINT" \
      --out-dir "$OUT" --max-frames 0 --log-every 50 "$@"
    ;;
  *)
    echo "Usage: bash $0 smoke|full [output_dir] [extra args...]" >&2
    exit 2
    ;;
esac
