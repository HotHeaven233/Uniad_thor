#!/usr/bin/env bash
set -euo pipefail
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH="$PWD:$PWD/tools/test:$PWD/tools/test/cudagraph:${PYTHONPATH:-}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-11.0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OPENBLAS_NUM_THREADS=1
P=tools/test/launch_bound_census_v2
OUT=${1:-work_dirs/launch_bound_census_v2/roi_batch_low_pilot}
shift || true
exec python -u "$P/collect_launch_shapes.py" \
  projects/configs/stage2_e2e/base_e2e_tf32.py \
  ckpts/uniad_base_e2e.pth \
  --stage pilot --out-dir "$OUT" \
  --pilot-modes roi_batch --pilot-budgets 2.5,5,10,20 \
  --scenes 6 --repeats 2 --warmup 1 "$@"
