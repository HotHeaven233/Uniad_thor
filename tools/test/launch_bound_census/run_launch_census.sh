#!/usr/bin/env bash
set -euo pipefail
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH="$PWD:$PWD/tools/test:$PWD/tools/test/cudagraph:${PYTHONPATH:-}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-11.0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OPENBLAS_NUM_THREADS=1
P=tools/test/launch_bound_census
C=${CONFIG:-projects/configs/stage2_e2e/base_e2e_tf32.py}
W=${CHECKPOINT:-ckpts/uniad_base_e2e.pth}
STAGE=${1:?Usage: $0 smoke|pilot|census OUT [args...]}
OUT=${2:?Specify a fresh output directory}
shift 2
case "$STAGE" in
 smoke) exec python -u "$P/collect_launch_shapes.py" "$C" "$W" --stage pilot --out-dir "$OUT" --scenes 1 --repeats 1 --max-pilot-budget 40 "$@";;
 pilot|census) exec python -u "$P/collect_launch_shapes.py" "$C" "$W" --stage "$STAGE" --out-dir "$OUT" "$@";;
 *) echo 'Unknown stage' >&2; exit 2;;
esac
