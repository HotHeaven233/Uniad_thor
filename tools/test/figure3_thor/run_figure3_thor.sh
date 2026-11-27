#!/usr/bin/env bash
set -euo pipefail
cd /home/uestc/uniad
source /home/uestc/.venvs/uniad/bin/activate
export PYTHONPATH=/home/uestc/uniad:${PYTHONPATH:-}
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=11.0
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
P=tools/test/figure3_thor
C=${CONFIG:-projects/configs/stage2_e2e/base_e2e_tf32.py}
W=${CHECKPOINT:-ckpts/uniad_base_e2e.pth}
STAGE=${1:-smoke}
R=${2:-work_dirs/figure3_thor_${STAGE}}
mkdir -p "$R"
run(){ local name=$1; shift; python -u "$P/profile_figure3_thor.py" "$C" "$W" --out-dir "$R/$name" "$@" 2>&1 | tee "$R/$name.log"; }
case "$STAGE" in
 smoke)
   run summary --profile summary --scenes 1 --frames-per-scene 2 --warmup 2 --repeats 1
   run detail --profile detail --manifest "$R/summary/manifest.json" --repeats 1
   run area --profile area --scenes 1 --frames-per-scene 1 --warmup 2 --repeats 1 --shapes 128x128,384x384,768x768
   python "$P/plot_figure3_thor.py" --summary "$R/summary" --detail "$R/detail" --area "$R/area" --out-dir "$R/report"
   ;;
 full)
   run summary --profile summary --scenes 12 --frames-per-scene 8 --warmup 3 --repeats 2
   run plain --profile plain --manifest "$R/summary/manifest.json" --repeats 2
   run detail --profile detail --scenes 6 --frames-per-scene 4 --warmup 3 --repeats 2
   run area --profile area --scenes 3 --frames-per-scene 1 --warmup 3 --repeats 2
   python "$P/plot_figure3_thor.py" --summary "$R/summary" --detail "$R/detail" --area "$R/area" --plain "$R/plain" --out-dir "$R/report"
   ;;
 trace)
   command -v nsys >/dev/null || { echo 'Nsight Systems nsys unavailable; keep panel b pending'; exit 2; }
   nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop --force-overwrite=true -o "$R/nsys" \
     "$(which python)" -u "$P/profile_figure3_thor.py" "$C" "$W" --out-dir "$R/trace" --profile summary --scenes 1 --frames-per-scene 1 --warmup 3 --repeats 1 --trace 2>&1 | tee "$R/nsys.log"
   nsys export --type sqlite --force-overwrite=true --output "$R/nsys.sqlite" "$R/nsys.nsys-rep" 2>&1 | tee "$R/export.log"
   ;;
 *) echo 'Usage: run_figure3_thor.sh smoke|full|trace [out-dir]'; exit 2;;
esac
