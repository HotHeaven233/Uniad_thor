#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <config.py> <checkpoint.pth> <out_dir> [extra args...]"
  exit 2
fi
CONFIG="$1"
CHECKPOINT="$2"
OUT_DIR="$3"
shift 3

python tools/test/backbone_budget_sweep/collect_backbone_budget_sweep.py \
  "$CONFIG" "$CHECKPOINT" \
  --stage sweep \
  --out-dir "$OUT_DIR" \
  --budget-axis packed \
  --sweep-budgets "0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100" \
  "$@"
