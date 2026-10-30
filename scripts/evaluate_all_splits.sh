#!/usr/bin/env bash
# Run the staged pipeline on the three evaluation sets of the paper and compute the metrics.
#
#   bash scripts/evaluate_all_splits.sh <model> <mesh>
#   bash scripts/evaluate_all_splits.sh transolver coarse
#
# Needs  checkpoints/<mesh>/<model>/stage{1,2,3}   (see checkpoints/README.md)
#        datasets/data/<mesh>/<mesh>_1500_{train_val,extrapolation,unseen}.h5   (see datasets/README.md)
# Results land in outputs/eval/<mesh>/<model>/{in_distribution,extrapolation,unseen}/
# Add "--one_step" as a 3rd argument to use the one-step protocol instead of the rolled-out one.
set -euo pipefail

MODEL="${1:?usage: $0 <model> <mesh> [--one_step]}"
MESH="${2:?usage: $0 <model> <mesh> [--one_step]}"
EXTRA="${3:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/inference/${MODEL}_${MESH}.json"
DATA_DIR="$ROOT/datasets/data/${MESH}"

cd "$ROOT/surrogateAI"

run() {  # name file split
  local name="$1" file="$2" split="$3"
  if [ ! -f "$file" ]; then echo "[skip] $file not found"; return; fi
  python inference.py --config "$CONFIG" --data "$file" --split "$split" \
      --out "$ROOT/outputs/eval/${MESH}/${MODEL}/${name}" --evaluate $EXTRA
}

run in_distribution "$DATA_DIR/${MESH}_1500_train_val.h5"     val    # 35 validation trajectories
run extrapolation   "$DATA_DIR/${MESH}_1500_extrapolation.h5" test   # 13 trajectories
run unseen          "$DATA_DIR/${MESH}_1500_unseen.h5"        test   # 20 trajectories
