#!/usr/bin/env bash
# Train the three structural stage networks of PressNet++ for one architecture.
# The stages are independent (no shared weights), so they can also be launched
# on separate GPUs in parallel: set STAGES="1" / "2" / "3" in three shells.
#
#   bash scripts/train_all_stages.sh <model> <mesh> [extra train.py flags]
#   bash scripts/train_all_stages.sh transolver coarse
#   STAGES="4" bash scripts/train_all_stages.sh transolver coarse     # monolithic baseline
#
# <model> : gcn | reg_dgcnn | meshgraphnet | dilated_dgcnn | transolver
# <mesh>  : coarse | medium | fine   (selects datasets/data/<mesh>/<mesh>_1500_train_val.h5)
set -euo pipefail

MODEL="${1:?usage: $0 <model> <mesh> [extra train.py flags]}"
MESH="${2:?usage: $0 <model> <mesh> [extra train.py flags]}"
shift 2

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGES="${STAGES:-1 2 3}"
DATA="$ROOT/datasets/data/${MESH}/${MESH}_1500_train_val.h5"
CONFIG="$ROOT/configs/train/${MODEL}.json"

[ -f "$CONFIG" ] || { echo "No config $CONFIG"; exit 1; }
[ -f "$DATA" ]   || { echo "Dataset not found: $DATA (see datasets/README.md)"; exit 1; }

cd "$ROOT/surrogateAI"
for STAGE in $STAGES; do
  echo "=== ${MODEL} / ${MESH} / stage ${STAGE} ==="
  python train.py --config "$CONFIG" \
      --train_val_data_path "$DATA" \
      --output_dir "$ROOT/outputs/${MESH}" \
      --stage "$STAGE" \
      --no_wandb "$@"
done
