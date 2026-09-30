#!/usr/bin/env bash
# Run Marigold inference in a detached tmux session (survives terminal/Cursor disconnect).
set -euo pipefail

SESSION="marigold-cegled"
ROOT="$(cd "$(dirname "$0")" && pwd)"
IMAGE_DIR="/home/hasan-spark/workspace/projects/data/universal_model/cegled_econo/cegled_econo_flat_8mats/chip_modeling/train/images"
OUTPUT_ROOT="/home/hasan-spark/workspace/projects/data/universal_model/cegled_econo/cegled_econo_flat_8mats/chip_modeling/train"
LOG="$ROOT/cegled_econo_infer.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists."
  echo "  attach:  tmux attach -t $SESSION"
  echo "  log:     tail -f $LOG"
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "$ROOT/.venv/bin/python run_infer_modality_folders.py \
    --image_dir '$IMAGE_DIR' \
    --output_root '$OUTPUT_ROOT' \
    --stop-vllm \
    --restart-vllm \
    2>&1 | tee -a '$LOG'"

echo "Started in tmux session: $SESSION"
echo "  attach:  tmux attach -t $SESSION"
echo "  detach:  Ctrl-b then d"
echo "  log:     tail -f $LOG"
