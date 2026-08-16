#!/usr/bin/env bash
# Launch DR-Grade training in a detached tmux session (works without a scheduler).
#
# Usage:
#   ./run_train.sh --data-root split_dataset_cropped --output-dir outputs \
#                  --mixed-precision --cache
#   tmux attach -t drgrade        # watch the run
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
# shellcheck disable=SC1091
source "$ROOT_DIR/setenv.sh"

OUTPUT_DIR="$(echo "$@" | grep -oE '\-\-output-dir[= ][^ ]+' | awk -F'[= ]' '{print $NF}' || echo outputs)"
mkdir -p "$OUTPUT_DIR"

SESSION="drgrade"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "⚠️  tmux session '$SESSION' already exists."
    echo "   Attach:  tmux attach -t $SESSION"
    echo "   Or kill: tmux kill-session -t $SESSION"
    exit 1
fi

echo "🚀 Launching training in tmux session '$SESSION' ..."
echo "   Log: $ROOT_DIR/train.log"
tmux new-session -d -s "$SESSION" -n train \
    "python train.py $* 2>&1 | tee $ROOT_DIR/train.log"
tmux set-option -t "$SESSION" remain-on-exit on
echo "   Attach:  tmux attach -t $SESSION"
echo "   GPU:     watch -n1 nvidia-smi"
echo "   Logs:    tail -f $ROOT_DIR/train.log"
