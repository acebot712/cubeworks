#!/bin/bash
# Wait for the deadlock sweep to release the GPU, then run the three-arm loss
# comparison. Both halves are idempotent, so if this is interrupted -- lid
# closed, crash, reboot -- re-running it picks up from the last finished run
# rather than starting over.
cd "$(dirname "$0")" || exit 1

while pgrep -f "depth_vs_size.py --steps" >/dev/null; do sleep 60; done
echo "=== sweep finished $(date '+%H:%M:%S'), starting P3" >&2

exec ../.venv-mlx/bin/python p3_losses.py --steps 12000 --seeds 3
