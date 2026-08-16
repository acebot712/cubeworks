#!/bin/bash
# The variance penalty was a null at lambda=1, where it is roughly balanced
# against the squared-error term it is fighting: the bootstrap target has real
# within-shell spread of its own, so shrinking the network's spread means
# fitting that target worse. This sweeps stronger weights to find out whether
# lambda=1 was simply too weak or whether the whole idea does not work.
#
# Waits for the main comparison to release the GPU, and writes to its own result
# file -- p3_losses.py rewrites its output wholesale, so sharing one would
# replace the main comparison with the sweep.
cd "$(dirname "$0")" || exit 1

while pgrep -f "p3_losses.py --steps 12000 --seeds 3$" >/dev/null; do sleep 60; done
echo "=== main comparison finished $(date '+%H:%M'), starting lambda sweep" >&2

exec ../.venv-mlx/bin/python p3_losses.py --steps 12000 --seeds 3 \
     --configs wings-k6 --arms "dprime:3,dprime:10,dprime:30" \
     --out p3-lambda.json
