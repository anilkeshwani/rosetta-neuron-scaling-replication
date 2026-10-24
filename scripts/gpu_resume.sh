#!/usr/bin/env bash
# Let the GPU queue start items again after gpu_pause.sh.
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
rm -f "${1:-logs/gpu_queue.txt}.pause" && echo "[resume] queue will start its next item within a minute"
