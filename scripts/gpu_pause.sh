#!/usr/bin/env bash
# Free the GPUs for another job: stop the queue from starting new items and kill the running
# match_lm workers. The interrupted item goes back on the queue (its line is dropped from
# QUEUE_FILE.done), so `scripts/gpu_resume.sh` reruns it from scratch.
#
# Usage: gpu_pause.sh [QUEUE_FILE]   (default logs/gpu_queue.txt)
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
QUEUE="${1:-logs/gpu_queue.txt}"
touch "$QUEUE.pause"
running=$(grep '^\[queue\] run: ' logs/gpu_queue_runner.log 2>/dev/null | tail -1 | sed 's/^\[queue\] run: //; s/ ([^)]*)$//')
pkill -f "match_lm.py" && echo "[pause] killed match_lm workers"
pkill -f "run_phase1_small_and_null.sh" && echo "[pause] stopped the phase 1 null schedule"
# Wait for the runner to record the interrupted item, then requeue it
for _ in $(seq 60); do
  if [ -n "$running" ] && grep -qF -- "$running"$'\t' "$QUEUE.done" 2>/dev/null; then
    grep -vF -- "$running"$'\t' "$QUEUE.done" > "$QUEUE.done.tmp" && mv "$QUEUE.done.tmp" "$QUEUE.done"
    echo "[pause] requeued: $running"; break
  fi
  [ -z "$running" ] && break
  sleep 2
done
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
