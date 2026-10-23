#!/usr/bin/env bash
# Run experiment pairs one after another from an editable queue file.
#
# Usage: gpu_queue_runner.sh QUEUE_FILE [WAIT_FILE WAIT_PATTERN]
#
# Each queue line is "EXPERIMENT_YAML ONLY_FILTER [VAR=VALUE ...]"; blank lines and lines starting
# with # are skipped, and a line "STOP" ends the runner. Finished lines are appended to
# QUEUE_FILE.done with their exit code, so the queue can be edited (reordered, extended) while
# the runner works. If WAIT_FILE and WAIT_PATTERN are given, start only once the pattern
# appears in that file (e.g. after another schedule finishes with the GPUs).
set -uo pipefail
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
QUEUE="$1"; DONE="$QUEUE.done"
touch "$DONE"
if [ "$#" -ge 3 ]; then
  until grep -q -- "$3" "$2" 2>/dev/null; do sleep 60; done
fi
echo "[queue] start $(date -Is)"
while true; do
  line=$(grep -v -e '^\s*#' -e '^\s*$' "$QUEUE" | while IFS= read -r l; do
           grep -qxF -- "$l" <(cut -f1 "$DONE") || { echo "$l"; break; }
         done)
  if [ -z "$line" ]; then sleep 60; continue; fi
  [ "$line" = "STOP" ] && { echo "[queue] stop $(date -Is)"; break; }
  read -r yaml filter extra <<<"$line"
  start=$(date -Is)
  echo "[queue] run: $line ($start)"
  env NPROC_PER_NODE=4 $extra uv run python scripts/run_experiment.py "$yaml" --only "$filter" \
    > "logs/queue_$(basename "$yaml" .yaml)__$filter.log" 2>&1
  rc=$?
  printf '%s\texit=%s\tstart=%s\tend=%s\n' "$line" "$rc" "$start" "$(date -Is)" >> "$DONE"
  echo "[queue] exit=$rc: $line ($(date -Is))"
done
