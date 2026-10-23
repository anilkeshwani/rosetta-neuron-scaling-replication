#!/usr/bin/env bash
# Kill match_lm workers that hit CUDA OOM and then hang (NCCL keeps the other ranks waiting for
# up to the 4-hour process-group timeout). A run counts as hung when its run.log mentions
# OutOfMemoryError and has not grown for 10 minutes. The killed run then fails normally, so
# the queue moves on and the run can be retried with a smaller BATCH_SIZE.
cd "$(dirname -- "${BASH_SOURCE[0]}")/.."
while true; do
  now=$(date +%s)
  for log in outputs/*/*/run.log; do
    dir=$(dirname "$log")
    [ -f "$dir/DONE" ] && continue
    age=$(( now - $(stat -c %Y "$log") ))
    if [ "$age" -gt 600 ] && grep -aq "OutOfMemoryError" "$log"; then
      if pgrep -f "match_lm.py.*--save_dir $PWD/$dir/correlations" >/dev/null; then
        echo "[watchdog] $(date -Is) killing hung OOM run: $dir"
        pkill -f "match_lm.py.*--save_dir $PWD/$dir/correlations"
      fi
    fi
  done
  sleep 120
done
