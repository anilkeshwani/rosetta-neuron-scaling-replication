#!/usr/bin/env bash
# Run the upstream matching pipeline on 2+ models and record a manifest next to the outputs.
#
# Usage: run_pair.sh MODEL1 MODEL2 [MODEL3 ...] -- RUN_NAME [EXPERIMENT]
#    or: run_pair.sh MODEL1 MODEL2 RUN_NAME            (two models, experiment "adhoc")
#
# Outputs go to $ROSETTA_OUTPUTS/<EXPERIMENT>/<RUN_NAME>/. Knobs are upstream's env vars
# (NUM_SAMPLES, K_NEIGHBORS, K_NEIGHBORS_LIST, BATCH_SIZE, B_BLOCK, USE_FSDP, ...).
# A finished run leaves a DONE marker and is skipped unless FORCE=1.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

if [[ " $* " == *" -- "* ]]; then
  MODELS=(); while [ "$1" != "--" ]; do MODELS+=("$1"); shift; done; shift
  RUN_NAME="$1"; EXPERIMENT="${2:-adhoc}"
else
  [ "$#" -eq 3 ] || { sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 1; }
  MODELS=("$1" "$2"); RUN_NAME="$3"; EXPERIMENT="adhoc"
fi

OUT="$ROSETTA_OUTPUTS/$EXPERIMENT/$RUN_NAME"
if [ -f "$OUT/DONE" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "[run_pair] already done: $OUT (FORCE=1 to rerun)"; exit 0
fi
mkdir -p "$OUT"
[ -s "$DATASET/val.jsonl.zst" ] || { echo "[run_pair] missing $DATASET/val.jsonl.zst; run scripts/download_pile_val.sh" >&2; exit 1; }

# shellcheck disable=SC1091
source "$ROSETTA_REPO/.venv/bin/activate"
export OUTPUT_BASE="$OUT"

python - "$OUT/run_manifest.json" "${MODELS[@]}" <<'PY'
import datetime, json, os, socket, subprocess, sys
path, models = sys.argv[1], sys.argv[2:]
def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"
knobs = ["NUM_SAMPLES", "K_NEIGHBORS", "K_NEIGHBORS_LIST", "BATCH_SIZE", "B_BLOCK", "USE_FSDP",
         "DEPTH_NEIGHBORS", "SEQ_LENGTH", "DTYPE", "SPAN_POOL", "SPLIT", "DATASET", "NPROC_PER_NODE",
         "STOP_AFTER_ROSETTA", "NCCL_P2P_DISABLE"]
repo = os.environ["ROSETTA_REPO"]
manifest = {
    "models": models,
    "experiment_run_dir": os.path.dirname(path),
    "started_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "host": socket.gethostname(),
    "knobs": {k: os.environ.get(k) for k in knobs},
    "repo_sha": sh("git", "-C", repo, "rev-parse", "HEAD"),
    "repo_dirty": bool(sh("git", "-C", repo, "status", "--porcelain")),
    "upstream_sha": sh("git", "-C", os.environ["UPSTREAM_DIR"], "rev-parse", "HEAD"),
    "upstream_dirty": bool(sh("git", "-C", os.environ["UPSTREAM_DIR"], "status", "--porcelain")),
    "gpus": sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
}
with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
PY

start=$(date +%s)
cd "$UPSTREAM_LANG"
bash match.sh "${MODELS[@]}" 2>&1 | tee "$OUT/run.log"
elapsed=$(( $(date +%s) - start ))

python - "$OUT/run_manifest.json" "$elapsed" <<'PY'
import datetime, json, sys
path, elapsed = sys.argv[1], int(sys.argv[2])
m = json.load(open(path))
m["finished_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
m["elapsed_s"] = elapsed
json.dump(m, open(path, "w"), indent=2)
PY
touch "$OUT/DONE"
echo "[run_pair] done in ${elapsed}s: $OUT"
