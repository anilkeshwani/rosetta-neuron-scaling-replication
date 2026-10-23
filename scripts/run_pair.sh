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
# Upstream reuses any token cache and neighbor files it finds in OUT, so a forced rerun or a
# retry after a failure starts from an empty directory. The old one is kept, not deleted.
if [ -d "$OUT" ]; then
  old="$OUT.superseded-$(date +%Y%m%dT%H%M%S)"
  mv "$OUT" "$old"
  echo "[run_pair] moved previous attempt to $old"
fi
mkdir -p "$OUT"
[ -s "$DATASET/val.jsonl.zst" ] || { echo "[run_pair] missing $DATASET/val.jsonl.zst; run scripts/download_pile_val.sh" >&2; exit 1; }

# shellcheck disable=SC1091
source "$ROSETTA_REPO/.venv/bin/activate"
export OUTPUT_BASE="$OUT"

python - "$OUT/run_manifest.json" "${MODELS[@]}" <<'PY'
import datetime, glob, json, os, socket, subprocess, sys
path, models = sys.argv[1], sys.argv[2:]
def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"
knobs = ["NUM_SAMPLES", "K_NEIGHBORS", "K_NEIGHBORS_LIST", "BATCH_SIZE", "B_BLOCK", "USE_FSDP",
         "DEPTH_NEIGHBORS", "SEQ_LENGTH", "DTYPE", "SPAN_POOL", "SPLIT", "DATASET", "NPROC_PER_NODE",
         "STOP_AFTER_ROSETTA", "NCCL_P2P_DISABLE", "TOKEN_CACHE", "CUDA_VISIBLE_DEVICES",
         "MASTER_PORT_MATCH", "NPROC_MATCH", "REVISION1", "REVISION2", "TOKENIZER1", "TOKENIZER2",
         "PILE_SUBSETS", "TOKENIZE_BATCH", "SAVE_NEIGHBORS_TOPK", "ROSETTA_CORR_ACCUM_DTYPE",
         "ROSETTA_PERMUTE_B_SPANS"]
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
    # Untracked files there are only __pycache__ from importing upstream modules
    "upstream_dirty": bool(sh("git", "-C", os.environ["UPSTREAM_DIR"], "status", "--porcelain", "--untracked-files=no")),
    # Our patches currently applied to upstream (each is opt-in; unset knobs keep upstream behavior)
    "upstream_patches": sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(repo, "patches", "*.patch"))
        if subprocess.run(["git", "-C", os.environ["UPSTREAM_DIR"], "apply", "--reverse", "--check", p],
                          capture_output=True).returncode == 0
    ),
    "gpus": sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
}
if manifest["gpus"].startswith("unavailable"):  # NVML unavailable; ask torch
    import torch
    manifest["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
cache = os.environ.get("TOKEN_CACHE")
if cache and os.path.exists(cache.removesuffix(".pt") + ".json"):
    manifest["token_cache_provenance"] = json.load(open(cache.removesuffix(".pt") + ".json"))

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
