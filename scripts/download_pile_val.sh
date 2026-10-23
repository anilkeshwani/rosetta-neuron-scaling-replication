#!/usr/bin/env bash
# Fetch the Pile validation set the authors use (their Google Drive folder, ~450 MB)
# into $DATASET/val.jsonl.zst, the layout upstream's sampler expects.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

FOLDER_URL="https://drive.google.com/drive/folders/1klmBdOi5hh5C508vbMFmpizqHgUS3zmR"
mkdir -p "$DATASET"

if [ -s "$DATASET/val.jsonl.zst" ]; then
  echo "[pile] already present: $DATASET/val.jsonl.zst"
else
  tmp="$(mktemp -d "$ROSETTA_DATA/.pile-dl.XXXX")"
  uv run --project "$ROSETTA_REPO" gdown --folder "$FOLDER_URL" -O "$tmp"
  found="$(find "$tmp" -name 'val.jsonl.zst' -print -quit)"
  [ -n "$found" ] || { echo "[pile] val.jsonl.zst not found in download: $(find "$tmp" -type f)" >&2; exit 1; }
  mv "$found" "$DATASET/val.jsonl.zst"
  rm -rf "$tmp"
fi

ls -lh "$DATASET/val.jsonl.zst"
sha256sum "$DATASET/val.jsonl.zst" | tee "$DATASET/val.jsonl.zst.sha256"
echo "[pile] first record subset: $({ zstdcat "$DATASET/val.jsonl.zst" || true; } | head -1 | uv run --project "$ROSETTA_REPO" python -c 'import json,sys; print(json.loads(sys.stdin.readline())["meta"])')"
