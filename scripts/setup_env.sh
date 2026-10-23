#!/usr/bin/env bash
# One-time setup: submodule, upstream patches, Python env, system checks, unit tests.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
cd "$ROSETTA_REPO"

command -v uv >/dev/null || { echo "uv not found; install from https://docs.astral.sh/uv/" >&2; exit 1; }
# Upstream's Pile reader shells out to zstdcat
command -v zstdcat >/dev/null || { echo "zstdcat not found; install the zstd package" >&2; exit 1; }

git submodule update --init --recursive

# Upstream = pinned commit + patches/*.patch in name order. Patches can touch the same file,
# so on any mismatch reset upstream's tracked files and apply the whole series again.
shopt -s nullglob
if command -v uv >/dev/null && uv run python scripts/check_upstream_patches.py >/dev/null 2>&1; then
  echo "[setup] upstream already matches the patch series"
else
  git -C "$UPSTREAM_DIR" checkout -- .
  for p in patches/*.patch; do
    git -C "$UPSTREAM_DIR" apply "$ROSETTA_REPO/$p"
    echo "[setup] applied: $p"
  done
fi

uv sync

uv run python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), transformers {transformers.__version__}")
try:
    n = torch.cuda.device_count()
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.0f} GiB")
    if n == 0:
        raise RuntimeError("no CUDA devices visible")
except Exception as e:
    print(f"[setup] WARNING: CUDA unavailable ({e}). If nvidia-smi reports a driver/library "
          "version mismatch, the machine needs a reboot after the driver upgrade.")
PY

uv run pytest -q
mkdir -p "$ROSETTA_DATA" "$ROSETTA_OUTPUTS"
echo "[setup] done"
