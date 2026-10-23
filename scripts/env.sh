# Shared environment for pipeline scripts. Bash scripts in scripts/ source this themselves;
# source it manually (bash/zsh) only when calling upstream scripts by hand.

ROSETTA_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export ROSETTA_REPO
export ROSETTA_DATA="${ROSETTA_DATA:-$ROSETTA_REPO/data}"
export ROSETTA_OUTPUTS="${ROSETTA_OUTPUTS:-$ROSETTA_REPO/outputs}"
export UPSTREAM_DIR="$ROSETTA_REPO/third_party/rosetta-neuron-scaling"
export UPSTREAM_LANG="$UPSTREAM_DIR/language"

# Upstream expects a directory containing val.jsonl.zst
export DATASET="${DATASET:-$ROSETTA_DATA/pile}"
export SPLIT="${SPLIT:-val}"

# GPUs per run
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"

# uv lives in ~/.local/bin, which non-login shells do not have on PATH
case ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) export PATH="$HOME/.local/bin:$PATH" ;; esac

export TOKENIZERS_PARALLELISM=false

# If NCCL hangs at init or during all-reduce, uncomment:
# export NCCL_P2P_DISABLE=1
