# Patches to upstream

Changes we need in `third_party/rosetta-neuron-scaling` live here as `git format-patch` files, applied by `scripts/setup_env.sh` with `git apply`. Each patch gets a one-line entry below and a fuller note in `docs/UPSTREAM_CODE_NOTES.md`.

| Patch | Purpose |
|---|---|
| `0001-opt-in-fp32-correlation-accumulator.patch` | `ROSETTA_CORR_ACCUM_DTYPE=float32` accumulates the pass-2 outer products in fp32 instead of bf16. Diagnostic only; unset keeps upstream behavior exactly |
