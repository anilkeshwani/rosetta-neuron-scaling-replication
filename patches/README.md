# Patches to upstream

Changes we need in `third_party/rosetta-neuron-scaling` live here as `git format-patch` files, applied in name order by `scripts/setup_env.sh` with `git apply`; `scripts/check_upstream_patches.py` checks that upstream equals the pinned commit plus the series. Each patch gets a one-line entry below and a fuller note in `docs/UPSTREAM_CODE_NOTES.md`.

| Patch | Purpose |
|---|---|
| `0001-opt-in-fp32-correlation-accumulator.patch` | `ROSETTA_CORR_ACCUM_DTYPE=float32` accumulates the pass-2 outer products in fp32 instead of bf16. Diagnostic only; unset keeps upstream behavior exactly |
| `0002-gemma3-multimodal-decoder-path.patch` | Loads the text decoder of multimodal Gemma 3 checkpoints (4B and up) as Gemma3ForCausalLM and wraps Gemma3DecoderLayer under FSDP. Other models load as before |
| `0003-opt-in-input-permutation-null.patch` | `ROSETTA_PERMUTE_B_SPANS=<seed>` shuffles model B's pooled span rows within each batch before the outer product: the input-permutation null of paper App. C.3 (Fig. 26) |
