# Compute notes

## What fits

The matching step holds both models at once. bf16 weights take 2 bytes per parameter:

| Pair | Weights (bf16) | Setting |
|---|---|---|
| ≤ 2.8B + 2.7B | ≤ 11 GB | One copy per GPU, data-parallel |
| 6.9B + 6.7B | ~27 GB | `USE_FSDP=true` across the 4 GPUs |
| 12B + 13B, Gemma-3-12B + Qwen2.5-14B | ~50-54 GB | `USE_FSDP=true`, `BATCH_SIZE=1`, `B_BLOCK=1` |
| 30B + 32B, 27B + 32B, anything with 70B | ≥ 118 GB | Does not fit; cloud (`requires: cloud` in the experiment YAMLs) |

Steps 4-5 of upstream's pipeline (activation collection for the viewer) load one un-sharded model per rank, so they cannot run for models above about 10B on our GPUs. Scaling runs set `STOP_AFTER_ROSETTA=true` and skip those steps.

## How long runs should take


Back-of-envelope for 10M tokens, counting the correlation matmul (2·T·n_A·n_B) plus one forward pass of both models per B-layer block, at 50% utilization:

| Pair | Estimate |
|---|---|
| Pythia-160M / OPT-125M | ~15 min |
| Pythia-1.4B / OPT-1.3B | ~5 h |
| Pythia-6.9B / OPT-6.7B | ~3 days |
| Pythia-12B / OPT-13B | ~1 week |

Treat these as rough until the phase 1 small runs give measured times (`elapsed_s` in each `run_manifest.json`).

## Ways to cut cost without breaking the comparison

- **Keep `NUM_SAMPLES` fixed within a curve.** The count of matches depends on the token budget (paper Fig. 33), so mixing budgets across sizes changes the exponent. A cheaper curve at, say, 2M tokens is fine if every point uses 2M. Check it against the 10M curve at the small sizes first.
- **`DEPTH_NEIGHBORS`** gives about a 10× speedup but drops cross-depth matches. It changes the count, so use it only for exploration, never for a reported fit.
- **Several small pairs at once** with `NPROC_PER_NODE=1` and `CUDA_VISIBLE_DEVICES` set per run. Give each run its own `MASTER_PORT_MATCH`/`MASTER_PORT_ACT`/`MASTER_PORT_CROSS`.
- **Long runs:** launch under `tmux` or `nohup`, since 12B pairs take days.
