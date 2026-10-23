# Notes on the upstream code

Upstream: [avdravid/rosetta-neuron-scaling](https://github.com/avdravid/rosetta-neuron-scaling), vendored at `third_party/rosetta-neuron-scaling`, pinned at `15cd3c7` ("Update paper title in README for NeurIPS 2026").

## What it provides

- `language/match.sh`: the entry point. With 2 models it runs the pairwise pipeline; with 3+ it runs the anchor pipeline (the first model is the anchor, and a Rosetta neuron must be a best buddy in every other model). This is how the paper's "collapsed" 3-family curves are made.
- Step 1 (`match_lm.py`): byte-aligned canonical spans, per-neuron mean/std (pass 1), blockwise correlation across all A layers × B-layer blocks (pass 2). Only the top-K neighbors per neuron are saved (`SAVE_NEIGHBORS_TOPK=100`), never the full matrix.
- Step 2 (`find_best_buddies.py`): mutual top-K. `K_NEIGHBORS_LIST="1 2 5 10"` reuses the step-1 neighbors to produce the Fig. 25 sweep for free.
- `REVISION1`/`REVISION2`: load a specific Hub revision, e.g. Pythia training checkpoints (`step143000`). Only wired into step 1, which is all the scaling counts need.
- Vision pipeline under `vision/` (not used yet).

## Gaps relative to the paper

| Paper element | Upstream status | Our plan |
|---|---|---|
| Random-init null (Fig. 4) | No code | `scripts/make_random_init_checkpoints.py` builds local checkpoints; `phase1_null_random_init.yaml` runs them |
| Input-permutation null (Fig. 26) | No code | Needs a patch to `find_pairs/match_corr.py` that permutes one model's span activations before the outer product. Not written yet |
| Power-law fits | No code | `scripts/collect_rosetta_counts.py`, `scripts/fit_power_law.py` |
| Vocabulary-space kurtosis (Fig. 6a, 28) | No code | To write: cosine of each neuron's W_out row with W_U, excess kurtosis, split Rosetta / non-Rosetta |
| Document-type firing (Fig. 7, 29, 30) | Partly: step 4 collects top-activating contexts | To write: map Pile `pile_set_name` to the 5 categories of Table 4 and normalize by token share |
| JavaScript filtering (Table 1) | No code | Later; needs CodeSearchNet and LoRA continued pretraining of GPT2-1.5B |

## Our patches

All patches are opt-in through an environment variable; with it unset the code path is upstream's. `run_pair.sh` records the applied patches and the knobs in each manifest.

- `0001-opt-in-fp32-correlation-accumulator.patch`: upstream sums the pass-2 outer products in the compute dtype (bf16), so over ~1,200 batches small late increments are rounded away. `ROSETTA_CORR_ACCUM_DTYPE=float32` keeps the inputs in bf16 but accumulates in fp32. Used to test whether bf16 accumulation explains part of the Pythia-410M/OPT-350M gap; not for reported fits.
- `0002-gemma3-multimodal-decoder-path.patch`: Gemma 3 at 4B and up ships as Gemma3ForConditionalGeneration, with the decoder under `model.language_model.layers`, which upstream's hook lookup does not find. The patch loads only the text decoder as Gemma3ForCausalLM from `config.text_config`, giving the 270M/1B layout (checkpoint keys load with no missing or unexpected decoder keys; activations are bit-identical to hooking the full model). It also adds Gemma3DecoderLayer to the FSDP wrap set, since without it upstream wraps the whole model as one FSDP unit. Upstream's activation-collection steps (4-5) still fail for these checkpoints; the counts do not use them.
- `0003-opt-in-input-permutation-null.patch`: the paper's input-permutation null (App. C.3) "randomly permute[s] the flattened activation positions for one model". Our streaming pipeline cannot permute across the whole dataset, so `ROSETTA_PERMUTE_B_SPANS=<seed>` permutes model B's pooled span rows within each batch (8 sequences, several thousand spans). That breaks the input alignment and keeps each neuron's marginal distribution. The permutation depends only on the seed and the batch index, so all B-block passes and ranks see the same one.

## Pitfalls

- **The token sample depends on the GPU count.** `match_pipeline.py` builds the token cache in shards: rank r draws `num_samples // world_size` tokens with seed `42 + r`, and rank 0 concatenates them. A 4-GPU run therefore sees different tokens from the authors' 8-GPU runs (at 200k tokens: 196 sequences instead of 200, and only 74% of the authors' smoke-test pairs reproduced). `scripts/build_token_cache.py` rebuilds the cache on CPU for any simulated rank count with upstream's own sampler, and experiments that set `token_cache_world_size: 8` pass it as `TOKEN_CACHE`. With a fixed cache, the GPU count still shifts the pass-1 batch partition and reduction order, which moved the smoke-test count by 3 (2 GPUs) and 6 (1 GPU) out of 2,663. Keep `NPROC_PER_NODE` fixed within a curve.
- **Reruns reuse whatever is in the output directory.** Upstream loads an existing token cache without checking its provenance, and `find_best_buddies.py` picks up every `nn_*_top*.pt` file it finds. `scripts/run_pair.sh` therefore moves a previous attempt aside to `<run>.superseded-<timestamp>` before a forced rerun or a retry.
- **bf16 throughout pass 2.** Span pooling, the means and standard deviations and the correlation accumulator are all bf16, and top-1 ties at bf16 resolution are common. Our runs are bit-identical on repeat on our machine, but a different GPU or library version flips some matches (see the phase 0 result in the handoff).
- **Family dispatch by substring.** `get_mlp_post_activation_modules` checks the lowercased model id/path for `"pythia"`, then `"gpt2"`, then `"opt"`, then `"llama"`/`"gemma"`/`"qwen"`. A local path containing the wrong substring gets the wrong hooks. `tests/test_registry.py` checks every registry id; local random-init directories keep the family name in their basename.
- **Gated MLPs.** For Llama, Gemma and Qwen the hook captures the input to `down_proj`, i.e. act(gate)·up. That product is signed and can be large in both directions, unlike post-ReLU OPT activations. Pearson correlation is scale-invariant, so matching is fine, but per-neuron statistics differ by family.
- **Gemma 3 at 4B and up** loads as `Gemma3ForConditionalGeneration`, with the decoder under `model.language_model.layers`. Upstream looks for `model.model.layers` or `model.layers` and will raise. Needs a patch (or loading `Gemma3ForCausalLM` from the text config). The 270M and 1B checkpoints are text-only and should work as is.
- **FSDP wrap classes.** `_resolve_transformer_layer_classes` names `GemmaDecoderLayer` only, and its Gemma branch returns before the generic "...DecoderLayer" fallback, so Gemma 2 and (unpatched) Gemma 3 are wrapped as one FSDP unit with no per-layer sharding. Patch 0002 fixes this for Gemma 3; Gemma 2 is not used here.
- **Pile data.** The sampler reads `val.jsonl.zst` by shelling out to `zstdcat`, and needs the `meta.pile_set_name` field for its per-subset sampling. HF mirrors without that field (or with subsets removed, like "pile-uncopyrighted") are not drop-in replacements and would also shift the document-category shares of Table 4.
- **NCCL hangs at init.** Upstream commit `d4b88b1` fixed an NCCL P2P deadlock at init. If runs hang, set `NCCL_P2P_DISABLE=1` (commented out in `scripts/env.sh`).
