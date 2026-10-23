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

## Pitfalls

- **Family dispatch by substring.** `get_mlp_post_activation_modules` checks the lowercased model id/path for `"pythia"`, then `"gpt2"`, then `"opt"`, then `"llama"`/`"gemma"`/`"qwen"`. A local path containing the wrong substring gets the wrong hooks. `tests/test_registry.py` checks every registry id; local random-init directories keep the family name in their basename.
- **Gated MLPs.** For Llama, Gemma and Qwen the hook captures the input to `down_proj`, i.e. act(gate)·up. That product is signed and can be large in both directions, unlike post-ReLU OPT activations. Pearson correlation is scale-invariant, so matching is fine, but per-neuron statistics differ by family.
- **Gemma 3 at 4B and up** loads as `Gemma3ForConditionalGeneration`, with the decoder under `model.language_model.layers`. Upstream looks for `model.model.layers` or `model.layers` and will raise. Needs a patch (or loading `Gemma3ForCausalLM` from the text config). The 270M and 1B checkpoints are text-only and should work as is.
- **FSDP wrap classes.** `_resolve_transformer_layer_classes` names `GemmaDecoderLayer` only. Gemma 2/3 fall through to the generic "...DecoderLayer" class-name fallback, which should work but is worth confirming on the first Gemma FSDP run.
- **Pile data.** The sampler reads `val.jsonl.zst` by shelling out to `zstdcat`, and needs the `meta.pile_set_name` field for its per-subset sampling. HF mirrors without that field (or with subsets removed, like "pile-uncopyrighted") are not drop-in replacements and would also shift the document-category shares of Table 4.
- **NCCL hangs at init.** Upstream commit `d4b88b1` fixed an NCCL P2P deadlock at init. If runs hang, set `NCCL_P2P_DISABLE=1` (commented out in `scripts/env.sh`).
