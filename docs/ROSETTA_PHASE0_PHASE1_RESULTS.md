# Rosetta Neuron scaling replication: phase 0 and phase 1 results

Status as of 2026-10-24. This replicates the Pythia-OPT curve of Dravid et al., "Neuron Populations Exhibit Divergent Selectivity with Scale" ([arXiv 2606.03990](https://arxiv.org/abs/2606.03990)), with the authors' own code ([avdravid/rosetta-neuron-scaling](https://github.com/avdravid/rosetta-neuron-scaling) at `15cd3c7`, unmodified). The paper reports |R| = 2.20·x^0.68 for the number of Rosetta Neurons (cross-model mutual-nearest-neighbor MLP neurons) against x, the mean MLP neuron count of the pair.

## Summary

- **The pipeline reproduces the authors' own example run** once it is fed the same token sample: 2,663 Pythia-160M/OPT-125M pairs against their 2,684, with 95.4% of pairs identical.
- **Two of three Pythia-OPT sizes match the paper:** 160M within 1 pair and 1.4B within 3.5%.
- **The 410M/350M pair is 36% below the paper** at k=1 (3,444 vs 5,418) and low at every k. We have not found a pipeline cause. We did not tune any setting to close the gap.
- With that point the three-point fit is 1.64·x^0.69 (paper exponent 0.68), but R² is 0.85, below our pass criterion of 0.98.
- The random-init null (9 pairs) is running.

## Setup

| Item | Value |
|---|---|
| Hardware | 4× RTX 3090 24 GB (PCIe, no NVLink), 2× Xeon Silver 4316, 128 GB RAM. The paper used 8× A100 80 GB per pair |
| Software | torch 2.14.1+cu130, transformers 4.57.6 |
| Data | Pile validation set from the authors' Google Drive folder, 214,670 docs |
| Matching | Upstream defaults: k=1 mutual nearest neighbors, mean span pooling, full layer grid, bf16, seq length 1024, seed 42 |
| Token budget | 10M tokens per pair (200k for the phase 0 smoke test) |
| Orchestration | `scripts/run_pair.sh` writes a `run_manifest.json` next to every run's outputs |

### The token sample depends on the GPU count

Upstream builds its token sample in shards: rank r draws `num_samples / world_size` tokens with seed `42 + r`. A 4-GPU run therefore reads different text from the authors' 8-GPU runs. `scripts/build_token_cache.py` rebuilds the cache on CPU for any simulated rank count, using upstream's own sampler. Its 4-rank output is tensor-identical to the cache of a real 4-GPU run. All phase 1 runs use the 8-rank sample (`token_cache_world_size: 8` in the experiment YAMLs).

## Phase 0: smoke test against the authors' example run

Pythia-160M vs OPT-125M at 200k tokens. The authors ship this run in `example_neurons/` (2,684 pairs).

| Run | Pairs | Identical to authors' pairs |
|---|---|---|
| Native 4-GPU token sample | 2,710 | 74.1% |
| Authors' 8-rank token sample | **2,663** | **95.4%** |

On the pairs both runs share, our correlations agree with the authors' to 0.003 on average. The remaining differences are numerical:

- **Repeat runs:** bit-identical on our machine.
- **transformers 5.20.2 instead of 4.57.6:** the identical pair list. The authors' log suggests they used 5.x.
- **GPU count with the token sample fixed:** 2 GPUs gave +3 pairs and 1 GPU gave +6 (pass-1 batch partition and reduction order).
- **BATCH_SIZE 4 instead of 8:** +9 pairs, 94% overlap.
- **B_BLOCK and the CUDA allocator setting:** no effect.

So the residual is most likely bf16 rounding on different GPU kernels.

## Phase 1: Pythia-OPT scaling

The paper's per-pair counts are not printed anywhere. We decoded them from the marker positions in the vector figure PDFs of Fig. 25 (`topk_robustness.pdf`) in the arXiv source. They land on integers to within 0.01 and refit to the published 2.20·x^0.678.

| Pair | x | Ours k=1 | Paper k=1 | Ours / paper | Ours k=2 / 5 / 10 | Paper k=2 / 5 / 10 | Wall time |
|---|---|---|---|---|---|---|---|
| Pythia-160M / OPT-125M | 36,864 | 2,608 | 2,609 | 1.00 | 6,995 / 24,274 / 60,577 | 6,976 / 24,304 / 60,525 | 8 min |
| Pythia-410M / OPT-350M | 98,304 | 3,444 | 5,418 | **0.64** | 9,470 / 34,965 / 92,476 | 12,307 / 40,171 / 100,971 | 38 min |
| Pythia-1.4B / OPT-1.3B | 196,608 | 8,688 | 9,001 | 0.97 | 24,007 / 85,523 / 217,499 | 23,671 / 90,019 / 235,511 | 2.5 h |

The 1.4B pair ran at BATCH_SIZE 4 because 8 ran out of memory on a 24 GB RTX 3090. A rerun of the 160M pair at BATCH_SIZE 4 and 10M tokens is queued to measure that effect.

### Fits

From `results/phase1_pythia_opt/power_law_fit.json`:

| k | Ours | R² | Paper |
|---|---|---|---|
| 1 | 1.64·x^0.69 | 0.85 | 2.20·x^0.68 |
| 2 | 3.64·x^0.71 | 0.86 | 2.51·x^0.75 |
| 5 | 10.49·x^0.73 | 0.89 | 6.09·x^0.78 |
| 10 | 22.77·x^0.74 | 0.92 | 35.39·x^0.71 |

Three points cannot pin an exponent; the standard error on α is about 0.25 to 0.29. The slope from 160M straight to 1.4B is 0.72 for us and 0.74 for the paper.

## The 410M/350M gap

What we checked, on CPU:

- **Hooks:** the OPT hook captures relu(fc1) on all 24 OPT-350M layers. The forward pass skips only modules after the last hook (`project_out`, LM head), so OPT-350M's post-LayerNorm layout and 512-d embedding projection do not affect the captured activations.
- **Per-layer counts:** no empty layers, and the shape across depth matches the 160M run.
- **Dead neurons:** OPT-350M has 26 dead units, against 2,907 in OPT-125M.
- **bf16 ties:** the top-1 tie rate is similar in both runs (8.8% vs 8.1%).
- **bf16 accumulation (open lead):** upstream sums the correlation matrix in bf16 over about 1,200 batches (`find_pairs/match_corr.py:179-182`), so later small increments get rounded away. A CPU emulation on a subset of layers showed this moves the mutual-NN count by about 4%. That is too small to explain 36% unless it scales badly with model size. The next test is a diagnostic fp32-accumulator patch on the 160M and 410M pairs. It would not be used for reported fits.

## Oddities in the paper's plotted data

1. For the three largest Pythia-OPT pairs, the k=2/5/10 counts share their last three digits with k=1 (…686, …276, …282). Raw counts would not do that.
2. The 2.8B point is plotted at x = 360,448. Both Pythia-2.8B and OPT-2.7B have 32 × 10,240 = 327,680 MLP neurons, and every other plotted x is the exact pair mean.
3. The OPT-Qwen2.5 series in Fig. 3a includes round values (24,000, 27,000 and 45,000).

## Still to do

- Random-init null: 9 pairs (3 sizes × 3 seeds), running.
- The BATCH_SIZE 4 check on the 160M pair.
- The fp32-accumulator diagnostic for the 410M gap.
- Pythia-2.8B, 6.9B and 12B pairs. Token caches are prebuilt; 6.9B and 12B need FSDP across the 4 GPUs.
- Phases 2 to 4 per `docs/REPLICATION_PLAN.md`.

## Reproducing

```bash
bash scripts/setup_env.sh
bash scripts/download_pile_val.sh
uv run python scripts/download_models.py --experiment configs/experiments/phase1_pythia_opt.yaml
uv run python scripts/run_experiment.py configs/experiments/phase1_pythia_opt.yaml --caches-only
uv run python scripts/run_experiment.py configs/experiments/phase1_pythia_opt.yaml --only pythia-160m__opt-125m
uv run python scripts/collect_rosetta_counts.py outputs/phase1_pythia_opt
uv run python scripts/fit_power_law.py results/phase1_pythia_opt/rosetta_counts.csv --reference language.pythia_opt
uv run python scripts/compare_best_buddies.py \
  third_party/rosetta-neuron-scaling/example_neurons/facebook_opt-125m/correlations/best_buddies.json \
  outputs/phase0_smoke/<run>/correlations/best_buddies.json
```
