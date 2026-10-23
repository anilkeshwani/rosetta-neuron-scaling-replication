# Rosetta Neuron Scaling: Replication and Extension

Replicating and extending Dravid, Bahri, Efros and Gandelsman, "Neuron Populations Exhibit Divergent Selectivity with Scale" ([arXiv 2606.03990](https://arxiv.org/abs/2606.03990), NeurIPS 2026). The paper finds that **Rosetta Neurons** (MLP neurons that are mutual nearest neighbors by activation correlation across independently trained models) grow as a sublinear power law in model size, |R| = c·x^α with α ≈ 0.5 to 0.7, and become more selective and domain-specialized as models grow.

The authors' code ([avdravid/rosetta-neuron-scaling](https://github.com/avdravid/rosetta-neuron-scaling)) is vendored as a git submodule at `third_party/rosetta-neuron-scaling`, pinned to the commit we replicate against. This repo adds the pieces the release lacks: a model registry with verified neuron counts, run orchestration with manifests, the null baselines, power-law fitting against the published fits and the Llama/Gemma extension.

## Quickstart

```bash
git clone --recurse-submodules <this repo> && cd rosetta-neuron-scaling-replication
bash scripts/setup_env.sh                       # uv sync, submodule, system checks, GPU check
source scripts/env.sh                           # sets ROSETTA_DATA etc. (bash scripts source it themselves)
bash scripts/download_pile_val.sh               # Pile validation set -> data/pile/val.jsonl.zst (~450 MB)
uv run python scripts/verify_model_registry.py  # check configs/models.yaml against HF configs
uv run python scripts/download_models.py --experiment configs/experiments/phase1_pythia_opt.yaml

# Smoke test: one tiny pair on a small token budget
NUM_SAMPLES=200000 bash scripts/run_pair.sh EleutherAI/pythia-160m facebook/opt-125m smoke

# A full phase, then fit
uv run python scripts/run_experiment.py configs/experiments/phase1_pythia_opt.yaml
uv run python scripts/collect_rosetta_counts.py outputs/phase1_pythia_opt
uv run python scripts/fit_power_law.py results/phase1_pythia_opt/rosetta_counts.csv --reference language.pythia_opt
```

## Layout

| Path | Contents |
|---|---|
| `configs/models.yaml` | Model registry: HF id, family, layers, MLP width, neuron count, MLP type, distillation lineage |
| `configs/paper_reference_fits.yaml` | Published power-law fits (Fig. 3, Fig. 25) to compare against |
| `configs/experiments/` | One YAML per phase: the size-matched pairs to run and the reference fit to compare with |
| `scripts/` | Setup, downloads, the run wrapper, nulls, count collection and fitting |
| `src/rosetta_repl/` | Small library: registry loading, power-law fitting |
| `docs/` | Paper notes, the replication plan, compute notes, notes on the upstream code |
| `third_party/rosetta-neuron-scaling/` | Upstream code (submodule) |
| `data/`, `outputs/` | Gitignored: Pile data, random-init checkpoints, run outputs |
| `results/` | Tracked: per-experiment `rosetta_counts.csv`, fit JSON and plots |

## Results so far

Phase 0 reproduces the authors' example run (95.4% of pairs identical). Of the first three Pythia-OPT sizes, 160M and 1.4B match the paper within 3.5%; 410M/350M is 36% low and under investigation. See [`docs/ROSETTA_PHASE0_PHASE1_RESULTS.md`](docs/ROSETTA_PHASE0_PHASE1_RESULTS.md).

## Plan in one paragraph

Phase 1 reproduces the Pythia–OPT curve (published fit 2.20·x^0.68) and the random-init null locally. Phase 2 adds OPT–Qwen2.5 (12.43·x^0.57), which exercises the gated-MLP path. Phase 3 is the extension: Gemma 3 vs Qwen2.5 across matched sizes, Llama 3.x as sparse points, and a test of whether distilled checkpoints inflate Rosetta counts. Downstream analyses (vocabulary-space kurtosis, document-type firing, the JavaScript filtering study) follow once the matches exist. Details and open questions in [`docs/REPLICATION_PLAN.md`](docs/REPLICATION_PLAN.md).

## Things to know before running

- Our runs use a workstation with 4× RTX 3090 (24 GB each, PCIe, no NVLink) and 128 GB RAM. Pairs above about 14B do not fit; see [`docs/COMPUTE_NOTES.md`](docs/COMPUTE_NOTES.md).
- The upstream pipeline picks hook points by substring of the model id (`"pythia"`, `"opt"`, `"qwen"`, ...). Local checkpoint directories must keep the family name in their path. See [`docs/UPSTREAM_CODE_NOTES.md`](docs/UPSTREAM_CODE_NOTES.md).
- Llama and Gemma repos are gated on the Hub. Accept the licenses with the account whose Hugging Face token you use before downloading.
