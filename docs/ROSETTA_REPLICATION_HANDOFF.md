# Handoff: Rosetta Neuron scaling replication

Written 2026-10-23 by the Claude Code session that scaffolded this repo. You are taking over on the GPU machine. Read this file, then `README.md`, `CLAUDE.md` and the four files in `docs/`.

## Goal

Replicate, then extend, Dravid et al. 2026, "Neuron Populations Exhibit Divergent Selectivity with Scale" (arXiv 2606.03990). The headline claim is that the number of Rosetta Neurons (cross-model mutual-nearest-neighbor MLP neurons) grows as |R| = c·x^α with α ≈ 0.5 to 0.7. The first target is the Pythia–OPT curve, **2.20·x^0.68**. Next comes OPT–Qwen2.5, then the extension to Gemma 3 and Llama 3.x, including a check of whether distilled checkpoints inflate the counts. `docs/ROSETTA_PAPER_NOTES.md` has the paper's numbers; `docs/REPLICATION_PLAN.md` has the phases and pass criteria.

## State of the repo

- Location: `~/projects/rosetta-neuron-scaling-replication`, local git only (no remote yet; ask the user before creating one).
- The authors' code is the submodule `third_party/rosetta-neuron-scaling`, pinned at `15cd3c7`. Do not edit it in place; patches go in `patches/` (see `CLAUDE.md`).
- Written but **not yet exercised against a real run**: `scripts/run_pair.sh`, `run_experiment.py`, `collect_rosetta_counts.py`, `fit_power_law.py`, `make_random_init_checkpoints.py`, `download_models.py`, `download_pile_val.sh`. Expect bugs. The smoke test exists to find them.
- `configs/models.yaml` holds layer counts and MLP widths written from memory. `scripts/verify_model_registry.py` checks them against the Hub; fix any MISMATCH lines before trusting x values.
- The paper-fit pairings in `configs/paper_reference_fits.yaml` are inferred from legend order. Only `pythia_opt` and `diffusion_openclip` are confirmed.

## Verified on 2026-10-23

- `bash scripts/setup_env.sh` ran clean: uv venv (Python 3.12), torch with CUDA 13.0, the submodule checked out and 7/7 unit tests passing.
- `verify_model_registry.py`: all 36 registry entries match their Hub configs. That includes the gated Llama and Gemma repos, so the local Hugging Face token already has access.
- After that run, `transformers` was pinned to `>=4.50,<5`. The first sync resolved to 5.x, while upstream targets the 4.x API. Run `uv sync` once to apply the pin.
- Not done yet: Pile download, model downloads, any pipeline run.

## Blockers
None open.


## Next steps, in order

1. `bash scripts/download_pile_val.sh`. It uses gdown on the authors' Google Drive folder. If Drive throttles or blocks it, stop and tell the user. Do not substitute an HF mirror without `meta.pile_set_name` (see `docs/UPSTREAM_CODE_NOTES.md`).
2. `uv sync` to apply the transformers pin, then `uv run pytest -q`.
3. `uv run python scripts/download_models.py --experiment configs/experiments/phase1_pythia_opt.yaml`. Also run `uv run python scripts/make_random_init_checkpoints.py EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1.4b facebook/opt-125m facebook/opt-350m facebook/opt-1.3b`.
4. Once the GPUs work: `uv run python scripts/run_experiment.py configs/experiments/phase0_smoke.yaml`. **Acceptance check:** the authors' own run with identical settings (200k tokens, seed 42) found **2,684** Pythia-160M/OPT-125M pairs; see `third_party/rosetta-neuron-scaling/example_neurons/facebook_opt-125m/correlations/best_buddies.json`. Compare the total and the pair list. A large gap means something differs (data file, transformers version, dtype) and must be explained before moving on.
5. Phase 1 small pairs (160M, 410M, 1.4B) plus the random-init null. Record wall times; the estimates in `docs/COMPUTE_NOTES.md` are back-of-envelope. Use `tmux` for anything over an hour.
6. Fit, then report to the user: `collect_rosetta_counts.py outputs/<experiment>` and `fit_power_law.py results/<experiment>/rosetta_counts.csv --reference language.pythia_opt`. Commit `results/`.

## Ground rules from the user

- **Writing style:** American English. No em-dashes or en-dashes. No serial comma. No filler phrases. The user dictates, so read odd words as likely transcription errors.
- **Git:** atomic commits with one concern each. Rebase, never merge. Never amend or force-push anything that has been pushed.
- **Python:** `uv run`; add dependencies with `uv add`.
- **Linear tickets:** cite as code plus title in quotes, e.g. "ML-1535 Plot WER vs Speaker Similarity", never the bare code.
- **Ask before** anything outward-facing or hard to undo: creating a GitHub repo, pushing, rebooting, deleting outputs, launching cloud jobs (SkyPilot is installed, but the 30B+ pairs need explicit approval).
- **Report faithfully:** if a number disagrees with the paper, say so with the evidence. Don't tune settings until it matches.
