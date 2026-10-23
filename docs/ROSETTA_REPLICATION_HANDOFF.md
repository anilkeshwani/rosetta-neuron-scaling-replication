# Handoff: Rosetta Neuron scaling replication

Written 2026-10-23 by the Claude Code session that scaffolded this repo. You are taking over on the GPU machine. Read this file, then `README.md`, `CLAUDE.md` and the four files in `docs/`.

## Goal

Replicate, then extend, Dravid et al. 2026, "Neuron Populations Exhibit Divergent Selectivity with Scale" (arXiv 2606.03990). The headline claim is that the number of Rosetta Neurons (cross-model mutual-nearest-neighbor MLP neurons) grows as |R| = c·x^α with α ≈ 0.5 to 0.7. The first target is the Pythia–OPT curve, **2.20·x^0.68**. Next comes OPT–Qwen2.5, then the extension to Gemma 3 and Llama 3.x, including a check of whether distilled checkpoints inflate the counts. `docs/ROSETTA_PAPER_NOTES.md` has the paper's numbers; `docs/REPLICATION_PLAN.md` has the phases and pass criteria.

## State of the repo

- Location: `~/projects/rosetta-neuron-scaling-replication`. Remote: https://github.com/anilkeshwani/rosetta-neuron-scaling-replication (public, created 2026-10-24 at the user's request; history rewritten once that day to drop a work email, see `docs/GIT_HISTORY_REWRITE_SHA_MAP.md`).
- The authors' code is the submodule `third_party/rosetta-neuron-scaling`, pinned at `15cd3c7`. Do not edit it in place; patches go in `patches/` (see `CLAUDE.md`).
- Written but **not yet exercised against a real run**: `scripts/run_pair.sh`, `run_experiment.py`, `collect_rosetta_counts.py`, `fit_power_law.py`, `make_random_init_checkpoints.py`, `download_models.py`, `download_pile_val.sh`. Expect bugs. The smoke test exists to find them.
- `configs/models.yaml` holds layer counts and MLP widths written from memory. `scripts/verify_model_registry.py` checks them against the Hub; fix any MISMATCH lines before trusting x values.
- The paper-fit pairings in `configs/paper_reference_fits.yaml` are inferred from legend order. Only `pythia_opt` and `diffusion_openclip` are confirmed.

## Verified on 2026-10-23

- `bash scripts/setup_env.sh` ran clean: uv venv (Python 3.12), torch with CUDA 13.0, the submodule checked out and 7/7 unit tests passing.
- `verify_model_registry.py`: all 36 registry entries match their Hub configs. That includes the gated Llama and Gemma repos, so the local Hugging Face token already has access.
- `transformers` is pinned to `>=4.50,<5` (now 4.57.6): an unpinned sync resolved to 5.x, while upstream targets the 4.x API.
- Not done yet: Pile download, model downloads, any pipeline run.

## Blockers
None open.


## Next steps, in order

1. `bash scripts/download_pile_val.sh`. It uses gdown on the authors' Google Drive folder. If Drive throttles or blocks it, stop and tell the user. Do not substitute an HF mirror without `meta.pile_set_name` (see `docs/UPSTREAM_CODE_NOTES.md`).
2. Pick up any changes since this handoff: `git log --oneline` (scaffold commit `2dec7e1`), `uv sync`, `uv run pytest -q`.
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

## Status, 2026-10-23 22:30

**What ran**
- Pile val downloaded from the authors' Drive folder in 23 s (450 MB, sha256 `264c875d…fab92`, 214,670 docs, `meta.pile_set_name` present). Phase 1 models (160M through 12B/13B, 249 GB in the HF cache) and the 18 random-init checkpoints (3 seeds × 6 models) are on disk.
- Code review by Codex (GPT-6 Astra) of our scripts found seven bugs; all are fixed and committed (forced reruns reused old caches and neighbor files, stale k-sweep files were counted, rank-deficient fits, `opt_qwen2.5` reference keys, manifest gaps).

**Phase 0 (acceptance check).** Upstream's token sample depends on the GPU count (rank r draws N/world_size tokens with seed 42+r). Run natively on 4 GPUs, the smoke test gives 2,710 pairs, only 74.1% identical to the authors' 2,684. `scripts/build_token_cache.py` rebuilds the authors' 8-rank sample on CPU (verified tensor-identical to upstream's own cache at 4 ranks). With it we get **2,663 pairs (-0.8%), 95.4% identical to theirs**; shared pairs agree to 0.003 in correlation. Our runs are bit-identical on repeat. transformers 5.20.2 gives the same list as 4.57.6, so the residual 4.6% is bf16 numerics on different GPUs/kernels. Knob sensitivities (160M pair, 200k tokens): 2 GPUs +3 pairs, 1 GPU +6, BATCH_SIZE 4 +9 (94% overlap), B_BLOCK and the allocator setting none. I judged this a pass.

**Phase 1 so far** (10M tokens, authors' 8-rank sample, 4 GPUs, k=1/2/5/10; paper values decoded from its Fig. 25, see `ROSETTA_PAPER_NOTES.md`):

| Pair | x | ours k=1 | paper k=1 | ours k=2/5/10 | paper k=2/5/10 | wall time |
|---|---|---|---|---|---|---|
| Pythia-160M / OPT-125M | 36,864 | 2,608 | 2,609 | 6,995 / 24,274 / 60,577 | 6,976 / 24,304 / 60,525 | 7.8 min |
| Pythia-410M / OPT-350M | 98,304 | **3,444** | **5,418** | 9,470 / 34,965 / 92,476 | 12,307 / 40,171 / 100,971 | 38 min |
| Pythia-1.4B / OPT-1.3B | 196,608 | 8,688 | 9,001 | 24,007 / 85,523 / 217,499 | 23,671 / 90,019 / 235,511 | 2.5 h (BATCH_SIZE 4) |

**Update 00:20:** the 1.4B pair finished 3.5% below the paper. The three-point k=1 fit is 1.64·x^0.69 (paper 2.20·x^0.68), but R² is only 0.85 because of the low 410M point, so the pass criterion (R² ≥ 0.98) is not met. Slope from 160M to 1.4B, skipping 410M: 0.72 ours, 0.74 paper. Fit in `results/phase1_pythia_opt/power_law_fit.{json,png}`. The null pairs started at 00:20.

The 160M pair matches the paper almost exactly. **The 410M pair is 36% below the paper at k=1** (local slope 0.28 against 0.68). A CPU diagnosis found nothing specific to OPT-350M: hooks capture relu(fc1) on all 24 layers, there are few dead units and no empty layers. It did find that pass 2 accumulates correlations in bf16 over ~1,200 batches, which drops late increments; emulated on a layer subset this changed the mutual-NN count by ~4%. Not tuned, per the rules.

**Running now (tmux)**
- `rosetta-phase1` (`scripts/run_phase1_small_and_null.sh`, log `logs/phase1_driver_attempt2.log`): Pythia-1.4B/OPT-1.3B finished (see update above). The first attempt ran out of memory in pass 1 at BATCH_SIZE 8 and hung in NCCL; it now uses BATCH_SIZE 4 (recorded in the experiment YAML). Then the 9 random-init null pairs as 4 single-GPU workers (rough guess: 10-15 h, the 1.4B nulls dominate).
- `rosetta-followup` (`logs/followup_batchsize_check.sh`): after everything else, reruns the 160M pair at 10M tokens with BATCH_SIZE 4, into `outputs/phase1_diagnostics/`, to measure whether the halved batch biases the 1.4B count (more bf16 additions).

**Next**
1. When the driver logs `all done`: `collect_rosetta_counts.py outputs/phase1_null_random_init` and commit. Check the BATCH_SIZE 4 rerun of the 160M pair against 2,608 to see whether the 1.4B count needs a correction note.
2. Decide with the user on a diagnostic fp32-accumulator patch (`find_pairs/match_corr.py:179-182`), run on 160M and 410M only, to test whether bf16 accumulation explains the 410M gap. It would go in `patches/` and not be used for reported fits.
3. Raise the paper's data oddities with the user (2.8B plotted at x=360,448 instead of 327,680; shared trailing digits across k at the largest sizes).
4. 2.8B, 6.9B and 12B pairs: token caches are prebuilt under `data/token_caches/phase1_pythia_opt/`. Expect memory pressure; 6.9B and 12B use FSDP.
