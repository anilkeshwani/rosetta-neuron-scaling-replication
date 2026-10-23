# Notes for agents working in this repo

- Python env is managed by uv. Run everything with `uv run ...`; add dependencies with `uv add`, never pip.
- Source `scripts/env.sh` before running pipeline scripts. It sets `ROSETTA_REPO`, `DATASET`, the HF cache and NCCL settings.
- Never edit files under `third_party/rosetta-neuron-scaling/` in place. If upstream code needs changing, put a patch in `patches/` with a note in `docs/UPSTREAM_CODE_NOTES.md`, so the pinned commit plus our patches stays reproducible.
- Every run goes through `scripts/run_pair.sh`, which writes `run_manifest.json` (models, knobs, git SHAs, GPU info) next to the outputs. Do not launch `match.sh` directly for results we intend to report.
- Neuron counts come from `configs/models.yaml`, which `scripts/verify_model_registry.py` checks against the Hub configs. Do not hard-code counts elsewhere.
- Results that go into a fit must use the default matching settings (k=1, mean span pooling, full layer grid, no `DEPTH_NEIGHBORS`) unless the experiment YAML says otherwise and the deviation is recorded.
- Large artifacts live under `data/` and `outputs/` (gitignored). Commit configs, scripts, summaries (`rosetta_counts.csv`, fit JSON, plots), not raw neighbor tensors.
- Use base (pretrained) checkpoints only, never instruction-tuned variants, to match the paper.
