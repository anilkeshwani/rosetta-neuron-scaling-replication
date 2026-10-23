# Replication plan

Each phase has an experiment YAML in `configs/experiments/` and a short list of checks that decide whether we move on.

## Phase 0: smoke test

`phase0_smoke.yaml`: Pythia-160M vs OPT-125M on 200k tokens.

- [ ] `setup_env.sh` passes, including the CUDA check
- [ ] The Pile validation set downloads and the sampler reads it
- [ ] The run produces `correlations/best_buddies.json` and `collect_rosetta_counts.py` parses it
- [ ] **Match the upstream example.** `third_party/rosetta-neuron-scaling/example_neurons` is the authors' own run with the same settings as our smoke test (`NUM_SAMPLES=200000`, seed 42, Pythia-160M as model 1). It found **2,684** Pythia-160M/OPT-125M pairs (`facebook_opt-125m/correlations/best_buddies.json`) and 2,672 Pythia-160M/GPT-2 pairs. Our count should be the same or very close; compare the pair lists as well as the totals

## Phase 1: replicate Pythia–OPT and the random-init null

`phase1_pythia_opt.yaml`, `phase1_null_random_init.yaml`.

- [ ] Run the three small pairs (160M, 410M, 1.4B) at 10M tokens and measure wall time
- [ ] Random-init null at the same three sizes, 3 seeds each: expect roughly 0-40 matches and no trend (Fig. 4)
- [ ] Extend to 2.8B and 6.9B; 12B if the measured times allow
- [ ] **Pass criterion:** k=1 exponent within ±0.05 of 0.68 with R² ≥ 0.98. The prefactor can differ more, since it depends on the token sample
- [ ] k sweep (1, 2, 5, 10): exponents should stay in a narrow sublinear band (Fig. 25: 0.68-0.78)
- [ ] Token-budget check at one size (1M, 3M, 10M tokens): how much does the count move? This tells us whether a cheaper budget is usable for the large pairs (Fig. 33)

## Phase 2: second family pair and gated MLPs

`phase2_opt_qwen.yaml`.

- [ ] OPT–Qwen2.5 at 1.3B/1.5B through 13B/14B. Published fit 12.43·x^0.57 (pairing inferred from legend order, so a mismatch in c could be ours or the inference)
- [ ] The 3-model intersection Pythia–OPT–Qwen2.5 (anchor pipeline) at the sizes where all three exist; published 1.99·x^0.60
- [ ] Write the input-permutation null patch and run it at 2 sizes (Fig. 26)

## Phase 3: extension to Gemma and Llama

`phase3_gemma3_qwen.yaml`, `phase3_llama_qwen.yaml`, `phase3_distillation_probe.yaml`.

- [ ] Patch upstream for Gemma 3 ≥ 4B (`Gemma3ForConditionalGeneration` layer path)
- [ ] Gemma 3 vs Qwen2.5 at 270M/0.5B through 12B/14B. Main question: does the exponent stay in 0.5-0.7 for two modern, gated-MLP, much more heavily trained families? These models saw 2-36T tokens against the Pile's ~300B, so a shift in α would be informative in itself
- [ ] Llama 3.x vs Qwen2.5 at 1B, 3B and 8B
- [ ] Distillation probe: Llama-3.2-1B/3B against their teacher (Llama-3.1-8B) and against an unrelated model of similar size (Qwen2.5-7B). If the teacher match count is much higher, distilled students are not independent samples and within-family curves that mix distilled and scratch-trained sizes are biased

## Phase 4: properties of the Rosetta population

These reuse the matches from phases 1-3.

- [ ] Vocabulary-space excess kurtosis of W_out rows against W_U, Rosetta vs non-Rosetta, across Pythia sizes (Fig. 6a). Cheap: weights only, no forward passes
- [ ] Document-type firing by Pile subset category (Fig. 7, Table 4), with the random non-Rosetta control (1000 neurons, 3 draws)
- [ ] Optional: the JavaScript filtering case study (Table 1)

## Open directions beyond the paper

- **Training dynamics.** Pythia publishes 154 checkpoints per size, and upstream supports `REVISION1`/`REVISION2`. When do Rosetta neurons appear, and does the count follow its own power law in training steps? The paper names this as future work.
- **Post-training.** Base vs instruct for Qwen2.5 or Llama: do Rosetta neurons survive instruction tuning?
- **Fitting β directly.** The theory predicts α = 1/β and Rosetta purification ∝ N^((β−1)/β). The kurtosis trend gives an independent handle on the second exponent; check whether both agree with one β.
