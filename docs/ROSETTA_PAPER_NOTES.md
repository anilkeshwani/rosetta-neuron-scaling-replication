# Paper notes: "Neuron Populations Exhibit Divergent Selectivity with Scale"

Dravid, Bahri, Efros (UC Berkeley), Gandelsman (TTIC). [arXiv 2606.03990](https://arxiv.org/abs/2606.03990), June 2026; NeurIPS 2026. Follows "Rosetta Neurons: Mining the common units in a model zoo" (Dravid et al., ICCV 2023). Download the PDF from arXiv if you need figures; these notes hold the numbers we compare against.

## Method (Section 3, Appendix B)

- **Neuron:** one coordinate of the MLP intermediate activation, m = φ(W_in h + b), indexed by (layer, channel). For gated MLPs, upstream uses the input to `down_proj`.
- **Alignment:** text is aligned by the token boundaries both tokenizers share in UTF-8 byte space; activations are mean-pooled within each aligned span. Vision grids are bilinearly resampled to a shared grid.
- **Similarity:** Pearson correlation over all aligned positions in the dataset (about 10M Pile validation tokens for language, 50k diffusion-generated ImageNet-class images for vision).
- **Rosetta Neurons:** mutual nearest neighbors (k=1) under that correlation. 3+-family "collapsed" curves intersect the matches found in the shared model.
- **Model families:** language uses Pythia, GPT-2, OPT and Qwen2.5 base models (about 100M to 30B parameters, 40k to 2M neurons). Vision uses OpenCLIP, DINOv2 and Pixio matched against one-step DiT generators (pMF, Sana, Flux.2).

## Results we are replicating

| Claim | Evidence | Number |
|---|---|---|
| Sublinear power law | Fig. 3 | |R| = c·x^α, α ≈ 0.5 to 0.7 in every family pair, R² ≈ 0.99; x = mean MLP neuron count of the pair |
| Pythia–OPT fit | Fig. 3a, Fig. 25 | **2.20·x^0.68** (k=1); k=2: 0.75, k=5: 0.78, k=10: 0.71 |
| Random-init null | Fig. 4 | 0 to 40 matches, no trend with size |
| Permutation null | Fig. 26 | 20 to 100 matches, no trend |
| Token-budget stability | Fig. 33 | Rosetta set stabilizes as tokens grow toward 1e7 to 1e8 (Pythia-6.9B/OPT-6.7B) |
| Polarization (language) | Fig. 6a, 28 | Mean excess kurtosis of cos(W_out row, W_U) rises with size for Rosetta neurons (up to ~15-20 in Pythia); non-Rosetta stays near 0 |
| Specialization | Fig. 7, Table 4 | Top-20 contexts of Rosetta neurons shift toward code and math with size, normalized by category token share |
| Data filtering | Table 1 | One JavaScript Rosetta neuron (Pythia-6.9B L16/U11168): F1 0.98; GPT2-1.5B test PPL 3.02 vs oracle 3.01, random 3.59 |
| Failure case | Fig. 38 | DINOv3 does not follow the law |

All published fits are in `configs/paper_reference_fits.yaml`.

### Pythia-OPT per-pair counts behind Fig. 3a and Fig. 25

The paper prints no table of counts. These values were decoded from the vector marker positions in the arXiv figure PDFs (`topk_robustness.pdf`), calibrated against each figure's axis ticks; they land on integers to within 0.01, and an OLS fit in log-log space returns the published fits (2.20·x^0.678 at k=1). Fig. 3a's k=1 points equal Fig. 25's.

| Pair | x (plotted) | k=1 | k=2 | k=5 | k=10 |
|---|---|---|---|---|---|
| Pythia-160M / OPT-125M | 36,864 | 2,609 | 6,976 | 24,304 | 60,525 |
| Pythia-410M / OPT-350M | 98,304 | 5,418 | 12,307 | 40,171 | 100,971 |
| Pythia-1.4B / OPT-1.3B | 196,608 | 9,001 | 23,671 | 90,019 | 235,511 |
| Pythia-2.8B / OPT-2.7B | 360,448 | 13,686 | 34,686 | 120,686 | 300,686 |
| Pythia-6.9B / OPT-6.7B | 524,288 | 16,276 | 47,276 | 180,276 | 375,276 |
| Pythia-12B / OPT-13B | 778,240 | 20,282 | 67,282 | 250,282 | 490,282 |

Oddities worth raising with the authors: (1) for the three largest pairs the k=2/5/10 counts share their last three digits with k=1, which does not look like raw counts; (2) the 2.8B point sits at x = 360,448, but both Pythia-2.8B and OPT-2.7B have 32 × 10,240 = 327,680 MLP neurons (every other x is the exact pair mean); (3) the OPT-Qwen2.5 series in Fig. 3a includes round values (24,000, 27,000, 45,000). Settings stated in the paper: ~10M Pile validation tokens per pair for all sizes, 8 A100 80 GB GPUs per pair. Seed, sequence length and batch size are not stated.

## Theory (Section 4.3, Appendix D)

- **Assumptions:** features have importance w_r ∝ r^(−β) with β > 1, and there are more features than neurons, A(N) > N. Each feature gets an isolation score s_r, the signal-to-interference ratio in its most dedicated neuron.
- **Allocation problem:** maximize Σ w_r log(1 + s_r) subject to Σ s_r ≤ κN (water-filling). The solution is s*(r) = [(r₀/r)^β − 1]₊ with r₀ = Θ(N^(1/β)).
- **Count prediction:** a feature counts as Rosetta when s_r ≥ τ > 1, which gives |R| = Θ(N^(1/β)). The measured α ≈ 0.5 to 0.7 therefore implies β ≈ 1.4 to 2.
- **Polarization prediction:** mean Rosetta isolation grows as N^((β−1)/β), while non-Rosetta isolation goes to 0.
- **Why matching works (D.6):** the cross-model correlation is 1/√((1+1/s)(1+1/s′)). The superposed tail is packed differently in each model, so only well-isolated features produce matches.

## Weak points to keep in mind

- The theory is phenomenological: its assumptions are chosen, not derived from training.
- Selectivity is measured through proxies: output-weight kurtosis for language and a GPT-5.4 yes/no judge for vision.
- The vision setup relies on generated images.
- The filtering result is a single neuron.
