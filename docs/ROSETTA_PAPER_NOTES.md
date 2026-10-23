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
