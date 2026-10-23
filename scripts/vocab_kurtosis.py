"""Vocabulary-space selectivity of Rosetta vs non-Rosetta Pythia neurons (paper Sec. 5.1, Fig. 6a, Fig. 28).

For each MLP neuron u = (layer, channel) of the Pythia model in a Pythia-OPT run, the paper computes
s_v^u = cos(W_out^u, W_U[v]) for every vocabulary row v of the unembedding and takes the excess
kurtosis of {s_v^u}_v as the neuron's output-side selectivity. It then averages this separately over
Rosetta neurons (the Pythia side of correlations/best_buddies.json) and all other neurons.

For GPT-NeoX, W_out^u = layers[l].mlp.dense_4h_to_h.weight[:, u] and W_U = embed_out.weight.

Variants (one row each in the summary):
  raw - exactly the paper's formula on the checkpoint weights (all W_U rows, padding rows included).
  tl  - TransformerLens-style processed weights, as used by Gurnee et al. (2024) whom the paper cites:
        W_U columns scaled by the final LayerNorm gain, W_U centered over the vocabulary, W_out
        centered over d_model. The paper does not say whether it does this.

Also reported, as our addition (the paper uses no control for this metric): mean kurtosis of random
non-Rosetta subsets of the same size as the Rosetta set, several draws.

Runs on CPU in float32, in neuron chunks.

    uv run scripts/vocab_kurtosis.py outputs/phase1_pythia_opt/pythia-160m__opt-125m ...
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from rosetta_repl import REPO_ROOT
from rosetta_repl.registry import load_registry, resolve

VARIANTS = ("raw", "tl")


class Weights:
    """Read single tensors from a (possibly sharded) local safetensors checkpoint."""

    def __init__(self, model_id: str):
        self.root = Path(snapshot_download(model_id, local_files_only=True, allow_patterns=["*.json", "*.safetensors"]))
        index = self.root / "model.safetensors.index.json"
        self.weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None

    def get(self, name: str) -> torch.Tensor:
        fname = self.weight_map[name] if self.weight_map else "model.safetensors"
        with safe_open(self.root / fname, framework="pt") as f:
            return f.get_tensor(name).float()


def excess_kurtosis(s: torch.Tensor) -> torch.Tensor:
    """Population (biased) excess kurtosis along dim 1, as scipy.stats.kurtosis(fisher=True, bias=True)."""
    c = s - s.mean(dim=1, keepdim=True)
    c2 = c.square()
    m2 = c2.mean(dim=1)
    m4 = c2.square().mean(dim=1)
    return m4 / m2.square() - 3.0


def unit_rows(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def neuron_kurtosis(model_id: str, n_layers: int, d_mlp: int, chunk: int) -> dict[str, np.ndarray]:
    """Return {variant: array (n_layers, d_mlp)} of excess kurtosis over the vocabulary."""
    w = Weights(model_id)
    w_u = w.get("embed_out.weight")  # (vocab, d_model)
    gamma = w.get("gpt_neox.final_layer_norm.weight")
    w_u_tl = w_u * gamma  # fold the final LN gain into the unembedding
    w_u_tl = w_u_tl - w_u_tl.mean(dim=0, keepdim=True)  # center over vocab
    unemb = {"raw": unit_rows(w_u).T.contiguous(), "tl": unit_rows(w_u_tl).T.contiguous()}
    out = {v: np.empty((n_layers, d_mlp), dtype=np.float32) for v in VARIANTS}
    for layer in range(n_layers):
        w_out = w.get(f"gpt_neox.layers.{layer}.mlp.dense_4h_to_h.weight").T.contiguous()  # (d_mlp, d_model)
        assert w_out.shape[0] == d_mlp, (w_out.shape, d_mlp)
        rows = {"raw": unit_rows(w_out), "tl": unit_rows(w_out - w_out.mean(dim=1, keepdim=True))}
        for v in VARIANTS:
            for start in range(0, d_mlp, chunk):
                sims = rows[v][start : start + chunk] @ unemb[v]  # (chunk, vocab) cosine similarities
                out[v][layer, start : start + chunk] = excess_kurtosis(sims).numpy()
        print(f"  {model_id} layer {layer + 1}/{n_layers}", flush=True)
    return out


def rosetta_mask(run_dir: Path, side: int, n_layers: int, d_mlp: int) -> np.ndarray:
    buddies = json.loads((run_dir / "correlations" / "best_buddies.json").read_text())
    mask = np.zeros((n_layers, d_mlp), dtype=bool)
    for b in buddies:
        mask[b[f"model{side}_layer"], b[f"model{side}_neuron"]] = True
    return mask


def pythia_side(models: list[str], registry) -> int:
    sides = [i + 1 for i, m in enumerate(models) if resolve(m, registry).family == "pythia"]
    if len(sides) != 1:
        raise ValueError(f"Expected exactly one Pythia model in {models}")
    return sides[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", type=Path, nargs="+", help="finished Pythia-OPT run directories")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results" / "phase4_vocab_kurtosis")
    parser.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "outputs" / "phase4_vocab_kurtosis",
                        help="per-neuron kurtosis arrays (.npz), reused when present")
    parser.add_argument("--chunk", type=int, default=2048, help="neurons per matmul chunk")
    parser.add_argument("--n-draws", type=int, default=5, help="random non-Rosetta control draws")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    rng = np.random.default_rng(args.seed)
    summary, per_layer = [], []

    for run_dir in args.run_dirs:
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        side = pythia_side(manifest["models"], registry)
        entry = resolve(manifest["models"][side - 1], registry)
        mask = rosetta_mask(run_dir, side, entry.n_layers, entry.d_mlp)
        cache = args.cache_dir / f"{entry.short_name}_vocab_kurtosis.npz"
        if cache.exists():
            kurt = dict(np.load(cache))
        else:
            print(f"{run_dir.name}: computing {entry.id}", flush=True)
            kurt = neuron_kurtosis(entry.id, entry.n_layers, entry.d_mlp, args.chunk)
            np.savez_compressed(cache, **kurt)

        n_ros = int(mask.sum())
        non_idx = np.flatnonzero(~mask.ravel())
        draws = [rng.choice(non_idx, size=n_ros, replace=False) for _ in range(args.n_draws)]
        for v in VARIANTS:
            k = kurt[v]
            ros, non = k[mask], k[~mask]
            ctrl = np.array([k.ravel()[d].mean() for d in draws])
            summary.append({
                "run": run_dir.name, "model": entry.id, "variant": v, "n_neurons": entry.n_neurons,
                "n_rosetta": n_ros, "rosetta_frac": n_ros / entry.n_neurons,
                "rosetta_mean": ros.mean(), "rosetta_median": np.median(ros), "rosetta_sem": ros.std(ddof=1) / np.sqrt(n_ros),
                "nonrosetta_mean": non.mean(), "nonrosetta_median": np.median(non),
                "all_mean": k.mean(),
                "ctrl_random_nonrosetta_mean": ctrl.mean(), "ctrl_random_nonrosetta_std": ctrl.std(ddof=1),
                "ctrl_n_draws": args.n_draws,
            })
            for layer in range(entry.n_layers):
                m = mask[layer]
                per_layer.append({
                    "model": entry.id, "variant": v, "layer": layer, "n_rosetta": int(m.sum()),
                    "rosetta_mean": k[layer][m].mean() if m.any() else np.nan,
                    "nonrosetta_mean": k[layer][~m].mean(),
                })

    df = pd.DataFrame(summary).sort_values(["variant", "n_neurons"])
    df.to_csv(args.out_dir / "vocab_kurtosis_summary.csv", index=False, float_format="%.6g")
    pd.DataFrame(per_layer).to_csv(args.out_dir / "vocab_kurtosis_per_layer.csv", index=False, float_format="%.6g")
    print(df.to_string(index=False))

    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(5 * len(VARIANTS), 4), sharey=False)
    for ax, v in zip(axes, VARIANTS):
        d = df[df.variant == v]
        ax.errorbar(d.n_neurons, d.rosetta_mean, yerr=d.rosetta_sem, marker="o", label="Rosetta")
        ax.plot(d.n_neurons, d.nonrosetta_mean, marker="s", label="Non-Rosetta (all)")
        ax.errorbar(d.n_neurons, d.ctrl_random_nonrosetta_mean, yerr=d.ctrl_random_nonrosetta_std, marker="^",
                    ls="--", label=f"Random non-Rosetta, |R| each ({args.n_draws} draws)")
        ax.set_xscale("log")
        ax.set_xlabel("Pythia MLP neurons")
        ax.set_ylabel("Mean excess kurtosis over vocabulary")
        ax.set_title(f"Pythia vocab-space selectivity ({v})")
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out_dir / "vocab_kurtosis_vs_size.png", dpi=150)


if __name__ == "__main__":
    main()
