"""Fit |R| = c * x^alpha per k to a rosetta_counts.csv and compare with the paper's fit.

Writes <csv dir>/power_law_fit.json and power_law_fit.png.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from rosetta_repl import REPO_ROOT
from rosetta_repl.scaling import fit_power_law


def reference(key: str | None) -> dict | None:
    if not key:
        return None
    refs = yaml.safe_load((REPO_ROOT / "configs" / "paper_reference_fits.yaml").read_text())
    section, name = key.split(".") if "." in key else ("language", key)
    return refs[section][name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("counts_csv", type=Path)
    parser.add_argument("--reference", help="key in paper_reference_fits.yaml, e.g. language.pythia_opt")
    args = parser.parse_args()

    df = pd.read_csv(args.counts_csv)
    df = df[df["done"]]
    ref = reference(args.reference)
    results = {"reference": {"key": args.reference, **ref} if ref else None, "fits": {}}

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, g in df.groupby("k"):
        g = g.sort_values("x_mean_neurons")
        if len(g) < 2:
            continue
        fit = fit_power_law(g["x_mean_neurons"], g["n_rosetta"])
        results["fits"][f"k{k}"] = fit.to_dict()
        pts = ax.scatter(g["x_mean_neurons"], g["n_rosetta"], label=f"k={k}: {fit.c:.2f}x^{fit.alpha:.2f} "
                         f"(R²={fit.r2:.3f})")
        xs = np.geomspace(g["x_mean_neurons"].min(), g["x_mean_neurons"].max(), 50)
        ax.plot(xs, fit.predict(xs), "--", color=pts.get_facecolor()[0])
    if ref:
        xs = np.geomspace(df["x_mean_neurons"].min(), df["x_mean_neurons"].max(), 50)
        ax.plot(xs, ref["c"] * xs ** ref["alpha"], "k:", label=f"paper: {ref['c']}x^{ref['alpha']}")
        if "k1" in results["fits"]:
            results["alpha_delta_k1"] = results["fits"]["k1"]["alpha"] - ref["alpha"]

    ax.set(xscale="log", yscale="log", xlabel="Model size (# MLP neurons, pair mean)",
           ylabel="# Rosetta Neurons", title=args.counts_csv.parent.name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.counts_csv.parent / "power_law_fit.png", dpi=150)

    (args.counts_csv.parent / "power_law_fit.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
