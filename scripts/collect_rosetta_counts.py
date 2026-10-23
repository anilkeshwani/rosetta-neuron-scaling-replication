"""Gather Rosetta Neuron counts from finished runs into results/<experiment>/rosetta_counts.csv.

One row per run and k. Pairwise runs count best-buddy pairs in
correlations/best_buddies[_kN].json; 3+-model runs count anchors in rosetta_anchors.json.
x (the paper's model-size axis) is the mean total MLP neuron count of the matched models.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from rosetta_repl import REPO_ROOT
from rosetta_repl.registry import load_registry, resolve

K_FILE = re.compile(r"best_buddies(?:_k(\d+))?\.json$")


def count(path: Path) -> int:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("buddies", data.get("anchors", data))
    return len(data)


def counts_for_run(run_dir: Path, k_default: int) -> dict[int, int]:
    anchors = sorted(run_dir.glob("**/rosetta_anchors.json"))
    if anchors:  # multi-model run; k-sweep anchors live in sibling dirs named with k<N>
        out = {}
        for p in anchors:
            m = re.search(r"k(\d+)", p.parent.name)
            out[int(m.group(1)) if m else k_default] = count(p)
        return out
    out = {}
    for p in (run_dir / "correlations").glob("best_buddies*.json"):
        if m := K_FILE.search(p.name):
            out[int(m.group(1)) if m.group(1) else k_default] = count(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path, help="outputs/<experiment>")
    parser.add_argument("--out", type=Path, help="default: results/<experiment>/rosetta_counts.csv")
    args = parser.parse_args()

    registry = load_registry()
    rows = []
    for manifest_path in sorted(args.experiment_dir.glob("*/run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        entries = [resolve(m, registry) for m in manifest["models"]]
        x = sum(e.n_neurons for e in entries) / len(entries)
        k_default = int(manifest["knobs"].get("K_NEIGHBORS") or 1)
        for k, n in sorted(counts_for_run(run_dir, k_default).items()):
            rows.append({
                "run": run_dir.name,
                "models": " | ".join(manifest["models"]),
                "families": "-".join(e.family for e in entries),
                "k": k,
                "n_rosetta": n,
                "x_mean_neurons": x,
                "fraction_of_min_model": n / min(e.n_neurons for e in entries),
                "done": (run_dir / "DONE").exists(),
                "elapsed_s": manifest.get("elapsed_s"),
                "num_samples": manifest["knobs"].get("NUM_SAMPLES"),
            })

    out = args.out or REPO_ROOT / "results" / args.experiment_dir.name / "rosetta_counts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False) if len(df) else "no finished runs found")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
