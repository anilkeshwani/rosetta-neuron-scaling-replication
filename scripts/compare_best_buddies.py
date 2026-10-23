"""Compare two best_buddies.json files: totals, pair-list overlap and correlation agreement.

Used for the phase 0 acceptance check against upstream's example run:
    uv run python scripts/compare_best_buddies.py \
        third_party/rosetta-neuron-scaling/example_neurons/facebook_opt-125m/correlations/best_buddies.json \
        outputs/phase0_smoke/pythia-160m__opt-125m/correlations/best_buddies.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[tuple[int, int, int, int], float]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("buddies", data)
    return {
        (b["model1_layer"], b["model1_neuron"], b["model2_layer"], b["model2_neuron"]): b["correlation"]
        for b in data
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", type=Path)
    parser.add_argument("ours", type=Path)
    args = parser.parse_args()

    ref, ours = load(args.reference), load(args.ours)
    shared = ref.keys() & ours.keys()
    union = ref.keys() | ours.keys()
    ref_m1 = {k[:2] for k in ref}
    ours_m1 = {k[:2] for k in ours}
    print(f"reference pairs: {len(ref):,}")
    print(f"our pairs:       {len(ours):,} ({len(ours) - len(ref):+,})")
    print(f"identical pairs: {len(shared):,} ({len(shared) / len(ref):.1%} of reference, Jaccard {len(shared) / len(union):.3f})")
    print(f"model1 neurons matched in both runs (any partner): {len(ref_m1 & ours_m1):,}")
    if shared:
        a = np.array([ref[k] for k in sorted(shared)])
        b = np.array([ours[k] for k in sorted(shared)])
        print(f"shared pairs: corr(ref, ours)={np.corrcoef(a, b)[0, 1]:.3f}, mean |diff|={np.abs(a - b).mean():.4f}")
    for name, d in (("reference", ref), ("ours", ours)):
        c = np.array(list(d.values()))
        print(f"{name:9s} correlation: mean={c.mean():.3f} std={c.std():.3f} min={c.min():.3f} max={c.max():.3f}")


if __name__ == "__main__":
    main()
