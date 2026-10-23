"""Check n_layers and d_mlp in configs/models.yaml against each model's config.json on the Hub.

Gated repos fail with an access error until the license is accepted for the token in use;
those are reported, not treated as mismatches.
"""

import argparse
import json
import sys

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

from rosetta_repl.registry import load_registry


def mlp_shape(cfg: dict) -> tuple[int, int]:
    cfg = cfg.get("text_config", cfg)  # Gemma 3 >= 4B nests the decoder config
    layers = cfg.get("num_hidden_layers", cfg.get("n_layer"))
    if "intermediate_size" in cfg:
        d_mlp = cfg["intermediate_size"]
    elif "ffn_dim" in cfg:  # OPT
        d_mlp = cfg["ffn_dim"]
    else:  # GPT-2: n_inner is null -> 4 * n_embd
        d_mlp = cfg.get("n_inner") or 4 * cfg["n_embd"]
    return layers, d_mlp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", help="only check this family")
    args = parser.parse_args()

    bad = 0
    for entry in load_registry().values():
        if args.family and entry.family != args.family:
            continue
        try:
            with open(hf_hub_download(entry.id, "config.json")) as f:
                layers, d_mlp = mlp_shape(json.load(f))
        except (GatedRepoError, RepositoryNotFoundError) as e:
            print(f"NO ACCESS  {entry.id}: {type(e).__name__}")
            continue
        ok = (layers, d_mlp) == (entry.n_layers, entry.d_mlp)
        bad += not ok
        status = "ok        " if ok else "MISMATCH  "
        print(f"{status}{entry.id}: hub=({layers}, {d_mlp}) registry=({entry.n_layers}, {entry.d_mlp}) "
              f"neurons={layers * d_mlp:,}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
