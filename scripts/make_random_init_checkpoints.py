"""Build untrained checkpoints for the random-initialization null (paper Fig. 4).

Each model is instantiated from its Hub config with the architecture's own init
(transformers' _init_weights) under a fixed seed and saved with its tokenizer to
$ROSETTA_DATA/random_init/<short-name>-seed<k>. Directory names keep the family string
(pythia, opt, ...) because upstream chooses hook points by substring of the model path.
"""

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", help="Hub ids, e.g. EleutherAI/pythia-160m facebook/opt-125m")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("ROSETTA_DATA", "data")) / "random_init")
    args = parser.parse_args()

    for model_id in args.models:
        config = AutoConfig.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        for seed in args.seeds:
            dest = args.out / f"{model_id.split('/')[-1]}-seed{seed}"
            if (dest / "config.json").exists():
                print(f"[random-init] exists: {dest}")
                continue
            torch.manual_seed(seed)
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
            model.save_pretrained(dest, safe_serialization=True)
            tokenizer.save_pretrained(dest)
            print(f"[random-init] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
