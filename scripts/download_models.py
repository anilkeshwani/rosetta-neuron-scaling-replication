"""Pre-download the checkpoints an experiment needs into the HF cache.

Fetches config, tokenizer and weights only (safetensors when the repo has them, else .bin).
Local paths (random-init checkpoints) and pairs marked `requires: cloud` are skipped.
"""

import argparse

import yaml
from huggingface_hub import list_repo_files, snapshot_download

TOKENIZER_AND_CONFIG = ["*.json", "*.model", "*.txt", "*.tiktoken", "tokenizer*"]


def models_in(experiment_path: str, include_cloud: bool) -> list[str]:
    with open(experiment_path) as f:
        exp = yaml.safe_load(f)
    out = []
    for pair in exp["pairs"]:
        if pair.get("requires") == "cloud" and not include_cloud:
            continue
        out += [m for m in pair["models"] if not m.startswith(("$", "/", "."))]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", action="append", default=[], help="experiment YAML (repeatable)")
    parser.add_argument("--model", action="append", default=[], help="Hub id (repeatable)")
    parser.add_argument("--include-cloud", action="store_true")
    args = parser.parse_args()

    models = list(args.model)
    for path in args.experiment:
        models += models_in(path, args.include_cloud)
    for model in dict.fromkeys(models):
        files = list_repo_files(model)
        weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in files) else ["*.bin"]
        print(f"[download] {model} ({weights[0]})", flush=True)
        snapshot_download(model, allow_patterns=TOKENIZER_AND_CONFIG + weights)


if __name__ == "__main__":
    main()
