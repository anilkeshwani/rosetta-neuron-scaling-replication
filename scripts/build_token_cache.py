"""Build upstream's byte-aligned token cache on CPU as an N-rank torchrun job would.

Upstream's match_lm builds the token sample in shards: rank r draws num_samples/world_size
tokens with seed 42 + r, and rank 0 concatenates the shards. The sample therefore depends
on the GPU count. The authors' example run (example_neurons/run.log) used 8 ranks, so a
4-GPU run sees different tokens. This script calls upstream's own cache builder
once per simulated rank and writes the merged file; pass it to run_pair.sh as TOKEN_CACHE
and match_lm loads it instead of sampling.

The per-tokenizer Pile stats files (~/.cache/rosetta/pile_val_stats_*.json and the
.ge<seq_length>tok.json filtered variant) are built first if missing, with upstream's
single-process scan, so that parallel rank workers do not race to write them.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_LANG = REPO_ROOT / "third_party" / "rosetta-neuron-scaling" / "language"
sys.path.insert(0, str(UPSTREAM_LANG))

from data.pile_sampler import (  # noqa: E402
    _load_or_build_filtered_token_stats,
    _load_or_build_val_stats,
    resolve_pile_stats_path,
)
from find_pairs.match_cache import _split_token_budget, pretokenize_pile_two_models_bytecache_once  # noqa: E402


def load_tokenizer(name: str):
    # Same settings as find_pairs/match_pipeline.run
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    tok.truncation_side = "right"
    return tok


def build_shard(job: tuple) -> dict:
    args, rank, stats_path = job
    tok1, tok2 = load_tokenizer(args.model1), load_tokenizer(args.model2)
    budget = _split_token_budget(args.num_samples, rank, args.world_size)
    print(f"[token-cache] rank {rank}/{args.world_size}: budget={budget} seed={args.seed + rank}", flush=True)
    return pretokenize_pile_two_models_bytecache_once(
        tok1,
        tok2,
        total_token_budget=budget,
        dataset_name=args.dataset,
        max_tokens=args.seq_length,
        max_bytes=args.max_bytes,
        text_batch=args.tokenize_batch,
        seed=args.seed + rank,
        split=args.split,
        pile_subsets=None,
        stats_path=stats_path,
        ratio_by="tokens",
        use_padding=False,
        allow_empty=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model1")
    parser.add_argument("model2")
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", default=os.environ.get("DATASET", str(REPO_ROOT / "data" / "pile")))
    parser.add_argument("--split", default="val")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=8, help="ranks built in parallel CPU processes")
    # Defaults below are run_pipeline.sh's / match_cli's
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument("--max-bytes", type=int, default=8096)
    parser.add_argument("--tokenize-batch", type=int, default=256)
    args = parser.parse_args()

    tok1 = load_tokenizer(args.model1)
    stats_path = resolve_pile_stats_path(args.dataset, tok1)
    _load_or_build_val_stats(args.dataset, tok1, stats_path=stats_path)
    _load_or_build_filtered_token_stats(args.dataset, tok1, context_length=args.seq_length, stats_path=stats_path)

    with ProcessPoolExecutor(max_workers=min(args.jobs, args.world_size)) as pool:
        results = list(pool.map(build_shard, [(args, rank, stats_path) for rank in range(args.world_size)]))
    shards = [s for s in results if s]  # rank order, as upstream's rank-0 merge

    merged = {k: torch.cat([s[k] for s in shards], dim=0) for k in shards[0]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.out)
    print(f"[token-cache] wrote {args.out}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in merged.items()))


if __name__ == "__main__":
    main()
