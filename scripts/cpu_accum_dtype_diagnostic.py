"""Measure on CPU how much bf16 accumulation of the correlation sum changes the Rosetta count.

Upstream's pass 2 adds each batch's outer product into a bf16 accumulator (on GPU, addmm_
computes C + A@B with fp32 internal accumulation and rounds the result to bf16). This script
replays pass 2 for one pair on CPU and keeps two accumulators fed by the same fp32 GEMM of the
same bf16 inputs:

  acc_fp32  += P                          (patch 0001's fp32 accumulation)
  acc_bf16   = bf16(acc_bf16 + P)         (emulates upstream's bf16 addmm_)

so the only difference between the two counts is the accumulator precision. It reuses the
GPU run's token cache and its saved pass-1 statistics (stats_{A,B}_layer*.pt, cast to bf16 as
upstream does), pools and standardizes in bf16 with upstream's own functions, and uses the
same batch size and order. Forward passes run in --forward-dtype on CPU (fp32 by default,
since this CPU has no bf16 matrix units), so the bf16 count also shows how close the CPU
replay is to the GPU count. Mutual top-k is computed on the full neuron-by-neuron matrices.

Example:
  uv run python scripts/cpu_accum_dtype_diagnostic.py outputs/phase1_pythia_opt/pythia-160m__opt-125m
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "rosetta-neuron-scaling" / "language"))

from find_pairs.match_cache import make_dataloader_from_cache  # noqa: E402
from find_pairs.match_model import (  # noqa: E402
    MultiActivationCapture,
    _load_model,
    get_mlp_post_activation_modules,
    make_backbone_forward,
)
from find_pairs.match_stats import (  # noqa: E402
    _aggregate_spans_vectorized,
    _build_canonical_spans_for_batch,
    _precompute_span_flat_indices,
)


def load_stats(corr_dir: Path, side: str, n_layers: int, dtype: torch.dtype):
    means, invstds = [], []
    for i in range(n_layers):
        s = torch.load(corr_dir / f"stats_{side}_layer{i}.pt", map_location="cpu")
        means.append(s["mean"].to(dtype))
        invstds.append(1.0 / s["std"].to(dtype))  # as match_pipeline: 1 / std after the cast
    return means, invstds


def mutual_topk(m: torch.Tensor, k: int) -> int:
    """Count (a, b) with b in a's top-k over all B neurons and a in b's top-k over all A neurons."""
    top_b = torch.topk(m, k, dim=1).indices  # (nA, k)
    top_a = torch.topk(m, k, dim=0).indices  # (k, nB)
    is_top_a = torch.zeros_like(m, dtype=torch.bool)
    is_top_a.scatter_(0, top_a, True)
    return int(is_top_a.gather(1, top_b).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="finished GPU run, e.g. outputs/phase1_pythia_opt/<pair>")
    parser.add_argument("--forward-dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--a-chunk", type=int, default=8192, help="A neurons per GEMM chunk (memory)")
    parser.add_argument("--max-batches", type=int, default=0, help="stop early (for timing tests)")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--out", type=Path, help="default: <run_dir>/cpu_accum_dtype_diagnostic.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    manifest = json.loads((args.run_dir / "run_manifest.json").read_text())
    model1, model2 = manifest["models"]
    knobs = manifest["knobs"]
    batch_size = int(knobs.get("BATCH_SIZE") or 8)
    corr_dir = args.run_dir / "correlations"
    cache_path = knobs.get("TOKEN_CACHE") or str(corr_dir / "cache" / "bytecache.pt")
    compute = torch.bfloat16
    fwd_dtype = getattr(torch, args.forward_dtype)

    cpu = torch.device("cpu")
    m1 = _load_model(model1, fwd_dtype, cpu).eval()
    m2 = _load_model(model2, fwd_dtype, cpu).eval()
    fwd1, fwd2 = make_backbone_forward(m1), make_backbone_forward(m2)
    layers_a = get_mlp_post_activation_modules(m1, model1)
    layers_b = get_mlp_post_activation_modules(m2, model2)
    capA = MultiActivationCapture([m for _, m, _ in layers_a], capture_output=[c for _, _, c in layers_a]).register()
    capB = MultiActivationCapture([m for _, m, _ in layers_b], capture_output=[c for _, _, c in layers_b]).register()
    meansA, invA = load_stats(corr_dir, "A", len(layers_a), compute)
    meansB, invB = load_stats(corr_dir, "B", len(layers_b), compute)
    nA, nB = sum(m.numel() for m in meansA), sum(m.numel() for m in meansB)
    print(f"[diag] {model1} ({nA:,} neurons) vs {model2} ({nB:,}); batch {batch_size}; cache {cache_path}", flush=True)

    cache = torch.load(cache_path, map_location="cpu")
    loader = make_dataloader_from_cache(cache, batch_size)
    acc32 = torch.zeros(nA, nB, dtype=torch.float32)
    acc16 = torch.zeros(nA, nB, dtype=compute)
    n_spans, n_batches, t0 = 0, 0, time.time()
    with torch.inference_mode():
        for bi, batch in enumerate(loader):
            if args.max_batches and bi >= args.max_batches:
                break
            span_b, span_a_tok, span_b_tok = _build_canonical_spans_for_batch(
                batch["byte_offsets1"], batch["attention_mask1"], batch["byte_offsets2"], batch["attention_mask2"],
                batch["aligned_len"])
            U = len(span_b)
            if U == 0:
                continue
            ids1 = batch["input_ids1"].long()
            fwd1(input_ids=ids1, attention_mask=batch["attention_mask1"])
            actsA = capA.get_and_clear(expected_batch=int(ids1.shape[0]), expected_seq=int(ids1.shape[1]))
            ids2 = batch["input_ids2"].long()
            fwd2(input_ids=ids2, attention_mask=batch["attention_mask2"])
            actsB = capB.get_and_clear(expected_batch=int(ids2.shape[0]), expected_seq=int(ids2.shape[1]))
            fa = _precompute_span_flat_indices(span_b, span_a_tok, U, cpu)
            fb = _precompute_span_flat_indices(span_b, span_b_tok, U, cpu)
            xa = torch.cat([_aggregate_spans_vectorized(a, *fa, U, "mean", compute_dtype=compute).sub_(mu).mul_(iv)
                            for a, mu, iv in zip(actsA, meansA, invA)], dim=1).float()
            xb = torch.cat([_aggregate_spans_vectorized(b, *fb, U, "mean", compute_dtype=compute).sub_(mu).mul_(iv)
                            for b, mu, iv in zip(actsB, meansB, invB)], dim=1).float()
            for s in range(0, nA, args.a_chunk):
                p = xa[:, s:s + args.a_chunk].T @ xb
                acc32[s:s + args.a_chunk] += p
                acc16[s:s + args.a_chunk] = (acc16[s:s + args.a_chunk].float() + p).to(compute)
            n_spans += U
            n_batches += 1
            if bi % 20 == 0:
                rate = (time.time() - t0) / (bi + 1)
                print(f"[diag] batch {bi + 1}/{len(loader)} spans {n_spans:,} {rate:.1f} s/batch "
                      f"eta {rate * (len(loader) - bi - 1) / 3600:.1f} h", flush=True)

    result = {"run_dir": str(args.run_dir), "models": [model1, model2], "batches": n_batches, "spans": n_spans,
              "forward_dtype": args.forward_dtype, "elapsed_s": round(time.time() - t0),
              "gpu_counts": {}, "cpu_bf16_accum": {}, "cpu_fp32_accum": {}}
    for k in args.ks:
        f = corr_dir / ("best_buddies.json" if k == int(knobs.get("K_NEIGHBORS") or 1) else f"best_buddies_k{k}.json")
        if f.exists():
            result["gpu_counts"][k] = len(json.loads(f.read_text()))
        result["cpu_bf16_accum"][k] = mutual_topk(acc16, k)  # top-k on the bf16 sums, as upstream
        result["cpu_fp32_accum"][k] = mutual_topk(acc32, k)
        print(f"[diag] k={k}: GPU {result['gpu_counts'].get(k)}  CPU bf16-accum {result['cpu_bf16_accum'][k]}  "
              f"CPU fp32-accum {result['cpu_fp32_accum'][k]}", flush=True)
    top1_16, top1_32 = acc16.argmax(1), acc32.argmax(1)
    result["top1_agreement_bf16_vs_fp32"] = float((top1_16 == top1_32).float().mean())
    out = args.out or args.run_dir / "cpu_accum_dtype_diagnostic.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
