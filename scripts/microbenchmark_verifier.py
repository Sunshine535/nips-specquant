"""Verifier microbenchmark: HBM traffic, kernel latency at various context lengths.

Phase 4 experiment: measures the raw cost of verification with and without
SpecQuant KV compression.
"""

import argparse
import json
import logging
import os
import time

import torch

from src.turboquant_kv import QuantizedKVCache, HadamardRotation, ScalarQuantizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark_attention_kernel(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    bits: int,
    block_size: int = 128,
    num_trials: int = 10,
    warmup: int = 3,
) -> dict:
    """Benchmark attention computation with and without quantization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=torch.float16)

    if bits == 0:
        for _ in range(warmup):
            if num_heads != num_kv_heads:
                k_exp = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
                v_exp = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            else:
                k_exp, v_exp = k, v
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_exp)
            torch.cuda.synchronize()

        times = []
        for _ in range(num_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if num_heads != num_kv_heads:
                k_exp = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
                v_exp = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
            else:
                k_exp, v_exp = k, v
            scores = torch.matmul(q, k_exp.transpose(-2, -1)) / (head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_exp)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        kv_bytes = 2 * num_kv_heads * seq_len * head_dim * 2
    else:
        qkv_cache = QuantizedKVCache(
            num_layers=1, num_kv_heads=num_kv_heads,
            head_dim=head_dim, bits=bits, block_size=block_size,
        )
        qkv_cache.compress_and_store(0, k, v)

        for _ in range(warmup):
            out = qkv_cache.compressed_attention(0, q)
            torch.cuda.synchronize()

        times = []
        for _ in range(num_trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = qkv_cache.compressed_attention(0, q)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        kv_bytes = qkv_cache.memory_bytes()

    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time)**2 for t in times) / len(times)) ** 0.5

    fp_bytes = 2 * num_kv_heads * seq_len * head_dim * 2
    compression = fp_bytes / kv_bytes if kv_bytes > 0 else 1.0

    return {
        "bits": bits,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "mean_latency_ms": mean_time * 1000,
        "std_latency_ms": std_time * 1000,
        "kv_memory_bytes": kv_bytes,
        "fp_memory_bytes": fp_bytes,
        "compression_ratio": compression,
        "bandwidth_gbps": kv_bytes / mean_time / 1e9 if mean_time > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Verifier Microbenchmark")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3.5-14B")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--bits", type=int, nargs="+", default=[0, 3, 4])
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="results/microbenchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    num_heads = 40
    num_kv_heads = 8
    head_dim = 128

    if "14B" in args.target_model:
        num_heads = 40
        num_kv_heads = 8
        head_dim = 128
    elif "9B" in args.target_model:
        num_heads = 32
        num_kv_heads = 8
        head_dim = 128

    all_results = {"config": vars(args), "results": []}

    for seq_len in args.seq_lengths:
        for bits in args.bits:
            label = f"{'fp16' if bits == 0 else f'{bits}bit'}_seq{seq_len}"
            logger.info(f"Running {label}...")

            result = benchmark_attention_kernel(
                batch_size=1,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                bits=bits,
                num_trials=args.num_trials,
            )
            all_results["results"].append(result)

            logger.info(
                f"  {label}: {result['mean_latency_ms']:.2f}ms, "
                f"compression={result['compression_ratio']:.1f}x, "
                f"BW={result['bandwidth_gbps']:.1f} GB/s"
            )

    output_file = os.path.join(
        args.output_dir,
        f"microbenchmark_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
