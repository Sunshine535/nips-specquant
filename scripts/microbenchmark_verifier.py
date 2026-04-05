"""Verifier microbenchmark: HBM traffic, kernel latency at various context lengths.

Phase 4 experiment: measures the raw cost of verification with and without
SpecQuant KV compression, RTN, and KIVI baselines.
"""

import argparse
import json
import logging
import os
import time

import torch

from src.turboquant_kv import QuantizedKVCache, HadamardRotation, ScalarQuantizer
from src.baselines import RTNKVCache, KIVIKVCache
from src.utils import aggregate_trials, save_results, format_with_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model architecture configs: (num_heads, num_kv_heads, head_dim)
# ---------------------------------------------------------------------------
ARCH_CONFIGS = {
    # Qwen3.5 family
    "Qwen/Qwen3.5-0.8B":  (16, 4, 64),
    "Qwen/Qwen3.5-4B":    (32, 8, 128),
    "Qwen/Qwen3.5-9B":    (32, 8, 128),
    "Qwen/Qwen3.5-14B":   (40, 8, 128),
    # Llama-3.1 family
    "Meta-Llama/Llama-3.1-8B":  (32, 8, 128),
    "Meta-Llama/Llama-3.1-70B": (64, 8, 128),
}

# Peak HBM bandwidth (GB/s) for common GPUs — used for roofline estimation
GPU_PEAK_BW = {
    "A100":     2039.0,
    "A100-80":  2039.0,
    "H100":     3352.0,
    "L40S":     864.0,
    "RTX 4090": 1008.0,
}


def _detect_gpu_peak_bw(device: torch.device) -> float:
    """Best-effort detection of peak HBM bandwidth for the active GPU.

    Falls back to A100 bandwidth when the GPU name is not recognised.
    """
    if device.type != "cuda":
        return 0.0
    name = torch.cuda.get_device_properties(device).name
    for key, bw in GPU_PEAK_BW.items():
        if key.lower().replace(" ", "") in name.lower().replace(" ", ""):
            return bw
    logger.warning("Unknown GPU '%s'; assuming A100 peak BW for roofline", name)
    return GPU_PEAK_BW["A100"]


def _run_fp16_attention(q, k, v, num_heads, num_kv_heads, head_dim):
    """Single forward pass of FP16 GQA attention (no quantization)."""
    if num_heads != num_kv_heads:
        k_exp = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_exp = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
    else:
        k_exp, v_exp = k, v
    scores = torch.matmul(q, k_exp.transpose(-2, -1)) / (head_dim ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, v_exp)


def benchmark_attention_kernel(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    bits: int,
    method: str = "specquant",
    block_size: int = 128,
    num_trials: int = 10,
    warmup: int = 3,
    device: str = "cuda:1",
) -> dict:
    """Benchmark attention computation across quantization methods.

    Args:
        method: one of "fp16", "specquant", "rtn", "kivi".
                When bits==0 *method* is forced to "fp16".
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    q = torch.randn(batch_size, num_heads, 1, head_dim, device=dev, dtype=torch.float16)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=dev, dtype=torch.float16)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=dev, dtype=torch.float16)

    if bits == 0 or method == "fp16":
        # --- FP16 baseline (no quantization) ---------------------------------
        for _ in range(warmup):
            _run_fp16_attention(q, k, v, num_heads, num_kv_heads, head_dim)
            torch.cuda.synchronize(dev)

        times = []
        for _ in range(num_trials):
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            _run_fp16_attention(q, k, v, num_heads, num_kv_heads, head_dim)
            torch.cuda.synchronize(dev)
            times.append(time.perf_counter() - t0)

        kv_bytes = 2 * num_kv_heads * seq_len * head_dim * 2  # K + V in fp16
        actual_method = "fp16"
    else:
        # --- Quantized cache (SpecQuant / RTN / KIVI) ------------------------
        cache_cls = {
            "specquant": QuantizedKVCache,
            "rtn":       RTNKVCache,
            "kivi":      KIVIKVCache,
        }[method]
        qkv_cache = cache_cls(
            num_layers=1, num_kv_heads=num_kv_heads,
            head_dim=head_dim, bits=bits, block_size=block_size,
        )
        qkv_cache.compress_and_store(0, k, v)

        for _ in range(warmup):
            qkv_cache.compressed_attention(0, q)
            torch.cuda.synchronize(dev)

        times = []
        for _ in range(num_trials):
            torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
            qkv_cache.compressed_attention(0, q)
            torch.cuda.synchronize(dev)
            times.append(time.perf_counter() - t0)

        kv_bytes = qkv_cache.memory_bytes()
        actual_method = method

    # --- Aggregate with CI ---------------------------------------------------
    latency_ms = [t * 1000 for t in times]
    trial_stats = aggregate_trials(latency_ms)

    fp_bytes = 2 * num_kv_heads * seq_len * head_dim * 2
    compression = fp_bytes / kv_bytes if kv_bytes > 0 else 1.0
    mean_time = trial_stats["mean"] / 1000.0  # back to seconds for BW calc

    # HBM traffic estimation: read KV + write output per verification step.
    # Q is tiny (1 token), so traffic is dominated by KV read.
    # Output: batch * num_heads * 1 * head_dim * 2 bytes (fp16)
    output_bytes = batch_size * num_heads * 1 * head_dim * 2
    hbm_read_bytes = kv_bytes        # KV cache read
    hbm_write_bytes = output_bytes   # attention output write
    hbm_total_bytes = hbm_read_bytes + hbm_write_bytes

    achieved_bw = hbm_total_bytes / mean_time / 1e9 if mean_time > 0 else 0.0
    peak_bw = _detect_gpu_peak_bw(dev)
    hw_utilization = achieved_bw / peak_bw if peak_bw > 0 else 0.0

    return {
        "method": actual_method,
        "bits": bits,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "mean_latency_ms": trial_stats["mean"],
        "std_latency_ms": trial_stats["std"],
        "ci_lower_ms": trial_stats["ci_lower"],
        "ci_upper_ms": trial_stats["ci_upper"],
        "n_trials": trial_stats["n_trials"],
        "kv_memory_bytes": kv_bytes,
        "fp_memory_bytes": fp_bytes,
        "compression_ratio": compression,
        "hbm_read_bytes": hbm_read_bytes,
        "hbm_write_bytes": hbm_write_bytes,
        "hbm_total_bytes": hbm_total_bytes,
        "achieved_bandwidth_gbps": achieved_bw,
        "peak_bandwidth_gbps": peak_bw,
        "hw_utilization": hw_utilization,
    }


def main():
    parser = argparse.ArgumentParser(description="Verifier Microbenchmark")
    parser.add_argument("--target-model", type=str, default="Qwen/Qwen3.5-14B")
    parser.add_argument("--seq-lengths", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384])
    parser.add_argument("--bits", type=int, nargs="+", default=[0, 3, 4])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["fp16", "specquant", "rtn", "kivi"],
                        help="Quantization methods to benchmark")
    parser.add_argument("--num-trials", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda:1",
                        help="CUDA device (default cuda:1, target model GPU)")
    parser.add_argument("--output-dir", type=str, default="results/microbenchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Resolve architecture config --------------------------------------
    if args.target_model in ARCH_CONFIGS:
        num_heads, num_kv_heads, head_dim = ARCH_CONFIGS[args.target_model]
    else:
        # Fuzzy fallback: match the size suffix (e.g. "14B") in known keys
        matched = False
        for key, cfg in ARCH_CONFIGS.items():
            if key.split("-")[-1] in args.target_model:
                num_heads, num_kv_heads, head_dim = cfg
                logger.warning(
                    "No exact config for '%s'; using %s config (%d/%d/%d)",
                    args.target_model, key, *cfg,
                )
                matched = True
                break
        if not matched:
            num_heads, num_kv_heads, head_dim = 40, 8, 128
            logger.warning(
                "Unknown model '%s'; falling back to default config "
                "(num_heads=40, num_kv_heads=8, head_dim=128)",
                args.target_model,
            )

    logger.info(
        "Architecture: %s  (num_heads=%d, num_kv_heads=%d, head_dim=%d)",
        args.target_model, num_heads, num_kv_heads, head_dim,
    )

    all_results = {"config": vars(args), "results": []}

    for seq_len in args.seq_lengths:
        for method in args.methods:
            bits_list = [0] if method == "fp16" else args.bits
            # Skip bits==0 for quantized methods (handled by fp16 method)
            bits_list = [b for b in bits_list if not (b == 0 and method != "fp16")]

            for bits in bits_list:
                label = f"{method}_{bits}bit_seq{seq_len}" if bits > 0 else f"fp16_seq{seq_len}"
                logger.info("Running %s ...", label)

                result = benchmark_attention_kernel(
                    batch_size=1,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    seq_len=seq_len,
                    head_dim=head_dim,
                    bits=bits,
                    method=method,
                    num_trials=args.num_trials,
                    device=args.device,
                )
                all_results["results"].append(result)

                ci_str = format_with_ci(
                    result["mean_latency_ms"],
                    result["ci_lower_ms"],
                    result["ci_upper_ms"],
                    fmt=".3f",
                )
                logger.info(
                    "  %s: %s ms, compression=%.1fx, "
                    "HBM=%.2f MB, achieved BW=%.1f GB/s (%.1f%% util)",
                    label,
                    ci_str,
                    result["compression_ratio"],
                    result["hbm_total_bytes"] / 1e6,
                    result["achieved_bandwidth_gbps"],
                    result["hw_utilization"] * 100,
                )

    # --- Save with atomic writer ------------------------------------------
    filename = f"microbenchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    save_results(all_results, args.output_dir, filename)
    logger.info("Done. %d configurations benchmarked.", len(all_results["results"]))


if __name__ == "__main__":
    main()
