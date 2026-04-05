"""Benchmark SpecQuant: compare vanilla spec decode, SpecQuant, and baselines.

Primary evaluation script for Claim 1 and Claim 2.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.baselines import BaselineDecoder
from src.speculative_decode import SpeculativeDecoder
from src.turboquant_kv import QuantizedKVCache, compute_tv_bound
from src.utils import aggregate_trials, mean_confidence_interval, validate_dual_gpu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

PROMPTS = {
    "reasoning": [
        "Solve the following math problem step by step: What is the sum of the first 100 prime numbers?",
        "A train travels at 60 mph for the first half of the journey and 40 mph for the second half. What is the average speed for the entire journey?",
        "If 3 machines take 3 minutes to make 3 widgets, how many minutes would 100 machines take to make 100 widgets?",
    ],
    "code": [
        "Write a Python function to find the longest common subsequence of two strings. Include type hints and docstring.",
        "Implement a binary search tree in Python with insert, delete, and search operations.",
        "Write a Python function that checks if a given graph is bipartite using BFS.",
    ],
    "long_context": [
        "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves. " * 20,
    ],
}


def load_models(
    draft_model_name: str,
    target_model_name: str,
    draft_device: str = "cuda:0",
    target_device: str = "cuda:1",
    dtype: torch.dtype = torch.float16,
) -> tuple:
    logger.info(f"Loading draft model: {draft_model_name} on {draft_device}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=dtype,
        device_map=draft_device,
        trust_remote_code=True,
    )

    logger.info(f"Loading target model: {target_model_name} on {target_device}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=dtype,
        device_map=target_device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return draft_model, target_model, tokenizer


def run_benchmark(
    decoder: SpeculativeDecoder,
    prompts: List[str],
    tokenizer,
    max_new_tokens: int = 128,
    gamma: int = 5,
    temperature: float = 1.0,
    num_warmup: int = 1,
    num_trials: int = 3,
) -> Dict:
    per_prompt_results = []
    all_throughputs = []
    all_acceptance_rates = []
    all_wall_times = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        for _ in range(num_warmup):
            decoder.generate(
                input_ids, max_new_tokens=min(16, max_new_tokens), gamma=gamma
            )

        trial_results = []
        for trial in range(num_trials):
            torch.cuda.synchronize()
            out = decoder.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                gamma=gamma,
                temperature=temperature,
            )
            torch.cuda.synchronize()
            trial_results.append(out.to_dict())

        prompt_throughputs = [t["throughput_tokens_per_sec"] for t in trial_results]
        prompt_acceptance = [t["acceptance_rate"] for t in trial_results]
        prompt_wall_times = [t["wall_time_seconds"] for t in trial_results]

        all_throughputs.extend(prompt_throughputs)
        all_acceptance_rates.extend(prompt_acceptance)
        all_wall_times.extend(prompt_wall_times)

        per_prompt_results.append({
            "prompt_len": input_ids.shape[1],
            "num_trials": num_trials,
            "throughput": aggregate_trials(prompt_throughputs),
            "acceptance_rate": aggregate_trials(prompt_acceptance),
            "wall_time": aggregate_trials(prompt_wall_times),
        })

    tp_mean, tp_ci_lo, tp_ci_hi = mean_confidence_interval(all_throughputs)
    acc_mean, acc_ci_lo, acc_ci_hi = mean_confidence_interval(all_acceptance_rates)
    wt_mean, wt_ci_lo, wt_ci_hi = mean_confidence_interval(all_wall_times)

    return {
        "mean_throughput": tp_mean,
        "throughput_ci": [tp_ci_lo, tp_ci_hi],
        "mean_acceptance_rate": acc_mean,
        "acceptance_rate_ci": [acc_ci_lo, acc_ci_hi],
        "mean_wall_time": wt_mean,
        "wall_time_ci": [wt_ci_lo, wt_ci_hi],
        "per_prompt": per_prompt_results,
    }


def run_autoregressive_baseline(
    decoder: SpeculativeDecoder,
    prompts: List[str],
    tokenizer,
    max_new_tokens: int = 128,
    num_trials: int = 3,
) -> Dict:
    throughputs = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        for _ in range(num_trials):
            torch.cuda.synchronize()
            _, wall = decoder.generate_autoregressive(
                input_ids, max_new_tokens=max_new_tokens
            )
            torch.cuda.synchronize()
            throughputs.append(max_new_tokens / wall)

    return {
        "mean_throughput": sum(throughputs) / len(throughputs),
        "num_samples": len(throughputs),
    }


def main():
    parser = argparse.ArgumentParser(description="SpecQuant Benchmark")
    parser.add_argument("--draft-model", type=str,
                        default=os.environ.get("QWEN35_0_8B", "Qwen/Qwen3.5-0.8B"))
    parser.add_argument("--target-model", type=str,
                        default=os.environ.get("QWEN35_9B", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--quant-bits", type=int, nargs="+", default=[0, 3, 4])
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--num-trials", type=int, default=3)
    parser.add_argument("--num-warmup", type=int, default=1)
    parser.add_argument("--prompt-type", type=str, default="reasoning",
                        choices=["reasoning", "code", "long_context", "all"])
    parser.add_argument("--output-dir", type=str, default="results/benchmark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draft-device", type=str, default="cuda:0",
                        help="Device for draft model (default: cuda:0)")
    parser.add_argument("--target-device", type=str, default="cuda:1",
                        help="Device for target model (default: cuda:1)")
    parser.add_argument("--baselines", type=str, nargs="*",
                        default=["rtn", "kivi", "absmax"],
                        help="Baseline methods to compare (default: rtn kivi absmax)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    validate_dual_gpu()

    draft_model, target_model, tokenizer = load_models(
        args.draft_model, args.target_model,
        draft_device=args.draft_device,
        target_device=args.target_device,
    )

    if args.prompt_type == "all":
        prompts = []
        for v in PROMPTS.values():
            prompts.extend(v)
    else:
        prompts = PROMPTS[args.prompt_type]

    all_results = {
        "config": vars(args),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "methods": {},
    }

    logger.info("=== Autoregressive baseline ===")
    decoder = SpeculativeDecoder(
        draft_model, target_model, tokenizer, quant_bits=0
    )
    ar_result = run_autoregressive_baseline(
        decoder, prompts, tokenizer,
        max_new_tokens=args.max_new_tokens,
        num_trials=args.num_trials,
    )
    all_results["methods"]["autoregressive"] = ar_result
    logger.info(f"  Throughput: {ar_result['mean_throughput']:.1f} tok/s")

    for bits in args.quant_bits:
        label = f"specquant_{bits}bit" if bits > 0 else "vanilla_spec"
        logger.info(f"=== {label} (gamma={args.gamma}) ===")

        decoder = SpeculativeDecoder(
            draft_model, target_model, tokenizer,
            quant_bits=bits,
            quant_block_size=args.block_size,
        )
        result = run_benchmark(
            decoder, prompts, tokenizer,
            max_new_tokens=args.max_new_tokens,
            gamma=args.gamma,
            temperature=args.temperature,
            num_warmup=args.num_warmup,
            num_trials=args.num_trials,
        )
        all_results["methods"][label] = result
        logger.info(
            f"  Throughput: {result['mean_throughput']:.1f} tok/s, "
            f"Acceptance: {result['mean_acceptance_rate']:.3f}"
        )

    # --- Baseline comparisons ---
    if args.baselines:
        for baseline_name in args.baselines:
            label = f"baseline_{baseline_name}"
            logger.info(f"=== {label} (gamma={args.gamma}) ===")

            baseline_decoder = BaselineDecoder(
                draft_model, target_model, tokenizer,
                baseline_type=baseline_name,
                quant_bits=args.quant_bits[0] if args.quant_bits else 3,
                quant_block_size=args.block_size,
            )
            result = run_benchmark(
                baseline_decoder, prompts, tokenizer,
                max_new_tokens=args.max_new_tokens,
                gamma=args.gamma,
                temperature=args.temperature,
                num_warmup=args.num_warmup,
                num_trials=args.num_trials,
            )
            all_results["methods"][label] = result
            logger.info(
                f"  Throughput: {result['mean_throughput']:.1f} tok/s, "
                f"Acceptance: {result['mean_acceptance_rate']:.3f}"
            )

    output_file = os.path.join(
        args.output_dir,
        f"benchmark_{Path(args.target_model).name}_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    # --- Formatted summary table ---
    logger.info("\n" + "=" * 90)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 90)
    header = f"{'Method':<25} {'Throughput (tok/s)':<25} {'Acceptance Rate':<25} {'Speedup':<10}"
    logger.info(header)
    logger.info("-" * 90)

    ar_tp = all_results["methods"]["autoregressive"]["mean_throughput"]
    logger.info(
        f"{'autoregressive':<25} {ar_tp:>8.1f}{'':<17} {'N/A':<25} {'1.00x':<10}"
    )

    for method, res in all_results["methods"].items():
        if method == "autoregressive":
            continue
        tp = res["mean_throughput"]
        speedup = tp / ar_tp if ar_tp > 0 else 0

        tp_ci = res.get("throughput_ci")
        if tp_ci:
            tp_str = f"{tp:>8.1f} [{tp_ci[0]:.1f}, {tp_ci[1]:.1f}]"
        else:
            tp_str = f"{tp:>8.1f}"

        acc = res.get("mean_acceptance_rate")
        acc_ci = res.get("acceptance_rate_ci")
        if acc is not None and acc_ci:
            acc_str = f"{acc:>8.3f} [{acc_ci[0]:.3f}, {acc_ci[1]:.3f}]"
        elif acc is not None:
            acc_str = f"{acc:>8.3f}"
        else:
            acc_str = "N/A"

        logger.info(f"{method:<25} {tp_str:<25} {acc_str:<25} {speedup:.2f}x")

    logger.info("=" * 90)


if __name__ == "__main__":
    main()
