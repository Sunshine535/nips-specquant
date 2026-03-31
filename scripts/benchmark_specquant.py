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

from src.speculative_decode import SpeculativeDecoder
from src.turboquant_kv import QuantizedKVCache, compute_tv_bound

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
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple:
    logger.info(f"Loading draft model: {draft_model_name}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    logger.info(f"Loading target model: {target_model_name}")
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=dtype,
        device_map=device,
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
    results = []

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

        avg_result = {}
        for key in trial_results[0]:
            if isinstance(trial_results[0][key], (int, float)):
                avg_result[key] = sum(t[key] for t in trial_results) / len(trial_results)
            elif isinstance(trial_results[0][key], list):
                avg_result[key] = [
                    sum(t[key][i] for t in trial_results) / len(trial_results)
                    for i in range(len(trial_results[0][key]))
                ]
        avg_result["prompt_len"] = input_ids.shape[1]
        avg_result["num_trials"] = num_trials
        results.append(avg_result)

    return {
        "mean_throughput": sum(r["throughput_tokens_per_sec"] for r in results) / len(results),
        "mean_acceptance_rate": sum(r["acceptance_rate"] for r in results) / len(results),
        "mean_wall_time": sum(r["wall_time_seconds"] for r in results) / len(results),
        "per_prompt": results,
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
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    draft_model, target_model, tokenizer = load_models(
        args.draft_model, args.target_model, device=args.device
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

    output_file = os.path.join(
        args.output_dir,
        f"benchmark_{Path(args.target_model).name}_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {output_file}")

    logger.info("\n=== Summary ===")
    ar_tp = all_results["methods"]["autoregressive"]["mean_throughput"]
    for method, res in all_results["methods"].items():
        if method == "autoregressive":
            continue
        tp = res["mean_throughput"]
        acc = res.get("mean_acceptance_rate", "N/A")
        speedup = tp / ar_tp if ar_tp > 0 else 0
        logger.info(f"  {method}: {tp:.1f} tok/s ({speedup:.2f}x), acc={acc}")


if __name__ == "__main__":
    main()
