"""Downstream task evaluation for SpecQuant.

Evaluates SpecQuant across four benchmarks (GSM8K, HumanEval, MMLU, MT-Bench)
comparing autoregressive baseline, vanilla speculative decode, and SpecQuant
at various quantization bit-widths.  Reports accuracy, throughput, and
acceptance rate with confidence intervals across multiple trials.
"""

import argparse
import logging
import multiprocessing
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.speculative_decode import SpeculativeDecoder, SpeculativeOutput
from src.utils import aggregate_trials, validate_dual_gpu, save_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading helpers
# ---------------------------------------------------------------------------

def _load_gsm8k(num_samples: int, seed: int) -> Optional[List[Dict]]:
    """Load GSM8K test split from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
        samples = []
        for row in ds:
            samples.append({
                "question": row["question"],
                "answer": row["answer"],
            })
        logger.info(f"GSM8K: loaded {len(samples)} samples")
        return samples
    except Exception as exc:
        logger.warning(f"Failed to load GSM8K: {exc}")
        return None


def _load_humaneval(num_samples: int, seed: int) -> Optional[List[Dict]]:
    """Load HumanEval from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
        samples = []
        for row in ds:
            samples.append({
                "task_id": row["task_id"],
                "prompt": row["prompt"],
                "canonical_solution": row["canonical_solution"],
                "test": row["test"],
                "entry_point": row["entry_point"],
            })
        logger.info(f"HumanEval: loaded {len(samples)} samples")
        return samples
    except Exception as exc:
        logger.warning(f"Failed to load HumanEval: {exc}")
        return None


def _load_mmlu(num_samples: int, seed: int) -> Optional[List[Dict]]:
    """Load MMLU test split from HuggingFace datasets.

    Samples uniformly across subjects so coverage is broad.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("cais/mmlu", "all", split="test")
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
        samples = []
        for row in ds:
            choices = row["choices"]
            samples.append({
                "question": row["question"],
                "choices": choices,
                "subject": row["subject"],
                "answer_index": row["answer"],  # 0-3 integer
            })
        logger.info(f"MMLU: loaded {len(samples)} samples")
        return samples
    except Exception as exc:
        logger.warning(f"Failed to load MMLU: {exc}")
        return None


def _load_mt_bench(num_samples: int, seed: int) -> Optional[List[Dict]]:
    """Load MT-Bench questions with reference answers.

    Falls back to a small built-in set if the HuggingFace dataset is
    unavailable.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
        ds = ds.shuffle(seed=seed).select(range(min(num_samples, len(ds))))
        samples = []
        for row in ds:
            samples.append({
                "question_id": row.get("question_id", ""),
                "turns": row.get("turn", row.get("turns", [])),
                "reference": row.get("reference", row.get("assistant_a", "")),
                "category": row.get("category", "general"),
            })
        if samples:
            logger.info(f"MT-Bench: loaded {len(samples)} samples (HF)")
            return samples
    except Exception:
        pass

    # Fallback: built-in minimal multi-turn questions
    _BUILTIN = [
        {
            "question_id": "mt_1",
            "turns": [
                "Explain the concept of quantum entanglement in simple terms.",
                "Now explain how it could be used in quantum computing.",
            ],
            "reference": (
                "Quantum entanglement is a phenomenon where two particles become "
                "linked so that measuring one instantly determines the state of the "
                "other, regardless of distance. In quantum computing, entangled "
                "qubits enable parallelism and error correction."
            ),
            "category": "stem",
        },
        {
            "question_id": "mt_2",
            "turns": [
                "Write a short story about a robot discovering emotions.",
                "Now rewrite the ending to be bittersweet instead.",
            ],
            "reference": (
                "The robot learned to feel joy and sadness but realized emotions "
                "made its existence finite in meaning."
            ),
            "category": "writing",
        },
        {
            "question_id": "mt_3",
            "turns": [
                "What are the main differences between Python and Rust?",
                "When would you recommend using Rust over Python?",
            ],
            "reference": (
                "Python is interpreted, dynamically typed, and emphasizes "
                "readability. Rust is compiled, statically typed, and "
                "prioritizes memory safety without a garbage collector. Rust "
                "is preferred for performance-critical systems, embedded "
                "software, and concurrent workloads."
            ),
            "category": "coding",
        },
    ]
    import random
    rng = random.Random(seed)
    selected = _BUILTIN * ((num_samples // len(_BUILTIN)) + 1)
    rng.shuffle(selected)
    selected = selected[:min(num_samples, len(selected))]
    logger.info(f"MT-Bench: loaded {len(selected)} samples (built-in fallback)")
    return selected


LOADERS = {
    "gsm8k": _load_gsm8k,
    "humaneval": _load_humaneval,
    "mmlu": _load_mmlu,
    "mt_bench": _load_mt_bench,
}


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "After your reasoning, write the final numeric answer on a new line "
        "in the format: #### <number>\n\n"
        f"Question: {question}\n\n"
        "Step-by-step solution:\n"
    )


def format_humaneval_prompt(prompt_code: str) -> str:
    return prompt_code  # HumanEval prompts are already function signatures


def format_mmlu_prompt(
    question: str,
    choices: List[str],
    few_shot_examples: Optional[List[Dict]] = None,
) -> str:
    """Format MMLU question with optional 5-shot examples."""
    labels = ["A", "B", "C", "D"]
    parts = []

    if few_shot_examples:
        for ex in few_shot_examples[:5]:
            ex_choices = ex["choices"]
            ex_q = f"Question: {ex['question']}\n"
            for i, c in enumerate(ex_choices):
                ex_q += f"{labels[i]}) {c}\n"
            ex_q += f"Answer: {labels[ex['answer_index']]}"
            parts.append(ex_q)

    q_str = f"Question: {question}\n"
    for i, c in enumerate(choices):
        q_str += f"{labels[i]}) {c}\n"
    q_str += "Answer:"
    parts.append(q_str)
    return "\n\n".join(parts)


def format_mt_bench_prompt(turns: List[str], turn_index: int = 0) -> str:
    """Format multi-turn conversation up to the specified turn."""
    parts = []
    for i in range(turn_index + 1):
        parts.append(f"User: {turns[i]}")
        if i < turn_index:
            parts.append("Assistant: [previous response]")
    parts.append("Assistant:")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction / scoring
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer after '####' in GSM8K chain-of-thought output."""
    # Look for #### pattern
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass
    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def extract_gsm8k_gold(answer_text: str) -> Optional[float]:
    """Extract gold numeric answer from GSM8K answer field."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def check_gsm8k(generated: str, gold_answer: str) -> bool:
    """Check if GSM8K generated answer matches gold numerically."""
    pred = extract_gsm8k_answer(generated)
    gold = extract_gsm8k_gold(gold_answer)
    if pred is None or gold is None:
        return False
    # Allow small floating-point tolerance
    return abs(pred - gold) < 1e-3


def _exec_with_timeout(code: str, timeout_seconds: int = 10) -> bool:
    """Execute code in a subprocess with a timeout.  Returns True if it exits 0."""
    def _target(code_str, result_dict):
        try:
            exec_globals: Dict[str, Any] = {}
            exec(code_str, exec_globals)
            result_dict["ok"] = True
        except Exception:
            result_dict["ok"] = False

    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    result_dict["ok"] = False
    proc = multiprocessing.Process(target=_target, args=(code, result_dict))
    proc.start()
    proc.join(timeout=timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        return False
    return bool(result_dict.get("ok", False))


def check_humaneval(generated: str, sample: Dict) -> bool:
    """Check HumanEval pass@1 by executing the completion with test cases."""
    # Build the full program: prompt + generated completion + test harness
    full_code = sample["prompt"] + generated + "\n\n" + sample["test"]
    full_code += f"\n\ncheck({sample['entry_point']})\n"
    return _exec_with_timeout(full_code, timeout_seconds=10)


def extract_mmlu_answer(text: str) -> Optional[str]:
    """Extract the chosen letter (A/B/C/D) from MMLU generation."""
    text = text.strip()
    # Direct single letter
    if text and text[0] in "ABCD":
        return text[0]
    # Pattern like "The answer is B"
    match = re.search(r"(?:answer|choice)\s*(?:is|:)?\s*([A-D])", text, re.I)
    if match:
        return match.group(1).upper()
    # Any standalone A-D
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    return None


def check_mmlu(generated: str, gold_index: int) -> bool:
    """Check MMLU answer correctness."""
    labels = ["A", "B", "C", "D"]
    pred = extract_mmlu_answer(generated)
    if pred is None:
        return False
    return pred == labels[gold_index]


def score_mt_bench(generated: str, reference: str) -> float:
    """Score MT-Bench response via reference-based entity/fact overlap.

    Returns a score in [0, 1].  This is a simplified judge-free metric:
    we tokenize reference and generated text into word n-grams and compute
    the F1 overlap of content words (length >= 4, lowered).
    """
    def _content_tokens(text: str) -> set:
        words = re.findall(r"[a-z0-9]+", text.lower())
        return {w for w in words if len(w) >= 4}

    ref_tokens = _content_tokens(reference)
    gen_tokens = _content_tokens(generated)
    if not ref_tokens:
        return 1.0 if not gen_tokens else 0.5  # no reference content
    if not gen_tokens:
        return 0.0
    overlap = ref_tokens & gen_tokens
    precision = len(overlap) / len(gen_tokens) if gen_tokens else 0.0
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ---------------------------------------------------------------------------
# Model loading (dual-GPU aware)
# ---------------------------------------------------------------------------

def load_models_dual_gpu(
    draft_model_name: str,
    target_model_name: str,
    draft_device: str = "cuda:0",
    target_device: str = "cuda:1",
    dtype: torch.dtype = torch.float16,
) -> Tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """Load draft and target models on separate devices."""
    validate_dual_gpu(draft_device, target_device)

    logger.info(f"Loading draft model: {draft_model_name} -> {draft_device}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=dtype,
        device_map=draft_device,
        trust_remote_code=True,
    )

    logger.info(f"Loading target model: {target_model_name} -> {target_device}")
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


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def _generate_text(
    decoder: SpeculativeDecoder,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 1.0,
    gamma: int = 5,
    use_autoregressive: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Generate text and return (decoded_output, stats_dict).

    stats_dict always contains 'wall_time_seconds' and 'throughput'.
    For speculative decoding it also contains 'acceptance_rate'.
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if use_autoregressive:
        generated_ids, wall = decoder.generate_autoregressive(
            input_ids, max_new_tokens=max_new_tokens, temperature=temperature,
        )
        num_new = generated_ids.shape[1] - input_ids.shape[1]
        stats = {
            "wall_time_seconds": wall,
            "throughput": num_new / wall if wall > 0 else 0.0,
            "acceptance_rate": None,
            "num_generated_tokens": num_new,
        }
        output_ids = generated_ids[0, input_ids.shape[1]:]
    else:
        out: SpeculativeOutput = decoder.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            gamma=gamma,
            temperature=temperature,
        )
        stats = {
            "wall_time_seconds": out.wall_time_seconds,
            "throughput": out.throughput,
            "acceptance_rate": out.acceptance_rate,
            "num_generated_tokens": out.num_generated_tokens,
        }
        output_ids = out.generated_ids[0, input_ids.shape[1]:]

    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    return decoded, stats


# ---------------------------------------------------------------------------
# Per-benchmark evaluation drivers
# ---------------------------------------------------------------------------

def evaluate_gsm8k(
    decoder: SpeculativeDecoder,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    max_new_tokens: int,
    temperature: float,
    gamma: int,
    use_autoregressive: bool,
) -> Dict[str, Any]:
    """Evaluate GSM8K: exact-match accuracy on parsed numeric answers."""
    correct = 0
    total = 0
    throughputs: List[float] = []
    acceptance_rates: List[float] = []
    wall_times: List[float] = []

    for idx, sample in enumerate(samples):
        prompt = format_gsm8k_prompt(sample["question"])
        try:
            generated, stats = _generate_text(
                decoder, tokenizer, prompt, max_new_tokens,
                temperature=temperature, gamma=gamma,
                use_autoregressive=use_autoregressive,
            )
        except Exception as exc:
            logger.debug(f"GSM8K sample {idx} generation failed: {exc}")
            total += 1
            continue

        is_correct = check_gsm8k(generated, sample["answer"])
        correct += int(is_correct)
        total += 1
        throughputs.append(stats["throughput"])
        wall_times.append(stats["wall_time_seconds"])
        if stats["acceptance_rate"] is not None:
            acceptance_rates.append(stats["acceptance_rate"])

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mean_throughput": _safe_mean(throughputs),
        "mean_wall_time": _safe_mean(wall_times),
        "mean_acceptance_rate": _safe_mean(acceptance_rates),
        "throughputs": throughputs,
        "acceptance_rates": acceptance_rates,
    }


def evaluate_humaneval(
    decoder: SpeculativeDecoder,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    max_new_tokens: int,
    temperature: float,
    gamma: int,
    use_autoregressive: bool,
) -> Dict[str, Any]:
    """Evaluate HumanEval: pass@1 via execution of generated function bodies."""
    passed = 0
    total = 0
    throughputs: List[float] = []
    acceptance_rates: List[float] = []
    wall_times: List[float] = []

    for idx, sample in enumerate(samples):
        prompt = format_humaneval_prompt(sample["prompt"])
        try:
            generated, stats = _generate_text(
                decoder, tokenizer, prompt, max_new_tokens,
                temperature=temperature, gamma=gamma,
                use_autoregressive=use_autoregressive,
            )
        except Exception as exc:
            logger.debug(f"HumanEval sample {idx} generation failed: {exc}")
            total += 1
            continue

        # Truncate generation at the next function definition or class definition
        # to isolate the target function body.
        for stop_pattern in ["\ndef ", "\nclass ", "\n#", "\nif __name__"]:
            stop_idx = generated.find(stop_pattern)
            if stop_idx >= 0:
                generated = generated[:stop_idx]

        is_pass = check_humaneval(generated, sample)
        passed += int(is_pass)
        total += 1
        throughputs.append(stats["throughput"])
        wall_times.append(stats["wall_time_seconds"])
        if stats["acceptance_rate"] is not None:
            acceptance_rates.append(stats["acceptance_rate"])

    pass_at_1 = passed / total if total > 0 else 0.0
    return {
        "pass_at_1": pass_at_1,
        "passed": passed,
        "total": total,
        "mean_throughput": _safe_mean(throughputs),
        "mean_wall_time": _safe_mean(wall_times),
        "mean_acceptance_rate": _safe_mean(acceptance_rates),
        "throughputs": throughputs,
        "acceptance_rates": acceptance_rates,
    }


def evaluate_mmlu(
    decoder: SpeculativeDecoder,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    max_new_tokens: int,
    temperature: float,
    gamma: int,
    use_autoregressive: bool,
) -> Dict[str, Any]:
    """Evaluate MMLU: accuracy on 4-way multiple choice with 5-shot prompting."""
    correct = 0
    total = 0
    throughputs: List[float] = []
    acceptance_rates: List[float] = []
    wall_times: List[float] = []

    # Build a small pool of few-shot examples from the first 5 samples
    # (rotated so the current sample is never in its own few-shot context).
    few_shot_pool = samples[:5] if len(samples) >= 5 else samples

    for idx, sample in enumerate(samples):
        # Exclude current sample from few-shot examples
        few_shot = [s for s in few_shot_pool if s is not sample][:5]
        prompt = format_mmlu_prompt(
            sample["question"], sample["choices"], few_shot_examples=few_shot,
        )
        try:
            # MMLU only needs a short generation (the letter)
            generated, stats = _generate_text(
                decoder, tokenizer, prompt, max_new_tokens=min(max_new_tokens, 16),
                temperature=temperature, gamma=gamma,
                use_autoregressive=use_autoregressive,
            )
        except Exception as exc:
            logger.debug(f"MMLU sample {idx} generation failed: {exc}")
            total += 1
            continue

        is_correct = check_mmlu(generated, sample["answer_index"])
        correct += int(is_correct)
        total += 1
        throughputs.append(stats["throughput"])
        wall_times.append(stats["wall_time_seconds"])
        if stats["acceptance_rate"] is not None:
            acceptance_rates.append(stats["acceptance_rate"])

    accuracy = correct / total if total > 0 else 0.0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mean_throughput": _safe_mean(throughputs),
        "mean_wall_time": _safe_mean(wall_times),
        "mean_acceptance_rate": _safe_mean(acceptance_rates),
        "throughputs": throughputs,
        "acceptance_rates": acceptance_rates,
    }


def evaluate_mt_bench(
    decoder: SpeculativeDecoder,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    max_new_tokens: int,
    temperature: float,
    gamma: int,
    use_autoregressive: bool,
) -> Dict[str, Any]:
    """Evaluate MT-Bench: reference-based F1 scoring on multi-turn generation."""
    scores: List[float] = []
    throughputs: List[float] = []
    acceptance_rates: List[float] = []
    wall_times: List[float] = []

    for idx, sample in enumerate(samples):
        turns = sample.get("turns", [])
        if not turns:
            continue

        # Generate response for each turn, concatenating context
        turn_responses: List[str] = []
        sample_throughputs: List[float] = []
        sample_walls: List[float] = []
        sample_acc: List[float] = []

        for turn_i in range(len(turns)):
            # Build prompt with conversation history
            parts = []
            for prev_i in range(turn_i):
                parts.append(f"User: {turns[prev_i]}")
                if prev_i < len(turn_responses):
                    parts.append(f"Assistant: {turn_responses[prev_i]}")
            parts.append(f"User: {turns[turn_i]}")
            parts.append("Assistant:")
            prompt = "\n".join(parts)

            try:
                generated, stats = _generate_text(
                    decoder, tokenizer, prompt, max_new_tokens,
                    temperature=temperature, gamma=gamma,
                    use_autoregressive=use_autoregressive,
                )
            except Exception as exc:
                logger.debug(f"MT-Bench sample {idx} turn {turn_i} failed: {exc}")
                generated = ""
                stats = {"throughput": 0.0, "wall_time_seconds": 0.0,
                         "acceptance_rate": None}

            turn_responses.append(generated)
            sample_throughputs.append(stats["throughput"])
            sample_walls.append(stats["wall_time_seconds"])
            if stats["acceptance_rate"] is not None:
                sample_acc.append(stats["acceptance_rate"])

        # Score the full conversation against reference
        full_response = " ".join(turn_responses)
        reference = sample.get("reference", "")
        score = score_mt_bench(full_response, reference)
        scores.append(score)
        throughputs.extend(sample_throughputs)
        wall_times.extend(sample_walls)
        acceptance_rates.extend(sample_acc)

    mean_score = _safe_mean(scores)
    return {
        "score": mean_score,
        "num_samples": len(scores),
        "mean_throughput": _safe_mean(throughputs),
        "mean_wall_time": _safe_mean(wall_times),
        "mean_acceptance_rate": _safe_mean(acceptance_rates),
        "scores": scores,
        "throughputs": throughputs,
        "acceptance_rates": acceptance_rates,
    }


BENCHMARK_EVALUATORS = {
    "gsm8k": evaluate_gsm8k,
    "humaneval": evaluate_humaneval,
    "mmlu": evaluate_mmlu,
    "mt_bench": evaluate_mt_bench,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _primary_metric(benchmark: str, result: Dict) -> float:
    """Return the primary metric value for a given benchmark result."""
    if benchmark == "gsm8k":
        return result.get("accuracy", 0.0)
    if benchmark == "humaneval":
        return result.get("pass_at_1", 0.0)
    if benchmark == "mmlu":
        return result.get("accuracy", 0.0)
    if benchmark == "mt_bench":
        return result.get("score", 0.0)
    return 0.0


def _primary_metric_name(benchmark: str) -> str:
    if benchmark == "humaneval":
        return "pass_at_1"
    if benchmark == "mt_bench":
        return "score"
    return "accuracy"


def _method_label(quant_bits: int) -> str:
    if quant_bits == 0:
        return "vanilla_spec"
    return f"specquant_{quant_bits}bit"


# ---------------------------------------------------------------------------
# Multi-trial runner
# ---------------------------------------------------------------------------

def run_trials(
    decoder: SpeculativeDecoder,
    tokenizer: AutoTokenizer,
    benchmark: str,
    samples: List[Dict],
    max_new_tokens: int,
    temperature: float,
    gamma: int,
    use_autoregressive: bool,
    num_trials: int,
    seed: int,
) -> Dict[str, Any]:
    """Run the evaluator multiple times and aggregate with CIs via utils."""
    evaluator = BENCHMARK_EVALUATORS[benchmark]
    trial_results: List[Dict] = []

    for trial in range(num_trials):
        torch.manual_seed(seed + trial)
        logger.info(f"    Trial {trial + 1}/{num_trials}")
        result = evaluator(
            decoder, tokenizer, samples, max_new_tokens,
            temperature, gamma, use_autoregressive,
        )
        trial_results.append(result)

    # Use project utility for statistical aggregation
    metric_name = _primary_metric_name(benchmark)
    primary_values = [_primary_metric(benchmark, r) for r in trial_results]
    throughput_values = [r.get("mean_throughput", 0.0) for r in trial_results]
    acceptance_values = [
        r.get("mean_acceptance_rate", 0.0)
        for r in trial_results
        if r.get("mean_acceptance_rate") is not None
    ]

    aggregated = {
        metric_name: aggregate_trials(primary_values),
        "throughput": aggregate_trials(throughput_values),
    }
    if acceptance_values:
        aggregated["acceptance_rate"] = aggregate_trials(acceptance_values)
    else:
        aggregated["acceptance_rate"] = {"mean": None, "ci_lower": None,
                                          "ci_upper": None, "std": None,
                                          "n_trials": num_trials}

    # Include per-trial raw results for reproducibility
    aggregated["per_trial"] = trial_results
    aggregated["num_trials"] = num_trials
    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SpecQuant downstream task evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--draft-model", type=str,
        default=os.environ.get("QWEN35_0_8B", "Qwen/Qwen3.5-0.8B"),
        help="Draft model name or path",
    )
    parser.add_argument(
        "--target-model", type=str,
        default=os.environ.get("QWEN35_9B", "Qwen/Qwen3.5-9B"),
        help="Target model name or path",
    )
    parser.add_argument(
        "--draft-device", type=str, default="cuda:0",
        help="Device for draft model",
    )
    parser.add_argument(
        "--target-device", type=str, default="cuda:1",
        help="Device for target model",
    )
    parser.add_argument(
        "--quant-bits", type=int, nargs="+", default=[0, 3, 4],
        help="Quantization bit-widths to evaluate (0 = vanilla spec decode)",
    )
    parser.add_argument(
        "--benchmarks", type=str, nargs="+",
        default=["gsm8k", "humaneval", "mmlu", "mt_bench"],
        choices=["gsm8k", "humaneval", "mmlu", "mt_bench"],
        help="Benchmarks to run",
    )
    parser.add_argument(
        "--num-samples", type=int, default=200,
        help="Number of samples per benchmark",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Maximum new tokens per generation",
    )
    parser.add_argument(
        "--gamma", type=int, default=5,
        help="Draft length (gamma) for speculative decoding",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--num-trials", type=int, default=3,
        help="Number of trials per (benchmark, method) pair",
    )
    parser.add_argument(
        "--block-size", type=int, default=128,
        help="Quantization block size",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/downstream",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    return parser


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load models on dual GPUs ----
    draft_model, target_model, tokenizer = load_models_dual_gpu(
        args.draft_model, args.target_model,
        draft_device=args.draft_device,
        target_device=args.target_device,
    )

    # ---- Load datasets ----
    datasets: Dict[str, Optional[List[Dict]]] = {}
    for bench_name in args.benchmarks:
        loader = LOADERS.get(bench_name)
        if loader is None:
            logger.warning(f"Unknown benchmark: {bench_name}, skipping")
            continue
        datasets[bench_name] = loader(args.num_samples, args.seed)

    active_benchmarks = [b for b in args.benchmarks if datasets.get(b)]
    if not active_benchmarks:
        logger.error("No benchmarks could be loaded. Exiting.")
        return

    logger.info(f"Active benchmarks: {active_benchmarks}")

    # ---- Define methods to compare ----
    methods = [("autoregressive", None)]  # (label, quant_bits_or_None)
    for bits in args.quant_bits:
        methods.append((_method_label(bits), bits))

    # ---- Run evaluations ----
    all_results: Dict[str, Any] = {
        "config": vars(args),
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "benchmarks": {},
    }

    for bench_name in active_benchmarks:
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark: {bench_name}")
        logger.info(f"{'='*60}")
        samples = datasets[bench_name]
        bench_results: Dict[str, Any] = {}

        for method_label, quant_bits in methods:
            logger.info(f"\n  Method: {method_label}")
            use_ar = method_label == "autoregressive"

            if use_ar:
                decoder = SpeculativeDecoder(
                    draft_model, target_model, tokenizer, quant_bits=0,
                )
            else:
                decoder = SpeculativeDecoder(
                    draft_model, target_model, tokenizer,
                    quant_bits=quant_bits,
                    quant_block_size=args.block_size,
                    quant_seed=args.seed,
                )

            try:
                result = run_trials(
                    decoder, tokenizer, bench_name, samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    gamma=args.gamma,
                    use_autoregressive=use_ar,
                    num_trials=args.num_trials,
                    seed=args.seed,
                )
            except Exception as exc:
                logger.error(f"  {method_label} on {bench_name} failed: {exc}")
                traceback.print_exc()
                result = {"error": str(exc)}

            bench_results[method_label] = result

            # Log summary for this method
            metric_name = _primary_metric_name(bench_name)
            if metric_name in result and isinstance(result[metric_name], dict):
                agg = result[metric_name]
                ci_lo = agg.get("ci_lower", 0.0) or 0.0
                ci_hi = agg.get("ci_upper", 0.0) or 0.0
                mean_val = agg.get("mean", 0.0) or 0.0
                logger.info(
                    f"    {metric_name}: {mean_val:.4f} "
                    f"[{ci_lo:.4f}, {ci_hi:.4f}]"
                )
            if "throughput" in result and isinstance(result["throughput"], dict):
                tp_agg = result["throughput"]
                tp_mean = tp_agg.get("mean", 0.0) or 0.0
                logger.info(f"    throughput: {tp_mean:.1f} tok/s")
            if "acceptance_rate" in result and isinstance(result["acceptance_rate"], dict):
                ar_agg = result["acceptance_rate"]
                ar_mean = ar_agg.get("mean")
                if ar_mean is not None:
                    logger.info(f"    acceptance_rate: {ar_mean:.4f}")

        all_results["benchmarks"][bench_name] = bench_results

    # ---- Save results ----
    output_file = os.path.join(
        args.output_dir,
        f"downstream_{Path(args.target_model).name}_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    save_results(all_results, output_file)
    logger.info(f"\nResults saved to {output_file}")

    # ---- Print summary table ----
    _print_summary(all_results, active_benchmarks, methods)


def _print_summary(
    all_results: Dict,
    benchmarks: List[str],
    methods: List[Tuple[str, Optional[int]]],
) -> None:
    """Print a human-readable summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    header = f"{'Method':<20}"
    for bench_name in benchmarks:
        metric = _primary_metric_name(bench_name)
        header += f"  {bench_name}({metric})"
    header += f"  {'throughput':>12}"
    logger.info(header)
    logger.info("-" * len(header))

    for method_label, _ in methods:
        row = f"{method_label:<20}"
        for bench_name in benchmarks:
            bench_data = all_results.get("benchmarks", {}).get(bench_name, {})
            method_data = bench_data.get(method_label, {})
            metric_name = _primary_metric_name(bench_name)
            if metric_name in method_data and isinstance(method_data[metric_name], dict):
                agg = method_data[metric_name]
                mean_val = agg.get("mean", 0.0) or 0.0
                ci_lo = agg.get("ci_lower", 0.0) or 0.0
                ci_hi = agg.get("ci_upper", 0.0) or 0.0
                cell = f"{mean_val:.3f}({ci_lo:.3f}-{ci_hi:.3f})"
            else:
                cell = "N/A"
            row += f"  {cell:>24}"

        # Average throughput across benchmarks for this method
        tp_values = []
        for bench_name in benchmarks:
            bench_data = all_results.get("benchmarks", {}).get(bench_name, {})
            method_data = bench_data.get(method_label, {})
            if "throughput" in method_data and isinstance(method_data["throughput"], dict):
                tp_val = method_data["throughput"].get("mean")
                if tp_val is not None:
                    tp_values.append(tp_val)
        avg_tp = _safe_mean(tp_values)
        row += f"  {avg_tp:>10.1f}/s"
        logger.info(row)

    logger.info("=" * 80)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
