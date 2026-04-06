"""Comprehensive ablation study for SpecQuant.

Sweeps over block size, speculative length (gamma), temperature, Hadamard
rotation seed, and mixed-precision layer configurations.  Each sweep holds all
non-swept parameters at their defaults and reports acceptance_rate, throughput,
and tokens_per_round with 95% confidence intervals across multiple trials.

Usage:
    python -m scripts.run_ablations \
        --draft-model Qwen/Qwen3.5-0.8B \
        --target-model Qwen/Qwen3.5-9B \
        --ablation-type all

    python -m scripts.run_ablations \
        --ablation-type gamma \
        --num-trials 5 \
        --output-dir results/ablations
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.speculative_decode import SpeculativeDecoder
from src.utils import aggregate_trials, validate_dual_gpu, save_results, mean_confidence_interval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Sweep grids ──────────────────────────────────────────────────────────────

BLOCK_SIZE_GRID = [32, 64, 128, 256, 512]
GAMMA_GRID = [1, 2, 3, 5, 7, 9]
TEMPERATURE_GRID = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
SEED_GRID = [42, 123, 456, 789, 1024]

# Defaults for non-swept parameters
DEFAULTS = {
    "quant_bits": 3,
    "block_size": 128,
    "gamma": 5,
    "temperature": 1.0,
    "seed": 42,
}

EVAL_PROMPTS = [
    "Solve the following math problem step by step: What is the sum of the first 100 prime numbers?",
    "Write a Python function to find the longest common subsequence of two strings. Include type hints and docstring.",
    "A train travels at 60 mph for the first half of the journey and 40 mph for the second half. What is the average speed for the entire journey?",
    "Explain the theory of general relativity in detail, covering spacetime curvature, the equivalence principle, and gravitational waves.",
]


# ── Model loading ────────────────────────────────────────────────────────────

def load_models(
    draft_model_name: str,
    target_model_name: str,
    draft_device: str,
    target_device: str,
    dtype: torch.dtype = torch.float16,
):
    """Load draft and target models onto their respective devices."""
    logger.info("Loading draft model: %s -> %s", draft_model_name, draft_device)
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_name,
        torch_dtype=dtype,
        device_map=draft_device,
        trust_remote_code=True,
    )

    logger.info("Loading target model: %s -> %s", target_model_name, target_device)
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


# ── Single-configuration runner ──────────────────────────────────────────────

def _run_single_config(
    draft_model,
    target_model,
    tokenizer,
    prompts: List[str],
    quant_bits: int,
    block_size: int,
    gamma: int,
    temperature: float,
    seed: int,
    max_new_tokens: int,
    num_trials: int,
) -> Dict[str, Any]:
    """Run multiple trials of speculative decoding with a single config.

    Returns a dict containing aggregated statistics (with 95% CIs) for
    acceptance_rate, throughput, and tokens_per_round across all
    prompt x trial combinations.
    """
    decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        quant_bits=quant_bits,
        quant_block_size=block_size,
        quant_seed=seed,
    )

    all_acceptance_rates: List[float] = []
    all_throughputs: List[float] = []
    all_tokens_per_round: List[float] = []
    all_wall_times: List[float] = []
    per_prompt_results: List[Dict[str, Any]] = []

    for prompt_idx, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        # Warmup (one short generation to fill caches / JIT paths)
        decoder.generate(
            input_ids,
            max_new_tokens=min(16, max_new_tokens),
            gamma=gamma,
            temperature=max(temperature, 0.01),  # avoid division by zero in warmup
        )

        prompt_trial_results = []
        for trial in range(num_trials):
            torch.cuda.synchronize()
            out = decoder.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                gamma=gamma,
                temperature=temperature,
            )
            torch.cuda.synchronize()

            result = out.to_dict()
            prompt_trial_results.append(result)

            all_acceptance_rates.append(result["acceptance_rate"])
            all_throughputs.append(result["throughput_tokens_per_sec"])
            all_tokens_per_round.append(result["tokens_per_round"])
            all_wall_times.append(result["wall_time_seconds"])

        per_prompt_results.append({
            "prompt_idx": prompt_idx,
            "prompt_len": input_ids.shape[1],
            "trials": prompt_trial_results,
        })

    return {
        "config": {
            "quant_bits": quant_bits,
            "block_size": block_size,
            "gamma": gamma,
            "temperature": temperature,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "num_trials": num_trials,
            "num_prompts": len(prompts),
        },
        "acceptance_rate": aggregate_trials(all_acceptance_rates),
        "throughput": aggregate_trials(all_throughputs),
        "tokens_per_round": aggregate_trials(all_tokens_per_round),
        "wall_time": aggregate_trials(all_wall_times),
        "per_prompt": per_prompt_results,
    }


# ── Individual ablation sweeps ───────────────────────────────────────────────

def ablation_block_size(
    draft_model,
    target_model,
    tokenizer,
    prompts: List[str],
    quant_bits: int,
    max_new_tokens: int,
    num_trials: int,
) -> Dict[str, Any]:
    """Sweep block_size: measures how block granularity affects quantization
    error and acceptance rate.
    """
    logger.info("=" * 60)
    logger.info("ABLATION: block_size sweep %s", BLOCK_SIZE_GRID)
    logger.info("=" * 60)

    results_by_block_size = {}
    for bs in BLOCK_SIZE_GRID:
        logger.info("  block_size=%d ...", bs)
        res = _run_single_config(
            draft_model, target_model, tokenizer, prompts,
            quant_bits=quant_bits,
            block_size=bs,
            gamma=DEFAULTS["gamma"],
            temperature=DEFAULTS["temperature"],
            seed=DEFAULTS["seed"],
            max_new_tokens=max_new_tokens,
            num_trials=num_trials,
        )
        results_by_block_size[str(bs)] = res
        logger.info(
            "    acceptance_rate=%.4f [%.4f, %.4f], throughput=%.1f tok/s",
            res["acceptance_rate"]["mean"],
            res["acceptance_rate"]["ci_lower"],
            res["acceptance_rate"]["ci_upper"],
            res["throughput"]["mean"],
        )

    return {
        "ablation_type": "block_size",
        "sweep_values": BLOCK_SIZE_GRID,
        "fixed_params": {
            "quant_bits": quant_bits,
            "gamma": DEFAULTS["gamma"],
            "temperature": DEFAULTS["temperature"],
            "seed": DEFAULTS["seed"],
        },
        "results": results_by_block_size,
    }


def ablation_gamma(
    draft_model,
    target_model,
    tokenizer,
    prompts: List[str],
    quant_bits: int,
    max_new_tokens: int,
    num_trials: int,
) -> Dict[str, Any]:
    """Sweep gamma (speculative length): measures optimal draft length
    with quantization enabled.
    """
    logger.info("=" * 60)
    logger.info("ABLATION: gamma sweep %s", GAMMA_GRID)
    logger.info("=" * 60)

    results_by_gamma = {}
    for g in GAMMA_GRID:
        logger.info("  gamma=%d ...", g)
        res = _run_single_config(
            draft_model, target_model, tokenizer, prompts,
            quant_bits=quant_bits,
            block_size=DEFAULTS["block_size"],
            gamma=g,
            temperature=DEFAULTS["temperature"],
            seed=DEFAULTS["seed"],
            max_new_tokens=max_new_tokens,
            num_trials=num_trials,
        )
        results_by_gamma[str(g)] = res
        logger.info(
            "    acceptance_rate=%.4f [%.4f, %.4f], tokens_per_round=%.2f",
            res["acceptance_rate"]["mean"],
            res["acceptance_rate"]["ci_lower"],
            res["acceptance_rate"]["ci_upper"],
            res["tokens_per_round"]["mean"],
        )

    # Also run without quantization (baseline) for comparison
    logger.info("  gamma sweep (no quantization baseline) ...")
    baseline_by_gamma = {}
    for g in GAMMA_GRID:
        logger.info("    gamma=%d (fp16 baseline) ...", g)
        res = _run_single_config(
            draft_model, target_model, tokenizer, prompts,
            quant_bits=0,
            block_size=DEFAULTS["block_size"],
            gamma=g,
            temperature=DEFAULTS["temperature"],
            seed=DEFAULTS["seed"],
            max_new_tokens=max_new_tokens,
            num_trials=num_trials,
        )
        baseline_by_gamma[str(g)] = res

    return {
        "ablation_type": "gamma",
        "sweep_values": GAMMA_GRID,
        "fixed_params": {
            "quant_bits": quant_bits,
            "block_size": DEFAULTS["block_size"],
            "temperature": DEFAULTS["temperature"],
            "seed": DEFAULTS["seed"],
        },
        "results_quantized": results_by_gamma,
        "results_fp_baseline": baseline_by_gamma,
    }


def ablation_temperature(
    draft_model,
    target_model,
    tokenizer,
    prompts: List[str],
    quant_bits: int,
    max_new_tokens: int,
    num_trials: int,
) -> Dict[str, Any]:
    """Sweep temperature: how sampling temperature affects quantization
    robustness (higher temperature amplifies quantization noise).
    """
    logger.info("=" * 60)
    logger.info("ABLATION: temperature sweep %s", TEMPERATURE_GRID)
    logger.info("=" * 60)

    results_quantized = {}
    results_fp = {}
    for temp in TEMPERATURE_GRID:
        logger.info("  temperature=%.1f ...", temp)

        # Quantized run
        res_q = _run_single_config(
            draft_model, target_model, tokenizer, prompts,
            quant_bits=quant_bits,
            block_size=DEFAULTS["block_size"],
            gamma=DEFAULTS["gamma"],
            temperature=temp,
            seed=DEFAULTS["seed"],
            max_new_tokens=max_new_tokens,
            num_trials=num_trials,
        )
        results_quantized[str(temp)] = res_q

        # Full-precision baseline at same temperature
        res_fp = _run_single_config(
            draft_model, target_model, tokenizer, prompts,
            quant_bits=0,
            block_size=DEFAULTS["block_size"],
            gamma=DEFAULTS["gamma"],
            temperature=temp,
            seed=DEFAULTS["seed"],
            max_new_tokens=max_new_tokens,
            num_trials=num_trials,
        )
        results_fp[str(temp)] = res_fp

        delta_acc = (
            res_q["acceptance_rate"]["mean"] - res_fp["acceptance_rate"]["mean"]
        )
        logger.info(
            "    quantized: acc=%.4f, fp: acc=%.4f, delta=%.4f",
            res_q["acceptance_rate"]["mean"],
            res_fp["acceptance_rate"]["mean"],
            delta_acc,
        )

    return {
        "ablation_type": "temperature",
        "sweep_values": TEMPERATURE_GRID,
        "fixed_params": {
            "quant_bits": quant_bits,
            "block_size": DEFAULTS["block_size"],
            "gamma": DEFAULTS["gamma"],
            "seed": DEFAULTS["seed"],
        },
        "results_quantized": results_quantized,
        "results_fp_baseline": results_fp,
    }


def ablation_seed(
    draft_model,
    target_model,
    tokenizer,
    prompts: List[str],
    quant_bits: int,
    max_new_tokens: int,
    num_trials: int,
) -> Dict[str, Any]:
    """Sweep Hadamard rotation seed: the random sign vector determines
    how well quantization distributes error across coordinates.
    """
    logger.info("=" * 60)
    logger.info("ABLATION: seed sensitivity %s", SEED_GRID)
    logger.info("=" * 60)

    results_by_seed = {}
    for s in SEED_GRID:
        logger.info("  seed=%d ...", s)
        res = _run_single_config(
            draft_model, target_model, tokenizer, prompts,
            quant_bits=quant_bits,
            block_size=DEFAULTS["block_size"],
            gamma=DEFAULTS["gamma"],
            temperature=DEFAULTS["temperature"],
            seed=s,
            max_new_tokens=max_new_tokens,
            num_trials=num_trials,
        )
        results_by_seed[str(s)] = res
        logger.info(
            "    acceptance_rate=%.4f [%.4f, %.4f]",
            res["acceptance_rate"]["mean"],
            res["acceptance_rate"]["ci_lower"],
            res["acceptance_rate"]["ci_upper"],
        )

    # Compute cross-seed variance for the main metrics
    seed_acc_means = [
        results_by_seed[str(s)]["acceptance_rate"]["mean"] for s in SEED_GRID
    ]
    seed_tp_means = [
        results_by_seed[str(s)]["throughput"]["mean"] for s in SEED_GRID
    ]
    seed_tpr_means = [
        results_by_seed[str(s)]["tokens_per_round"]["mean"] for s in SEED_GRID
    ]

    cross_seed_summary = {
        "acceptance_rate": aggregate_trials(seed_acc_means),
        "throughput": aggregate_trials(seed_tp_means),
        "tokens_per_round": aggregate_trials(seed_tpr_means),
    }
    logger.info(
        "  Cross-seed acceptance_rate: mean=%.4f, std=%.4f",
        cross_seed_summary["acceptance_rate"]["mean"],
        cross_seed_summary["acceptance_rate"]["std"],
    )

    return {
        "ablation_type": "seed",
        "sweep_values": SEED_GRID,
        "fixed_params": {
            "quant_bits": quant_bits,
            "block_size": DEFAULTS["block_size"],
            "gamma": DEFAULTS["gamma"],
            "temperature": DEFAULTS["temperature"],
        },
        "results": results_by_seed,
        "cross_seed_summary": cross_seed_summary,
    }


def _capture_real_kv(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    seq_length: int = 256,
) -> Dict[int, tuple]:
    """Run a forward pass and capture real K, V tensors from all layers via hooks.

    Returns dict mapping layer_idx -> (key_tensor, value_tensor) on CPU float32.
    """
    config = model.config
    device = next(model.parameters()).device

    # Discover attention modules
    attn_modules = []
    inner = getattr(model, "model", None)
    if inner is not None:
        layers = getattr(inner, "layers", None)
        if layers is not None:
            for i, layer in enumerate(layers):
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    attn_modules.append((i, attn))
    if not attn_modules:
        transformer = getattr(model, "transformer", None)
        if transformer is not None:
            h = getattr(transformer, "h", None)
            if h is not None:
                for i, layer in enumerate(h):
                    attn = getattr(layer, "attn", getattr(layer, "self_attn", None))
                    if attn is not None:
                        attn_modules.append((i, attn))

    captured: Dict[int, tuple] = {}
    handles = []

    def make_hook(layer_idx):
        def hook_fn(module, args, output):
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, tuple) and len(item) == 2:
                        k_cand, v_cand = item
                        if (isinstance(k_cand, torch.Tensor) and
                                isinstance(v_cand, torch.Tensor) and
                                k_cand.ndim == 4 and v_cand.ndim == 4):
                            captured[layer_idx] = (
                                k_cand.detach().cpu().float(),
                                v_cand.detach().cpu().float(),
                            )
                            break
        return hook_fn

    for idx, attn_mod in attn_modules:
        handles.append(attn_mod.register_forward_hook(make_hook(idx)))

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=seq_length)
    input_ids = inputs["input_ids"].to(device)

    try:
        with torch.no_grad():
            model(input_ids, use_cache=True)
    finally:
        for h in handles:
            h.remove()

    return captured


def _interpolate_bracket_results(
    res_3bit: Dict[str, Any],
    res_4bit: Dict[str, Any],
    frac_4bit: float,
) -> Dict[str, Any]:
    """Linearly interpolate metrics between uniform 3-bit and 4-bit results.

    Used for mixed-precision configs where end-to-end measurement is not
    possible (SpeculativeDecoder only supports a single quant_bits value).
    The interpolated values are supplementary estimates, not direct
    measurements.

    Args:
        res_3bit: aggregated result dict from uniform 3-bit decoding
        res_4bit: aggregated result dict from uniform 4-bit decoding
        frac_4bit: fraction of layers at 4-bit (0.0 to 1.0)
    """
    frac_3bit = 1.0 - frac_4bit
    interpolated = {}

    for key in res_3bit:
        v3 = res_3bit[key]
        v4 = res_4bit.get(key)
        if isinstance(v3, (int, float)) and isinstance(v4, (int, float)):
            interpolated[key] = frac_3bit * v3 + frac_4bit * v4
        elif isinstance(v3, dict) and isinstance(v4, dict):
            # Recursively interpolate numeric fields in nested dicts
            inner = {}
            for sub_key in v3:
                sv3 = v3[sub_key]
                sv4 = v4.get(sub_key)
                if isinstance(sv3, (int, float)) and isinstance(sv4, (int, float)):
                    inner[sub_key] = frac_3bit * sv3 + frac_4bit * sv4
                else:
                    inner[sub_key] = sv3  # keep non-numeric as-is
            interpolated[key] = inner
        else:
            interpolated[key] = v3  # keep non-numeric as-is

    interpolated["_interpolation_note"] = (
        "Estimated via linear interpolation between uniform 3-bit and 4-bit "
        "bracket results. NOT an end-to-end mixed-precision measurement."
    )
    return interpolated


def ablation_mixed_precision(
    draft_model,
    target_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    num_trials: int,
) -> Dict[str, Any]:
    """Mixed-precision analysis: identify per-layer sensitivity from real
    model activations, then evaluate configurations where sensitive layers
    use 4-bit while the rest use 3-bit.

    Strategy:
    1. Capture real K, V activations from a representative prompt via hooks.
    2. Measure per-layer quantization error at 3-bit and 4-bit on real data.
    3. Rank layers by sensitivity and define mixed-precision configs.
    4. Evaluate all configs (uniform and mixed) with real decoding.
       Mixed configs run at the rounded effective bit-width (3 or 4) since
       SpeculativeDecoder uses a single quant_bits.  Mixed-precision results
       are estimated via linear interpolation between uniform 3-bit and 4-bit
       bracket results, weighted by the fraction of 4-bit layers.
    """
    logger.info("=" * 60)
    logger.info("ABLATION: mixed-precision analysis")
    logger.info("=" * 60)

    # ── Step 1: Capture real per-layer KV activations ───────────────────
    config = target_model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    logger.info(
        "  Probing %d layers (kv_heads=%d, head_dim=%d) with real activations ...",
        num_layers, num_kv_heads, head_dim,
    )

    from src.turboquant_kv import HadamardRotation, ScalarQuantizer
    import torch.nn.functional as F

    rotation = HadamardRotation(head_dim, seed=DEFAULTS["seed"])
    quantizer_3b = ScalarQuantizer(bits=3, block_size=DEFAULTS["block_size"])
    quantizer_4b = ScalarQuantizer(bits=4, block_size=DEFAULTS["block_size"])

    # Use the first prompt to capture real activations
    probe_prompt = prompts[0] if prompts else "The quick brown fox jumps over the lazy dog."
    captured_kv = _capture_real_kv(target_model, tokenizer, probe_prompt, seq_length=256)

    layer_mse_3bit = []
    layer_mse_4bit = []

    for layer_idx in range(num_layers):
        if layer_idx in captured_kv:
            k, v = captured_kv[layer_idx]
        else:
            # Fallback for layers not captured (should not happen normally)
            logger.warning("  Layer %d: no real KV captured, using zeros", layer_idx)
            k = torch.zeros(1, num_kv_heads, 1, head_dim)
            v = torch.zeros(1, num_kv_heads, 1, head_dim)

        k_rot = rotation.rotate(k)
        v_rot = rotation.rotate(v)

        # 3-bit round-trip error on real activations
        kc3, ks3, kz3 = quantizer_3b.quantize(k_rot)
        vc3, vs3, vz3 = quantizer_3b.quantize(v_rot)
        k_deq3 = rotation.inverse_rotate(quantizer_3b.dequantize(kc3, ks3, kz3))
        v_deq3 = rotation.inverse_rotate(quantizer_3b.dequantize(vc3, vs3, vz3))
        mse3 = (F.mse_loss(k_deq3, k).item() + F.mse_loss(v_deq3, v).item()) / 2.0

        # 4-bit round-trip error on real activations
        kc4, ks4, kz4 = quantizer_4b.quantize(k_rot)
        vc4, vs4, vz4 = quantizer_4b.quantize(v_rot)
        k_deq4 = rotation.inverse_rotate(quantizer_4b.dequantize(kc4, ks4, kz4))
        v_deq4 = rotation.inverse_rotate(quantizer_4b.dequantize(vc4, vs4, vz4))
        mse4 = (F.mse_loss(k_deq4, k).item() + F.mse_loss(v_deq4, v).item()) / 2.0

        layer_mse_3bit.append(mse3)
        layer_mse_4bit.append(mse4)

    # Rank layers by 3-bit sensitivity (descending MSE)
    sensitivity_order = sorted(
        range(num_layers), key=lambda i: layer_mse_3bit[i], reverse=True
    )

    logger.info("  Top-5 most sensitive layers (3-bit MSE on real activations):")
    for rank, li in enumerate(sensitivity_order[:5]):
        logger.info(
            "    rank %d: layer %d, MSE_3b=%.6f, MSE_4b=%.6f",
            rank, li, layer_mse_3bit[li], layer_mse_4bit[li],
        )

    # ── Step 2: Define mixed-precision configs ───────────────────────────
    n_quarter = max(1, num_layers // 4)
    n_half = max(1, num_layers // 2)

    configs = {
        "uniform_3bit": {"layers_4bit": [], "description": "All layers at 3-bit"},
        f"top{n_quarter}_4bit": {
            "layers_4bit": sensitivity_order[:n_quarter],
            "description": f"Top {n_quarter} sensitive layers at 4-bit, rest at 3-bit",
        },
        f"top{n_half}_4bit": {
            "layers_4bit": sensitivity_order[:n_half],
            "description": f"Top {n_half} sensitive layers at 4-bit, rest at 3-bit",
        },
        "uniform_4bit": {
            "layers_4bit": list(range(num_layers)),
            "description": "All layers at 4-bit",
        },
    }

    # ── Step 3: Evaluate each config ──────────────────────────────────────
    # Uniform configs (all-3bit, all-4bit) are measured end-to-end with real
    # decoding.  Mixed configs cannot be run end-to-end because
    # SpeculativeDecoder accepts only a single quant_bits value -- there is
    # no per-layer bit assignment in the current decoder.  For mixed configs
    # we bracket performance using the two uniform measurements and
    # linearly interpolate weighted by the fraction of 4-bit layers.
    # These are clearly labelled as "estimated" (not end-to-end mixed).
    eval_results = {}

    # First, run the two uniform bracket configs with real decoding
    bracket_results = {}
    for bracket_bits in (3, 4):
        bracket_label = f"uniform_{bracket_bits}bit"
        logger.info(
            "  Running uniform %d-bit bracket with real decoding ...",
            bracket_bits,
        )
        bracket_results[bracket_bits] = _run_single_config(
            draft_model, target_model, tokenizer, prompts,
            quant_bits=bracket_bits,
            block_size=DEFAULTS["block_size"],
            gamma=DEFAULTS["gamma"],
            temperature=DEFAULTS["temperature"],
            seed=DEFAULTS["seed"],
            max_new_tokens=max_new_tokens,
            num_trials=num_trials,
        )

    for config_name, cfg in configs.items():
        layers_4bit = set(cfg["layers_4bit"])
        n_4bit = len(layers_4bit)
        n_3bit = num_layers - n_4bit

        if n_4bit == 0:
            effective_bits = 3
        elif n_4bit == num_layers:
            effective_bits = 4
        else:
            effective_bits = (3 * n_3bit + 4 * n_4bit) / num_layers

        is_uniform = (n_4bit == 0 or n_4bit == num_layers)
        actual_bits = 3 if n_4bit == 0 else (4 if n_4bit == num_layers else None)

        if is_uniform:
            # Uniform configs: use real end-to-end decoding results
            res = bracket_results[actual_bits]
            measurement_type = "end_to_end"
            logger.info(
                "  Config '%s': uniform %d-bit, end-to-end measured",
                config_name, actual_bits,
            )
        else:
            # Mixed configs: linear interpolation between bracket results
            # NOTE: This is estimated, NOT end-to-end mixed-precision decoding.
            # The current SpeculativeDecoder does not support per-layer bit
            # assignment.  We linearly interpolate acceptance rate and
            # throughput between the uniform-3bit and uniform-4bit brackets,
            # weighted by the fraction of layers at each bit-width.
            frac_4bit = n_4bit / num_layers
            res = _interpolate_bracket_results(
                bracket_results[3], bracket_results[4], frac_4bit,
            )
            measurement_type = "estimated_from_per_layer_analysis"
            logger.info(
                "  Config '%s': %d layers@3bit, %d layers@4bit (eff=%.2f bits) "
                "-- ESTIMATED via linear interpolation between uniform 3-bit and "
                "4-bit brackets (weighted by fraction of 4-bit layers), not "
                "end-to-end mixed-precision decoding",
                config_name, n_3bit, n_4bit, effective_bits,
            )

        # Compute per-layer MSE from real activations for this config
        weighted_mse_values = []
        for li in range(num_layers):
            if li in layers_4bit:
                weighted_mse_values.append(layer_mse_4bit[li])
            else:
                weighted_mse_values.append(layer_mse_3bit[li])
        avg_mse = sum(weighted_mse_values) / num_layers

        memory_ratio_vs_4bit = (3 * n_3bit + 4 * n_4bit) / (4 * num_layers)

        eval_results[config_name] = {
            "measured": is_uniform,
            "measurement_type": measurement_type,
            "effective_bits": effective_bits,
            "layers_4bit": sorted(layers_4bit),
            "n_layers_3bit": n_3bit,
            "n_layers_4bit": n_4bit,
            "real_activation_avg_mse": avg_mse,
            "memory_ratio_vs_uniform4bit": memory_ratio_vs_4bit,
            "results": res,
        }

    return {
        "ablation_type": "mixed_precision",
        "num_layers": num_layers,
        "sensitivity_order": sensitivity_order,
        "layer_mse_3bit": layer_mse_3bit,
        "layer_mse_4bit": layer_mse_4bit,
        "activation_source": "real_model_activations",
        "configs": {k: v.get("description", "") for k, v in configs.items()},
        "results": eval_results,
        "note": (
            "Uniform configs (all-3bit, all-4bit) are measured end-to-end. "
            "Mixed configs are estimated via linear interpolation between "
            "uniform 3-bit and 4-bit bracket results, weighted by the fraction "
            "of 4-bit layers -- NOT from end-to-end mixed-precision decoding "
            "(the current decoder does not support per-layer bit assignment)."
        ),
    }


# ── Unified runner ───────────────────────────────────────────────────────────

ABLATION_REGISTRY = {
    "block_size": "ablation_block_size",
    "gamma": "ablation_gamma",
    "temperature": "ablation_temperature",
    "seed": "ablation_seed",
    "mixed_precision": "ablation_mixed_precision",
}


def run_ablations(args):
    """Dispatch to individual ablation functions based on --ablation-type."""
    # Validate dual-GPU setup
    try:
        validate_dual_gpu()
    except RuntimeError as e:
        logger.warning("Dual-GPU validation failed: %s", e)
        logger.warning(
            "Falling back to single device. Performance numbers will "
            "reflect model-parallel overhead on one GPU."
        )

    draft_model, target_model, tokenizer = load_models(
        args.draft_model,
        args.target_model,
        args.draft_device,
        args.target_device,
    )

    prompts = EVAL_PROMPTS
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if args.ablation_type == "all":
        types_to_run = list(ABLATION_REGISTRY.keys())
    else:
        types_to_run = [args.ablation_type]

    all_outputs = {}
    for abl_type in types_to_run:
        logger.info("\n>>> Starting ablation: %s", abl_type)
        t0 = time.perf_counter()

        if abl_type == "mixed_precision":
            result = ablation_mixed_precision(
                draft_model, target_model, tokenizer, prompts,
                max_new_tokens=args.max_new_tokens,
                num_trials=args.num_trials,
            )
        elif abl_type == "block_size":
            result = ablation_block_size(
                draft_model, target_model, tokenizer, prompts,
                quant_bits=args.quant_bits,
                max_new_tokens=args.max_new_tokens,
                num_trials=args.num_trials,
            )
        elif abl_type == "gamma":
            result = ablation_gamma(
                draft_model, target_model, tokenizer, prompts,
                quant_bits=args.quant_bits,
                max_new_tokens=args.max_new_tokens,
                num_trials=args.num_trials,
            )
        elif abl_type == "temperature":
            result = ablation_temperature(
                draft_model, target_model, tokenizer, prompts,
                quant_bits=args.quant_bits,
                max_new_tokens=args.max_new_tokens,
                num_trials=args.num_trials,
            )
        elif abl_type == "seed":
            result = ablation_seed(
                draft_model, target_model, tokenizer, prompts,
                quant_bits=args.quant_bits,
                max_new_tokens=args.max_new_tokens,
                num_trials=args.num_trials,
            )
        else:
            logger.error("Unknown ablation type: %s", abl_type)
            continue

        elapsed = time.perf_counter() - t0
        result["elapsed_seconds"] = elapsed
        result["timestamp"] = timestamp

        # Save individual JSON per sweep type
        filename = f"ablation_{abl_type}_{timestamp}.json"
        save_results(result, args.output_dir, filename)
        logger.info(
            ">>> Ablation '%s' complete in %.1f seconds, saved to %s/%s",
            abl_type, elapsed, args.output_dir, filename,
        )

        all_outputs[abl_type] = result

    # Save a combined summary if running "all"
    if len(types_to_run) > 1:
        summary = _build_summary(all_outputs)
        save_results(summary, args.output_dir, f"ablation_summary_{timestamp}.json")
        _print_summary_table(summary)

    return all_outputs


def _build_summary(all_outputs: Dict[str, Any]) -> Dict[str, Any]:
    """Build a condensed cross-ablation summary for quick inspection."""
    summary: Dict[str, Any] = {"ablation_types": list(all_outputs.keys())}

    for abl_type, data in all_outputs.items():
        section: Dict[str, Any] = {}

        if abl_type in ("block_size", "seed"):
            results = data.get("results", {})
            section["sweep_values"] = data.get("sweep_values", [])
            section["acceptance_rates"] = {
                k: v["acceptance_rate"]["mean"] for k, v in results.items()
            }
            section["throughputs"] = {
                k: v["throughput"]["mean"] for k, v in results.items()
            }

        elif abl_type == "gamma":
            results_q = data.get("results_quantized", {})
            results_fp = data.get("results_fp_baseline", {})
            section["sweep_values"] = data.get("sweep_values", [])
            section["acceptance_rates_quantized"] = {
                k: v["acceptance_rate"]["mean"] for k, v in results_q.items()
            }
            section["acceptance_rates_fp"] = {
                k: v["acceptance_rate"]["mean"] for k, v in results_fp.items()
            }

        elif abl_type == "temperature":
            results_q = data.get("results_quantized", {})
            results_fp = data.get("results_fp_baseline", {})
            section["sweep_values"] = data.get("sweep_values", [])
            section["acceptance_rate_deltas"] = {
                k: (
                    results_q[k]["acceptance_rate"]["mean"]
                    - results_fp[k]["acceptance_rate"]["mean"]
                )
                for k in results_q
            }

        elif abl_type == "mixed_precision":
            section["sensitivity_order_top5"] = data.get("sensitivity_order", [])[:5]
            eval_results = data.get("results", {})
            for cfg_name, cfg_data in eval_results.items():
                if cfg_data.get("measured"):
                    section[cfg_name] = {
                        "acceptance_rate": cfg_data["results"]["acceptance_rate"]["mean"],
                        "throughput": cfg_data["results"]["throughput"]["mean"],
                    }
                else:
                    section[cfg_name] = {
                        "estimated_mse": cfg_data.get("estimated_avg_mse"),
                        "memory_ratio": cfg_data.get("memory_ratio_vs_uniform4bit"),
                    }

        summary[abl_type] = section

    return summary


def _print_summary_table(summary: Dict[str, Any]):
    """Print a human-readable summary table to the logger."""
    logger.info("\n" + "=" * 70)
    logger.info("ABLATION STUDY SUMMARY")
    logger.info("=" * 70)

    for abl_type in summary.get("ablation_types", []):
        section = summary.get(abl_type, {})
        logger.info("\n--- %s ---", abl_type)

        if abl_type == "block_size":
            logger.info("  %-12s %-15s %-15s", "block_size", "acc_rate", "throughput")
            acc = section.get("acceptance_rates", {})
            tp = section.get("throughputs", {})
            for k in sorted(acc.keys(), key=lambda x: int(x)):
                logger.info(
                    "  %-12s %-15.4f %-15.1f",
                    k, acc[k], tp.get(k, 0),
                )

        elif abl_type == "gamma":
            logger.info(
                "  %-8s %-18s %-18s",
                "gamma", "acc_rate (quant)", "acc_rate (fp)",
            )
            acc_q = section.get("acceptance_rates_quantized", {})
            acc_fp = section.get("acceptance_rates_fp", {})
            for k in sorted(acc_q.keys(), key=lambda x: int(x)):
                logger.info(
                    "  %-8s %-18.4f %-18.4f",
                    k, acc_q[k], acc_fp.get(k, 0),
                )

        elif abl_type == "temperature":
            logger.info("  %-12s %-18s", "temperature", "acc_rate_delta")
            deltas = section.get("acceptance_rate_deltas", {})
            for k in sorted(deltas.keys(), key=lambda x: float(x)):
                logger.info("  %-12s %-+18.4f", k, deltas[k])

        elif abl_type == "seed":
            logger.info("  %-8s %-15s %-15s", "seed", "acc_rate", "throughput")
            acc = section.get("acceptance_rates", {})
            tp = section.get("throughputs", {})
            for k in sorted(acc.keys(), key=lambda x: int(x)):
                logger.info(
                    "  %-8s %-15.4f %-15.1f",
                    k, acc[k], tp.get(k, 0),
                )

        elif abl_type == "mixed_precision":
            logger.info("  Most sensitive layers: %s", section.get("sensitivity_order_top5"))
            for cfg_name in ("uniform_3bit", "uniform_4bit"):
                if cfg_name in section:
                    vals = section[cfg_name]
                    if "acceptance_rate" in vals:
                        logger.info(
                            "  %s: acc=%.4f, tp=%.1f",
                            cfg_name,
                            vals["acceptance_rate"],
                            vals["throughput"],
                        )

    logger.info("=" * 70)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SpecQuant Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--draft-model", type=str,
        default=os.environ.get("QWEN35_0_8B", "Qwen/Qwen3.5-0.8B"),
        help="HuggingFace model ID for the draft model",
    )
    parser.add_argument(
        "--target-model", type=str,
        default=os.environ.get("QWEN35_9B", "Qwen/Qwen3.5-9B"),
        help="HuggingFace model ID for the target model",
    )
    parser.add_argument(
        "--draft-device", type=str, default="cuda:0",
        help="Device for the draft model",
    )
    parser.add_argument(
        "--target-device", type=str, default="cuda:1",
        help="Device for the target model",
    )
    parser.add_argument(
        "--ablation-type", type=str, default="all",
        choices=["block_size", "gamma", "temperature", "seed", "mixed_precision", "all"],
        help="Which ablation sweep to run",
    )
    parser.add_argument(
        "--quant-bits", type=int, default=3,
        help="Quantization bit-width for non-mixed-precision sweeps",
    )
    parser.add_argument(
        "--num-trials", type=int, default=3,
        help="Number of trials per configuration",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Maximum new tokens to generate per run",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/ablations",
        help="Directory for ablation result JSON files",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("SpecQuant ablation study")
    logger.info("  draft_model  = %s", args.draft_model)
    logger.info("  target_model = %s", args.target_model)
    logger.info("  draft_device = %s", args.draft_device)
    logger.info("  target_device= %s", args.target_device)
    logger.info("  ablation_type= %s", args.ablation_type)
    logger.info("  quant_bits   = %d", args.quant_bits)
    logger.info("  num_trials   = %d", args.num_trials)
    logger.info("  max_new_tok  = %d", args.max_new_tokens)
    logger.info("  output_dir   = %s", args.output_dir)

    run_ablations(args)


if __name__ == "__main__":
    main()
