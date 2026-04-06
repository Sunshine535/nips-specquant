"""Evaluate empirical TV distance between full-precision and quantized verification logits.

Claim 3 validation: compare measured TV with theoretical bound from Proposition 1.

Three measurement modes:
1. **End-to-end** — uses SpeculativeDecoder.measure_tv_distance() to compare FP vs
   quantized logits via prefix/suffix split (the ground-truth empirical measurement).
2. **Per-layer** — registers forward hooks on each transformer layer's attention module
   to capture K, V tensors, quantizes them with SpecQuant and baselines, recomputes
   attention, and measures per-layer output TV distance.
3. **Theoretical bound** — computes Proposition 1 bound and validates empirical <= bound.

Output: JSON with per-bitwidth statistics, per-layer analysis, and per-position TV.
"""

import argparse
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.speculative_decode import SpeculativeDecoder
from src.quantized_verifier import _apply_rotary_pos_emb
from src.turboquant_kv import (
    HadamardRotation,
    QuantizedKVCache,
    ScalarQuantizer,
    compute_tv_bound,
)
from src.baselines import RTNKVCache, KIVIKVCache
from src.utils import aggregate_trials, save_results, validate_dual_gpu

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diverse prompt suite for robustness
# ---------------------------------------------------------------------------

DIVERSE_PROMPTS = [
    # Math / reasoning
    "Let x be a positive integer. If x^2 + 3x - 10 = 0, then the value of x is",
    "The derivative of f(x) = x^3 * ln(x) with respect to x can be computed as",
    "Consider a random variable X ~ N(0,1). The probability that X > 1.96 is",
    "In combinatorics, the number of ways to arrange n distinct objects in a circle is",
    "The eigenvalues of the matrix [[2, 1], [1, 2]] are computed by solving",
    # Code / programming
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if",
    "import torch\nimport torch.nn as nn\n\nclass TransformerBlock(nn.Module):\n    def __init__(self, d_model, nhead):\n        super().__init__()\n        self.attention =",
    "# Binary search implementation\ndef binary_search(sorted_list, target):\n    low, high = 0, len(sorted_list) - 1\n    while low <= high:\n        mid =",
    # Long-form / expository text
    "The theory of general relativity, proposed by Albert Einstein in 1915, fundamentally changed our understanding of gravity. Instead of treating gravity as a force between masses, Einstein described it as the curvature of spacetime caused by",
    "In the field of natural language processing, the transformer architecture introduced by Vaswani et al. in 2017 revolutionized sequence modeling by replacing recurrent connections with",
    "The history of quantum computing begins with Richard Feynman's 1982 observation that classical computers cannot efficiently simulate quantum mechanical systems. This insight led to",
    # Dialogue / conversational
    "User: Can you explain how gradient descent works in neural networks?\nAssistant: Gradient descent is an optimization algorithm that",
    "User: What are the key differences between GPT and BERT?\nAssistant: The main architectural difference is that GPT uses a",
    "User: How does attention mechanism work in transformers?\nAssistant: The attention mechanism computes a weighted sum of values, where the weights are determined by",
    # Technical / scientific
    "In distributed systems, the CAP theorem states that a distributed data store cannot simultaneously provide more than two of the following three guarantees:",
    "The Boltzmann distribution describes the probability of a system being in a state with energy E at temperature T as P(E) proportional to",
]


# ---------------------------------------------------------------------------
# Attention-layer discovery
# ---------------------------------------------------------------------------

def _find_attention_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """Find all attention sub-modules across common HuggingFace architectures.

    Supports Qwen, Llama, Mistral, GPT-NeoX, Falcon, and similar architectures
    that use 'self_attn' or 'attention' as the attention module name within
    each transformer layer.
    """
    attn_layers = []
    for name, module in model.named_modules():
        # Most HF models: model.layers[i].self_attn
        # Some: model.h[i].attn or model.transformer.h[i].self_attention
        basename = name.split(".")[-1]
        if basename in ("self_attn", "attn", "self_attention", "attention"):
            attn_layers.append((name, module))
    return attn_layers


def _find_transformer_layers(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    """Find the top-level transformer layer blocks (e.g. model.model.layers[i]).

    We look for the repeated decoder blocks that contain self_attn + mlp.
    """
    layers = []
    for name, module in model.named_modules():
        # Common patterns: model.model.layers.0, model.transformer.h.0
        # Check if this module has a self_attn child
        children_names = {n for n, _ in module.named_children()}
        if ("self_attn" in children_names or "attn" in children_names or
                "self_attention" in children_names):
            # Confirm it also has mlp/feed_forward
            if any(n in children_names for n in ("mlp", "feed_forward", "ffn")):
                layers.append((name, module))
    return layers


# ---------------------------------------------------------------------------
# Per-layer KV capture and TV measurement
# ---------------------------------------------------------------------------

class _KVCaptureHook:
    """Forward hook that captures Q, K, V tensors from an attention module.

    Works by intercepting the output of the attention layer. For HuggingFace
    models, attention modules return (attn_output, attn_weights, past_key_value)
    or similar tuples. We extract the past_key_value component which contains
    the K, V tensors for the current layer.

    Q is reconstructed from the module's input hidden_states via q_proj when
    available, giving real query tensors for accurate TV measurement.
    """

    def __init__(self, num_q_heads: int = 0, head_dim: int = 0):
        self.captured_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.captured_q: Optional[torch.Tensor] = None
        self.handle: Optional[torch.utils.hooks.RemovableHook] = None
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim

    def hook_fn(self, module, args, output):
        """Extract Q, K, V from attention module output."""
        # HF attention modules typically return:
        # (attn_output, attn_weights, past_key_value)
        # where past_key_value is (key, value) each of shape
        # (batch, num_kv_heads, seq_len, head_dim)
        if isinstance(output, tuple) and len(output) >= 3:
            past_kv = output[2]
            if isinstance(past_kv, tuple) and len(past_kv) == 2:
                self.captured_kv = (past_kv[0].detach().clone(),
                                    past_kv[1].detach().clone())
            elif hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
                # DynamicCache or similar
                if len(past_kv.key_cache) > 0:
                    self.captured_kv = (
                        past_kv.key_cache[-1].detach().clone(),
                        past_kv.value_cache[-1].detach().clone(),
                    )
        elif isinstance(output, tuple) and len(output) == 2:
            # Some models return (attn_output, past_key_value)
            past_kv = output[1]
            if isinstance(past_kv, tuple) and len(past_kv) == 2:
                k, v = past_kv
                if isinstance(k, torch.Tensor) and k.dim() == 4:
                    self.captured_kv = (k.detach().clone(), v.detach().clone())

        # Capture Q from hidden_states via q_proj if available
        q_proj = getattr(module, "q_proj", None)
        if q_proj is not None and len(args) > 0:
            hidden_states = args[0]
            if isinstance(hidden_states, torch.Tensor):
                with torch.no_grad():
                    q = q_proj(hidden_states)
                    B, S = hidden_states.shape[:2]
                    if self.num_q_heads > 0 and self.head_dim > 0:
                        q = q.view(B, S, self.num_q_heads, self.head_dim)
                        q = q.transpose(1, 2)  # (B, num_q_heads, S, head_dim)

                    # Apply RoPE to Q for RoPE-based models (Llama, Qwen,
                    # Mistral).  Without RoPE, Q has no positional signal
                    # and TV distance metrics are meaningless.
                    # Handle both new HF API (position_embeddings kwarg)
                    # and older API (module.rotary_emb callable) with
                    # try/except fallback for different call signatures.
                    rotary_emb = getattr(module, "rotary_emb", None)
                    if rotary_emb is not None and q.dim() == 4:
                        seq_len = q.shape[2]
                        position_ids = torch.arange(
                            seq_len, device=q.device,
                        ).unsqueeze(0)
                        try:
                            # New HF API: rotary_emb(x, position_ids)
                            cos, sin = rotary_emb(q, position_ids)
                        except TypeError:
                            # Older API: rotary_emb(x, seq_len)
                            cos, sin = rotary_emb(q, seq_len)
                            if cos.dim() == 4:
                                cos = cos[:, :, :seq_len, :]
                                sin = sin[:, :, :seq_len, :]
                            elif cos.dim() == 2:
                                cos = cos[:seq_len]
                                sin = sin[:seq_len]
                            elif cos.dim() == 3:
                                cos = cos[:, :seq_len, :]
                                sin = sin[:, :seq_len, :]
                        q = _apply_rotary_pos_emb(q, cos, sin)

                    self.captured_q = q.detach().clone()

    def register(self, module: torch.nn.Module):
        self.handle = module.register_forward_hook(self.hook_fn)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def reset(self):
        self.captured_kv = None
        self.captured_q = None


def _quantize_kv_specquant(
    key: torch.Tensor,
    value: torch.Tensor,
    bits: int,
    block_size: int,
    head_dim: int,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round-trip quantize K, V using SpecQuant (Hadamard + scalar)."""
    rotation = HadamardRotation(head_dim, seed=seed)
    quantizer = ScalarQuantizer(bits=bits, block_size=block_size)

    k_rot = rotation.rotate(key.float())
    v_rot = rotation.rotate(value.float())

    k_codes, k_scales, k_zeros = quantizer.quantize(k_rot)
    v_codes, v_scales, v_zeros = quantizer.quantize(v_rot)

    k_deq = quantizer.dequantize(k_codes, k_scales, k_zeros)
    v_deq = quantizer.dequantize(v_codes, v_scales, v_zeros)

    k_out = rotation.inverse_rotate(k_deq).to(key.dtype)
    v_out = rotation.inverse_rotate(v_deq).to(value.dtype)
    return k_out, v_out


def _quantize_kv_rtn(
    key: torch.Tensor,
    value: torch.Tensor,
    bits: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round-trip quantize K, V using RTN baseline (no rotation)."""
    cache = RTNKVCache(
        num_layers=1, num_kv_heads=num_kv_heads,
        head_dim=head_dim, bits=bits, block_size=block_size,
    )
    cache.compress_and_store(0, key.float(), value.float())
    k_deq, v_deq = cache.get_rotated_kv(0)  # "rotated" is a misnomer for RTN, just deq
    return k_deq.to(key.dtype), v_deq.to(value.dtype)


def _quantize_kv_kivi(
    key: torch.Tensor,
    value: torch.Tensor,
    bits: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round-trip quantize K, V using KIVI baseline."""
    cache = KIVIKVCache(
        num_layers=1, num_kv_heads=num_kv_heads,
        head_dim=head_dim, bits=bits, block_size=block_size,
    )
    cache.compress_and_store(0, key.float(), value.float())
    k_deq, v_deq = cache.get_rotated_kv(0)
    return k_deq.to(key.dtype), v_deq.to(value.dtype)


def _recompute_attention_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Compute scaled dot-product attention from Q, K, V tensors.

    Args:
        query:  (batch, num_q_heads, q_len, head_dim)
        key:    (batch, num_kv_heads, kv_len, head_dim)
        value:  (batch, num_kv_heads, kv_len, head_dim)
        head_dim: dimension per head for scale factor

    Returns:
        Attention output (batch, num_q_heads, q_len, head_dim)
    """
    scale = 1.0 / math.sqrt(head_dim)

    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_q_heads != num_kv_heads:
        repeat_factor = num_q_heads // num_kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value.float())


# ---------------------------------------------------------------------------
# Per-layer TV measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_per_layer_tv(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    bits: int,
    block_size: int = 128,
    seed: int = 42,
) -> Dict[str, Any]:
    """Measure per-layer TV distance introduced by KV quantization.

    Registers forward hooks on each attention layer to capture K, V tensors.
    For each layer independently, quantizes the captured K, V with SpecQuant
    and baselines, recomputes attention output, and measures the TV distance
    between full-precision and quantized attention outputs.

    Args:
        model: target model (already on device)
        input_ids: (1, seq_len) token IDs
        bits: quantization bit-width
        block_size: block size for scalar quantizer
        seed: seed for Hadamard rotation

    Returns:
        Dict with per-layer TV for each quantization method, plus layer-level
        statistics (which layers have highest TV, KV range stats).
    """
    device = next(model.parameters()).device
    config = model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    # Find attention modules
    attn_layers = _find_attention_layers(model)
    if len(attn_layers) == 0:
        logger.warning("Could not find attention layers in model; skipping per-layer TV")
        return {"error": "no_attention_layers_found", "bits": bits}

    logger.info(
        "Found %d attention layers for per-layer KV capture (expected %d)",
        len(attn_layers), num_layers,
    )

    # Register hooks on all attention layers (pass geometry for Q capture)
    num_q_heads = config.num_attention_heads
    hooks = []
    for _name, attn_module in attn_layers:
        hook = _KVCaptureHook(num_q_heads=num_q_heads, head_dim=head_dim)
        hook.register(attn_module)
        hooks.append(hook)

    # Forward pass with use_cache=True to get K, V in attention outputs
    try:
        _output = model(input_ids.to(device), use_cache=True)
    finally:
        # Always clean up hooks
        for hook in hooks:
            hook.remove()

    # Collect per-layer results
    per_layer_results = []
    captured_count = 0

    for layer_idx, hook in enumerate(hooks):
        layer_result = {"layer_idx": layer_idx}

        if hook.captured_kv is None:
            layer_result["status"] = "no_kv_captured"
            per_layer_results.append(layer_result)
            continue

        captured_count += 1
        k_fp, v_fp = hook.captured_kv  # (batch, num_kv_heads, seq_len, head_dim)
        layer_result["status"] = "ok"
        layer_result["k_shape"] = list(k_fp.shape)
        layer_result["v_shape"] = list(v_fp.shape)

        # Compute KV range statistics (useful for theoretical bound calibration)
        k_range = (k_fp.max() - k_fp.min()).item()
        v_range = (v_fp.max() - v_fp.min()).item()
        v_fnorm = v_fp.float().norm().item()
        layer_result["k_range"] = k_range
        layer_result["v_range"] = v_range
        layer_result["v_frobenius_norm"] = v_fnorm
        layer_result["k_std"] = k_fp.float().std().item()
        layer_result["v_std"] = v_fp.float().std().item()

        # Use real Q tensors when captured; fall back to K as synthetic Q.
        if hook.captured_q is not None:
            q_for_attn = hook.captured_q
        else:
            q_for_attn = k_fp.clone()

        # Full-precision attention output
        attn_fp = _recompute_attention_output(q_for_attn, k_fp, v_fp, head_dim)

        # SpecQuant quantized attention
        k_sq, v_sq = _quantize_kv_specquant(
            k_fp, v_fp, bits, block_size, head_dim, seed,
        )
        attn_sq = _recompute_attention_output(q_for_attn, k_sq, v_sq, head_dim)

        # RTN baseline
        k_rtn, v_rtn = _quantize_kv_rtn(
            k_fp, v_fp, bits, block_size, num_kv_heads, head_dim,
        )
        attn_rtn = _recompute_attention_output(q_for_attn, k_rtn, v_rtn, head_dim)

        # KIVI baseline
        k_kivi, v_kivi = _quantize_kv_kivi(
            k_fp, v_fp, bits, block_size, num_kv_heads, head_dim,
        )
        attn_kivi = _recompute_attention_output(q_for_attn, k_kivi, v_kivi, head_dim)

        # Normalize attention outputs to probability distributions (softmax over
        # the head_dim axis) so TV distance is meaningful
        def _to_probs(x: torch.Tensor) -> torch.Tensor:
            # x: (batch, heads, seq_len, head_dim) -> treat last dim as logits
            return F.softmax(x.float(), dim=-1)

        probs_fp = _to_probs(attn_fp)
        probs_sq = _to_probs(attn_sq)
        probs_rtn = _to_probs(attn_rtn)
        probs_kivi = _to_probs(attn_kivi)

        # TV per position: avg over batch and heads
        # shape: (batch, heads, seq_len)
        tv_sq = 0.5 * (probs_fp - probs_sq).abs().sum(dim=-1).mean(dim=(0, 1))
        tv_rtn = 0.5 * (probs_fp - probs_rtn).abs().sum(dim=-1).mean(dim=(0, 1))
        tv_kivi = 0.5 * (probs_fp - probs_kivi).abs().sum(dim=-1).mean(dim=(0, 1))

        layer_result["specquant_tv_mean"] = tv_sq.mean().item()
        layer_result["specquant_tv_std"] = tv_sq.std().item() if tv_sq.numel() > 1 else 0.0
        layer_result["specquant_tv_per_position"] = tv_sq.cpu().tolist()

        layer_result["rtn_tv_mean"] = tv_rtn.mean().item()
        layer_result["rtn_tv_std"] = tv_rtn.std().item() if tv_rtn.numel() > 1 else 0.0

        layer_result["kivi_tv_mean"] = tv_kivi.mean().item()
        layer_result["kivi_tv_std"] = tv_kivi.std().item() if tv_kivi.numel() > 1 else 0.0

        # Also compute raw L2 distance of attention outputs (not TV, but useful)
        layer_result["specquant_l2"] = (attn_fp - attn_sq).float().norm().item()
        layer_result["rtn_l2"] = (attn_fp - attn_rtn).float().norm().item()
        layer_result["kivi_l2"] = (attn_fp - attn_kivi).float().norm().item()

        # Per-layer theoretical bound with actual measured ranges
        layer_result["theoretical_tv_bound"] = compute_tv_bound(
            w_o_fnorm=1.0,
            range_k=k_range,
            range_v=v_range,
            v_fnorm=v_fnorm,
            dim=head_dim,
            bits=bits,
            block_size=block_size,
            temperature=1.0,
        )

        per_layer_results.append(layer_result)

    # Clean up captured tensors
    for hook in hooks:
        hook.reset()

    return {
        "bits": bits,
        "block_size": block_size,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "num_layers_captured": captured_count,
        "seq_len": input_ids.shape[1],
        "per_layer": per_layer_results,
    }


# ---------------------------------------------------------------------------
# End-to-end empirical TV via SpeculativeDecoder.measure_tv_distance()
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_e2e_tv(
    decoder: SpeculativeDecoder,
    input_ids: torch.Tensor,
    num_positions: int,
) -> Dict[str, Any]:
    """Thin wrapper around SpeculativeDecoder.measure_tv_distance().

    This is the ground-truth empirical TV measurement: prefix/suffix split,
    FP vs quantized KV, compare full output logit distributions.
    """
    return decoder.measure_tv_distance(input_ids, num_positions=num_positions)


# ---------------------------------------------------------------------------
# Theoretical bound computation with measured statistics
# ---------------------------------------------------------------------------

def compute_calibrated_bound(
    per_layer_results: List[Dict[str, Any]],
    bits: int,
    block_size: int,
    head_dim: int,
) -> Dict[str, Any]:
    """Compute theoretical TV bound using empirically measured KV range statistics.

    Instead of assuming range_k=4, range_v=4, uses the actual measured ranges
    from per-layer KV capture for a tighter bound estimate.
    """
    k_ranges = []
    v_ranges = []
    v_fnorms = []

    for layer in per_layer_results:
        if layer.get("status") != "ok":
            continue
        k_ranges.append(layer["k_range"])
        v_ranges.append(layer["v_range"])
        v_fnorms.append(layer["v_frobenius_norm"])

    if not k_ranges:
        return {"error": "no_valid_layers"}

    # Use worst-case (max) ranges for the bound
    bound_worst = compute_tv_bound(
        w_o_fnorm=1.0,
        range_k=max(k_ranges),
        range_v=max(v_ranges),
        v_fnorm=max(v_fnorms),
        dim=head_dim,
        bits=bits,
        block_size=block_size,
        temperature=1.0,
    )

    # Also compute with average ranges for comparison
    bound_avg = compute_tv_bound(
        w_o_fnorm=1.0,
        range_k=float(np.mean(k_ranges)),
        range_v=float(np.mean(v_ranges)),
        v_fnorm=float(np.mean(v_fnorms)),
        dim=head_dim,
        bits=bits,
        block_size=block_size,
        temperature=1.0,
    )

    # Default assumption bound (range=4, fnorm=1) for comparison
    bound_default = compute_tv_bound(
        w_o_fnorm=1.0,
        range_k=4.0,
        range_v=4.0,
        v_fnorm=1.0,
        dim=head_dim,
        bits=bits,
        block_size=block_size,
        temperature=1.0,
    )

    return {
        "bound_worst_case": bound_worst,
        "bound_avg_case": bound_avg,
        "bound_default_assumption": bound_default,
        "measured_k_range_max": max(k_ranges),
        "measured_k_range_mean": float(np.mean(k_ranges)),
        "measured_v_range_max": max(v_ranges),
        "measured_v_range_mean": float(np.mean(v_ranges)),
        "measured_v_fnorm_max": max(v_fnorms),
        "measured_v_fnorm_mean": float(np.mean(v_fnorms)),
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_models(
    target_model_name: str,
    draft_model_name: Optional[str],
    draft_device: str,
    target_device: str,
) -> Tuple[AutoModelForCausalLM, Optional[AutoModelForCausalLM], AutoTokenizer]:
    """Load target (and optionally draft) models on specified devices."""
    logger.info("Loading target model: %s -> %s", target_model_name, target_device)
    target_model = AutoModelForCausalLM.from_pretrained(
        target_model_name,
        torch_dtype=torch.float16,
        device_map=target_device,
        trust_remote_code=True,
    )
    target_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_name, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    draft_model = None
    if draft_model_name:
        logger.info("Loading draft model: %s -> %s", draft_model_name, draft_device)
        draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_name,
            torch_dtype=torch.float16,
            device_map=draft_device,
            trust_remote_code=True,
        )
        draft_model.eval()

    return target_model, draft_model, tokenizer


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    target_model: AutoModelForCausalLM,
    draft_model: Optional[AutoModelForCausalLM],
    tokenizer: AutoTokenizer,
    bits_list: List[int],
    block_size: int,
    num_samples: int,
    seq_length: int,
    seed: int,
) -> Dict[str, Any]:
    """Run full TV distance evaluation across bit-widths, prompts, and layers.

    Returns a structured result dict with:
    - per_bitwidth: empirical TV stats, theoretical bounds, bound validation
    - per_layer_analysis: which layers contribute most TV
    - per_position_analysis: how TV varies across sequence positions
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Select prompts (cycle through DIVERSE_PROMPTS if num_samples > len)
    prompts = []
    for i in range(num_samples):
        prompts.append(DIVERSE_PROMPTS[i % len(DIVERSE_PROMPTS)])

    results = {
        "per_bitwidth": {},
        "per_layer_analysis": {},
        "per_position_analysis": {},
    }

    for bits in bits_list:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating %d-bit quantization (%d samples)", bits, num_samples)
        logger.info("=" * 60)

        # -- End-to-end TV via SpeculativeDecoder --
        e2e_tv_values = []
        e2e_tv_per_position_all = []

        if draft_model is not None:
            decoder = SpeculativeDecoder(
                draft_model=draft_model,
                target_model=target_model,
                tokenizer=tokenizer,
                quant_bits=bits,
                quant_block_size=block_size,
                quant_seed=seed,
            )

            for sample_idx, prompt in enumerate(prompts):
                input_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                    max_length=seq_length,
                ).input_ids

                tv_result = measure_e2e_tv(decoder, input_ids, num_positions=seq_length)
                e2e_tv_values.append(tv_result["tv_mean"])
                e2e_tv_per_position_all.append(tv_result["tv_per_position"])

                if (sample_idx + 1) % 10 == 0 or sample_idx == 0:
                    logger.info(
                        "  [e2e] sample %d/%d: TV mean=%.6f, std=%.6f",
                        sample_idx + 1, num_samples,
                        tv_result["tv_mean"], tv_result["tv_std"],
                    )

            del decoder
            torch.cuda.empty_cache()

        # -- Per-layer TV via hooks --
        per_layer_tv_samples = []  # List of per-layer result dicts, one per sample

        for sample_idx, prompt in enumerate(prompts):
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=seq_length,
            ).input_ids

            layer_result = measure_per_layer_tv(
                target_model, input_ids,
                bits=bits, block_size=block_size, seed=seed,
            )
            per_layer_tv_samples.append(layer_result)

            if (sample_idx + 1) % 10 == 0 or sample_idx == 0:
                # Report aggregate per-layer TV for this sample
                ok_layers = [
                    lr for lr in layer_result.get("per_layer", [])
                    if lr.get("status") == "ok"
                ]
                if ok_layers:
                    mean_tv = np.mean([lr["specquant_tv_mean"] for lr in ok_layers])
                    logger.info(
                        "  [per-layer] sample %d/%d: avg layer TV=%.6f (%d layers)",
                        sample_idx + 1, num_samples, mean_tv, len(ok_layers),
                    )

        # -- Aggregate e2e TV statistics --
        e2e_stats = {}
        if e2e_tv_values:
            e2e_stats = aggregate_trials(e2e_tv_values)
            e2e_stats["per_sample_values"] = e2e_tv_values
            logger.info(
                "  [e2e aggregate] %d-bit: TV mean=%.6f, std=%.6f, CI=(%.6f, %.6f)",
                bits, e2e_stats["mean"], e2e_stats["std"],
                e2e_stats["ci_lower"], e2e_stats["ci_upper"],
            )

        # -- Aggregate per-layer TV statistics --
        # Collect per-layer means across all samples
        num_layers_found = 0
        if per_layer_tv_samples:
            num_layers_found = per_layer_tv_samples[0].get("num_layers_captured", 0)

        layer_agg = {}  # layer_idx -> {method -> aggregate_trials result}
        for layer_idx in range(num_layers_found):
            sq_vals, rtn_vals, kivi_vals = [], [], []
            k_ranges, v_ranges = [], []

            for sample in per_layer_tv_samples:
                layers = sample.get("per_layer", [])
                if layer_idx < len(layers) and layers[layer_idx].get("status") == "ok":
                    lr = layers[layer_idx]
                    sq_vals.append(lr["specquant_tv_mean"])
                    rtn_vals.append(lr["rtn_tv_mean"])
                    kivi_vals.append(lr["kivi_tv_mean"])
                    k_ranges.append(lr["k_range"])
                    v_ranges.append(lr["v_range"])

            layer_entry = {"layer_idx": layer_idx}
            if len(sq_vals) >= 2:
                layer_entry["specquant"] = aggregate_trials(sq_vals)
                layer_entry["rtn"] = aggregate_trials(rtn_vals)
                layer_entry["kivi"] = aggregate_trials(kivi_vals)
                layer_entry["k_range_mean"] = float(np.mean(k_ranges))
                layer_entry["v_range_mean"] = float(np.mean(v_ranges))
            elif len(sq_vals) == 1:
                layer_entry["specquant"] = {"mean": sq_vals[0], "std": 0.0, "n_trials": 1}
                layer_entry["rtn"] = {"mean": rtn_vals[0], "std": 0.0, "n_trials": 1}
                layer_entry["kivi"] = {"mean": kivi_vals[0], "std": 0.0, "n_trials": 1}
                layer_entry["k_range_mean"] = k_ranges[0]
                layer_entry["v_range_mean"] = v_ranges[0]

            layer_agg[layer_idx] = layer_entry

        # Find highest-TV layers
        layer_tv_ranking = sorted(
            [(idx, entry.get("specquant", {}).get("mean", 0.0))
             for idx, entry in layer_agg.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        # -- Per-position TV analysis --
        position_analysis = {}
        if e2e_tv_per_position_all:
            # Align positions across samples (they may differ in length)
            max_pos = max(len(pp) for pp in e2e_tv_per_position_all)
            position_means = []
            position_stds = []
            for pos in range(max_pos):
                vals = [
                    pp[pos] for pp in e2e_tv_per_position_all
                    if pos < len(pp)
                ]
                if vals:
                    position_means.append(float(np.mean(vals)))
                    position_stds.append(float(np.std(vals)) if len(vals) > 1 else 0.0)
                else:
                    position_means.append(0.0)
                    position_stds.append(0.0)

            position_analysis = {
                "num_positions": max_pos,
                "tv_by_position_mean": position_means,
                "tv_by_position_std": position_stds,
                # Summarize trend: early vs late positions
                "early_half_tv_mean": float(np.mean(position_means[: max_pos // 2])) if max_pos > 1 else 0.0,
                "late_half_tv_mean": float(np.mean(position_means[max_pos // 2 :])) if max_pos > 1 else 0.0,
            }

        # -- Theoretical bound with calibrated ranges --
        all_per_layer = []
        for sample in per_layer_tv_samples:
            all_per_layer.extend(sample.get("per_layer", []))

        config = target_model.config
        head_dim = config.hidden_size // config.num_attention_heads
        calibrated_bound = compute_calibrated_bound(
            all_per_layer, bits, block_size, head_dim,
        )

        # -- Bound validation: does empirical <= theoretical? --
        bound_validation = {}
        if e2e_stats and "error" not in calibrated_bound:
            empirical_tv = e2e_stats["mean"]
            for bound_name in ["bound_worst_case", "bound_avg_case", "bound_default_assumption"]:
                bound_val = calibrated_bound[bound_name]
                bound_validation[bound_name] = {
                    "theoretical": bound_val,
                    "empirical": empirical_tv,
                    "bound_holds": empirical_tv <= bound_val,
                    "margin": bound_val - empirical_tv,
                    "tightness_ratio": empirical_tv / bound_val if bound_val > 0 else 0.0,
                }

        # -- Assemble per-bitwidth results --
        bit_result = {
            "bits": bits,
            "block_size": block_size,
            "num_samples": num_samples,
            "e2e_tv": e2e_stats,
            "per_layer": {str(k): v for k, v in layer_agg.items()},
            "layer_tv_ranking_top5": layer_tv_ranking[:5],
            "layer_tv_ranking_bottom5": layer_tv_ranking[-5:] if len(layer_tv_ranking) > 5 else [],
            "calibrated_bounds": calibrated_bound,
            "bound_validation": bound_validation,
        }

        results["per_bitwidth"][f"{bits}bit"] = bit_result
        results["per_position_analysis"][f"{bits}bit"] = position_analysis

        # Log bound validation summary
        if bound_validation:
            for bname, bval in bound_validation.items():
                status = "HOLDS" if bval["bound_holds"] else "VIOLATED"
                logger.info(
                    "  [bound check] %d-bit %s: empirical=%.6f, bound=%.6f -> %s (tightness=%.2f%%)",
                    bits, bname, bval["empirical"], bval["theoretical"],
                    status, bval["tightness_ratio"] * 100,
                )

    # -- Cross-bitwidth per-layer analysis --
    results["per_layer_analysis"] = {
        str(bits): {
            str(k): v
            for k, v in results["per_bitwidth"].get(f"{bits}bit", {}).get("per_layer", {}).items()
        }
        for bits in bits_list
    }

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Empirical TV distance validation (Claim 3 / Proposition 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Full evaluation with draft model for e2e TV
  python -m scripts.eval_tv_distance \\
      --target-model Qwen/Qwen3.5-14B \\
      --draft-model Qwen/Qwen3.5-0.8B \\
      --bits 2 3 4 --num-samples 50

  # Per-layer only (no draft model needed)
  python -m scripts.eval_tv_distance \\
      --target-model Qwen/Qwen3.5-14B \\
      --bits 3 4 --num-samples 20
""",
    )
    parser.add_argument(
        "--target-model", type=str, required=True,
        help="Target model name or path (e.g. Qwen/Qwen3.5-14B)",
    )
    parser.add_argument(
        "--draft-model", type=str, default=None,
        help="Draft model for decoder-based e2e TV measurement (optional)",
    )
    parser.add_argument(
        "--bits", type=int, nargs="+", default=[2, 3, 4],
        help="Quantization bit-widths to evaluate",
    )
    parser.add_argument(
        "--block-size", type=int, default=128,
        help="Block size for scalar quantizer",
    )
    parser.add_argument(
        "--num-samples", type=int, default=100,
        help="Number of prompt samples per bit-width",
    )
    parser.add_argument(
        "--seq-length", type=int, default=512,
        help="Maximum sequence length for tokenized prompts",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/tv_validation",
        help="Directory for output JSON files",
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
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # -- Pre-flight checks --
    logger.info("TV Distance Validation")
    logger.info("  target_model  = %s", args.target_model)
    logger.info("  draft_model   = %s", args.draft_model or "(none -- per-layer only)")
    logger.info("  bits          = %s", args.bits)
    logger.info("  block_size    = %d", args.block_size)
    logger.info("  num_samples   = %d", args.num_samples)
    logger.info("  seq_length    = %d", args.seq_length)
    logger.info("  draft_device  = %s", args.draft_device)
    logger.info("  target_device = %s", args.target_device)
    logger.info("  seed          = %d", args.seed)

    if args.draft_model:
        try:
            validate_dual_gpu()
        except RuntimeError as e:
            logger.warning("Dual-GPU validation failed: %s", e)
            logger.warning("Falling back to single GPU; e2e TV may be slow")

    # -- Load models --
    target_model, draft_model, tokenizer = load_models(
        target_model_name=args.target_model,
        draft_model_name=args.draft_model,
        draft_device=args.draft_device,
        target_device=args.target_device,
    )

    # -- Run evaluation --
    t_start = time.perf_counter()

    eval_results = run_evaluation(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        bits_list=args.bits,
        block_size=args.block_size,
        num_samples=args.num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
    )

    wall_time = time.perf_counter() - t_start

    # -- Assemble final output --
    output = {
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "bits": args.bits,
            "block_size": args.block_size,
            "num_samples": args.num_samples,
            "seq_length": args.seq_length,
            "draft_device": args.draft_device,
            "target_device": args.target_device,
            "seed": args.seed,
        },
        "wall_time_seconds": wall_time,
        "results": eval_results,
    }

    # -- Save --
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_tag = args.target_model.replace("/", "_")
    filename = f"tv_validation_{model_tag}_{timestamp}.json"
    save_results(output, args.output_dir, filename)

    # -- Print summary --
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Wall time: %.1f seconds", wall_time)

    for bits in args.bits:
        key = f"{bits}bit"
        bit_res = eval_results.get("per_bitwidth", {}).get(key, {})
        e2e = bit_res.get("e2e_tv", {})
        bv = bit_res.get("bound_validation", {})

        if e2e:
            logger.info(
                "  %d-bit e2e TV: mean=%.6f, std=%.6f, CI=(%.6f, %.6f)",
                bits, e2e.get("mean", 0), e2e.get("std", 0),
                e2e.get("ci_lower", 0), e2e.get("ci_upper", 0),
            )

        ranking = bit_res.get("layer_tv_ranking_top5", [])
        if ranking:
            logger.info(
                "  %d-bit highest-TV layers: %s",
                bits,
                ", ".join(f"L{idx}={tv:.6f}" for idx, tv in ranking[:3]),
            )

        for bname, bval in bv.items():
            status = "PASS" if bval["bound_holds"] else "FAIL"
            logger.info(
                "  %d-bit %s: %s (empirical=%.6f <= bound=%.6f, tightness=%.1f%%)",
                bits, bname, status,
                bval["empirical"], bval["theoretical"],
                bval["tightness_ratio"] * 100,
            )

    logger.info("Results saved to %s/%s", args.output_dir, filename)


if __name__ == "__main__":
    main()
