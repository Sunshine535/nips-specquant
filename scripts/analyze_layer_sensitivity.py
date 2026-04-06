"""Per-layer sensitivity analysis under KV quantization using REAL model activations.

Phase 5: Register forward hooks on a HuggingFace model to capture real K, V
tensors during inference.  For each layer, compare quantization error across
SpecQuant (Hadamard + scalar), RTN, and KIVI baselines.  Report per-layer
MSE, cosine similarity, max absolute error, attention output distortion,
activation statistics (range, kurtosis), and identify sensitive layers.
"""

import argparse
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sp_stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.baselines import RTNKVCache, KIVIKVCache
from src.quantized_verifier import _apply_rotary_pos_emb
from src.turboquant_kv import HadamardRotation, QuantizedKVCache, ScalarQuantizer
from src.utils import aggregate_trials, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diverse prompts for multi-prompt statistical analysis
# ---------------------------------------------------------------------------

DIVERSE_PROMPTS = [
    # Reasoning / math
    "Let me solve this step by step. A train travels at 60 km/h for 2 hours, then 80 km/h for 3 hours. The total distance is",
    # Code
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\n# Testing the function with",
    # Creative writing
    "The old lighthouse keeper climbed the spiral staircase one last time. The beam had guided ships for a hundred years, but tonight the automated system would take over. He paused at the top and",
    # Science / technical
    "Quantum entanglement occurs when two particles become correlated such that the quantum state of one instantly influences the other, regardless of the distance separating them. This phenomenon was described by Einstein as",
    # Dialogue / conversation
    "Alice: Have you finished the quarterly report yet?\nBob: Almost. I'm still waiting on the sales figures from the Tokyo office.\nAlice: Can you send me what you have so far? I need to",
    # Legal / formal
    "WHEREAS the parties have agreed to enter into this binding agreement on the terms set forth herein, and WHEREAS the consideration exchanged is deemed sufficient and adequate, the parties hereby agree to the following provisions:",
    # News / factual
    "The International Space Station orbits Earth at an altitude of approximately 408 kilometers, traveling at a speed of about 28,000 kilometers per hour. This means it completes one full orbit every",
    # Poetry / literary
    "In the twilight of a summer evening, the fields stretched golden and endless, each blade of wheat catching the last rays of the setting sun. A lone figure walked along the dirt path,",
    # Instruction following
    "Please explain the difference between supervised and unsupervised machine learning. In supervised learning, the model is trained on labeled data where the correct output is known. The algorithm learns to map inputs to outputs by",
    # Multi-lingual context
    "The concept of 'wabi-sabi' in Japanese aesthetics embraces imperfection and transience. Similarly, the Danish concept of 'hygge' emphasizes comfort and contentment. These cultural philosophies suggest that happiness comes from",
    # Mathematical proof
    "Theorem: For any prime p > 2, the sum 1 + 2 + ... + (p-1) is divisible by p. Proof: We can pair each integer k with p-k. Since k + (p-k) = p, and there are (p-1)/2 such pairs,",
    # Technical documentation
    "To configure the distributed training pipeline, set the following environment variables: MASTER_ADDR for the master node IP, MASTER_PORT for the communication port, WORLD_SIZE for the total number of processes, and RANK for",
]


# ---------------------------------------------------------------------------
# Hook-based KV capture
# ---------------------------------------------------------------------------

class KVCaptureHook:
    """Register forward hooks on attention layers to capture real Q, K, V tensors.

    Supports common HuggingFace architectures (Qwen2, Llama, Mistral, etc.)
    where the attention module stores key/value states as
    ``(batch, num_kv_heads, seq_len, head_dim)`` tensors.

    Q is reconstructed from the module's input hidden_states via q_proj when
    available, giving real query tensors for accurate attention distortion
    measurement.
    """

    def __init__(self, num_q_heads: int = 0, head_dim: int = 0):
        self.captured_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.captured_q: Dict[int, torch.Tensor] = {}
        self._hooks: List[Any] = []
        self.num_q_heads = num_q_heads
        self.head_dim = head_dim

    def _make_hook(self, layer_idx: int):
        """Create a hook function that captures Q, K, V from the attention module."""
        capture = self  # closure reference

        def hook_fn(module, args, output):
            # HuggingFace attention modules return (attn_output, attn_weights, past_key_value)
            # or (attn_output, past_key_value) depending on config.
            # The past_key_value is a tuple of (key, value) each shaped
            # (batch, num_kv_heads, seq_len, head_dim).
            past_kv = None
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, tuple) and len(item) == 2:
                        k_cand, v_cand = item
                        if (isinstance(k_cand, torch.Tensor) and
                                isinstance(v_cand, torch.Tensor) and
                                k_cand.ndim == 4 and v_cand.ndim == 4):
                            past_kv = (k_cand, v_cand)
                            break
            if past_kv is not None:
                k, v = past_kv
                capture.captured_kv[layer_idx] = (k.detach().cpu().float(),
                                                   v.detach().cpu().float())

            # Capture Q from hidden_states via q_proj if available
            q_proj = getattr(module, "q_proj", None)
            if q_proj is not None and len(args) > 0:
                hidden_states = args[0]
                if isinstance(hidden_states, torch.Tensor):
                    with torch.no_grad():
                        q = q_proj(hidden_states)
                        B, S = hidden_states.shape[:2]
                        if capture.num_q_heads > 0 and capture.head_dim > 0:
                            q = q.view(B, S, capture.num_q_heads, capture.head_dim)
                            q = q.transpose(1, 2)  # (B, num_q_heads, S, head_dim)

                        # Apply RoPE to Q for RoPE-based models (Llama, Qwen,
                        # Mistral).  Without RoPE, Q has no positional signal
                        # and attention distortion metrics are meaningless.
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

                        capture.captured_q[layer_idx] = q.detach().cpu().float()

        return hook_fn

    def register(self, model: torch.nn.Module):
        """Find and hook all attention layers in the model."""
        self.clear()

        # Try common HuggingFace patterns
        attn_modules = []

        # Pattern 1: model.model.layers[i].self_attn (Qwen2, Llama, Mistral)
        inner = getattr(model, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
            if layers is not None:
                for i, layer in enumerate(layers):
                    attn = getattr(layer, "self_attn", None)
                    if attn is not None:
                        attn_modules.append((i, attn))

        # Pattern 2: model.transformer.h[i].attn (GPT-2 style)
        if not attn_modules:
            transformer = getattr(model, "transformer", None)
            if transformer is not None:
                h = getattr(transformer, "h", None)
                if h is not None:
                    for i, layer in enumerate(h):
                        attn = getattr(layer, "attn", getattr(layer, "self_attn", None))
                        if attn is not None:
                            attn_modules.append((i, attn))

        if not attn_modules:
            raise RuntimeError(
                "Could not find attention modules in the model. "
                "Supported architectures: Qwen2, Llama, Mistral, GPT-2."
            )

        for layer_idx, attn_module in attn_modules:
            handle = attn_module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(handle)

        logger.info("Registered KV capture hooks on %d attention layers", len(attn_modules))

    def clear(self):
        """Remove all hooks and discard captured data."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self.captured_kv.clear()
        self.captured_q.clear()

    def clear_captures(self):
        """Discard captured KV data but keep hooks active."""
        self.captured_kv.clear()
        self.captured_q.clear()


# ---------------------------------------------------------------------------
# Per-layer quantization error measurement
# ---------------------------------------------------------------------------

def quantize_specquant(
    tensor: torch.Tensor,
    rotation: HadamardRotation,
    quantizer: ScalarQuantizer,
) -> torch.Tensor:
    """SpecQuant: Hadamard rotate -> quantize -> dequantize -> inverse rotate."""
    rotated = rotation.rotate(tensor)
    codes, scales, zeros = quantizer.quantize(rotated)
    deq = quantizer.dequantize(codes, scales, zeros)
    return rotation.inverse_rotate(deq)


def quantize_rtn(
    tensor: torch.Tensor,
    bits: int,
    block_size: int,
) -> torch.Tensor:
    """RTN baseline: per-channel min/max quantization (no rotation)."""
    from src.baselines import _per_channel_minmax, _per_channel_dequant
    codes, scales, zeros = _per_channel_minmax(tensor.float(), bits, block_size, axis=-1)
    return _per_channel_dequant(codes, scales, zeros, block_size, axis=-1)


def quantize_kivi(
    key: torch.Tensor,
    value: torch.Tensor,
    bits: int,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KIVI baseline: keys per-channel, values per-token."""
    from src.baselines import _per_channel_minmax, _per_channel_dequant
    k_codes, k_scales, k_zeros = _per_channel_minmax(key.float(), bits, block_size, axis=-1)
    v_codes, v_scales, v_zeros = _per_channel_minmax(value.float(), bits, block_size, axis=-2)
    k_deq = _per_channel_dequant(k_codes, k_scales, k_zeros, block_size, axis=-1)
    v_deq = _per_channel_dequant(v_codes, v_scales, v_zeros, block_size, axis=-2)
    return k_deq, v_deq


def cosine_similarity_tensors(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute mean cosine similarity between two tensors, flattened per-sample."""
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    cos = F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=-1)
    return cos.item()


def compute_attention_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Compute scaled dot-product attention output.

    All inputs are (batch, num_heads, seq_len, head_dim).
    Handles GQA by repeating K, V heads if needed.
    """
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_q_heads != num_kv_heads:
        repeat_factor = num_q_heads // num_kv_heads
        key = key.repeat_interleave(repeat_factor, dim=1)
        value = value.repeat_interleave(repeat_factor, dim=1)

    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, value.float())


def analyze_single_layer(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_idx: int,
    rotation: HadamardRotation,
    quantizer: ScalarQuantizer,
    bits: int,
    block_size: int,
    query: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Compute quantization error metrics for a single layer's K, V.

    Args:
        key: (batch, num_kv_heads, seq_len, head_dim) real K activations
        value: (batch, num_kv_heads, seq_len, head_dim) real V activations
        layer_idx: layer index (for reporting)
        rotation: HadamardRotation instance
        quantizer: ScalarQuantizer instance
        bits: quantization bit width
        block_size: quantization block size
        query: (batch, num_q_heads, seq_len, head_dim) real Q activations.
            When provided, uses real Q for attention distortion measurement.
            Falls back to using the last K position as a synthetic query.

    Returns:
        Dict with per-method error metrics and activation statistics.
    """
    device = key.device

    # --- Activation statistics ---
    k_range = (key.max() - key.min()).item()
    v_range = (value.max() - value.min()).item()
    k_flat = key.reshape(-1).cpu().numpy()
    v_flat = value.reshape(-1).cpu().numpy()
    k_kurtosis = float(sp_stats.kurtosis(k_flat, fisher=True))
    v_kurtosis = float(sp_stats.kurtosis(v_flat, fisher=True))
    k_std = float(np.std(k_flat))
    v_std = float(np.std(v_flat))

    results: Dict[str, Any] = {
        "layer_idx": layer_idx,
        "activation_range_k": k_range,
        "activation_range_v": v_range,
        "kurtosis_k": k_kurtosis,
        "kurtosis_v": v_kurtosis,
        "std_k": k_std,
        "std_v": v_std,
    }

    # --- SpecQuant ---
    k_sq = quantize_specquant(key, rotation, quantizer)
    v_sq = quantize_specquant(value, rotation, quantizer)
    results["specquant"] = {
        "mse_k": F.mse_loss(k_sq, key).item(),
        "mse_v": F.mse_loss(v_sq, value).item(),
        "cosine_k": cosine_similarity_tensors(k_sq, key),
        "cosine_v": cosine_similarity_tensors(v_sq, value),
        "max_abs_error_k": (k_sq - key).abs().max().item(),
        "max_abs_error_v": (v_sq - value).abs().max().item(),
    }

    # --- RTN ---
    k_rtn = quantize_rtn(key, bits, block_size)
    v_rtn = quantize_rtn(value, bits, block_size)
    results["rtn"] = {
        "mse_k": F.mse_loss(k_rtn, key).item(),
        "mse_v": F.mse_loss(v_rtn, value).item(),
        "cosine_k": cosine_similarity_tensors(k_rtn, key),
        "cosine_v": cosine_similarity_tensors(v_rtn, value),
        "max_abs_error_k": (k_rtn - key).abs().max().item(),
        "max_abs_error_v": (v_rtn - value).abs().max().item(),
    }

    # --- KIVI ---
    k_kivi, v_kivi = quantize_kivi(key, value, bits, block_size)
    results["kivi"] = {
        "mse_k": F.mse_loss(k_kivi, key).item(),
        "mse_v": F.mse_loss(v_kivi, value).item(),
        "cosine_k": cosine_similarity_tensors(k_kivi, key),
        "cosine_v": cosine_similarity_tensors(v_kivi, value),
        "max_abs_error_k": (k_kivi - key).abs().max().item(),
        "max_abs_error_v": (v_kivi - value).abs().max().item(),
    }

    # --- Attention output comparison ---
    # Use real Q tensors when available for accurate attention distortion
    # measurement.  Falls back to using the last K position as a synthetic
    # query when real Q is not captured.
    head_dim = key.shape[-1]
    if query is not None:
        # Use real Q; take the last position for a focused comparison
        attn_query = query[:, :, -1:, :]
    else:
        attn_query = key[:, :, -1:, :]  # fallback: single query from last K position

    attn_orig = compute_attention_output(attn_query, key, value, head_dim)

    attn_sq = compute_attention_output(attn_query, k_sq, v_sq, head_dim)
    results["specquant"]["attn_output_mse"] = F.mse_loss(attn_sq, attn_orig).item()
    results["specquant"]["attn_output_cosine"] = cosine_similarity_tensors(attn_sq, attn_orig)

    attn_rtn = compute_attention_output(attn_query, k_rtn, v_rtn, head_dim)
    results["rtn"]["attn_output_mse"] = F.mse_loss(attn_rtn, attn_orig).item()
    results["rtn"]["attn_output_cosine"] = cosine_similarity_tensors(attn_rtn, attn_orig)

    attn_kivi = compute_attention_output(attn_query, k_kivi, v_kivi, head_dim)
    results["kivi"]["attn_output_mse"] = F.mse_loss(attn_kivi, attn_orig).item()
    results["kivi"]["attn_output_cosine"] = cosine_similarity_tensors(attn_kivi, attn_orig)

    return results


# ---------------------------------------------------------------------------
# Multi-prompt analysis with statistics
# ---------------------------------------------------------------------------

def run_single_prompt(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    seq_length: int,
    device: str,
    kv_hook: KVCaptureHook,
    rotation: HadamardRotation,
    quantizer: ScalarQuantizer,
    bits: int,
    block_size: int,
) -> List[Dict[str, Any]]:
    """Run a single prompt through the model and analyze all layers.

    Returns a list of per-layer dicts (one per transformer layer).
    """
    kv_hook.clear_captures()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=seq_length,
        padding="max_length" if seq_length > 0 else False,
    )
    input_ids = inputs["input_ids"][:, :seq_length].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask[:, :seq_length].to(device)

    with torch.no_grad():
        model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=False,
        )

    if not kv_hook.captured_kv:
        raise RuntimeError(
            "No KV pairs captured. The model architecture may not be supported. "
            "Check that forward hooks are correctly placed on attention layers."
        )

    layer_results = []
    for layer_idx in sorted(kv_hook.captured_kv.keys()):
        k, v = kv_hook.captured_kv[layer_idx]
        q = kv_hook.captured_q.get(layer_idx)  # may be None if q_proj unavailable
        # Move to CPU for analysis to avoid GPU OOM on large models
        result = analyze_single_layer(
            k, v, layer_idx, rotation, quantizer, bits, block_size,
            query=q,
        )
        layer_results.append(result)

    return layer_results


def aggregate_across_prompts(
    all_prompt_results: List[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Aggregate per-layer results across multiple prompts.

    Args:
        all_prompt_results: list of (per-prompt list of per-layer dicts)

    Returns:
        Aggregated results with means, CIs, and sensitive layer detection.
    """
    num_prompts = len(all_prompt_results)
    num_layers = len(all_prompt_results[0])
    methods = ["specquant", "rtn", "kivi"]

    # Collect per-layer, per-prompt values
    per_layer: Dict[str, List[List[float]]] = {}
    metric_keys = [
        "activation_range_k", "activation_range_v",
        "kurtosis_k", "kurtosis_v",
        "std_k", "std_v",
    ]
    method_metric_keys = [
        "mse_k", "mse_v", "cosine_k", "cosine_v",
        "max_abs_error_k", "max_abs_error_v",
        "attn_output_mse", "attn_output_cosine",
    ]

    # Initialize
    for key in metric_keys:
        per_layer[key] = [[] for _ in range(num_layers)]
    for method in methods:
        for mk in method_metric_keys:
            full_key = f"{method}_{mk}"
            per_layer[full_key] = [[] for _ in range(num_layers)]

    # Collect
    for prompt_results in all_prompt_results:
        for li, layer_data in enumerate(prompt_results):
            for key in metric_keys:
                per_layer[key][li].append(layer_data[key])
            for method in methods:
                for mk in method_metric_keys:
                    full_key = f"{method}_{mk}"
                    per_layer[full_key][li].append(layer_data[method][mk])

    # Aggregate with CIs
    aggregated_layers = []
    for li in range(num_layers):
        layer_agg: Dict[str, Any] = {"layer_idx": li}

        for key in metric_keys:
            values = per_layer[key][li]
            layer_agg[key] = aggregate_trials(values) if len(values) >= 2 else {
                "mean": values[0], "std": 0.0, "n_trials": 1,
                "ci_lower": values[0], "ci_upper": values[0],
            }

        for method in methods:
            layer_agg[method] = {}
            for mk in method_metric_keys:
                full_key = f"{method}_{mk}"
                values = per_layer[full_key][li]
                layer_agg[method][mk] = aggregate_trials(values) if len(values) >= 2 else {
                    "mean": values[0], "std": 0.0, "n_trials": 1,
                    "ci_lower": values[0], "ci_upper": values[0],
                }

        aggregated_layers.append(layer_agg)

    # --- Identify sensitive layers (MSE > mean + 2*std across layers) ---
    sensitive_layers = _identify_sensitive_layers(aggregated_layers, methods)

    # --- Build per-layer summary arrays for top-level output ---
    summary: Dict[str, Any] = {
        "num_layers": num_layers,
        "num_prompts": num_prompts,
        "per_layer_details": aggregated_layers,
        "sensitive_layers": sensitive_layers,
    }

    # Flat per-layer arrays (mean values) for easy plotting
    for method in methods:
        summary[f"per_layer_mse_k_{method}"] = [
            aggregated_layers[li][method]["mse_k"]["mean"] for li in range(num_layers)
        ]
        summary[f"per_layer_mse_v_{method}"] = [
            aggregated_layers[li][method]["mse_v"]["mean"] for li in range(num_layers)
        ]
        summary[f"per_layer_cosine_k_{method}"] = [
            aggregated_layers[li][method]["cosine_k"]["mean"] for li in range(num_layers)
        ]
        summary[f"per_layer_cosine_v_{method}"] = [
            aggregated_layers[li][method]["cosine_v"]["mean"] for li in range(num_layers)
        ]
        summary[f"per_layer_attn_mse_{method}"] = [
            aggregated_layers[li][method]["attn_output_mse"]["mean"] for li in range(num_layers)
        ]

    summary["per_layer_activation_range_k"] = [
        aggregated_layers[li]["activation_range_k"]["mean"] for li in range(num_layers)
    ]
    summary["per_layer_activation_range_v"] = [
        aggregated_layers[li]["activation_range_v"]["mean"] for li in range(num_layers)
    ]
    summary["per_layer_kurtosis_k"] = [
        aggregated_layers[li]["kurtosis_k"]["mean"] for li in range(num_layers)
    ]
    summary["per_layer_kurtosis_v"] = [
        aggregated_layers[li]["kurtosis_v"]["mean"] for li in range(num_layers)
    ]

    # --- Method comparison summary ---
    summary["method_comparison"] = _build_method_comparison(aggregated_layers, methods)

    return summary


def _identify_sensitive_layers(
    aggregated_layers: List[Dict[str, Any]],
    methods: List[str],
) -> Dict[str, Any]:
    """Identify layers where MSE exceeds mean + 2*std (outlier detection)."""
    sensitive: Dict[str, Any] = {}
    num_layers = len(aggregated_layers)

    for method in methods:
        for kv_type in ["k", "v"]:
            mse_key = f"mse_{kv_type}"
            layer_means = [
                aggregated_layers[li][method][mse_key]["mean"]
                for li in range(num_layers)
            ]
            arr = np.array(layer_means)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            threshold = mean_val + 2.0 * std_val

            flagged = []
            for li in range(num_layers):
                if layer_means[li] > threshold:
                    flagged.append({
                        "layer_idx": li,
                        "mse": layer_means[li],
                        "threshold": threshold,
                        "kurtosis": aggregated_layers[li][f"kurtosis_{kv_type}"]["mean"],
                        "activation_range": aggregated_layers[li][f"activation_range_{kv_type}"]["mean"],
                        "reason": _diagnose_sensitivity(
                            aggregated_layers[li], kv_type,
                        ),
                    })

            tag = f"{method}_{kv_type}"
            sensitive[tag] = {
                "threshold": threshold,
                "mean_mse": mean_val,
                "std_mse": std_val,
                "sensitive_layer_indices": [f["layer_idx"] for f in flagged],
                "details": flagged,
            }

    return sensitive


def _diagnose_sensitivity(layer_agg: Dict[str, Any], kv_type: str) -> str:
    """Diagnose why a layer is quantization-sensitive."""
    kurtosis = layer_agg[f"kurtosis_{kv_type}"]["mean"]
    act_range = layer_agg[f"activation_range_{kv_type}"]["mean"]
    std = layer_agg[f"std_{kv_type}"]["mean"]

    reasons = []
    if kurtosis > 5.0:
        reasons.append(f"high kurtosis ({kurtosis:.1f}) indicates heavy-tailed outliers")
    if act_range > 10.0 * std:
        reasons.append(
            f"large activation range ({act_range:.2f}) relative to std ({std:.4f}) "
            "suggests extreme outlier channels"
        )
    if std > 1.0:
        reasons.append(f"high variance (std={std:.4f}) amplifies quantization error")

    if not reasons:
        reasons.append("marginally above threshold; no single dominant cause identified")

    return "; ".join(reasons)


def _build_method_comparison(
    aggregated_layers: List[Dict[str, Any]],
    methods: List[str],
) -> Dict[str, Any]:
    """Build a cross-method comparison summary."""
    num_layers = len(aggregated_layers)
    comparison: Dict[str, Any] = {}

    for method in methods:
        mean_mse_k = float(np.mean([
            aggregated_layers[li][method]["mse_k"]["mean"]
            for li in range(num_layers)
        ]))
        mean_mse_v = float(np.mean([
            aggregated_layers[li][method]["mse_v"]["mean"]
            for li in range(num_layers)
        ]))
        mean_cosine_k = float(np.mean([
            aggregated_layers[li][method]["cosine_k"]["mean"]
            for li in range(num_layers)
        ]))
        mean_cosine_v = float(np.mean([
            aggregated_layers[li][method]["cosine_v"]["mean"]
            for li in range(num_layers)
        ]))
        mean_attn_mse = float(np.mean([
            aggregated_layers[li][method]["attn_output_mse"]["mean"]
            for li in range(num_layers)
        ]))
        mean_attn_cosine = float(np.mean([
            aggregated_layers[li][method]["attn_output_cosine"]["mean"]
            for li in range(num_layers)
        ]))

        comparison[method] = {
            "mean_mse_k": mean_mse_k,
            "mean_mse_v": mean_mse_v,
            "mean_cosine_k": mean_cosine_k,
            "mean_cosine_v": mean_cosine_v,
            "mean_attn_output_mse": mean_attn_mse,
            "mean_attn_output_cosine": mean_attn_cosine,
        }

    # Relative improvement of SpecQuant over baselines
    sq = comparison["specquant"]
    for baseline in ["rtn", "kivi"]:
        bl = comparison[baseline]
        tag = f"specquant_vs_{baseline}"
        comparison[tag] = {
            "mse_k_reduction_pct": _pct_reduction(sq["mean_mse_k"], bl["mean_mse_k"]),
            "mse_v_reduction_pct": _pct_reduction(sq["mean_mse_v"], bl["mean_mse_v"]),
            "attn_mse_reduction_pct": _pct_reduction(
                sq["mean_attn_output_mse"], bl["mean_attn_output_mse"],
            ),
        }

    return comparison


def _pct_reduction(treatment: float, control: float) -> float:
    """Percentage reduction of treatment relative to control."""
    if abs(control) < 1e-12:
        return 0.0
    return (1.0 - treatment / control) * 100.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str,
    device: str,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.nn.Module, Any]:
    """Load a HuggingFace causal LM and its tokenizer."""
    logger.info("Loading tokenizer for %s ...", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model %s on %s (dtype=%s) ...", model_name, device, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded: %d layers, %d params",
                model.config.num_hidden_layers,
                sum(p.numel() for p in model.parameters()))
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Layer Sensitivity Analysis with Real Model Activations",
    )
    parser.add_argument("--target-model", type=str, required=True,
                        help="HuggingFace model name (e.g. Qwen/Qwen3.5-14B)")
    parser.add_argument("--target-device", type=str, default="cuda:1",
                        help="Device for the target model (default: cuda:1)")
    parser.add_argument("--bits", type=int, default=3,
                        help="Quantization bit width (default: 3)")
    parser.add_argument("--block-size", type=int, default=128,
                        help="Quantization block size (default: 128)")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Number of diverse prompts to evaluate (default: 10)")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Maximum sequence length for each prompt (default: 512)")
    parser.add_argument("--output-dir", type=str, default="results/robustness",
                        help="Output directory for results (default: results/robustness)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    model, tokenizer = load_model_and_tokenizer(
        args.target_model,
        args.target_device,
    )

    # Extract architecture info for quantization tools
    config = model.config
    head_dim = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers

    logger.info("Architecture: %d layers, head_dim=%d, bits=%d, block_size=%d",
                num_layers, head_dim, args.bits, args.block_size)

    # Set up quantization tools
    rotation = HadamardRotation(head_dim, seed=args.seed)
    quantizer = ScalarQuantizer(bits=args.bits, block_size=args.block_size)

    # Register hooks (pass geometry so Q can be reshaped correctly)
    num_q_heads = config.num_attention_heads
    kv_hook = KVCaptureHook(num_q_heads=num_q_heads, head_dim=head_dim)
    kv_hook.register(model)

    # Select prompts
    prompts = DIVERSE_PROMPTS[:args.num_prompts]
    if len(prompts) < args.num_prompts:
        logger.warning(
            "Requested %d prompts but only %d available; using all %d",
            args.num_prompts, len(prompts), len(prompts),
        )

    # Run analysis across prompts
    all_prompt_results: List[List[Dict[str, Any]]] = []
    for pi, prompt in enumerate(prompts):
        logger.info("Processing prompt %d/%d (len=%d chars) ...",
                     pi + 1, len(prompts), len(prompt))
        t0 = time.perf_counter()

        layer_results = run_single_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            seq_length=args.seq_length,
            device=args.target_device,
            kv_hook=kv_hook,
            rotation=rotation,
            quantizer=quantizer,
            bits=args.bits,
            block_size=args.block_size,
        )
        all_prompt_results.append(layer_results)
        elapsed = time.perf_counter() - t0
        logger.info("  Prompt %d: analyzed %d layers in %.1fs", pi + 1, len(layer_results), elapsed)

    # Clean up hooks
    kv_hook.clear()

    # Aggregate results
    logger.info("Aggregating results across %d prompts ...", len(all_prompt_results))
    summary = aggregate_across_prompts(all_prompt_results)

    # Add config metadata
    output = {
        "config": {
            "target_model": args.target_model,
            "target_device": args.target_device,
            "bits": args.bits,
            "block_size": args.block_size,
            "num_prompts": len(prompts),
            "seq_length": args.seq_length,
            "seed": args.seed,
            "num_layers": num_layers,
            "head_dim": head_dim,
            "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
            "num_attention_heads": config.num_attention_heads,
            "hidden_size": config.hidden_size,
        },
        "results": summary,
    }

    # Log top-level comparison
    mc = summary["method_comparison"]
    logger.info("=== Method Comparison (averaged across all layers and prompts) ===")
    for method in ["specquant", "rtn", "kivi"]:
        m = mc[method]
        logger.info(
            "  %s: MSE_K=%.6f  MSE_V=%.6f  Cosine_K=%.6f  Cosine_V=%.6f  AttnMSE=%.6f",
            method.upper(),
            m["mean_mse_k"], m["mean_mse_v"],
            m["mean_cosine_k"], m["mean_cosine_v"],
            m["mean_attn_output_mse"],
        )
    for tag in ["specquant_vs_rtn", "specquant_vs_kivi"]:
        if tag in mc:
            r = mc[tag]
            logger.info(
                "  %s: K MSE reduction=%.1f%%  V MSE reduction=%.1f%%  Attn MSE reduction=%.1f%%",
                tag, r["mse_k_reduction_pct"], r["mse_v_reduction_pct"],
                r["attn_mse_reduction_pct"],
            )

    # Log sensitive layers
    for tag, info in summary["sensitive_layers"].items():
        if info["sensitive_layer_indices"]:
            logger.info("Sensitive layers [%s]: %s", tag, info["sensitive_layer_indices"])

    # Save
    filename = f"layer_sensitivity_{time.strftime('%Y%m%d_%H%M%S')}.json"
    save_results(output, args.output_dir, filename)
    logger.info("Done.")


if __name__ == "__main__":
    main()
