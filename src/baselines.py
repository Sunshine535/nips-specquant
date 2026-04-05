"""Quantization baselines for fair comparison with SpecQuant.

Implements RTN (Round-To-Nearest), KIVI (Key-Integer-Value-Integer), and
Absmax KV cache quantization.  All classes expose the same API as
:class:`QuantizedKVCache` so benchmark scripts can swap methods with a
single argument change.  :class:`BaselineDecoder` wraps a baseline cache
inside the same ``generate()`` interface used by :class:`SpeculativeDecoder`.
"""

import math
import time
import logging
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .turboquant_kv import ScalarQuantizer
from .speculative_decode import SpeculativeOutput, _trim_kv_cache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _per_channel_minmax(
    x: torch.Tensor,
    bits: int,
    block_size: int,
    axis: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Asymmetric per-channel (or per-block-along-*axis*) min/max quantization.

    Args:
        x: input tensor (..., seq_len, dim)
        bits: quantization bit-width
        block_size: block size along *axis*
        axis: -1 for head_dim (per-channel), -2 for seq_len (per-token)

    Returns:
        codes (uint8), scales (fp16), zeros (fp16)
    """
    n_levels = (1 << bits) - 1
    orig_shape = x.shape

    # Determine which dimension we block over.
    if axis == -1:
        # Quantize along head_dim (last axis) — group consecutive dim elements.
        dim = orig_shape[-1]
        seq_len = orig_shape[-2]
        batch_dims = orig_shape[:-2]
        n_blocks = (dim + block_size - 1) // block_size
        padded_dim = n_blocks * block_size
        if padded_dim > dim:
            x = F.pad(x, (0, padded_dim - dim))
        # (..., seq_len, n_blocks, block_size)
        x_blocks = x.view(*batch_dims, seq_len, n_blocks, block_size)
        block_min = x_blocks.amin(dim=-1, keepdim=True)   # (..., seq_len, n_blocks, 1)
        block_max = x_blocks.amax(dim=-1, keepdim=True)
        block_range = (block_max - block_min).clamp(min=1e-8)
        scale = block_range / n_levels
        x_norm = (x_blocks - block_min) / scale
        codes = x_norm.round().clamp(0, n_levels).to(torch.uint8)
        # Store scales/zeros squeezed: (..., seq_len, n_blocks)
        return (
            codes.view(*batch_dims, seq_len, padded_dim)[..., :dim],
            scale.squeeze(-1).half(),
            block_min.squeeze(-1).half(),
        )
    else:
        # axis == -2: quantize along seq_len (per-token groups).
        seq_len = orig_shape[-2]
        dim = orig_shape[-1]
        batch_dims = orig_shape[:-2]
        n_blocks = (seq_len + block_size - 1) // block_size
        padded_len = n_blocks * block_size
        if padded_len > seq_len:
            x = F.pad(x, (0, 0, 0, padded_len - seq_len))
        # (..., n_blocks, block_size, dim)
        x_blocks = x.view(*batch_dims, n_blocks, block_size, dim)
        block_min = x_blocks.amin(dim=-2, keepdim=True)   # (..., n_blocks, 1, dim)
        block_max = x_blocks.amax(dim=-2, keepdim=True)
        block_range = (block_max - block_min).clamp(min=1e-8)
        scale = block_range / n_levels
        x_norm = (x_blocks - block_min) / scale
        codes = x_norm.round().clamp(0, n_levels).to(torch.uint8)
        # Reshape back to (..., padded_len, dim) then trim
        return (
            codes.view(*batch_dims, padded_len, dim)[..., :seq_len, :],
            scale.squeeze(-2).half(),   # (..., n_blocks, dim)
            block_min.squeeze(-2).half(),
        )


def _per_channel_dequant(
    codes: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size: int,
    axis: int = -1,
) -> torch.Tensor:
    """Dequantize codes produced by :func:`_per_channel_minmax`."""
    if axis == -1:
        orig_dim = codes.shape[-1]
        seq_len = codes.shape[-2]
        batch_dims = codes.shape[:-2]
        n_blocks = scales.shape[-1]
        padded_dim = n_blocks * block_size
        if padded_dim > orig_dim:
            codes = F.pad(codes, (0, padded_dim - orig_dim))
        # (..., seq_len, n_blocks, block_size)
        codes_blocks = codes.view(*batch_dims, seq_len, n_blocks, block_size).float()
        # scales/zeros: (..., seq_len, n_blocks) -> unsqueeze last
        x = codes_blocks * scales.float().unsqueeze(-1) + zeros.float().unsqueeze(-1)
        return x.reshape(*batch_dims, seq_len, padded_dim)[..., :orig_dim]
    else:
        seq_len = codes.shape[-2]
        dim = codes.shape[-1]
        batch_dims = codes.shape[:-2]
        n_blocks = scales.shape[-2]
        padded_len = n_blocks * block_size
        if padded_len > seq_len:
            codes = F.pad(codes, (0, 0, 0, padded_len - seq_len))
        # (..., n_blocks, block_size, dim)
        codes_blocks = codes.view(*batch_dims, n_blocks, block_size, dim).float()
        # scales/zeros: (..., n_blocks, dim) -> unsqueeze -2
        x = codes_blocks * scales.float().unsqueeze(-2) + zeros.float().unsqueeze(-2)
        return x.reshape(*batch_dims, padded_len, dim)[..., :seq_len, :]


# =========================================================================
# 1. RTN — Round-To-Nearest per-channel min/max (no rotation)
# =========================================================================

class RTNKVCache:
    """Simplest KV cache quantization baseline.

    Per-channel (along head_dim) asymmetric min/max quantization with
    block-wise scales.  No Hadamard rotation — direct scalar quantization.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        bits: int = 3,
        block_size: int = 128,
        **_kwargs,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bits = bits
        self.block_size = block_size

        self.k_codes: list = [None] * num_layers
        self.k_scales: list = [None] * num_layers
        self.k_zeros: list = [None] * num_layers
        self.v_codes: list = [None] * num_layers
        self.v_scales: list = [None] * num_layers
        self.v_zeros: list = [None] * num_layers

        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    # -- core API (mirrors QuantizedKVCache) --------------------------------

    def compress_and_store(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Quantize K, V with per-channel (head_dim) min/max.

        Args:
            layer_idx: transformer layer index
            key: (batch, num_kv_heads, seq_len, head_dim)
            value: (batch, num_kv_heads, seq_len, head_dim)
        """
        k_codes, k_scales, k_zeros = _per_channel_minmax(
            key.float(), self.bits, self.block_size, axis=-1,
        )
        v_codes, v_scales, v_zeros = _per_channel_minmax(
            value.float(), self.bits, self.block_size, axis=-1,
        )
        self.k_codes[layer_idx] = k_codes
        self.k_scales[layer_idx] = k_scales
        self.k_zeros[layer_idx] = k_zeros
        self.v_codes[layer_idx] = v_codes
        self.v_scales[layer_idx] = v_scales
        self.v_zeros[layer_idx] = v_zeros
        self._seq_len = key.shape[-2]

    def get_rotated_kv(
        self, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return K, V in original space (no rotation)."""
        k = _per_channel_dequant(
            self.k_codes[layer_idx],
            self.k_scales[layer_idx],
            self.k_zeros[layer_idx],
            self.block_size, axis=-1,
        )
        v = _per_channel_dequant(
            self.v_codes[layer_idx],
            self.v_scales[layer_idx],
            self.v_zeros[layer_idx],
            self.block_size, axis=-1,
        )
        return k, v

    def compressed_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute attention using dequantized KV (no rotation trick).

        Args:
            layer_idx: transformer layer index
            query: (batch, num_q_heads, q_len, head_dim)
            scale: attention scale (default 1/sqrt(head_dim))

        Returns:
            Attention output (batch, num_q_heads, q_len, head_dim)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        k, v = self.get_rotated_kv(layer_idx)

        num_q_heads = query.shape[1]
        num_kv_heads = k.shape[1]
        if num_q_heads != num_kv_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        scores = torch.matmul(query.float(), k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    # -- memory accounting --------------------------------------------------

    def memory_bytes(self) -> int:
        if self.k_codes[0] is None:
            return 0
        total = 0
        for i in range(self.num_layers):
            codes_bytes = self.k_codes[i].numel() + self.v_codes[i].numel()
            scales_bytes = (
                self.k_scales[i].numel() + self.v_scales[i].numel()
                + self.k_zeros[i].numel() + self.v_zeros[i].numel()
            ) * 2  # fp16
            total += codes_bytes + scales_bytes
        return total

    def full_precision_bytes(self) -> int:
        if self.k_codes[0] is None:
            return 0
        seq_len = self.k_codes[0].shape[-2]
        per_layer = 2 * self.num_kv_heads * seq_len * self.head_dim * 2
        return per_layer * self.num_layers

    @property
    def compression_ratio(self) -> float:
        fp = self.full_precision_bytes()
        if fp == 0:
            return 0.0
        return fp / self.memory_bytes()


# =========================================================================
# 2. KIVI — per-channel Keys, per-token Values
# =========================================================================

class KIVIKVCache:
    """KIVI-style KV cache quantization (Hooper et al., 2024).

    Keys are quantized per-channel (along head_dim axis) and values are
    quantized per-token (along seq_len axis).  This asymmetric scheme
    exploits the observation that key channels have tight ranges while
    value tokens do.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        bits: int = 3,
        block_size: int = 128,
        **_kwargs,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bits = bits
        self.block_size = block_size

        self.k_codes: list = [None] * num_layers
        self.k_scales: list = [None] * num_layers
        self.k_zeros: list = [None] * num_layers
        self.v_codes: list = [None] * num_layers
        self.v_scales: list = [None] * num_layers
        self.v_zeros: list = [None] * num_layers

        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def compress_and_store(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        """Quantize keys per-channel, values per-token.

        Args:
            layer_idx: transformer layer index
            key: (batch, num_kv_heads, seq_len, head_dim)
            value: (batch, num_kv_heads, seq_len, head_dim)
        """
        # Keys: per-channel (head_dim axis) quantization
        k_codes, k_scales, k_zeros = _per_channel_minmax(
            key.float(), self.bits, self.block_size, axis=-1,
        )
        # Values: per-token (seq_len axis) quantization
        v_codes, v_scales, v_zeros = _per_channel_minmax(
            value.float(), self.bits, self.block_size, axis=-2,
        )
        self.k_codes[layer_idx] = k_codes
        self.k_scales[layer_idx] = k_scales
        self.k_zeros[layer_idx] = k_zeros
        self.v_codes[layer_idx] = v_codes
        self.v_scales[layer_idx] = v_scales
        self.v_zeros[layer_idx] = v_zeros
        self._seq_len = key.shape[-2]

    def get_rotated_kv(
        self, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return K, V (no rotation applied)."""
        k = _per_channel_dequant(
            self.k_codes[layer_idx],
            self.k_scales[layer_idx],
            self.k_zeros[layer_idx],
            self.block_size, axis=-1,
        )
        v = _per_channel_dequant(
            self.v_codes[layer_idx],
            self.v_scales[layer_idx],
            self.v_zeros[layer_idx],
            self.block_size, axis=-2,
        )
        return k, v

    def compressed_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute attention using KIVI-dequantized KV.

        Args:
            layer_idx: transformer layer index
            query: (batch, num_q_heads, q_len, head_dim)
            scale: attention scale (default 1/sqrt(head_dim))

        Returns:
            Attention output (batch, num_q_heads, q_len, head_dim)
        """
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        k, v = self.get_rotated_kv(layer_idx)

        num_q_heads = query.shape[1]
        num_kv_heads = k.shape[1]
        if num_q_heads != num_kv_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        scores = torch.matmul(query.float(), k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    def memory_bytes(self) -> int:
        if self.k_codes[0] is None:
            return 0
        total = 0
        for i in range(self.num_layers):
            codes_bytes = self.k_codes[i].numel() + self.v_codes[i].numel()
            scales_bytes = (
                self.k_scales[i].numel() + self.v_scales[i].numel()
                + self.k_zeros[i].numel() + self.v_zeros[i].numel()
            ) * 2
            total += codes_bytes + scales_bytes
        return total

    def full_precision_bytes(self) -> int:
        if self.k_codes[0] is None:
            return 0
        seq_len = self.k_codes[0].shape[-2]
        per_layer = 2 * self.num_kv_heads * seq_len * self.head_dim * 2
        return per_layer * self.num_layers

    @property
    def compression_ratio(self) -> float:
        fp = self.full_precision_bytes()
        if fp == 0:
            return 0.0
        return fp / self.memory_bytes()


# =========================================================================
# 3. Absmax — symmetric per-tensor quantization
# =========================================================================

class AbsmaxKVCache:
    """Simplest possible baseline: symmetric per-tensor absmax quantization.

    scale = max(|x|) / (2^{bits-1} - 1), zero = 0.  One scale per
    (layer, K/V, head) tensor — no blocking at all.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        bits: int = 3,
        block_size: int = 128,   # accepted for API compat, but unused
        **_kwargs,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bits = bits
        self.block_size = block_size  # stored for API compat
        self.n_levels = (1 << (bits - 1)) - 1  # symmetric: -(2^{b-1}-1) .. +(2^{b-1}-1)

        self.k_codes: list = [None] * num_layers
        self.k_scales: list = [None] * num_layers
        self.v_codes: list = [None] * num_layers
        self.v_scales: list = [None] * num_layers

        self._seq_len = 0

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def _quantize_absmax(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric per-head absmax quantization.

        Returns int8-range codes and per-head fp16 scales.  The scale
        tensor has shape (batch, num_kv_heads, 1, 1) — one scalar per
        head.
        """
        # Per-head absmax: reduce over (seq_len, head_dim).
        amax = x.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        scale = amax / self.n_levels                     # (..., 1, 1)
        codes = (x / scale).round().clamp(-self.n_levels, self.n_levels).to(torch.int8)
        return codes, scale.half()

    def _dequantize_absmax(
        self, codes: torch.Tensor, scales: torch.Tensor,
    ) -> torch.Tensor:
        return codes.float() * scales.float()

    # -- core API -----------------------------------------------------------

    def compress_and_store(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ):
        k_codes, k_scales = self._quantize_absmax(key.float())
        v_codes, v_scales = self._quantize_absmax(value.float())
        self.k_codes[layer_idx] = k_codes
        self.k_scales[layer_idx] = k_scales
        self.v_codes[layer_idx] = v_codes
        self.v_scales[layer_idx] = v_scales
        self._seq_len = key.shape[-2]

    def get_rotated_kv(
        self, layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return K, V (no rotation applied)."""
        k = self._dequantize_absmax(
            self.k_codes[layer_idx], self.k_scales[layer_idx],
        )
        v = self._dequantize_absmax(
            self.v_codes[layer_idx], self.v_scales[layer_idx],
        )
        return k, v

    def compressed_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        if scale is None:
            scale = 1.0 / math.sqrt(self.head_dim)

        k, v = self.get_rotated_kv(layer_idx)

        num_q_heads = query.shape[1]
        num_kv_heads = k.shape[1]
        if num_q_heads != num_kv_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        scores = torch.matmul(query.float(), k.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

    def memory_bytes(self) -> int:
        if self.k_codes[0] is None:
            return 0
        total = 0
        for i in range(self.num_layers):
            # int8 codes: 1 byte each
            codes_bytes = self.k_codes[i].numel() + self.v_codes[i].numel()
            # fp16 scales: one per head per K/V
            scales_bytes = (self.k_scales[i].numel() + self.v_scales[i].numel()) * 2
            total += codes_bytes + scales_bytes
        return total

    def full_precision_bytes(self) -> int:
        if self.k_codes[0] is None:
            return 0
        seq_len = self.k_codes[0].shape[-2]
        per_layer = 2 * self.num_kv_heads * seq_len * self.head_dim * 2
        return per_layer * self.num_layers

    @property
    def compression_ratio(self) -> float:
        fp = self.full_precision_bytes()
        if fp == 0:
            return 0.0
        return fp / self.memory_bytes()


# =========================================================================
# Factory
# =========================================================================

BASELINE_REGISTRY = {
    "rtn": RTNKVCache,
    "kivi": KIVIKVCache,
    "absmax": AbsmaxKVCache,
}


def make_baseline_cache(
    baseline_type: str,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    bits: int = 3,
    block_size: int = 128,
    **kwargs,
):
    """Instantiate a baseline KV cache by name.

    Args:
        baseline_type: one of "rtn", "kivi", "absmax"

    Returns:
        A cache object with the same API as :class:`QuantizedKVCache`.
    """
    cls = BASELINE_REGISTRY.get(baseline_type)
    if cls is None:
        raise ValueError(
            f"Unknown baseline_type={baseline_type!r}. "
            f"Choose from {list(BASELINE_REGISTRY.keys())}"
        )
    return cls(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        bits=bits,
        block_size=block_size,
        **kwargs,
    )


# =========================================================================
# 4. BaselineDecoder — SpeculativeDecoder-like wrapper for baselines
# =========================================================================

class BaselineDecoder:
    """Speculative decoding with a baseline KV cache quantization method.

    Drop-in replacement for :class:`SpeculativeDecoder`: same ``generate()``
    signature and return type, but uses RTN / KIVI / Absmax instead of
    the TurboQuant (Hadamard + scalar) cache.
    """

    def __init__(
        self,
        draft_model: PreTrainedModel,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        baseline_type: str = "rtn",
        quant_bits: int = 3,
        quant_block_size: int = 128,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.baseline_type = baseline_type
        self.quant_bits = quant_bits
        self.quant_block_size = quant_block_size

        self.draft_model.eval()
        self.target_model.eval()
        self.draft_device = next(draft_model.parameters()).device
        self.target_device = next(target_model.parameters()).device

        self.use_quant = quant_bits > 0
        if self.use_quant:
            config = target_model.config
            num_layers = config.num_hidden_layers
            num_kv_heads = getattr(
                config, "num_key_value_heads", config.num_attention_heads,
            )
            head_dim = config.hidden_size // config.num_attention_heads
            logger.info(
                f"Baseline {baseline_type} enabled: {quant_bits}-bit, "
                f"block_size={quant_block_size}, layers={num_layers}, "
                f"kv_heads={num_kv_heads}, head_dim={head_dim}"
            )

    def _build_cache(self):
        config = self.target_model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads,
        )
        head_dim = config.hidden_size // config.num_attention_heads
        return make_baseline_cache(
            baseline_type=self.baseline_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bits=self.quant_bits,
            block_size=self.quant_block_size,
        )

    def _compress_kv(self, kv: Any, cache) -> Any:
        """Quantize-dequantize the target model KV cache through the baseline."""
        if isinstance(kv, tuple):
            new_layers = []
            for i, layer in enumerate(kv):
                k, v = layer[0], layer[1]
                cache.compress_and_store(i, k.float(), v.float())
                k_deq, v_deq = cache.get_rotated_kv(i)
                new_layers.append((k_deq.to(k.dtype), v_deq.to(v.dtype)))
            return tuple(new_layers)

        if hasattr(kv, "key_cache") and hasattr(kv, "value_cache"):
            for i in range(min(len(kv.key_cache), cache.num_layers)):
                k = kv.key_cache[i]
                v = kv.value_cache[i]
                if k is None:
                    continue
                cache.compress_and_store(i, k.float(), v.float())
                k_deq, v_deq = cache.get_rotated_kv(i)
                kv.key_cache[i] = k_deq.to(k.dtype)
                kv.value_cache[i] = v_deq.to(v.dtype)
            return kv

        return kv

    # -- draft phase (identical to SpeculativeDecoder) ----------------------

    def _draft_phase(
        self,
        start_logits: torch.Tensor,
        kv: Any,
        gamma: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Any]:
        tokens: List[torch.Tensor] = []
        probs_list: List[torch.Tensor] = []
        logits = start_logits
        current_kv = kv

        for _ in range(gamma):
            p = F.softmax(logits / max(temperature, 1e-8), dim=-1).squeeze(0)
            tok = torch.multinomial(p, num_samples=1).squeeze(-1)
            tokens.append(tok.cpu())
            probs_list.append(p.cpu())

            out = self.draft_model(
                tok.view(1, 1).to(self.draft_device),
                past_key_values=current_kv,
                use_cache=True,
            )
            current_kv = out.past_key_values
            logits = out.logits[:, -1, :]

        return torch.stack(tokens), probs_list, current_kv

    @staticmethod
    def _rejection_sample(
        target_next_logits: torch.Tensor,
        verify_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: List[torch.Tensor],
        gamma: int,
        temperature: float,
    ) -> Tuple[int, torch.Tensor]:
        device = verify_logits.device
        temp = max(temperature, 1e-8)
        accepted: List[torch.Tensor] = []
        n_accepted = 0

        for i in range(gamma):
            if i == 0:
                tgt_logits_i = target_next_logits.squeeze(0)
            else:
                tgt_logits_i = verify_logits[:, i - 1, :].squeeze(0)

            tp = F.softmax(tgt_logits_i.to(device) / temp, dim=-1)
            dp = draft_probs[i].to(device)
            tok_id = draft_tokens[i].item()

            p_t = tp[tok_id]
            p_d = dp[tok_id].clamp(min=1e-10)

            if torch.rand(1, device=device).item() < min(1.0, (p_t / p_d).item()):
                accepted.append(draft_tokens[i])
                n_accepted += 1
            else:
                adjusted = (tp - dp).clamp(min=0)
                s = adjusted.sum()
                if s > 0:
                    adjusted = adjusted / s
                else:
                    adjusted = tp
                new_tok = torch.multinomial(adjusted, num_samples=1).squeeze(-1)
                accepted.append(new_tok.cpu())
                break
        else:
            bonus_logits = verify_logits[:, gamma - 1, :].squeeze(0)
            bonus_p = F.softmax(bonus_logits.to(device) / temp, dim=-1)
            bonus = torch.multinomial(bonus_p, num_samples=1).squeeze(-1)
            accepted.append(bonus.cpu())

        return n_accepted, torch.stack(accepted)

    # -- generate -----------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        gamma: int = 5,
        temperature: float = 1.0,
    ) -> SpeculativeOutput:
        """Run speculative decoding with a baseline quantized KV cache."""
        assert input_ids.shape[0] == 1, "Only batch_size=1 is supported"

        prefix_len = input_ids.shape[1]
        acc_by_pos = [0] * gamma
        rounds_by_pos = [0] * gamma
        total_draft = 0
        total_accepted = 0
        n_rounds = 0
        t_draft_total = 0.0
        t_verify_total = 0.0
        t_quant_total = 0.0

        # Prefill
        draft_out = self.draft_model(
            input_ids.to(self.draft_device), use_cache=True,
        )
        draft_kv = draft_out.past_key_values
        draft_next_logits = draft_out.logits[:, -1, :]

        target_out = self.target_model(
            input_ids.to(self.target_device), use_cache=True,
        )
        target_kv = target_out.past_key_values
        target_next_logits = target_out.logits[:, -1, :]

        all_token_ids = input_ids.cpu().clone()
        kv_len = prefix_len

        bl_cache = None
        if self.use_quant:
            bl_cache = self._build_cache()
            target_kv = self._compress_kv(target_kv, bl_cache)

        start = time.perf_counter()

        while all_token_ids.shape[1] - prefix_len < max_new_tokens:
            remaining = max_new_tokens - (all_token_ids.shape[1] - prefix_len)
            cur_gamma = min(gamma, remaining)
            if cur_gamma <= 0:
                break

            n_rounds += 1
            for k in range(cur_gamma):
                rounds_by_pos[k] += 1
            total_draft += cur_gamma

            # Draft
            t0 = time.perf_counter()
            draft_tokens, draft_probs, draft_kv = self._draft_phase(
                draft_next_logits, draft_kv, cur_gamma, temperature,
            )
            t_draft_total += time.perf_counter() - t0

            # Verify
            t0 = time.perf_counter()
            verify_out = self.target_model(
                draft_tokens.view(1, -1).to(self.target_device),
                past_key_values=target_kv,
                use_cache=True,
            )
            target_kv_ext = verify_out.past_key_values
            verify_logits = verify_out.logits

            n_acc, accepted = self._rejection_sample(
                target_next_logits,
                verify_logits,
                draft_tokens,
                draft_probs,
                cur_gamma,
                temperature,
            )
            t_verify_total += time.perf_counter() - t0

            total_accepted += n_acc
            for k in range(n_acc):
                acc_by_pos[k] += 1

            all_token_ids = torch.cat(
                [all_token_ids, accepted.view(1, -1).cpu()], dim=1,
            )

            new_kv_len = kv_len + n_acc
            draft_kv = _trim_kv_cache(draft_kv, new_kv_len)
            target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)

            last_tok = accepted[-1]

            d_out = self.draft_model(
                last_tok.view(1, 1).to(self.draft_device),
                past_key_values=draft_kv,
                use_cache=True,
            )
            draft_kv = d_out.past_key_values
            draft_next_logits = d_out.logits[:, -1, :]

            t_out = self.target_model(
                last_tok.view(1, 1).to(self.target_device),
                past_key_values=target_kv,
                use_cache=True,
            )
            target_kv = t_out.past_key_values
            target_next_logits = t_out.logits[:, -1, :]

            if bl_cache is not None:
                t0_q = time.perf_counter()
                target_kv = self._compress_kv(target_kv, bl_cache)
                t_quant_total += time.perf_counter() - t0_q

            kv_len = new_kv_len + 1

        wall = time.perf_counter() - start

        final_ids = all_token_ids[:, : prefix_len + max_new_tokens]
        return SpeculativeOutput(
            generated_ids=final_ids,
            num_generated_tokens=final_ids.shape[1] - prefix_len,
            num_draft_rounds=n_rounds,
            total_draft_tokens=total_draft,
            total_accepted_tokens=total_accepted,
            acceptance_counts_by_position=acc_by_pos,
            draft_rounds_by_position=rounds_by_pos,
            wall_time_seconds=wall,
            draft_time_seconds=t_draft_total,
            verify_time_seconds=t_verify_total,
            quantize_time_seconds=t_quant_total,
        )

    @torch.no_grad()
    def generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, float]:
        """Baseline autoregressive generation with target model only."""
        generated = input_ids.to(self.target_device)
        past = None

        start = time.perf_counter()
        for _ in range(max_new_tokens):
            if past is None:
                out = self.target_model(generated, use_cache=True)
            else:
                out = self.target_model(
                    generated[:, -1:], past_key_values=past, use_cache=True,
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, tok], dim=1)
        wall = time.perf_counter() - start
        return generated.cpu(), wall

    @torch.no_grad()
    def measure_tv_distance(
        self,
        input_ids: torch.Tensor,
        num_positions: int = 256,
    ) -> dict:
        """Empirical TV distance between full-precision and baseline-quantized logits."""
        if not self.use_quant:
            return {"tv_mean": 0.0, "tv_std": 0.0, "tv_per_position": []}

        device = self.target_device
        seq_len = min(input_ids.shape[1], num_positions)
        tokens = input_ids[:, :seq_len].to(device)

        split = max(1, seq_len // 2)
        prefix = tokens[:, :split]
        suffix = tokens[:, split:]

        if suffix.shape[1] == 0:
            return {"tv_mean": 0.0, "tv_std": 0.0, "tv_per_position": []}

        # Full-precision reference
        fp_out = self.target_model(prefix, use_cache=True)
        fp_kv = fp_out.past_key_values

        # Quantized via baseline cache
        q_out = self.target_model(prefix, use_cache=True)
        q_kv = q_out.past_key_values
        bl_cache = self._build_cache()
        q_kv = self._compress_kv(q_kv, bl_cache)

        fp_verify = self.target_model(
            suffix, past_key_values=fp_kv, use_cache=False,
        )
        fp_probs = F.softmax(fp_verify.logits.float(), dim=-1)

        q_verify = self.target_model(
            suffix, past_key_values=q_kv, use_cache=False,
        )
        q_probs = F.softmax(q_verify.logits.float(), dim=-1)

        tv_per_pos = 0.5 * (fp_probs - q_probs).abs().sum(dim=-1).squeeze(0)
        tv_values = tv_per_pos.cpu().tolist()
        if isinstance(tv_values, float):
            tv_values = [tv_values]

        tv_t = torch.tensor(tv_values)
        return {
            "tv_mean": tv_t.mean().item(),
            "tv_std": tv_t.std().item() if len(tv_values) > 1 else 0.0,
            "tv_per_position": tv_values,
            "num_positions": len(tv_values),
            "prefix_len": split,
            "quant_bits": self.quant_bits,
            "baseline_type": self.baseline_type,
        }
