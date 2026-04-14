"""TurboQuant KV cache quantization via Hadamard rotation + scalar quantization.

Implements compressed-domain KV storage for speculative decoding verification.
Core idea: rotate K/V into near-isotropic space via fast Walsh-Hadamard transform,
then apply per-coordinate scalar quantization. Attention is computed entirely in
the rotated space; inverse rotation is applied once per head to the output.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _next_power_of_2(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _random_sign_vector(d: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    signs = torch.randint(0, 2, (d,), generator=generator, dtype=torch.float32) * 2 - 1
    return signs



def _raw_wht(x: torch.Tensor) -> torch.Tensor:
    """Un-normalized fast Walsh-Hadamard butterfly on the last dimension."""
    d = x.shape[-1]
    assert d & (d - 1) == 0, f"Dimension must be power of 2, got {d}"

    x = x.contiguous().clone()
    flat = x.view(-1, d)
    n_vecs = flat.shape[0]

    h = 1
    while h < d:
        flat_view = flat.view(n_vecs, d // (2 * h), 2, h)
        a = flat_view[:, :, 0, :].clone()
        b = flat_view[:, :, 1, :].clone()
        flat_view[:, :, 0, :] = a + b
        flat_view[:, :, 1, :] = a - b
        flat = flat_view.view(n_vecs, d)
        h *= 2

    return flat.view(x.shape)


def fast_hadamard_transform(x: torch.Tensor, signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Normalized fast Walsh-Hadamard. Forward: signs first, then WHT."""
    if signs is not None:
        x = x * signs.to(x.device, x.dtype)
    return _raw_wht(x) / math.sqrt(x.shape[-1])


def fast_hadamard_inverse(x: torch.Tensor, signs: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Inverse normalized Hadamard. WHT first, then signs."""
    result = _raw_wht(x) / math.sqrt(x.shape[-1])
    if signs is not None:
        result = result * signs.to(result.device, result.dtype)
    return result


class HadamardRotation:
    """Manages a fixed random-sign Hadamard rotation for a given dimension."""

    def __init__(self, dim: int, seed: int = 42):
        self.dim = dim
        self.padded_dim = _next_power_of_2(dim)
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.signs = _random_sign_vector(self.padded_dim, generator=gen)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - x.shape[-1]))
        return fast_hadamard_transform(x, self.signs)

    def inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        result = fast_hadamard_inverse(x, self.signs)
        if self.dim < self.padded_dim:
            result = result[..., :self.dim]
        return result


class ScalarQuantizer:
    """Per-block uniform scalar quantizer for rotated KV vectors."""

    def __init__(self, bits: int = 3, block_size: int = 128):
        self.bits = bits
        self.block_size = block_size
        self.n_levels = (1 << bits) - 1

    def quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize tensor to b-bit codes with per-block scales.

        Args:
            x: shape (..., seq_len, dim)

        Returns:
            codes: uint8 tensor of quantized codes, shape (..., seq_len, dim)
            scales: fp16 per-block scale, shape (..., n_blocks, dim)
            zeros: fp16 per-block zero point, shape (..., n_blocks, dim)
        """
        orig_shape = x.shape
        seq_len = orig_shape[-2]
        dim = orig_shape[-1]
        batch_dims = orig_shape[:-2]

        n_blocks = (seq_len + self.block_size - 1) // self.block_size
        padded_len = n_blocks * self.block_size

        if padded_len > seq_len:
            x = F.pad(x, (0, 0, 0, padded_len - seq_len))

        x_blocks = x.view(*batch_dims, n_blocks, self.block_size, dim)
        block_min = x_blocks.amin(dim=-2)
        block_max = x_blocks.amax(dim=-2)

        block_range = (block_max - block_min).clamp(min=1e-8)
        scale = block_range / self.n_levels
        zero = block_min

        x_normalized = (x_blocks - zero.unsqueeze(-2)) / scale.unsqueeze(-2)
        codes = x_normalized.round().clamp(0, self.n_levels).to(torch.uint8)

        return codes.view(*batch_dims, padded_len, dim)[..., :seq_len, :], scale.half(), zero.half()

    def dequantize(
        self, codes: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize codes back to floating point.

        Args:
            codes: uint8 quantized codes (..., seq_len, dim)
            scales: fp16 per-block scale (..., n_blocks, dim)
            zeros: fp16 per-block zero (..., n_blocks, dim)

        Returns:
            Dequantized tensor in fp32 (..., seq_len, dim)
        """
        seq_len = codes.shape[-2]
        dim = codes.shape[-1]
        batch_dims = codes.shape[:-2]

        n_blocks = scales.shape[-2]
        padded_len = n_blocks * self.block_size

        if padded_len > seq_len:
            codes = F.pad(codes, (0, 0, 0, padded_len - seq_len))

        codes_blocks = codes.view(*batch_dims, n_blocks, self.block_size, dim).float()
        x = codes_blocks * scales.float().unsqueeze(-2) + zeros.float().unsqueeze(-2)

        return x.view(*batch_dims, padded_len, dim)[..., :seq_len, :]


class QuantizedKVCache:
    """Compressed KV cache using Hadamard rotation + scalar quantization.

    Stores K and V in quantized rotated space. Supports incremental append
    for new tokens during speculative verification.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        bits: int = 3,
        block_size: int = 128,
        seed: int = 42,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.bits = bits
        self.block_size = block_size

        self.rotation = HadamardRotation(head_dim, seed=seed)
        self.quantizer = ScalarQuantizer(bits=bits, block_size=block_size)

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
        """Rotate and quantize K, V for a given layer.

        Args:
            layer_idx: which transformer layer
            key: shape (batch, num_kv_heads, seq_len, head_dim)
            value: shape (batch, num_kv_heads, seq_len, head_dim)
        """
        k_rotated = self.rotation.rotate(key)
        v_rotated = self.rotation.rotate(value)

        k_codes, k_scales, k_zeros = self.quantizer.quantize(k_rotated)
        v_codes, v_scales, v_zeros = self.quantizer.quantize(v_rotated)

        self.k_codes[layer_idx] = k_codes
        self.k_scales[layer_idx] = k_scales
        self.k_zeros[layer_idx] = k_zeros
        self.v_codes[layer_idx] = v_codes
        self.v_scales[layer_idx] = v_scales
        self.v_zeros[layer_idx] = v_zeros

        self._seq_len = key.shape[-2]

    def get_rotated_kv(
        self, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize and return K, V in rotated space (for attention computation).

        Returns K, V still in Hadamard-rotated space — caller computes attention
        in this space and inverse-rotates the output.
        """
        k_deq = self.quantizer.dequantize(
            self.k_codes[layer_idx],
            self.k_scales[layer_idx],
            self.k_zeros[layer_idx],
        )
        v_deq = self.quantizer.dequantize(
            self.v_codes[layer_idx],
            self.v_scales[layer_idx],
            self.v_zeros[layer_idx],
        )
        return k_deq, v_deq

    def compressed_attention(
        self,
        layer_idx: int,
        query: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute attention with compressed KV cache in rotated domain.

        Args:
            layer_idx: transformer layer index
            query: shape (batch, num_q_heads, q_len, head_dim)
            scale: attention scale factor (default: 1/sqrt(head_dim))

        Returns:
            Attention output in original (non-rotated) space,
            shape (batch, num_q_heads, q_len, head_dim)
        """
        if scale is None:
            # Use original head_dim for attention scale, NOT padded_dim.
            # Padding is an implementation detail of the Hadamard transform;
            # the semantic scale factor must match the model's head_dim.
            scale = 1.0 / math.sqrt(self.head_dim)

        q_rotated = self.rotation.rotate(query.float())

        k_rotated, v_rotated = self.get_rotated_kv(layer_idx)

        num_q_heads = query.shape[1]
        num_kv_heads = k_rotated.shape[1]
        if num_q_heads != num_kv_heads:
            repeat_factor = num_q_heads // num_kv_heads
            k_rotated = k_rotated.repeat_interleave(repeat_factor, dim=1)
            v_rotated = v_rotated.repeat_interleave(repeat_factor, dim=1)

        scores = torch.matmul(q_rotated, k_rotated.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        out_rotated = torch.matmul(attn_weights, v_rotated)

        out = self.rotation.inverse_rotate(out_rotated)
        return out

    def memory_bytes(self) -> int:
        """Estimate total memory usage of the compressed cache."""
        if self.k_codes[0] is None:
            return 0

        total = 0
        for i in range(self.num_layers):
            codes_bytes = self.k_codes[i].numel() + self.v_codes[i].numel()
            scales_bytes = (
                self.k_scales[i].numel() + self.v_scales[i].numel() +
                self.k_zeros[i].numel() + self.v_zeros[i].numel()
            ) * 2
            total += codes_bytes + scales_bytes
        return total

    def full_precision_bytes(self) -> int:
        """Estimate equivalent full-precision KV cache memory."""
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


def compute_quant_error_bound(
    dim: int,
    bits: int,
) -> float:
    """Worst-case per-coordinate quantization error for b-bit uniform scalar quantization.

    For a per-block min-max quantizer with (2^b - 1) levels, the worst-case
    rounding error per coordinate is range / (2 * (2^b - 1)).  Across d
    coordinates, the worst-case L2 error is sqrt(d) times the per-coordinate
    bound (not sqrt(d / block_size) — block-shared scale does not reduce
    worst-case error).

    Returns the per-coordinate worst-case relative error factor: 1 / (2 * (2^b - 1)).
    """
    n_levels = (1 << bits) - 1
    return 1.0 / (2 * n_levels)


def estimate_tv_proxy(
    w_o_fnorm: float,
    range_k: float,
    range_v: float,
    v_fnorm: float,
    q_fnorm: float,
    dim: int,
    bits: int,
    temperature: float = 1.0,
) -> float:
    """Heuristic TV proxy for single-head, single-layer attention quantization.

    NOT a rigorous upper bound.  This estimates the scale of logit perturbation
    from quantizing one attention layer's KV cache, under a Gaussian-noise
    approximation of uniform scalar quantization error.

    Limitations (see PROOF_AUDIT.md Issues 1-4):
      - Uses RMS error model (sigma = range / (2*sqrt(3)*(2^b-1))), not worst-case.
      - Only covers one attention head; multi-layer error propagation is ignored.
      - Requires query norm (q_fnorm) for the key-perturbation term.
      - Temperature scaling applies to the output softmax only (single factor).

    For a proper bound, compose per-layer Lipschitz constants across the
    full transformer (which requires layer-norm Lipschitz, residual structure,
    and LM head norm — well beyond a single closed-form expression).
    """
    n_levels = (1 << bits) - 1
    # RMS per-coordinate quantization noise (Gaussian approx of uniform)
    sigma_per_coord = 1.0 / (2 * math.sqrt(3) * n_levels)

    # Value perturbation: ||delta_v|| ~ range_v * sigma * sqrt(d)
    v_err = range_v * sigma_per_coord * math.sqrt(dim)
    # Key perturbation: changes scores by q·epsilon_k, so depends on ||q||
    k_err = q_fnorm * range_k * sigma_per_coord * math.sqrt(dim)

    # Single-head output perturbation (heuristic, not a bound)
    proxy = w_o_fnorm * (v_err + v_fnorm * k_err) / temperature
    return proxy
