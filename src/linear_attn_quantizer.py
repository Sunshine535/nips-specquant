"""Quantized state cache and verifier for GatedDeltaNet linear attention (Qwen3.5).

Qwen3.5 uses GatedDeltaNet (linear attention) instead of standard MHA.
Its cache stores fixed-size recurrent states rather than growing KV sequences:
  - recurrent_states: [batch, num_heads, state_dim, state_dim] -- delta-rule state S
  - conv_states: [batch, intermediate_size, conv_width] -- causal convolution state

This module applies Hadamard rotation + scalar quantization to the recurrent state
matrices, providing compressed-domain verification analogous to QuantizedKVCache
for standard attention models.

Key insight: the state matrix S is fixed-size (d x d per head), so compression
ratio is constant regardless of sequence length -- a major advantage over KV cache
quantization where memory grows linearly with context.
"""

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .turboquant_kv import HadamardRotation, ScalarQuantizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture detection helpers
# ---------------------------------------------------------------------------

_LINEAR_ATTN_ARCH_KEYWORDS = (
    "gateddeltan",
    "gateddeltanet",
    "linear_attn",
    "qwen3.5",
    "qwen3_5",
)


def is_linear_attention_model(model) -> bool:
    """Detect whether a model uses GatedDeltaNet linear attention.

    Checks for:
    1. Architecture string containing known linear-attention identifiers.
    2. Presence of ``linear_attn`` submodules on transformer layers.
    3. A cache object that carries ``recurrent_states`` rather than key/value.
    """
    config = getattr(model, "config", None)
    if config is not None:
        arch_list = getattr(config, "architectures", []) or []
        model_type = getattr(config, "model_type", "") or ""
        combined = " ".join(arch_list).lower() + " " + model_type.lower()
        for kw in _LINEAR_ATTN_ARCH_KEYWORDS:
            if kw in combined:
                return True

    # Structural probe: look for linear_attn on the first decoder layer.
    layers = _get_decoder_layers(model)
    if layers and hasattr(layers[0], "linear_attn"):
        return True

    return False


def _get_decoder_layers(model) -> list:
    """Return the list of decoder layer modules from a HF causal LM."""
    # Common HF attribute paths for the layer list.
    for attr_chain in (
        ("model", "layers"),
        ("transformer", "h"),
        ("transformer", "layers"),
        ("gpt_neox", "layers"),
    ):
        obj = model
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            return list(obj)
    return []


# ---------------------------------------------------------------------------
# QuantizedStateCache
# ---------------------------------------------------------------------------

class QuantizedStateCache:
    """Compressed storage for GatedDeltaNet recurrent states.

    Applies Hadamard rotation to the last dimension (columns) of each row
    of the state matrix S in R^{d x d}, then scalar-quantizes to *b*-bit.

    Conceptually:
        S_rotated = S @ H^T            (rotate columns)
        codes, scales, zeros = quantize(S_rotated)
        S_approx = H @ dequantize(codes, scales, zeros)

    For a single head with state_dim=128 and 3-bit quantization:
        FP16:  128*128*2 = 32 KiB
        3-bit: 128*128*1 (codes) + metadata ~ 18 KiB  => ~1.8x compression per head

    Across all layers and heads, the savings compound.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        state_dim: int,
        bits: int = 3,
        block_size: int = 128,
        seed: int = 42,
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.state_dim = state_dim
        self.bits = bits
        self._seed = seed
        self.block_size = block_size

        # Rotation applied along the last (column) dimension of the state matrix.
        self.rotation = HadamardRotation(state_dim, seed=seed)
        self.quantizer = ScalarQuantizer(bits=bits, block_size=block_size)

        # Per-layer quantized recurrent state storage.
        self.state_codes: List[Optional[torch.Tensor]] = [None] * num_layers
        self.state_scales: List[Optional[torch.Tensor]] = [None] * num_layers
        self.state_zeros: List[Optional[torch.Tensor]] = [None] * num_layers

        # Per-layer conv state storage (small -- stored in fp16, not quantized
        # by default, but we provide an optional quantized path).
        self.conv_states_fp: List[Optional[torch.Tensor]] = [None] * num_layers
        self.conv_codes: List[Optional[torch.Tensor]] = [None] * num_layers
        self.conv_scales: List[Optional[torch.Tensor]] = [None] * num_layers
        self.conv_zeros: List[Optional[torch.Tensor]] = [None] * num_layers
        self._quantize_conv = False  # toggled by compress_conv

        self._batch_size = 0

    # ----- recurrent state compression -----

    def compress_state(
        self, layer_idx: int, state: torch.Tensor
    ) -> None:
        """Compress recurrent_states [batch, heads, d, d] -> quantized codes.

        We treat each (batch, head) slice as an independent 2-D matrix of shape
        [d, d].  The Hadamard rotation is applied along the last dimension
        (columns) so each row becomes a rotated vector suitable for scalar
        quantization.

        After rotation the matrix is [batch*heads, d, d] which we feed to the
        existing ScalarQuantizer that expects (..., seq_len, dim) -- here
        seq_len=d (rows) and dim=d (rotated columns).
        """
        b, h, d1, d2 = state.shape
        # Auto-adapt state_dim on first call if needed
        if d2 != self.state_dim:
            logger.info(f"Auto-adapting state_dim: {self.state_dim} -> {d2}")
            self.state_dim = d2
            self.rotation = HadamardRotation(d2, seed=self._seed if hasattr(self, '_seed') else 42)
        self._batch_size = b
        self._actual_heads = h
        self._d1 = d1  # May differ from state_dim for non-square states

        # Flatten batch and heads -> [b*h, d1, d2]
        flat = state.float().reshape(b * h, d1, d2)

        # Rotate along the last dimension (columns).
        rotated = self.rotation.rotate(flat)  # [b*h, d1, padded_d]

        # Quantize: treat as (..., seq_len=d1, dim=padded_d)
        codes, scales, zeros = self.quantizer.quantize(rotated)

        self.state_codes[layer_idx] = codes
        self.state_scales[layer_idx] = scales
        self.state_zeros[layer_idx] = zeros

    def decompress_state(self, layer_idx: int) -> torch.Tensor:
        """Decompress back to fp32 [batch, heads, d, d].

        Dequantizes and applies inverse Hadamard rotation.
        """
        codes = self.state_codes[layer_idx]
        scales = self.state_scales[layer_idx]
        zeros = self.state_zeros[layer_idx]
        if codes is None:
            raise ValueError(f"No compressed state stored for layer {layer_idx}")

        deq = self.quantizer.dequantize(codes, scales, zeros)  # [b*h, d, padded_d]
        unrotated = self.rotation.inverse_rotate(deq)  # [b*h, d, state_dim]

        b = self._batch_size
        h = getattr(self, '_actual_heads', self.num_heads)
        d1 = getattr(self, '_d1', self.state_dim)
        return unrotated.reshape(b, h, d1, self.state_dim)

    # ----- conv state compression -----

    def compress_conv(
        self, layer_idx: int, conv_states: torch.Tensor
    ) -> None:
        """Compress conv_states (optional, small).

        conv_states shape: [batch, intermediate_size, conv_width]
        These are very small (~1 MB total across all layers) so we simply
        store them in fp16.  If quantization is desired, set
        ``self._quantize_conv = True`` before calling.
        """
        if not self._quantize_conv:
            self.conv_states_fp[layer_idx] = conv_states.half()
            return

        # Quantize along the last dim (conv_width, typically 4).
        # ScalarQuantizer expects (..., seq_len, dim). Treat
        # intermediate_size as seq_len and conv_width as dim.
        flat = conv_states.float()  # [batch, inter, conv_w]
        codes, scales, zeros = self.quantizer.quantize(flat)
        self.conv_codes[layer_idx] = codes
        self.conv_scales[layer_idx] = scales
        self.conv_zeros[layer_idx] = zeros

    def decompress_conv(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Decompress conv_states back to fp32."""
        if self.conv_states_fp[layer_idx] is not None:
            return self.conv_states_fp[layer_idx].float()
        codes = self.conv_codes[layer_idx]
        if codes is None:
            return None
        return self.quantizer.dequantize(
            codes, self.conv_scales[layer_idx], self.conv_zeros[layer_idx]
        )

    # ----- memory reporting -----

    def memory_bytes(self) -> int:
        """Estimate total memory of compressed state cache (all layers)."""
        total = 0
        for i in range(self.num_layers):
            # Recurrent state
            if self.state_codes[i] is not None:
                total += self.state_codes[i].numel()  # uint8 -> 1 byte each
                total += self.state_scales[i].numel() * 2  # fp16
                total += self.state_zeros[i].numel() * 2  # fp16
            # Conv state
            if self.conv_states_fp[i] is not None:
                total += self.conv_states_fp[i].numel() * 2  # fp16
            elif self.conv_codes[i] is not None:
                total += self.conv_codes[i].numel()
                total += self.conv_scales[i].numel() * 2
                total += self.conv_zeros[i].numel() * 2
        return total

    def full_precision_bytes(self) -> int:
        """Estimate equivalent fp16 storage for all states."""
        # recurrent: [batch, heads, d, d] * fp16 per layer
        recurrent_per_layer = (
            self._batch_size * self.num_heads * self.state_dim * self.state_dim * 2
        )
        total = recurrent_per_layer * self.num_layers

        # conv states (if stored)
        for i in range(self.num_layers):
            if self.conv_states_fp[i] is not None:
                total += self.conv_states_fp[i].numel() * 2
            elif self.conv_codes[i] is not None:
                # Reconstruct original numel from codes shape
                total += self.conv_codes[i].numel() * 2
        return total

    @property
    def compression_ratio(self) -> float:
        fp = self.full_precision_bytes()
        if fp == 0:
            return 0.0
        compressed = self.memory_bytes()
        if compressed == 0:
            return 0.0
        return fp / compressed


# ---------------------------------------------------------------------------
# LinearAttnVerifier
# ---------------------------------------------------------------------------

class LinearAttnVerifier:
    """Compressed-state verification for GatedDeltaNet linear attention.

    Provides the same interface expected by SpeculativeDecoder:
    compress the target model's cache after prefill and after each
    verification round, reducing memory traffic at the cost of a small
    quantization error estimated by the same RMS-noise TV proxy analysis
    (adapted for fixed-size state matrices rather than growing KV).

    Usage within SpeculativeDecoder:
        verifier = LinearAttnVerifier(target_model, bits=3, ...)
        compressed_cache = verifier.compress_cache(raw_cache)
        # ... run verification forward pass with compressed_cache ...

    The delta-rule update is:
        S_new = decay * S_old + beta * v * k^T
    With compressed S_old we decompress, apply the update in fp32,
    then re-compress S_new.
    """

    def __init__(
        self,
        model,
        bits: int = 3,
        block_size: int = 128,
        seed: int = 42,
        quantize_conv: bool = False,
    ):
        self.model = model
        self.bits = bits
        self.block_size = block_size
        self.seed = seed
        self.quantize_conv = quantize_conv

        config = model.config
        # Handle nested text_config (multimodal models like Qwen3.5)
        if hasattr(config, "text_config"):
            config = config.text_config
        self.num_layers = config.num_hidden_layers
        # GatedDeltaNet state: [batch, num_value_heads, key_dim, value_dim]
        # Detect actual dimensions from a probe forward pass or from config.
        # Prefer probing the actual cache to be architecture-agnostic.
        self.num_heads = getattr(config, "linear_num_value_heads",
                                 getattr(config, "linear_num_key_heads",
                                         config.num_attention_heads))
        self.state_dim = getattr(config, "linear_key_head_dim",
                                 getattr(config, "linear_value_head_dim",
                                         getattr(config, "head_dim", 128)))

        self._state_cache: Optional[QuantizedStateCache] = None
        self._patches_installed = False
        self._original_forwards: Dict[int, Any] = {}

        logger.info(
            "LinearAttnVerifier: %d-bit, block=%d, layers=%d, heads=%d, "
            "state_dim=%d",
            bits, block_size, self.num_layers, self.num_heads, self.state_dim,
        )

    def _build_state_cache(self) -> QuantizedStateCache:
        cache = QuantizedStateCache(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            state_dim=self.state_dim,
            bits=self.bits,
            block_size=self.block_size,
            seed=self.seed,
        )
        cache._quantize_conv = self.quantize_conv
        return cache

    # ----- cache compression (round-trip) -----

    def compress_cache(self, cache: Any) -> Any:
        """Compress a GatedDeltaNet cache in-place (round-trip quantize).

        The cache object is expected to expose per-layer recurrent_states
        and conv_states.  We compress each, then write back the decompressed
        (lossy) version so the model can continue decoding.

        Supports two cache layouts:
        1. Tuple of tuples: ((recurrent_l0, conv_l0), (recurrent_l1, ...), ...)
        2. Object with .recurrent_states / .conv_states lists (HF DynamicCache
           style or custom Qwen3.5 cache).
        """
        if self._state_cache is None:
            self._state_cache = self._build_state_cache()

        self._last_t0 = time.perf_counter()
        sc = self._state_cache

        if isinstance(cache, (list, tuple)):
            return self._compress_tuple_cache(cache, sc)
        if hasattr(cache, "recurrent_states"):
            return self._compress_object_cache(cache, sc)

        # transformers >= 5.5.0: DynamicCache with .layers list of
        # LinearAttentionLayer objects that have .recurrent_states / .conv_states
        if hasattr(cache, "layers") and len(cache.layers) > 0:
            layer0 = cache.layers[0]
            if hasattr(layer0, "recurrent_states") and layer0.recurrent_states is not None:
                return self._compress_layers_cache(cache, sc)

        # Fallback: try to detect recurrent states in a DynamicCache that
        # repurposes key_cache / value_cache slots for state/conv.
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return self._compress_kv_style_cache(cache, sc)

        logger.warning(
            "LinearAttnVerifier: unrecognised cache type %s -- skipping compression",
            type(cache).__name__,
        )
        return cache

    def _compress_layers_cache(
        self, cache: Any, sc: QuantizedStateCache
    ) -> Any:
        """Compress a DynamicCache with .layers[i].recurrent_states layout."""
        for i in range(min(len(cache.layers), self.num_layers)):
            layer = cache.layers[i]
            if not hasattr(layer, "recurrent_states") or layer.recurrent_states is None:
                continue
            state = layer.recurrent_states  # [batch, heads, d, d]
            sc.compress_state(i, state.float())
            decompressed = sc.decompress_state(i).to(state.dtype)
            layer.recurrent_states = decompressed
            # Conv states are small — keep in fp16
            if hasattr(layer, "conv_states") and layer.conv_states is not None:
                sc.compress_conv(i, layer.conv_states.float())
        if not hasattr(self, 't_compress'):
            self.t_compress = 0.0
        self.t_compress += time.perf_counter() - self._last_t0
        return cache

    def _compress_tuple_cache(
        self, cache: tuple, sc: QuantizedStateCache
    ) -> tuple:
        new_layers = []
        for i, layer_cache in enumerate(cache):
            if i >= self.num_layers:
                new_layers.append(layer_cache)
                continue
            recurrent, conv = layer_cache[0], layer_cache[1]

            # Compress recurrent state.
            sc.compress_state(i, recurrent.float())
            approx_recurrent = sc.decompress_state(i).to(recurrent.dtype)

            # Compress conv state.
            if conv is not None:
                sc.compress_conv(i, conv.float())
                approx_conv = sc.decompress_conv(i)
                if approx_conv is not None:
                    approx_conv = approx_conv.to(conv.dtype)
                else:
                    approx_conv = conv
            else:
                approx_conv = None

            rest = layer_cache[2:] if len(layer_cache) > 2 else ()
            new_layers.append((approx_recurrent, approx_conv) + rest)
        return tuple(new_layers)

    def _compress_object_cache(
        self, cache: Any, sc: QuantizedStateCache
    ) -> Any:
        for i in range(min(len(cache.recurrent_states), self.num_layers)):
            rs = cache.recurrent_states[i]
            if rs is None:
                continue
            sc.compress_state(i, rs.float())
            cache.recurrent_states[i] = sc.decompress_state(i).to(rs.dtype)

            if hasattr(cache, "conv_states") and i < len(cache.conv_states):
                cs = cache.conv_states[i]
                if cs is not None:
                    sc.compress_conv(i, cs.float())
                    approx = sc.decompress_conv(i)
                    if approx is not None:
                        cache.conv_states[i] = approx.to(cs.dtype)
        return cache

    def _compress_kv_style_cache(
        self, cache: Any, sc: QuantizedStateCache
    ) -> Any:
        """Handle the case where recurrent state is stored in key_cache slots.

        Some HF wrappers for linear-attention models repurpose DynamicCache
        by storing recurrent_states in key_cache and conv_states in
        value_cache.  Detect this by checking tensor shapes.
        """
        for i in range(min(len(cache.key_cache), self.num_layers)):
            k = cache.key_cache[i]
            v = cache.value_cache[i]
            if k is None:
                continue

            # Heuristic: recurrent state has shape [b, heads, d, d]
            if k.dim() == 4 and k.shape[-1] == k.shape[-2]:
                sc.compress_state(i, k.float())
                cache.key_cache[i] = sc.decompress_state(i).to(k.dtype)

            if v is not None:
                sc.compress_conv(i, v.float())
                approx = sc.decompress_conv(i)
                if approx is not None:
                    cache.value_cache[i] = approx.to(v.dtype)
        return cache

    # ----- monkey-patching -----

    def install_patches(self) -> None:
        """Monkey-patch linear_attn forwards to inject compression.

        After patching, each layer's linear_attn.forward will:
        1. Call the original forward.
        2. Compress the returned recurrent state in-place.

        This is useful for intercepting intermediate states during a single
        forward pass (e.g., for profiling or per-layer analysis).
        """
        if self._patches_installed:
            return

        layers = _get_decoder_layers(self.model)
        if self._state_cache is None:
            self._state_cache = self._build_state_cache()

        for idx, layer in enumerate(layers):
            attn_mod = getattr(layer, "linear_attn", None)
            if attn_mod is None:
                continue

            original_forward = attn_mod.forward
            self._original_forwards[idx] = original_forward
            sc = self._state_cache
            layer_idx = idx

            def _make_patched(orig_fwd, li, state_cache):
                def patched_forward(*args, **kwargs):
                    output = orig_fwd(*args, **kwargs)
                    # Output may be a tuple: (hidden, new_cache, ...)
                    if isinstance(output, tuple) and len(output) >= 2:
                        new_cache = output[1]
                        if isinstance(new_cache, tuple) and len(new_cache) >= 1:
                            recurrent = new_cache[0]
                            if (
                                recurrent is not None
                                and recurrent.dim() == 4
                                and recurrent.shape[-1] == recurrent.shape[-2]
                            ):
                                state_cache.compress_state(li, recurrent.float())
                                approx = state_cache.decompress_state(li).to(
                                    recurrent.dtype
                                )
                                new_cache = (approx,) + new_cache[1:]
                                output = (output[0], new_cache) + output[2:]
                    return output
                return patched_forward

            attn_mod.forward = _make_patched(original_forward, layer_idx, sc)

        self._patches_installed = True
        logger.info("LinearAttnVerifier: installed patches on %d layers", len(self._original_forwards))

    def remove_patches(self) -> None:
        """Restore original forward methods."""
        if not self._patches_installed:
            return

        layers = _get_decoder_layers(self.model)
        for idx, original_forward in self._original_forwards.items():
            attn_mod = getattr(layers[idx], "linear_attn", None)
            if attn_mod is not None:
                attn_mod.forward = original_forward

        self._original_forwards.clear()
        self._patches_installed = False
        logger.info("LinearAttnVerifier: removed patches")

    # ----- high-level verification entry point -----

    @torch.no_grad()
    def verify(
        self,
        draft_tokens: torch.Tensor,
        past_cache: Any,
    ) -> Tuple[Any, Any]:
        """Run a verification forward pass with state compression.

        Args:
            draft_tokens: [1, gamma] tensor of drafted token ids.
            past_cache: the target model's cache from the previous step.

        Returns:
            (model_output, compressed_cache) where model_output contains
            logits for rejection sampling and compressed_cache is the
            round-tripped cache.
        """
        device = next(self.model.parameters()).device
        output = self.model(
            draft_tokens.to(device),
            past_key_values=past_cache,
            use_cache=True,
        )
        compressed = self.compress_cache(output.past_key_values)
        return output, compressed

    # ----- stats -----

    def get_memory_stats(self) -> Dict[str, Any]:
        """Return a dict of memory statistics for the compressed state."""
        if self._state_cache is None:
            return {
                "compressed_bytes": 0,
                "full_precision_bytes": 0,
                "compression_ratio": 0.0,
                "bits": self.bits,
            }
        sc = self._state_cache
        return {
            "compressed_bytes": sc.memory_bytes(),
            "full_precision_bytes": sc.full_precision_bytes(),
            "compression_ratio": sc.compression_ratio,
            "bits": self.bits,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "state_dim": self.state_dim,
        }


# ---------------------------------------------------------------------------
# Heuristic TV proxy for state quantization
# ---------------------------------------------------------------------------

def estimate_state_quant_tv_proxy(
    w_o_fnorm: float,
    state_range: float,
    state_dim: int,
    bits: int,
    block_size: int,
    temperature: float = 1.0,
) -> float:
    """Heuristic TV proxy for state-matrix quantization.

    NOT a rigorous upper bound (see PROOF_AUDIT.md Issues 1-4).
    Analogous to ``estimate_tv_proxy`` in turboquant_kv.py but adapted for
    the fixed-size recurrent state matrix.  Uses RMS noise model:

        sigma_eff = state_range / ((2^b - 1) * sqrt(12 * d))

    This is a heuristic scale estimate, not a deterministic bound.
    """
    n_levels = (1 << bits) - 1
    sigma = state_range / (n_levels * math.sqrt(12.0 * state_dim))
    noise_norm = sigma * state_dim  # summed over d rows of d columns
    tv = w_o_fnorm * noise_norm / temperature
    return min(tv, 1.0)
