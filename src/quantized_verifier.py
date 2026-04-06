"""Quantized verifier: monkey-patches target model attention for compressed-domain verification.

Instead of post-hooks (which run AFTER native SDPA already loaded full FP16
prefix KV from HBM, doubling bandwidth), this module REPLACES each attention
layer's forward() with a version that:

  1. Computes Q, K, V from hidden_states using the layer's own projections.
  2. Applies RoPE at correct positions.
  3. For prefix positions: reads compressed KV from QuantizedKVCache (small
     HBM read -- uint8 codes + fp16 scales instead of full fp16 KV).
  4. For new draft tokens: standard FP16 attention (tiny, only gamma tokens).
  5. Merges prefix + new attention via log-sum-exp weighted combination.
  6. Applies output projection.

The native SDPA NEVER runs on the prefix -- only on gamma new tokens.
Bandwidth savings = prefix_len / (prefix_len + gamma) * compression_ratio
"""

import math
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .turboquant_kv import QuantizedKVCache
from .utils import get_kv_tensors, get_num_kv_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture discovery
# ---------------------------------------------------------------------------

_ATTN_PATTERNS = [
    # (layers_accessor, attn_attr) -- covers Llama / Qwen / Mistral families
    ("model.layers", "self_attn"),       # Llama, Qwen, Mistral, Gemma
]


def _find_attention_layers(model: torch.nn.Module) -> List[Tuple[int, torch.nn.Module]]:
    """Discover attention sub-modules for Llama/Qwen/Mistral architectures.

    Returns a list of (layer_idx, attn_module) sorted by layer index.
    Raises RuntimeError when no known pattern matches.
    """
    for layers_path, attn_attr in _ATTN_PATTERNS:
        parts = layers_path.split(".")
        obj = model
        try:
            for p in parts:
                obj = getattr(obj, p)
        except AttributeError:
            continue

        # obj should be an nn.ModuleList (or similar iterable)
        if not hasattr(obj, "__len__"):
            continue

        results = []
        for idx, layer in enumerate(obj):
            attn = getattr(layer, attn_attr, None)
            if attn is not None:
                # Verify the module has the projections we need
                if not all(hasattr(attn, p) for p in ("q_proj", "k_proj", "v_proj", "o_proj")):
                    continue
                results.append((idx, attn))

        if results:
            logger.debug(
                "Found %d attention layers via pattern '%s.*.%s'",
                len(results), layers_path, attn_attr,
            )
            return results

    raise RuntimeError(
        f"Cannot discover attention layers for {type(model).__name__}. "
        "Supported families: Llama, Qwen, Mistral."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_model_geometry(model: torch.nn.Module) -> Dict[str, int]:
    """Pull num_layers, num_q_heads, num_kv_heads, head_dim from HF config."""
    cfg = model.config
    num_q_heads = cfg.num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", num_q_heads)
    head_dim = cfg.hidden_size // num_q_heads
    return {
        "num_layers": cfg.num_hidden_layers,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def _safe_log_sumexp_merge(
    attn_prefix: torch.Tensor,
    lse_prefix: torch.Tensor,
    attn_new: torch.Tensor,
    lse_new: torch.Tensor,
) -> torch.Tensor:
    """Numerically stable merge of two partial attention outputs.

    Given::

        out_prefix = softmax(Q @ K_prefix^T / s) @ V_prefix   (shape: B, H, Q, D)
        lse_prefix = log(sum(exp(Q @ K_prefix^T / s)))         (shape: B, H, Q)

    and similarly for _new tokens, the merged output over [prefix; new] is::

        w_p = exp(lse_prefix - lse_max)
        w_n = exp(lse_new   - lse_max)
        out = (w_p * out_prefix + w_n * out_new) / (w_p + w_n)

    This is exact -- not an approximation.

    All tensors must share the same dtype and device.
    """
    # lse shapes: (B, H, Q) -- unsqueeze last dim for broadcasting against D
    lse_max = torch.maximum(lse_prefix, lse_new).unsqueeze(-1)  # (B, H, Q, 1)
    w_p = torch.exp(lse_prefix.unsqueeze(-1) - lse_max)
    w_n = torch.exp(lse_new.unsqueeze(-1) - lse_max)
    merged = (w_p * attn_prefix + w_n * attn_new) / (w_p + w_n + 1e-12)
    return merged


def _compute_attention_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    causal_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard scaled dot-product attention that also returns log-sum-exp.

    Args:
        query:  (B, H, Q, D)
        key:    (B, H, S, D)
        value:  (B, H, S, D)
        scale:  1/sqrt(d) scaling factor
        causal_mask: optional (Q, S) bool mask -- True means *masked* (excluded)

    Returns:
        attn_output: (B, H, Q, D)
        lse:         (B, H, Q)  -- log of the softmax denominator per query position
    """
    # (B, H, Q, S)
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if causal_mask is not None:
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # log-sum-exp along key dimension
    lse = torch.logsumexp(scores, dim=-1)  # (B, H, Q)

    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, value)  # (B, H, Q, D)
    return attn_output, lse


# ---------------------------------------------------------------------------
# RoPE helpers (matches HF transformers internal implementation)
# ---------------------------------------------------------------------------

def _apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary positional embeddings to a tensor.

    Args:
        x: (B, H, S, D)
        cos, sin: broadcastable to (B, 1, S, D) or (1, 1, S, D)
    """
    # Ensure cos/sin have the right shape for broadcasting
    if cos.dim() == 2:
        # (S, D) -> (1, 1, S, D)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # (B, S, D) -> (B, 1, S, D)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    # else assume (B, H, S, D) or (1, 1, S, D) already

    # Standard rotary: split into halves, rotate
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]

    cos = cos[..., : d // 2]
    sin = sin[..., : d // 2]

    rotated = torch.cat(
        [x1 * cos - x2 * sin, x2 * cos + x1 * sin],
        dim=-1,
    )
    return rotated


def _apply_rotary_pos_emb_pair(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to both query and key tensors."""
    return _apply_rotary_pos_emb(q, cos, sin), _apply_rotary_pos_emb(k, cos, sin)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class QuantizedVerifier:
    """Monkey-patches target model attention for compressed-domain verification.

    Instead of post-hooks (which run AFTER native attention already loaded FP KV),
    this REPLACES each attention layer's forward() with a version that:
    1. Computes Q, K, V from hidden_states using the layer's own projections
    2. Applies RoPE at correct positions
    3. For prefix positions: reads compressed KV from QuantizedKVCache (small HBM read)
    4. For new draft tokens: standard FP16 attention (tiny, only gamma tokens)
    5. Merges prefix + new attention via log-sum-exp weighted combination
    6. Applies output projection

    The native SDPA NEVER runs on the prefix -- only on gamma new tokens.
    Bandwidth savings = prefix_len / (prefix_len + gamma) * compression_ratio
    """

    def __init__(
        self,
        model: torch.nn.Module,
        bits: int = 3,
        block_size: int = 128,
        seed: int = 42,
    ):
        geom = _extract_model_geometry(model)
        self.model = model
        self.device = next(model.parameters()).device
        self.num_layers = geom["num_layers"]
        self.num_q_heads = geom["num_q_heads"]
        self.num_kv_heads = geom["num_kv_heads"]
        self.head_dim = geom["head_dim"]
        self.bits = bits
        self.gqa_repeat = self.num_q_heads // self.num_kv_heads

        self.qkv_cache = QuantizedKVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            bits=bits,
            block_size=block_size,
            seed=seed,
        )

        self._attn_layers = _find_attention_layers(model)

        # Stored original forward methods (so we can restore them)
        self._originals: List[Tuple[torch.nn.Module, Any]] = []

        # Per-invocation state set by verify() before the forward pass
        self._active = False

        # Timing accumulators (seconds)
        self.t_compress = 0.0
        self.t_decompress_attn = 0.0
        self.t_patch_overhead = 0.0

        logger.info(
            "QuantizedVerifier: %d-bit, layers=%d, q_heads=%d, kv_heads=%d, "
            "head_dim=%d, GQA repeat=%d",
            bits, self.num_layers, self.num_q_heads, self.num_kv_heads,
            self.head_dim, self.gqa_repeat,
        )

    # ------------------------------------------------------------------
    # Patch management (replaces hook-based approach)
    # ------------------------------------------------------------------

    def install_patches(self) -> None:
        """Replace each attention layer's forward() with compressed-domain version.

        Unlike register_forward_hook (post-hook), monkey-patching ensures the
        native SDPA never loads full FP16 prefix KV from HBM.  The patched
        forward reads compressed KV (uint8 codes) for the prefix and only
        runs standard FP16 attention on the gamma new draft tokens.
        """
        self.remove_patches()  # idempotent
        for layer_idx, attn_module in self._attn_layers:
            original_forward = attn_module.forward
            self._originals.append((attn_module, original_forward))
            attn_module.forward = self._make_patched_forward(
                layer_idx, attn_module, original_forward,
            )
        logger.debug("Installed %d attention patches", len(self._originals))

    def remove_patches(self) -> None:
        """Restore all original attention forward methods."""
        for attn_module, original_forward in self._originals:
            attn_module.forward = original_forward
        self._originals.clear()

    # ------------------------------------------------------------------
    # Prefix KV compression
    # ------------------------------------------------------------------

    def compress_prefix_kv(self, kv_cache: Any) -> None:
        """Compress existing FP16 KV cache into the quantized store.

        Call this once after the initial prefill forward pass.  The original
        HF KV cache can be discarded afterwards (the compressed version is
        stored internally).

        Supports transformers 4.x (.key_cache/.value_cache),
        transformers >= 5.5 (.layers[i].keys/.values), and tuple caches.
        """
        t0 = time.perf_counter()

        num_layers = get_num_kv_layers(kv_cache)
        if num_layers == 0:
            raise TypeError(f"Unsupported KV cache type: {type(kv_cache)}")

        for i in range(min(num_layers, self.num_layers)):
            k, v = get_kv_tensors(kv_cache, i)
            if k is None:
                continue
            self.qkv_cache.compress_and_store(i, k.float(), v.float())

        self.t_compress += time.perf_counter() - t0
        logger.debug(
            "Prefix KV compressed: seq_len=%d, ratio=%.2fx, saved %.1f MB",
            self.qkv_cache.seq_len,
            self.qkv_cache.compression_ratio,
            (self.qkv_cache.full_precision_bytes() - self.qkv_cache.memory_bytes()) / 1e6,
        )

    def append_and_compress(self, kv_cache: Any) -> None:
        """Append new KV entries to the compressed cache after accepted tokens.

        The incoming kv_cache should contain *all* KV up to the current
        position (prefix + accepted).  We extract only the new portion
        (positions beyond self.qkv_cache.seq_len) and compress it.

        For simplicity and correctness we re-compress the full cache.
        A production implementation would do incremental append, but the
        cost is negligible relative to attention bandwidth.

        Supports transformers 4.x (.key_cache/.value_cache),
        transformers >= 5.5 (.layers[i].keys/.values), and tuple caches.
        """
        t0 = time.perf_counter()

        num_layers = get_num_kv_layers(kv_cache)
        if num_layers == 0:
            raise TypeError(f"Unsupported KV cache type: {type(kv_cache)}")

        for i in range(min(num_layers, self.num_layers)):
            k, v = get_kv_tensors(kv_cache, i)
            if k is None:
                continue
            self.qkv_cache.compress_and_store(i, k.float(), v.float())

        self.t_compress += time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Verification forward pass
    # ------------------------------------------------------------------

    def verify(
        self,
        draft_token_ids: torch.Tensor,
        target_kv: Any,
    ) -> Tuple[torch.Tensor, Any]:
        """Run the target model forward with compressed-domain prefix attention.

        This is the main entry point called during each speculative decoding
        round.  It:
          1. Sets internal state so patched forwards know to intercept.
          2. Runs model.forward() -- patched attention replaces prefix KV reads
             with compressed_attention().
          3. Returns (verify_logits, updated_kv).

        Args:
            draft_token_ids: (1, gamma) token ids to verify
            target_kv: HF past_key_values from previous round (will be
                replaced internally -- we pass it through so the model sees
                the correct positional information / sequence length).

        Returns:
            logits: (1, gamma, vocab_size) verification logits
            extended_kv: updated HF past_key_values (prefix + draft)
        """
        self._active = True

        try:
            with torch.no_grad():
                out = self.model(
                    draft_token_ids.to(self.device),
                    past_key_values=target_kv,
                    use_cache=True,
                )
        finally:
            self._active = False

        return out.logits, out.past_key_values

    # ------------------------------------------------------------------
    # The patched forward factory
    # ------------------------------------------------------------------

    def _make_patched_forward(
        self,
        layer_idx: int,
        module: torch.nn.Module,
        original_forward,
    ):
        """Build a patched forward() for one attention layer.

        When self._active is False, delegates to the original forward.
        When active, bypasses native SDPA entirely for the prefix and
        computes attention using compressed KV from QuantizedKVCache.

        The patched forward:
          - Projects hidden_states -> Q, K, V using the module's own weights
          - Applies RoPE (handles both old and new HF transformers API)
          - Updates the HF KV cache with new tokens (for position tracking)
          - Phase 1: compressed prefix attention via QuantizedKVCache
          - Phase 2: FP16 attention over just the gamma new tokens
          - Merges via log-sum-exp weighted combination
          - Applies o_proj and returns in the same format as original
        """
        verifier = self  # capture in closure

        num_q_heads = verifier.num_q_heads
        num_kv_heads = verifier.num_kv_heads
        head_dim = verifier.head_dim
        gqa_repeat = verifier.gqa_repeat
        scale = 1.0 / math.sqrt(head_dim)
        padded_scale = 1.0 / math.sqrt(verifier.qkv_cache.rotation.padded_dim)

        def patched_forward(
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_value: Any = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.Tensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
        ):
            # --- Inactive: delegate to original ---
            if not verifier._active:
                return original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            prefix_len = verifier.qkv_cache.seq_len
            if prefix_len == 0:
                # No compressed prefix yet -- run original
                return original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )

            t0 = time.perf_counter()

            B, L, _ = hidden_states.shape

            # ---- 1. Project Q, K, V using module's own weights ----
            q = module.q_proj(hidden_states)
            q = q.view(B, L, num_q_heads, head_dim).transpose(1, 2)
            k = module.k_proj(hidden_states)
            k = k.view(B, L, num_kv_heads, head_dim).transpose(1, 2)
            v = module.v_proj(hidden_states)
            v = v.view(B, L, num_kv_heads, head_dim).transpose(1, 2)

            # ---- 2. Apply RoPE ----
            # Handle both new HF API (position_embeddings kwarg) and older
            # API (module.rotary_emb callable).
            if position_embeddings is not None:
                cos, sin = position_embeddings
                q, k = _apply_rotary_pos_emb_pair(q, k, cos, sin)
            elif hasattr(module, "rotary_emb"):
                # Build position_ids if not provided
                if position_ids is None:
                    seq_start = prefix_len
                    position_ids = torch.arange(
                        seq_start, seq_start + L,
                        device=hidden_states.device, dtype=torch.long,
                    ).unsqueeze(0).expand(B, -1)
                try:
                    # New HF API: rotary_emb(x, position_ids)
                    cos, sin = module.rotary_emb(v, position_ids)
                except TypeError:
                    # Older API: rotary_emb(x, seq_len)
                    total_len = prefix_len + L
                    cos, sin = module.rotary_emb(v, total_len)
                    # Slice to only the positions we need
                    if cos.dim() == 4:
                        cos = cos[:, :, prefix_len:prefix_len + L, :]
                        sin = sin[:, :, prefix_len:prefix_len + L, :]
                    elif cos.dim() == 2:
                        cos = cos[prefix_len:prefix_len + L]
                        sin = sin[prefix_len:prefix_len + L]
                    elif cos.dim() == 3:
                        cos = cos[:, prefix_len:prefix_len + L, :]
                        sin = sin[:, prefix_len:prefix_len + L, :]
                q, k = _apply_rotary_pos_emb_pair(q, k, cos, sin)

            # ---- 3. Update HF KV cache with new tokens ----
            # We still update the HF cache so positional tracking is correct
            # downstream, but we do NOT use it for attention computation --
            # attention uses compressed prefix + FP new.
            if past_key_value is not None and use_cache:
                # DynamicCache.update() returns (all_k, all_v) but we only
                # use the new tokens for our FP attention phase.
                if not hasattr(past_key_value, "update"):
                    raise TypeError(
                        f"QuantizedVerifier requires a cache object with an "
                        f".update() method (e.g. DynamicCache), but got "
                        f"{type(past_key_value).__name__}. Tuple-based caches "
                        f"are not supported."
                    )
                past_key_value.update(k, v, layer_idx, cache_kwargs={
                    "cache_position": cache_position,
                } if cache_position is not None else {})

            # ---- 4. Phase 1: Compressed prefix attention ----
            t_decomp = time.perf_counter()

            # compressed_attention handles GQA internally and returns output
            # in original (non-rotated) space: (B, num_q_heads, L, head_dim)
            prefix_attn = verifier.qkv_cache.compressed_attention(
                layer_idx, q, scale=None,  # uses padded_dim scale internally
            )

            # Compute LSE for prefix to enable correct merge.
            # We need scores in the rotated domain to match compressed_attention.
            # Use float32 for all computations to avoid dtype mismatches.
            q_rotated = verifier.qkv_cache.rotation.rotate(q.float())
            k_rot, _ = verifier.qkv_cache.get_rotated_kv(layer_idx)

            # GQA expansion for score computation
            if gqa_repeat > 1:
                k_rot_exp = k_rot.repeat_interleave(gqa_repeat, dim=1)
            else:
                k_rot_exp = k_rot

            prefix_scores = torch.matmul(
                q_rotated, k_rot_exp.float().transpose(-2, -1)
            ) * padded_scale  # (B, num_q_heads, L, prefix_len)
            lse_prefix = torch.logsumexp(prefix_scores, dim=-1)  # (B, H, L)

            verifier.t_decompress_attn += time.perf_counter() - t_decomp

            # ---- 5. Phase 2: New-token attention (FP16, standard) ----
            # Only gamma new tokens -- trivially small, no bandwidth concern.
            if gqa_repeat > 1:
                k_new = k.repeat_interleave(gqa_repeat, dim=1)
                v_new = v.repeat_interleave(gqa_repeat, dim=1)
            else:
                k_new = k
                v_new = v

            # Causal mask: position i can attend to new positions [0..i]
            if L > 1:
                causal_mask = torch.triu(
                    torch.ones(L, L, device=q.device, dtype=torch.bool),
                    diagonal=1,
                )
            else:
                causal_mask = None

            new_attn, lse_new = _compute_attention_with_lse(
                q.float(), k_new.float(), v_new.float(),
                scale, causal_mask=causal_mask,
            )

            # ---- 6. Merge prefix + new ----
            merged = _safe_log_sumexp_merge(
                prefix_attn.float(), lse_prefix.float(),
                new_attn, lse_new,
            )

            # ---- 7. Output projection ----
            merged = merged.to(hidden_states.dtype)
            merged = merged.transpose(1, 2).contiguous().reshape(B, L, -1)
            output = module.o_proj(merged)

            verifier.t_patch_overhead += time.perf_counter() - t0

            # Return in the same format as the ORIGINAL attention forward.
            # Different HF model families return different tuple lengths:
            #   Qwen2: (attn_output, attn_weights) — 2 values
            #   Llama: (attn_output, attn_weights, past_kv) — 3 values
            # Match the original by inspecting its return convention.
            # Safe default: return (output, None) which matches Qwen2/Llama
            # when output_attentions=False. The decoder layer unpacks with:
            #   hidden_states, _ = self.self_attn(...)  — works with 2-tuple
            return output, None

        return patched_forward

    # ------------------------------------------------------------------
    # Memory / performance reporting
    # ------------------------------------------------------------------

    def get_memory_stats(self) -> Dict[str, Any]:
        """Report actual memory usage: compressed vs FP16 equivalent."""
        compressed = self.qkv_cache.memory_bytes()
        fp16_equiv = self.qkv_cache.full_precision_bytes()
        return {
            "compressed_bytes": compressed,
            "fp16_equivalent_bytes": fp16_equiv,
            "compression_ratio": self.qkv_cache.compression_ratio,
            "memory_saved_mb": (fp16_equiv - compressed) / 1e6,
            "seq_len": self.qkv_cache.seq_len,
            "bits": self.bits,
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
        }

    def get_timing_stats(self) -> Dict[str, float]:
        """Report accumulated timing for compression, decompression, and patch overhead."""
        return {
            "compress_seconds": self.t_compress,
            "decompress_attn_seconds": self.t_decompress_attn,
            "patch_overhead_seconds": self.t_patch_overhead,
            "total_verifier_seconds": (
                self.t_compress + self.t_decompress_attn + self.t_patch_overhead
            ),
        }

    def reset_timing(self) -> None:
        """Zero out all timing accumulators."""
        self.t_compress = 0.0
        self.t_decompress_attn = 0.0
        self.t_patch_overhead = 0.0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.install_patches()
        return self

    def __exit__(self, *exc):
        self.remove_patches()
        return False
