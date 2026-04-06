"""ThinkCompress: Adaptive KV cache compression for chain-of-thought reasoning.

CoT models (Qwen3, DeepSeek-R1) emit <think>...thinking tokens...</think> followed
by answer tokens. The thinking KV cache is 88-100% of total memory. ThinkCompress
compresses the thinking portion adaptively by tracking per-token importance via
attention patterns and assigning bit-widths proportional to utility.

Key ideas:
  1. Importance scoring: EMA of attention received by each thinking token.
  2. Adaptive bit-width: per-token assignment to maximize information under budget.
  3. Streaming eviction: lowest-importance tokens dropped entirely.
  4. Phase awareness: different policies for thinking generation vs answer generation.

Quantization primitives (HadamardRotation, ScalarQuantizer) are reused from
turboquant_kv to keep the compression path consistent with the SpecQuant pipeline.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .turboquant_kv import HadamardRotation, ScalarQuantizer
from .utils import get_kv_tensors, set_kv_tensors, get_num_kv_layers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bit-widths we actually support for quantized groups. FP16 tokens are stored
# uncompressed; evicted tokens consume zero bits.
SUPPORTED_BIT_WIDTHS = (2, 3, 4, 8)
FP16_BITS = 16
EVICTED_BITS = 0


# ---------------------------------------------------------------------------
# Phase tracking
# ---------------------------------------------------------------------------

class GenerationPhase(Enum):
    """Phase of chain-of-thought generation."""
    PREFIX = auto()      # prompt / system tokens before any generation
    THINKING = auto()    # inside <think>...</think>
    ANSWER = auto()      # after </think>, generating the answer


@dataclass
class PhaseState:
    """Tracks thinking vs answer boundaries during generation."""
    phase: GenerationPhase = GenerationPhase.PREFIX
    think_start_pos: int = -1   # sequence position of <think> token
    think_end_pos: int = -1     # sequence position of </think> token
    total_generated: int = 0

    @property
    def thinking_length(self) -> int:
        if self.think_start_pos < 0:
            return 0
        end = self.think_end_pos if self.think_end_pos >= 0 else (
            self.think_start_pos + self.total_generated
        )
        return max(0, end - self.think_start_pos - 1)


# ---------------------------------------------------------------------------
# Importance scoring
# ---------------------------------------------------------------------------

class ImportanceScorer:
    """Tracks per-token importance during generation.

    Three methods:
      1. EMA (default): exponential moving average of attention received.
         importance[t] = decay * importance[t] + (1-decay) * attention_from_new_token[t]
      2. Cumulative: running sum of all attention received since creation.
      3. Sink: attention sink detection -- first and last few tokens of
         each phase are always marked important regardless of attention.

    Scores are maintained per-layer, per-head, then averaged across layers/heads
    when final scores are requested.
    """

    METHODS = ("ema", "cumulative", "sink")

    def __init__(
        self,
        method: str = "ema",
        decay: float = 0.95,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        sink_size: int = 4,
        device: Optional[torch.device] = None,
    ):
        if method not in self.METHODS:
            raise ValueError(f"Unknown method '{method}'; choose from {self.METHODS}")
        self.method = method
        self.decay = decay
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.sink_size = sink_size
        self.device = device or torch.device("cpu")

        # Per-layer, per-head importance: list[Tensor(num_heads, max_pos)]
        # Lazily allocated on first update.
        self._scores: Optional[List[torch.Tensor]] = None
        self._update_count = 0

    def _ensure_allocated(self, seq_len: int) -> None:
        """Lazily allocate score tensors on first update."""
        if self._scores is not None:
            # Grow if needed
            current_len = self._scores[0].shape[-1]
            if seq_len > current_len:
                extra = seq_len - current_len
                for i in range(self.num_layers):
                    pad = torch.zeros(
                        self.num_heads, extra,
                        device=self.device, dtype=torch.float32,
                    )
                    self._scores[i] = torch.cat([self._scores[i], pad], dim=-1)
            return

        if self.num_layers is None or self.num_heads is None:
            raise RuntimeError(
                "num_layers and num_heads must be set before first update"
            )
        self._scores = [
            torch.zeros(self.num_heads, seq_len, device=self.device, dtype=torch.float32)
            for _ in range(self.num_layers)
        ]

    def update(
        self,
        attention_weights: torch.Tensor,
        layer_idx: int,
        position: int,
        thinking_start: int = 0,
        thinking_end: Optional[int] = None,
    ) -> None:
        """Update importance scores with attention from token at ``position``.

        Parameters
        ----------
        attention_weights : Tensor
            Shape (batch, num_heads, 1, seq_len) or (num_heads, seq_len) --
            the attention distribution from the newly generated token over all
            previous positions.  Only the thinking-range slice is used.
        layer_idx : int
            Transformer layer index.
        position : int
            Sequence position of the newly generated token.
        thinking_start : int
            Start of thinking range (inclusive).
        thinking_end : int or None
            End of thinking range (exclusive). None means up to ``position``.
        """
        # Normalise shape to (num_heads, seq_len)
        w = attention_weights
        if w.dim() == 4:
            w = w.squeeze(0).squeeze(-2)  # (num_heads, seq_len)
        elif w.dim() == 3:
            w = w.squeeze(-2)

        seq_len = w.shape[-1]
        self._ensure_allocated(seq_len)

        end = thinking_end if thinking_end is not None else position
        end = min(end, seq_len)
        start = max(thinking_start, 0)

        if start >= end:
            return

        scores_layer = self._scores[layer_idx]

        # Slice to thinking region
        attn_slice = w[:, start:end]  # (num_q_heads, thinking_len)
        # Handle GQA: average over query head groups if more heads than expected
        if attn_slice.shape[0] > self.num_heads:
            group_size = attn_slice.shape[0] // self.num_heads
            attn_slice = attn_slice.view(self.num_heads, group_size, -1).mean(dim=1)
        # Move to scores device (model may span multiple GPUs)
        attn_slice = attn_slice.to(scores_layer.device)
        current_slice = scores_layer[:, start:end]

        if self.method == "ema":
            scores_layer[:, start:end] = (
                self.decay * current_slice + (1.0 - self.decay) * attn_slice
            )
        elif self.method == "cumulative":
            scores_layer[:, start:end] = current_slice + attn_slice
        elif self.method == "sink":
            # Cumulative scoring, but first/last sink_size tokens get a bonus
            scores_layer[:, start:end] = current_slice + attn_slice
            # Sink bonus applied at retrieval time, not here

        self._update_count += 1

    def get_scores(
        self,
        thinking_start: int = 0,
        thinking_end: Optional[int] = None,
    ) -> torch.Tensor:
        """Return importance scores averaged across layers and heads.

        Returns
        -------
        Tensor of shape (thinking_length,) with per-token importance.
        """
        if self._scores is None:
            return torch.zeros(0, device=self.device)

        total_len = self._scores[0].shape[-1]
        end = thinking_end if thinking_end is not None else total_len
        end = min(end, total_len)
        start = max(thinking_start, 0)

        if start >= end:
            return torch.zeros(0, device=self.device)

        # Average across layers and heads
        stacked = torch.stack(
            [s[:, start:end] for s in self._scores], dim=0
        )  # (num_layers, num_heads, thinking_len)
        avg = stacked.mean(dim=(0, 1))  # (thinking_len,)

        # Apply sink bonus if method is "sink"
        if self.method == "sink":
            length = avg.shape[0]
            if length > 0:
                sink_n = min(self.sink_size, length)
                # Boost first and last sink_n tokens
                max_score = avg.max().item() if avg.numel() > 0 else 1.0
                bonus = max_score * 2.0
                avg[:sink_n] = avg[:sink_n] + bonus
                if length > sink_n:
                    avg[-sink_n:] = avg[-sink_n:] + bonus

        return avg

    def reset(self) -> None:
        """Clear all accumulated scores."""
        self._scores = None
        self._update_count = 0


# ---------------------------------------------------------------------------
# Adaptive bit allocation
# ---------------------------------------------------------------------------

class AdaptiveBitAllocator:
    """Assigns per-token bit-widths given importance scores and compression target.

    Algorithm:
      1. Compute bit budget from target compression ratio.
      2. Sort tokens by importance (ascending).
      3. Below eviction_threshold percentile: assign 0 bits (evict).
      4. Greedily assign bits from low to high importance, starting at min_bits,
         stepping up through SUPPORTED_BIT_WIDTHS, keeping within budget.
      5. Highest-importance tokens get FP16 (16 bits).

    The objective is to approximately solve:
        minimize  sum_t  quant_error(t, bits[t]) * importance[t]
        s.t.      sum_t  bits[t] <= budget
    via the greedy heuristic that low-importance tokens tolerate more error.
    """

    def allocate(
        self,
        importance_scores: torch.Tensor,
        target_ratio: float,
        min_bits: int = 2,
        max_bits: int = 16,
        eviction_threshold: float = 0.01,
    ) -> torch.Tensor:
        """Compute per-token bit assignments.

        Parameters
        ----------
        importance_scores : Tensor
            Shape (N,) with per-token importance.  Higher = more important.
        target_ratio : float
            Desired compression ratio (e.g., 4.0 means 4x compression).
        min_bits : int
            Minimum bits for any retained token.
        max_bits : int
            Maximum bits (FP16 = 16).
        eviction_threshold : float
            Fraction of tokens (by importance rank) to evict entirely.

        Returns
        -------
        Tensor of shape (N,) with int bit assignments (0 = evicted).
        """
        N = importance_scores.shape[0]
        if N == 0:
            return torch.zeros(0, dtype=torch.int32, device=importance_scores.device)

        device = importance_scores.device

        # Total bit budget: FP16 baseline is N * 16 bits per element.
        # At target_ratio compression, we have N * 16 / target_ratio bits.
        budget = N * FP16_BITS / target_ratio

        # Sort by importance (ascending -- least important first)
        sorted_indices = torch.argsort(importance_scores)
        assignments = torch.full((N,), min_bits, dtype=torch.int32, device=device)

        # Step 1: evict bottom tokens
        n_evict = int(N * eviction_threshold)
        if n_evict > 0:
            evict_idx = sorted_indices[:n_evict]
            assignments[evict_idx] = EVICTED_BITS

        # Step 2: assign bits to remaining tokens
        # Available bit-widths (sorted ascending), excluding evicted
        available_bw = sorted(
            bw for bw in SUPPORTED_BIT_WIDTHS if min_bits <= bw <= max_bits
        )
        if max_bits not in available_bw:
            available_bw.append(max_bits)
        available_bw = sorted(set(available_bw))

        # Start with min_bits for all non-evicted tokens.
        # Remaining budget after current assignment.
        current_cost = assignments.sum().item()

        if current_cost > budget:
            # Budget is too tight even for min_bits -- evict more
            # Evict from least important until budget is met
            remaining = sorted_indices[n_evict:]
            for idx_pos in range(len(remaining)):
                if current_cost <= budget:
                    break
                tok_idx = remaining[idx_pos].item()
                current_cost -= assignments[tok_idx].item()
                assignments[tok_idx] = EVICTED_BITS

        # Step 3: upgrade from highest importance downward
        # Iterate tokens from most important to least, upgrade bits if budget allows
        non_evicted = sorted_indices[assignments[sorted_indices] > 0]
        # Reverse so we process most-important first
        upgrade_order = non_evicted.flip(0)

        remaining_budget = budget - assignments.sum().item()

        for bw in reversed(available_bw):
            if bw <= min_bits:
                continue
            for tok_idx in upgrade_order:
                tok_idx_val = tok_idx.item()
                current_bw = assignments[tok_idx_val].item()
                if current_bw >= bw:
                    continue
                upgrade_cost = bw - current_bw
                if upgrade_cost <= remaining_budget:
                    assignments[tok_idx_val] = bw
                    remaining_budget -= upgrade_cost
                else:
                    break  # no budget for this bit-width tier

        return assignments

    def summarize(self, assignments: torch.Tensor) -> Dict[str, Any]:
        """Summary statistics for a bit assignment vector."""
        N = assignments.shape[0]
        if N == 0:
            return {"total_tokens": 0}

        unique, counts = torch.unique(assignments, return_counts=True)
        dist = {int(u): int(c) for u, c in zip(unique.tolist(), counts.tolist())}
        avg_bits = assignments.float().mean().item()
        effective_ratio = FP16_BITS / avg_bits if avg_bits > 0 else float("inf")

        return {
            "total_tokens": N,
            "bit_distribution": dist,
            "avg_bits": round(avg_bits, 2),
            "effective_ratio": round(effective_ratio, 2),
            "n_evicted": int((assignments == 0).sum().item()),
            "n_fp16": int((assignments == FP16_BITS).sum().item()),
        }


# ---------------------------------------------------------------------------
# Compressed thinking cache
# ---------------------------------------------------------------------------

@dataclass
class _QuantizedGroup:
    """One group of tokens sharing the same bit-width in a single layer."""
    bits: int
    positions: torch.Tensor        # original sequence positions
    k_codes: torch.Tensor           # quantized K codes
    k_scales: torch.Tensor
    k_zeros: torch.Tensor
    v_codes: torch.Tensor           # quantized V codes
    v_scales: torch.Tensor
    v_zeros: torch.Tensor


@dataclass
class _FP16Group:
    """FP16 tokens in a single layer (no quantization)."""
    positions: torch.Tensor
    keys: torch.Tensor              # (num_kv_heads, n_tokens, head_dim)
    values: torch.Tensor


@dataclass
class _LayerStorage:
    """Mixed-precision storage for one transformer layer."""
    quantized_groups: List[_QuantizedGroup] = field(default_factory=list)
    fp16_group: Optional[_FP16Group] = None
    evicted_positions: Optional[torch.Tensor] = None


class CompressedThinkingCache:
    """Stores thinking KV cache with mixed precision.

    Groups tokens by bit-width for efficient storage:
      - FP16 group: conclusion tokens (kept as-is)
      - Quantized groups: 2/3/4/8-bit (Hadamard rotation + scalar quantization)
      - Evicted group: not stored, masked in attention

    Reconstruction for attention:
      1. Dequantize each quantized group
      2. Scatter back to original positions
      3. Set evicted positions to zero (masked via attention_mask)
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 128,
        seed: int = 42,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size

        self.rotation = HadamardRotation(head_dim, seed=seed)
        self._quantizers: Dict[int, ScalarQuantizer] = {}

        self._layers: List[_LayerStorage] = [
            _LayerStorage() for _ in range(num_layers)
        ]
        self._total_positions = 0
        self._thinking_start = 0
        self._thinking_end = 0

    def _get_quantizer(self, bits: int) -> ScalarQuantizer:
        """Lazily create and cache a ScalarQuantizer for each bit-width."""
        if bits not in self._quantizers:
            self._quantizers[bits] = ScalarQuantizer(
                bits=bits, block_size=self.block_size
            )
        return self._quantizers[bits]

    def compress(
        self,
        kv_cache: Any,
        bit_assignments: torch.Tensor,
        thinking_start: int,
        thinking_end: int,
    ) -> None:
        """Compress thinking portion of KV cache according to bit assignments.

        Parameters
        ----------
        kv_cache : DynamicCache or similar
            The HuggingFace KV cache object.
        bit_assignments : Tensor
            Shape (thinking_length,) with per-token bit-width.
            0 = evicted, 2/3/4/8 = quantized, 16 = FP16.
        thinking_start : int
            Start of thinking range in the sequence (inclusive).
        thinking_end : int
            End of thinking range in the sequence (exclusive).
        """
        self._thinking_start = thinking_start
        self._thinking_end = thinking_end
        self._total_positions = thinking_end - thinking_start
        think_len = thinking_end - thinking_start

        # Handle off-by-one: scorer may have one fewer entry than thinking range
        # (last thinking token hasn't been attended to by subsequent tokens yet)
        if bit_assignments.shape[0] < think_len:
            # Pad with max_bits (preserve the last unseen tokens at FP16)
            pad = torch.full((think_len - bit_assignments.shape[0],), 16,
                             dtype=bit_assignments.dtype, device=bit_assignments.device)
            bit_assignments = torch.cat([bit_assignments, pad])
        elif bit_assignments.shape[0] > think_len:
            bit_assignments = bit_assignments[:think_len]

        num_layers = get_num_kv_layers(kv_cache)
        if num_layers < self.num_layers:
            logger.warning(
                "KV cache has %d layers but expected %d", num_layers, self.num_layers
            )

        # Find unique bit-widths and group positions
        unique_bits = torch.unique(bit_assignments).tolist()
        unique_bits = [int(b) for b in unique_bits]

        for layer_idx in range(min(num_layers, self.num_layers)):
            key, value = get_kv_tensors(kv_cache, layer_idx)
            # key, value shape: (batch, num_kv_heads, seq_len, head_dim)
            # Extract thinking range
            k_think = key[:, :, thinking_start:thinking_end, :]
            v_think = value[:, :, thinking_start:thinking_end, :]

            storage = _LayerStorage()

            for bw in unique_bits:
                mask = bit_assignments == bw
                positions = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if positions.numel() == 0:
                    continue

                if bw == EVICTED_BITS:
                    storage.evicted_positions = positions.to(k_think.device)
                    continue

                # Gather tokens at these positions
                # k_think: (batch, heads, think_len, dim)
                positions = positions.to(k_think.device)
                k_subset = k_think[:, :, positions, :]
                v_subset = v_think[:, :, positions, :]

                if bw == FP16_BITS:
                    storage.fp16_group = _FP16Group(
                        positions=positions,
                        keys=k_subset.clone(),
                        values=v_subset.clone(),
                    )
                else:
                    # Hadamard rotate then quantize
                    quantizer = self._get_quantizer(bw)

                    # Reshape for rotation: merge batch and heads
                    orig_shape = k_subset.shape
                    k_flat = k_subset.reshape(-1, positions.numel(), self.head_dim)
                    v_flat = v_subset.reshape(-1, positions.numel(), self.head_dim)

                    k_rot = self.rotation.rotate(k_flat.float())
                    v_rot = self.rotation.rotate(v_flat.float())

                    k_codes, k_scales, k_zeros = quantizer.quantize(k_rot)
                    v_codes, v_scales, v_zeros = quantizer.quantize(v_rot)

                    storage.quantized_groups.append(_QuantizedGroup(
                        bits=bw,
                        positions=positions,
                        k_codes=k_codes,
                        k_scales=k_scales,
                        k_zeros=k_zeros,
                        v_codes=v_codes,
                        v_scales=v_scales,
                        v_zeros=v_zeros,
                    ))

            self._layers[layer_idx] = storage

    def get_kv(
        self,
        layer_idx: int,
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct full K, V for the thinking range.

        Returns tensors of shape (1, num_kv_heads, thinking_length, head_dim)
        with evicted positions set to zero.
        """
        storage = self._layers[layer_idx]
        think_len = self._total_positions
        device = self._infer_device(storage)

        k_full = torch.zeros(
            1, self.num_kv_heads, think_len, self.head_dim,
            dtype=dtype, device=device,
        )
        v_full = torch.zeros_like(k_full)

        # Scatter FP16 group
        if storage.fp16_group is not None:
            g = storage.fp16_group
            k_full[:, :, g.positions, :] = g.keys.to(dtype)
            v_full[:, :, g.positions, :] = g.values.to(dtype)

        # Scatter quantized groups
        for qg in storage.quantized_groups:
            quantizer = self._get_quantizer(qg.bits)

            k_deq = quantizer.dequantize(qg.k_codes, qg.k_scales, qg.k_zeros)
            v_deq = quantizer.dequantize(qg.v_codes, qg.v_scales, qg.v_zeros)

            # Inverse Hadamard
            k_orig = self.rotation.inverse_rotate(k_deq)
            v_orig = self.rotation.inverse_rotate(v_deq)

            # Reshape back to (1, heads, n_tokens, dim)
            n_tok = qg.positions.numel()
            k_orig = k_orig.reshape(1, self.num_kv_heads, n_tok, self.head_dim)
            v_orig = v_orig.reshape(1, self.num_kv_heads, n_tok, self.head_dim)

            k_full[:, :, qg.positions, :] = k_orig.to(dtype)
            v_full[:, :, qg.positions, :] = v_orig.to(dtype)

        # Evicted positions stay as zeros
        return k_full, v_full

    def get_eviction_mask(
        self,
        layer_idx: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return a boolean mask where True = evicted (should be masked in attention).

        Shape: (thinking_length,)
        """
        storage = self._layers[layer_idx]
        think_len = self._total_positions
        dev = device or self._infer_device(storage)
        mask = torch.zeros(think_len, dtype=torch.bool, device=dev)
        if storage.evicted_positions is not None:
            mask[storage.evicted_positions] = True
        return mask

    def memory_bytes(self) -> int:
        """Total memory usage of compressed thinking cache."""
        total = 0
        for storage in self._layers:
            if storage.fp16_group is not None:
                g = storage.fp16_group
                total += g.keys.numel() * 2 + g.values.numel() * 2
            for qg in storage.quantized_groups:
                # Codes are uint8 (1 byte each)
                total += qg.k_codes.numel() + qg.v_codes.numel()
                # Scales and zeros are fp16 (2 bytes each)
                total += (
                    qg.k_scales.numel() + qg.k_zeros.numel()
                    + qg.v_scales.numel() + qg.v_zeros.numel()
                ) * 2
        return total

    def full_precision_bytes(self) -> int:
        """Equivalent FP16 memory for the entire thinking range."""
        return (
            self.num_layers
            * 2  # K and V
            * self.num_kv_heads
            * self._total_positions
            * self.head_dim
            * 2  # FP16 = 2 bytes
        )

    def compression_ratio(self) -> float:
        """Effective compression ratio of the thinking cache."""
        compressed = self.memory_bytes()
        if compressed == 0:
            return 0.0
        return self.full_precision_bytes() / compressed

    def _infer_device(self, storage: _LayerStorage) -> torch.device:
        """Infer device from stored tensors."""
        if storage.fp16_group is not None:
            return storage.fp16_group.keys.device
        for qg in storage.quantized_groups:
            return qg.k_codes.device
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Attention hook management
# ---------------------------------------------------------------------------

class _AttentionHookManager:
    """Installs and manages forward hooks on attention layers to capture weights.

    Supports two strategies:
      1. model.generate() with output_attentions=True (simpler but uses more memory)
      2. Forward hooks on attention modules (lower overhead, works with any generate)

    We use strategy 2 (hooks) for production and strategy 1 for analysis/debugging.
    """

    def __init__(self, model: Any):
        self.model = model
        self._hooks: List[Any] = []
        self._captured: Dict[int, torch.Tensor] = {}  # layer_idx -> attn weights

    def install(self) -> None:
        """Install forward hooks on all attention layers."""
        self._remove_all()
        self._captured.clear()

        # Find attention modules. Support common HuggingFace naming patterns.
        layers = self._find_attention_layers()
        for layer_idx, attn_module in enumerate(layers):
            hook = attn_module.register_forward_hook(
                self._make_hook(layer_idx)
            )
            self._hooks.append(hook)

        logger.debug("Installed attention hooks on %d layers", len(layers))

    def _make_hook(self, layer_idx: int):
        """Create a forward hook closure for a specific layer."""
        def hook_fn(module, input, output):
            # HuggingFace attention modules return (attn_output, attn_weights, ...)
            # when output_attentions=True, or just attn_output otherwise.
            # With our hook, we look for the attention weights in the output tuple.
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]
                if attn_weights is not None:
                    # Store only the last query position to save memory
                    # attn_weights shape: (batch, num_heads, q_len, kv_len)
                    self._captured[layer_idx] = attn_weights[:, :, -1:, :].detach()
        return hook_fn

    def get_captured(self) -> Dict[int, torch.Tensor]:
        """Return captured attention weights and clear the buffer."""
        result = dict(self._captured)
        self._captured.clear()
        return result

    def remove(self) -> None:
        """Remove all hooks."""
        self._remove_all()

    def _remove_all(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()

    def _find_attention_layers(self) -> List[Any]:
        """Locate attention sub-modules in common HuggingFace architectures."""
        # Try common attribute paths
        candidates = []

        # Qwen3, Llama, Mistral pattern: model.layers[i].self_attn
        base = getattr(self.model, "model", self.model)
        layers_attr = getattr(base, "layers", None)
        if layers_attr is not None:
            for layer in layers_attr:
                attn = getattr(layer, "self_attn", None)
                if attn is not None:
                    candidates.append(attn)
            if candidates:
                return candidates

        # GPT-NeoX pattern: model.gpt_neox.layers[i].attention
        gpt = getattr(base, "gpt_neox", None)
        if gpt is not None:
            for layer in gpt.layers:
                attn = getattr(layer, "attention", None)
                if attn is not None:
                    candidates.append(attn)
            if candidates:
                return candidates

        # Fallback: search for any module with "attn" or "attention" in its name
        for name, module in self.model.named_modules():
            if name.endswith(".self_attn") or name.endswith(".attention"):
                candidates.append(module)

        if not candidates:
            raise RuntimeError(
                "Could not find attention layers in model. "
                "Ensure the model follows HuggingFace conventions."
            )

        return candidates

    def __del__(self):
        self._remove_all()


# ---------------------------------------------------------------------------
# Think boundary detection
# ---------------------------------------------------------------------------

def _find_think_tokens(tokenizer: Any) -> Tuple[Optional[int], Optional[int]]:
    """Find <think> and </think> token IDs in the tokenizer vocabulary.

    Returns (think_start_id, think_end_id). Either may be None if not found.
    """
    # Try direct encoding first (works for Qwen3, DeepSeek-R1)
    start_id = None
    end_id = None

    for token_str in ("<think>", "<|think|>"):
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 1:
            start_id = ids[0]
            break

    for token_str in ("</think>", "<|/think|>"):
        ids = tokenizer.encode(token_str, add_special_tokens=False)
        if len(ids) == 1:
            end_id = ids[0]
            break

    # Fallback: search vocabulary directly
    if start_id is None or end_id is None:
        vocab = tokenizer.get_vocab()
        for token, tid in vocab.items():
            if start_id is None and "think" in token.lower() and "<" in token:
                if "/" not in token:
                    start_id = tid
            if end_id is None and "think" in token.lower() and "/" in token:
                end_id = tid

    if start_id is not None and end_id is not None:
        logger.info(
            "Found think tokens: start=%d (%s), end=%d (%s)",
            start_id, tokenizer.decode([start_id]),
            end_id, tokenizer.decode([end_id]),
        )
    else:
        logger.warning(
            "Could not find think boundary tokens (start=%s, end=%s). "
            "Heuristic boundary detection will be used.",
            start_id, end_id,
        )

    return start_id, end_id


# ---------------------------------------------------------------------------
# Main compressor
# ---------------------------------------------------------------------------

@dataclass
class CompressionStats:
    """Statistics from a ThinkCompress generation run."""
    total_tokens: int = 0
    thinking_tokens: int = 0
    answer_tokens: int = 0
    thinking_compression_ratio: float = 1.0
    thinking_memory_bytes: int = 0
    thinking_fp16_bytes: int = 0
    avg_bits_per_thinking_token: float = 16.0
    n_evicted: int = 0
    n_fp16_kept: int = 0
    importance_gini: float = 0.0
    importance_entropy: float = 0.0
    compression_time_ms: float = 0.0
    bit_distribution: Dict[int, int] = field(default_factory=dict)


class ThinkCompressor:
    """Adaptive KV cache compression for chain-of-thought reasoning.

    Integrates with HuggingFace model generation loop. During generation:
      1. Detects <think> and </think> boundaries in generated tokens.
      2. During thinking phase: tracks per-token importance via attention EMA.
      3. At thinking->answer transition: assigns per-token bit-widths.
      4. During answer phase: serves compressed thinking KV + full-precision answer KV.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace causal LM (e.g., Qwen3ForCausalLM).
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer.
    target_ratio : float
        Target compression ratio for thinking KV (e.g., 4.0 = 4x compression).
    importance_method : str
        Scoring method: "ema", "cumulative", or "sink".
    ema_decay : float
        EMA decay factor for importance scoring.
    min_bits : int
        Minimum bits for any retained thinking token.
    max_bits : int
        Maximum bits (16 = FP16, no compression).
    eviction_threshold : float
        Fraction of lowest-importance tokens to evict entirely.
    block_size : int
        Quantization block size for ScalarQuantizer.
    seed : int
        Random seed for Hadamard rotation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        target_ratio: float = 4.0,
        importance_method: str = "ema",
        ema_decay: float = 0.95,
        min_bits: int = 2,
        max_bits: int = 16,
        eviction_threshold: float = 0.01,
        block_size: int = 128,
        seed: int = 42,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.target_ratio = target_ratio
        self.importance_method = importance_method
        self.ema_decay = ema_decay
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.eviction_threshold = eviction_threshold
        self.block_size = block_size
        self.seed = seed

        # Detect model architecture
        config = getattr(model, "config", None)
        self._num_layers = getattr(config, "num_hidden_layers", 32)
        self._num_kv_heads = getattr(
            config, "num_key_value_heads",
            getattr(config, "num_attention_heads", 32),
        )
        self._num_attn_heads = getattr(config, "num_attention_heads", 32)
        self._head_dim = getattr(
            config, "head_dim",
            getattr(config, "hidden_size", 4096) // self._num_attn_heads,
        )

        # Think boundary token IDs
        self._think_start_id, self._think_end_id = _find_think_tokens(tokenizer)

        # Sub-components
        self._scorer = ImportanceScorer(
            method=importance_method,
            decay=ema_decay,
            num_layers=self._num_layers,
            num_heads=self._num_kv_heads,
            device=_model_device(model),
        )
        self._allocator = AdaptiveBitAllocator()
        self._hook_manager = _AttentionHookManager(model)

        # Phase tracking
        self._phase = PhaseState()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        do_sample: bool = True,
        compress_at_transition: bool = True,
        **generate_kwargs,
    ) -> Dict[str, Any]:
        """Generate text with adaptive thinking KV compression.

        Parameters
        ----------
        input_ids : Tensor
            Shape (1, seq_len) -- prompt token IDs.
        max_new_tokens : int
            Maximum tokens to generate.
        temperature : float
            Sampling temperature.
        top_p : float
            Nucleus sampling threshold.
        do_sample : bool
            Whether to sample (True) or use greedy decoding (False).
        compress_at_transition : bool
            If True, compress thinking KV at the think->answer transition.
            If False, compress after generation completes (for analysis).

        Returns
        -------
        Dict with keys:
          - "output_ids": full output token IDs (including prompt)
          - "generated_text": decoded text
          - "stats": CompressionStats dataclass
          - "importance_scores": Tensor of per-thinking-token importance
          - "bit_assignments": Tensor of per-thinking-token bit-widths
        """
        device = input_ids.device
        self._phase = PhaseState()
        self._scorer.reset()

        prompt_len = input_ids.shape[1]
        generated_ids = input_ids.clone()

        # Install attention hooks
        self._hook_manager.install()

        compressed_cache: Optional[CompressedThinkingCache] = None
        kv_cache = None
        stats = CompressionStats()
        importance_scores = None
        bit_assignments = None

        try:
            for step in range(max_new_tokens):
                # Forward pass with KV cache
                model_kwargs = {
                    "input_ids": generated_ids[:, -1:] if kv_cache is not None else generated_ids,
                    "past_key_values": kv_cache,
                    "use_cache": True,
                    "output_attentions": True,
                }
                # Ensure eager attention for output_attentions support
                if step == 0:
                    try:
                        self.model.set_attn_implementation("eager")
                    except (AttributeError, Exception):
                        pass  # Not all models support this
                outputs = self.model(**model_kwargs)

                logits = outputs.logits[:, -1, :]
                kv_cache = outputs.past_key_values

                # Sample next token
                next_token = self._sample_token(
                    logits, temperature=temperature, top_p=top_p,
                    do_sample=do_sample,
                )
                # Ensure next_token is 2D [1, 1] for concatenation
                nt = next_token.view(1, 1) if next_token.dim() == 0 else next_token.unsqueeze(0) if next_token.dim() == 1 else next_token
                if nt.dim() == 1:
                    nt = nt.unsqueeze(0)
                generated_ids = torch.cat([generated_ids, nt], dim=1)
                current_pos = generated_ids.shape[1] - 1
                next_token_id = next_token.item()

                # Update phase
                old_phase = self._phase.phase
                self._update_phase(next_token_id, current_pos)
                self._phase.total_generated += 1

                # Update importance scores during thinking phase
                if self._phase.phase == GenerationPhase.THINKING and outputs.attentions is not None:
                    for layer_idx, attn_w in enumerate(outputs.attentions):
                        # attn_w: (batch, num_heads, q_len, kv_len)
                        # Extract attention from last generated token to all previous
                        w = attn_w[0, :, -1, :]  # (num_heads, kv_len)
                        self._scorer.update(
                            w,
                            layer_idx=layer_idx,
                            position=current_pos,
                            thinking_start=self._phase.think_start_pos,
                        )

                # Compress at think->answer transition
                if (
                    compress_at_transition
                    and old_phase == GenerationPhase.THINKING
                    and self._phase.phase == GenerationPhase.ANSWER
                    and kv_cache is not None
                    and self.target_ratio > 1.01  # Skip compression at ratio ~1.0
                ):
                    t0 = time.monotonic()
                    importance_scores, bit_assignments, compressed_cache = (
                        self._compress_thinking_kv(kv_cache)
                    )
                    stats.compression_time_ms = (time.monotonic() - t0) * 1000

                # EOS check
                if next_token_id == self.tokenizer.eos_token_id:
                    break

            # If compression was not done at transition (no </think> found, or
            # compress_at_transition=False), compress now
            if compressed_cache is None and self._phase.think_start_pos >= 0 and self.target_ratio > 1.01:
                if self._phase.think_end_pos < 0:
                    self._phase.think_end_pos = generated_ids.shape[1]
                t0 = time.monotonic()
                importance_scores, bit_assignments, compressed_cache = (
                    self._compress_thinking_kv(kv_cache)
                )
                stats.compression_time_ms = (time.monotonic() - t0) * 1000

        finally:
            self._hook_manager.remove()

        # Populate stats
        stats.total_tokens = generated_ids.shape[1]
        stats.thinking_tokens = self._phase.thinking_length
        stats.answer_tokens = max(
            0, self._phase.total_generated - stats.thinking_tokens
        )

        if compressed_cache is not None:
            stats.thinking_compression_ratio = compressed_cache.compression_ratio()
            stats.thinking_memory_bytes = compressed_cache.memory_bytes()
            stats.thinking_fp16_bytes = compressed_cache.full_precision_bytes()

        # Always extract importance scores from scorer (even without compression)
        if importance_scores is None and self._phase.think_start_pos >= 0:
            think_start = self._phase.think_start_pos
            think_end = self._phase.think_end_pos if self._phase.think_end_pos > 0 else generated_ids.shape[1]
            importance_scores = self._scorer.get_scores(
                thinking_start=think_start, thinking_end=think_end,
            )
        if importance_scores is not None and importance_scores.numel() > 0:
            stats.importance_gini = _gini_coefficient(importance_scores)
            stats.importance_entropy = _entropy(importance_scores)

        if bit_assignments is not None:
            summary = self._allocator.summarize(bit_assignments)
            stats.avg_bits_per_thinking_token = summary["avg_bits"]
            stats.n_evicted = summary["n_evicted"]
            stats.n_fp16_kept = summary["n_fp16"]
            stats.bit_distribution = summary["bit_distribution"]

        # Decode output
        generated_text = self.tokenizer.decode(
            generated_ids[0, prompt_len:], skip_special_tokens=False,
        )

        return {
            "output_ids": generated_ids,
            "generated_text": generated_text,
            "stats": stats,
            "importance_scores": importance_scores,
            "bit_assignments": bit_assignments,
        }

    def _update_phase(self, token_id: int, position: int) -> None:
        """Update generation phase based on the newly generated token."""
        if token_id == self._think_start_id and self._phase.phase != GenerationPhase.ANSWER:
            self._phase.phase = GenerationPhase.THINKING
            self._phase.think_start_pos = position
        elif token_id == self._think_end_id and self._phase.phase == GenerationPhase.THINKING:
            self._phase.phase = GenerationPhase.ANSWER
            self._phase.think_end_pos = position

    def _compress_thinking_kv(
        self,
        kv_cache: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, CompressedThinkingCache]:
        """Score importance, allocate bits, compress thinking KV."""
        think_start = self._phase.think_start_pos
        think_end = self._phase.think_end_pos
        if think_end < 0:
            think_end = think_start + self._phase.thinking_length

        # Get importance scores
        importance = self._scorer.get_scores(
            thinking_start=think_start,
            thinking_end=think_end,
        )

        # Allocate bits
        assignments = self._allocator.allocate(
            importance,
            target_ratio=self.target_ratio,
            min_bits=self.min_bits,
            max_bits=self.max_bits,
            eviction_threshold=self.eviction_threshold,
        )

        # Compress
        cache = CompressedThinkingCache(
            num_layers=self._num_layers,
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            block_size=self.block_size,
            seed=self.seed,
        )
        cache.compress(kv_cache, assignments, think_start, think_end)

        logger.info(
            "Compressed thinking KV: %d tokens, ratio=%.1fx, memory %d -> %d bytes",
            importance.shape[0],
            cache.compression_ratio(),
            cache.full_precision_bytes(),
            cache.memory_bytes(),
        )

        return importance, assignments, cache

    @staticmethod
    def _sample_token(
        logits: torch.Tensor,
        temperature: float = 0.6,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """Sample a single token from logits with temperature and nucleus sampling."""
        if not do_sample:
            return logits.argmax(dim=-1).squeeze()

        logits = logits / max(temperature, 1e-8)

        # Top-p (nucleus) filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens above the threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back
        filtered_logits = torch.zeros_like(logits)
        filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

        probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(probs.squeeze(0), num_samples=1).squeeze()


# ---------------------------------------------------------------------------
# Utility: statistical measures for importance distributions
# ---------------------------------------------------------------------------

def _gini_coefficient(scores: torch.Tensor) -> float:
    """Gini coefficient of an importance score distribution.

    0 = perfectly uniform, 1 = maximally concentrated. High Gini means
    importance is concentrated in a few tokens -- good for compression.
    """
    if scores.numel() == 0:
        return 0.0
    s = scores.float().abs()
    n = s.numel()
    if n <= 1:
        return 0.0
    sorted_s, _ = torch.sort(s)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=s.device)
    return float(
        (2.0 * (index * sorted_s).sum() / (n * sorted_s.sum()) - (n + 1) / n)
    ) if sorted_s.sum() > 0 else 0.0


def _entropy(scores: torch.Tensor) -> float:
    """Shannon entropy of the normalized importance distribution (in nats)."""
    if scores.numel() == 0:
        return 0.0
    p = scores.float().abs()
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    # Avoid log(0)
    p = p.clamp(min=1e-12)
    return float(-(p * p.log()).sum())


def _model_device(model: Any) -> torch.device:
    """Infer the device a model resides on."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


# ---------------------------------------------------------------------------
# Analysis utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def analyze_thinking_attention(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
) -> Dict[str, Any]:
    """Generate CoT and analyze attention patterns without compression.

    Runs generation with importance scoring but no compression. Returns
    detailed per-token importance analysis for understanding attention
    structure in thinking sequences.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace causal LM with thinking capability.
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer.
    prompt : str
        Input prompt (will be formatted for chat if needed).
    max_new_tokens : int
        Maximum tokens to generate.
    temperature : float
        Sampling temperature.

    Returns
    -------
    Dict with keys:
      - "generated_text": full generated text
      - "thinking_text": extracted thinking content
      - "answer_text": extracted answer content
      - "importance_scores": Tensor of per-thinking-token importance
      - "thinking_length": number of thinking tokens
      - "answer_length": number of answer tokens
      - "token_ratio": thinking / answer ratio
      - "gini_coefficient": importance distribution Gini (higher = more skewed)
      - "entropy": importance distribution entropy
      - "top_k_pct": fraction of total importance in top-K% of tokens
      - "per_layer_gini": list of per-layer Gini coefficients
    """
    compressor = ThinkCompressor(
        model, tokenizer,
        target_ratio=1.0,  # no compression, just analysis
        importance_method="cumulative",
    )

    result = compressor.generate(
        input_ids=_encode_prompt(tokenizer, prompt),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        compress_at_transition=False,
    )

    generated = result["generated_text"]
    importance = result["importance_scores"]
    stats = result["stats"]

    # Extract thinking and answer text
    thinking_text = ""
    answer_text = generated
    if "<think>" in generated and "</think>" in generated:
        parts = generated.split("</think>", 1)
        thinking_text = parts[0].replace("<think>", "").strip()
        answer_text = parts[1].strip() if len(parts) > 1 else ""

    # Compute per-layer Gini coefficients
    per_layer_gini = []
    if compressor._scorer._scores is not None:
        think_start = compressor._phase.think_start_pos
        think_end = compressor._phase.think_end_pos
        if think_end < 0:
            think_end = think_start + stats.thinking_tokens
        for layer_scores in compressor._scorer._scores:
            layer_avg = layer_scores[:, think_start:think_end].mean(dim=0)
            per_layer_gini.append(_gini_coefficient(layer_avg))

    # Compute top-K importance concentration
    top_k_pct = {}
    if importance is not None and importance.numel() > 0:
        sorted_imp, _ = torch.sort(importance, descending=True)
        total_imp = sorted_imp.sum()
        if total_imp > 0:
            for k_pct in (1, 5, 10, 20, 50):
                k = max(1, int(importance.numel() * k_pct / 100))
                top_k_pct[f"top_{k_pct}pct"] = float(sorted_imp[:k].sum() / total_imp)

    analysis = {
        "generated_text": generated,
        "thinking_text": thinking_text,
        "answer_text": answer_text,
        "importance_scores": importance,
        "thinking_length": stats.thinking_tokens,
        "answer_length": stats.answer_tokens,
        "token_ratio": (
            stats.thinking_tokens / max(stats.answer_tokens, 1)
        ),
        "gini_coefficient": stats.importance_gini,
        "entropy": stats.importance_entropy,
        "top_k_pct": top_k_pct,
        "per_layer_gini": per_layer_gini,
    }

    return analysis


@torch.no_grad()
def compare_compression_methods(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    methods: Optional[List[str]] = None,
    target_ratios: Optional[List[float]] = None,
    max_new_tokens: int = 2048,
) -> Dict[str, Any]:
    """Compare multiple compression strategies across prompts and ratios.

    Runs each prompt with each (method, ratio) combination and collects
    accuracy and compression statistics for side-by-side comparison.

    Parameters
    ----------
    model : PreTrainedModel
        HuggingFace causal LM.
    tokenizer : PreTrainedTokenizer
        Corresponding tokenizer.
    prompts : list of str
        Evaluation prompts.
    methods : list of str
        Importance scoring methods to compare. Default: ["ema", "cumulative", "sink"].
    target_ratios : list of float
        Compression ratios to evaluate. Default: [2.0, 4.0, 6.0, 8.0].
    max_new_tokens : int
        Max generation length per prompt.

    Returns
    -------
    Dict with structure:
      {
        "results": [
          {
            "prompt_idx": int,
            "method": str,
            "target_ratio": float,
            "actual_ratio": float,
            "thinking_tokens": int,
            "answer_tokens": int,
            "avg_bits": float,
            "n_evicted": int,
            "gini": float,
            "generated_text": str,
          },
          ...
        ],
        "summary": {
          (method, ratio): {
            "mean_actual_ratio": float,
            "mean_avg_bits": float,
            "mean_eviction_pct": float,
          }
        },
        "baseline": [
          {"prompt_idx": int, "generated_text": str, "thinking_tokens": int},
          ...
        ],
      }
    """
    if methods is None:
        methods = ["ema", "cumulative", "sink"]
    if target_ratios is None:
        target_ratios = [2.0, 4.0, 6.0, 8.0]

    # First, generate baselines (FP16, no compression) for each prompt
    baseline_results = []
    for pidx, prompt in enumerate(prompts):
        compressor = ThinkCompressor(
            model, tokenizer, target_ratio=1.0, importance_method="ema",
        )
        out = compressor.generate(
            input_ids=_encode_prompt(tokenizer, prompt),
            max_new_tokens=max_new_tokens,
            compress_at_transition=False,
        )
        baseline_results.append({
            "prompt_idx": pidx,
            "generated_text": out["generated_text"],
            "thinking_tokens": out["stats"].thinking_tokens,
            "answer_tokens": out["stats"].answer_tokens,
        })
        logger.info(
            "Baseline prompt %d: %d thinking, %d answer tokens",
            pidx, out["stats"].thinking_tokens, out["stats"].answer_tokens,
        )

    # Run all (method, ratio) combinations
    all_results = []
    for method in methods:
        for ratio in target_ratios:
            for pidx, prompt in enumerate(prompts):
                compressor = ThinkCompressor(
                    model, tokenizer,
                    target_ratio=ratio,
                    importance_method=method,
                )
                out = compressor.generate(
                    input_ids=_encode_prompt(tokenizer, prompt),
                    max_new_tokens=max_new_tokens,
                )
                s = out["stats"]
                entry = {
                    "prompt_idx": pidx,
                    "method": method,
                    "target_ratio": ratio,
                    "actual_ratio": s.thinking_compression_ratio,
                    "thinking_tokens": s.thinking_tokens,
                    "answer_tokens": s.answer_tokens,
                    "avg_bits": s.avg_bits_per_thinking_token,
                    "n_evicted": s.n_evicted,
                    "gini": s.importance_gini,
                    "compression_time_ms": s.compression_time_ms,
                    "generated_text": out["generated_text"],
                }
                all_results.append(entry)
                logger.info(
                    "method=%s ratio=%.1f prompt=%d: actual_ratio=%.2f avg_bits=%.1f",
                    method, ratio, pidx, s.thinking_compression_ratio,
                    s.avg_bits_per_thinking_token,
                )

    # Build summary aggregated across prompts
    summary = {}
    for method in methods:
        for ratio in target_ratios:
            subset = [
                r for r in all_results
                if r["method"] == method and r["target_ratio"] == ratio
            ]
            if not subset:
                continue
            n = len(subset)
            summary[(method, ratio)] = {
                "mean_actual_ratio": sum(r["actual_ratio"] for r in subset) / n,
                "mean_avg_bits": sum(r["avg_bits"] for r in subset) / n,
                "mean_eviction_pct": sum(
                    r["n_evicted"] / max(r["thinking_tokens"], 1) for r in subset
                ) / n * 100,
                "mean_compression_time_ms": sum(
                    r["compression_time_ms"] for r in subset
                ) / n,
            }

    return {
        "results": all_results,
        "summary": summary,
        "baseline": baseline_results,
    }


# ---------------------------------------------------------------------------
# Prompt encoding helper
# ---------------------------------------------------------------------------

def _encode_prompt(tokenizer: Any, prompt: str) -> torch.Tensor:
    """Encode a prompt string into input_ids tensor.

    If the tokenizer supports chat templates (Qwen3-style), uses
    apply_chat_template with enable_thinking. Otherwise falls back to
    plain tokenization.

    Returns shape (1, seq_len) on the model device.
    """
    # Try chat template first (Qwen3, DeepSeek-R1 pattern)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=True,
            )
            if isinstance(input_ids, torch.Tensor):
                return input_ids
        except (TypeError, ValueError):
            # enable_thinking not supported; fall through
            pass

    # Fallback: plain tokenization
    encoded = tokenizer(prompt, return_tensors="pt")
    return encoded["input_ids"]


# ---------------------------------------------------------------------------
# Mixed-precision TV bound for ThinkCompress
# ---------------------------------------------------------------------------

def compute_mixed_precision_tv_bound(
    importance_scores: torch.Tensor,
    bit_assignments: torch.Tensor,
    w_o_fnorm: float,
    range_kv: float,
    head_dim: int,
    block_size: int = 128,
    temperature: float = 1.0,
) -> float:
    """Compute theoretical TV bound for mixed-precision thinking KV.

    Extends the SpecQuant TV bound (Proposition 1) to heterogeneous bit-widths.
    The bound is a weighted sum of per-group quantization errors, where weights
    are the group's total importance mass.

    Parameters
    ----------
    importance_scores : Tensor
        Shape (N,) normalized importance per thinking token.
    bit_assignments : Tensor
        Shape (N,) bit-widths per token (0 = evicted).
    w_o_fnorm : float
        Frobenius norm of the output projection matrix.
    range_kv : float
        Range of K/V values (max - min).
    head_dim : int
        Attention head dimension.
    block_size : int
        Quantization block size.
    temperature : float
        Softmax temperature.

    Returns
    -------
    Upper bound on TV(p, p_tilde) for the mixed-precision verification.
    """
    if importance_scores.numel() == 0:
        return 0.0

    # Normalise importance to sum to 1
    imp = importance_scores.float().abs()
    total_imp = imp.sum()
    if total_imp <= 0:
        return 0.0
    imp = imp / total_imp

    sigma = 1.0 / (2 * math.sqrt(3))
    tv_total = 0.0

    unique_bits = torch.unique(bit_assignments).tolist()
    for bw in unique_bits:
        bw = int(bw)
        mask = bit_assignments == bw
        group_imp = imp[mask].sum().item()

        if bw == 0:
            # Evicted tokens contribute their full importance as error
            tv_total += group_imp
        elif bw >= FP16_BITS:
            # FP16 tokens: negligible quantization error
            continue
        else:
            # Quantized tokens: error scales as 1/2^bits
            quant_std = sigma * math.sqrt(head_dim) / (2**bw * math.sqrt(block_size))
            tv_total += group_imp * range_kv * quant_std

    tv_total *= w_o_fnorm / temperature
    return tv_total
