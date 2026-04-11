"""AcceptSpec: Acceptance-Preserving KV Cache Management for Speculative Decoding.

Core module implementing:
  1. AcceptSensitivityOracle — measures per-token acceptance sensitivity via perturbation
  2. AcceptPredictor — zero-overhead predictor using draft attention × value norm
  3. MixedPrecisionKV — per-token precision tags with differential compression

The key insight: in speculative decoding, KV cache importance should be measured by
contribution to verifier acceptance probability, not by attention weight or perplexity.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .turboquant_kv import HadamardRotation, ScalarQuantizer
from .utils import get_kv_tensors, set_kv_tensors, get_num_kv_layers, get_kv_layer_indices

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-token precision tags (stored as 2-bit codes)
TAG_EVICTED = 0   # token KV discarded
TAG_2BIT = 1      # aggressive compression
TAG_4BIT = 2      # moderate compression
TAG_FP16 = 3      # full precision (acceptance-critical)


@dataclass
class SensitivityResult:
    """Result of oracle acceptance sensitivity measurement for one verification step."""
    step_idx: int
    num_kv_tokens: int
    # Per-token sensitivity: S_accept(i) = |alpha_full - alpha_perturbed_i|
    sensitivities: torch.Tensor       # [num_kv_tokens]
    # Per-token attention importance (sum across heads/layers)
    attention_importance: torch.Tensor  # [num_kv_tokens]
    # Full acceptance count (baseline)
    alpha_full: float
    # Fraction of tokens that are "critical" (top-k by sensitivity)
    gini: float


@dataclass
class OracleStudyResult:
    """Aggregated oracle study result across multiple problems."""
    num_problems: int
    num_steps: int
    # Cumulative sensitivity curve: fraction of tokens retained vs fraction of sensitivity captured
    retained_fractions: List[float]
    sensitivity_captured: List[float]
    # Top-k coverage: what fraction of sensitivity is in top-k% tokens
    top10_coverage: float
    top20_coverage: float
    top50_coverage: float
    # Gini coefficient of sensitivity distribution
    mean_gini: float
    std_gini: float
    # Per-layer sensitivity heatmap: [num_layers, num_bins]
    layer_sensitivity_heatmap: Optional[torch.Tensor] = None
    # Spearman correlation between acceptance sensitivity and attention importance
    spearman_rho: float = 0.0


# ---------------------------------------------------------------------------
# Oracle: Acceptance Sensitivity Measurement
# ---------------------------------------------------------------------------

class AcceptSensitivityOracle:
    """Measures per-token acceptance sensitivity via KV perturbation.

    For each verification step, estimates how much acceptance changes when
    each token's KV is quantized to 2-bit. Uses sampled perturbation for
    efficiency (not all tokens are perturbed).
    """

    def __init__(
        self,
        target_model: Any,
        quantizer_bits: int = 2,
        quantizer_block_size: int = 128,
        sample_fraction: float = 0.2,
        seed: int = 42,
    ):
        self.target_model = target_model
        self.quantizer = ScalarQuantizer(bits=quantizer_bits, block_size=quantizer_block_size)
        self.rotation = HadamardRotation(
            dim=target_model.config.hidden_size // target_model.config.num_attention_heads,
            seed=seed,
        )
        self.sample_fraction = sample_fraction
        self.rng = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def measure_step_sensitivity(
        self,
        target_kv: Any,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        target_next_logits: torch.Tensor,
        temperature: float = 1.0,
        num_samples: int = 50,
        coupled_seeds: Optional[torch.Tensor] = None,
    ) -> SensitivityResult:
        """Measure acceptance sensitivity for one verification step.

        Uses coupled randomness: same U_j values for all perturbations.
        """
        device = next(self.target_model.parameters()).device
        # Only operate on layers with standard KV cache (skip linear attention layers)
        kv_layers = get_kv_layer_indices(target_kv)
        if not kv_layers:
            return None
        k0, v0 = get_kv_tensors(target_kv, kv_layers[0])
        if k0 is None:
            return None
        num_kv_tokens = k0.shape[2]
        gamma = draft_tokens.shape[0]

        # Step 1: Compute baseline acceptance with full KV
        verify_out = self.target_model(
            draft_tokens.view(1, -1).to(device),
            past_key_values=target_kv,
            use_cache=True,
        )
        verify_logits = verify_out.logits

        # Compute per-position acceptance probabilities
        alpha_full, per_pos_accept = self._compute_acceptance(
            target_next_logits, verify_logits, draft_tokens, draft_probs,
            gamma, temperature, coupled_seeds,
        )

        # Step 2: Extract attention importance from verification
        # We hook into the model to get attention weights
        attention_importance = self._get_attention_importance(
            target_kv, draft_tokens, num_kv_tokens,
        )

        # Step 3: Sample tokens to perturb
        n_sample = min(num_samples, num_kv_tokens)
        sample_indices = torch.randperm(num_kv_tokens, generator=self.rng)[:n_sample]

        sensitivities = torch.zeros(num_kv_tokens, device='cpu')

        # Step 4: For each sampled token, perturb and re-measure
        for idx in sample_indices:
            idx_val = idx.item()
            # Save original KV for this token across MHA layers only
            orig_kvs = {}
            for layer_i in kv_layers:
                k, v = get_kv_tensors(target_kv, layer_i)
                if k is None:
                    continue
                orig_kvs[layer_i] = (
                    k[:, :, idx_val:idx_val+1, :].clone(),
                    v[:, :, idx_val:idx_val+1, :].clone(),
                )
                # Quantize this token's KV to 2-bit
                k_tok = k[:, :, idx_val:idx_val+1, :].float()
                v_tok = v[:, :, idx_val:idx_val+1, :].float()
                k_rot = self.rotation.rotate(k_tok)
                v_rot = self.rotation.rotate(v_tok)
                k_codes, k_scales, k_zeros = self.quantizer.quantize(k_rot)
                v_codes, v_scales, v_zeros = self.quantizer.quantize(v_rot)
                k_deq = self.rotation.inverse_rotate(
                    self.quantizer.dequantize(k_codes, k_scales, k_zeros)
                ).to(k.dtype)
                v_deq = self.rotation.inverse_rotate(
                    self.quantizer.dequantize(v_codes, v_scales, v_zeros)
                ).to(v.dtype)
                k[:, :, idx_val:idx_val+1, :] = k_deq
                v[:, :, idx_val:idx_val+1, :] = v_deq

            # Re-run verification with perturbed KV
            perturbed_out = self.target_model(
                draft_tokens.view(1, -1).to(device),
                past_key_values=target_kv,
                use_cache=True,
            )
            perturbed_logits = perturbed_out.logits
            alpha_perturbed, _ = self._compute_acceptance(
                target_next_logits, perturbed_logits, draft_tokens, draft_probs,
                gamma, temperature, coupled_seeds,
            )

            sensitivities[idx_val] = abs(alpha_full - alpha_perturbed)

            # Restore original KV
            for layer_i in kv_layers:
                k, v = get_kv_tensors(target_kv, layer_i)
                if k is None or layer_i not in orig_kvs:
                    continue
                orig_k, orig_v = orig_kvs[layer_i]
                k[:, :, idx_val:idx_val+1, :] = orig_k
                v[:, :, idx_val:idx_val+1, :] = orig_v

            # Trim extended KV from re-running model
            # (the model appends to cache during forward, need to reset)
            for layer_i in kv_layers:
                k, v = get_kv_tensors(target_kv, layer_i)
                if k is not None and k.shape[2] > num_kv_tokens:
                    set_kv_tensors(
                        target_kv, layer_i,
                        k[:, :, :num_kv_tokens, :],
                        v[:, :, :num_kv_tokens, :],
                    )

        # Extrapolate: for non-sampled tokens, use attention importance as proxy
        # (will validate this correlation in the analysis)
        sampled_mask = torch.zeros(num_kv_tokens, dtype=torch.bool)
        sampled_mask[sample_indices] = True

        gini = self._gini_coefficient(sensitivities[sampled_mask])

        return SensitivityResult(
            step_idx=0,
            num_kv_tokens=num_kv_tokens,
            sensitivities=sensitivities,
            attention_importance=attention_importance,
            alpha_full=alpha_full,
            gini=gini,
        )

    def _compute_acceptance(
        self,
        target_next_logits: torch.Tensor,
        verify_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        gamma: int,
        temperature: float,
        coupled_seeds: Optional[torch.Tensor] = None,
    ) -> Tuple[float, torch.Tensor]:
        """Compute acceptance rate using rejection sampling with coupled randomness."""
        device = verify_logits.device
        n_accepted = 0
        per_pos = torch.zeros(gamma)

        # Coupled random seeds for paired comparison
        if coupled_seeds is None:
            coupled_seeds = torch.rand(gamma)

        # First token uses target_next_logits from before draft
        prev_logits = target_next_logits.squeeze(0) if target_next_logits.dim() > 1 else target_next_logits
        if temperature > 0:
            prev_logits = prev_logits / temperature

        for j in range(gamma):
            # Target probability for draft token j
            target_p = F.softmax(prev_logits, dim=-1)
            tok = draft_tokens[j].item()
            p_target = target_p[tok].item()
            p_draft = draft_probs[j].item()

            # Acceptance probability
            if p_draft > 0:
                accept_prob = min(1.0, p_target / p_draft)
            else:
                accept_prob = 1.0

            # Coupled test
            if coupled_seeds[j].item() < accept_prob:
                n_accepted += 1
                per_pos[j] = 1.0
            else:
                break  # Standard rejection sampling: stop at first rejection

            # Next position's target logits
            if j < gamma - 1:
                prev_logits = verify_logits[0, j, :]
                if temperature > 0:
                    prev_logits = prev_logits / temperature

        alpha = n_accepted / gamma if gamma > 0 else 0.0
        return alpha, per_pos

    def _get_attention_importance(
        self,
        target_kv: Any,
        draft_tokens: torch.Tensor,
        num_kv_tokens: int,
    ) -> torch.Tensor:
        """Extract per-token attention importance from the target model.

        Uses a simple proxy: for each KV token, sum the attention weights it
        receives from the draft tokens across all heads and layers.

        This requires registering hooks on attention layers.
        """
        importance = torch.zeros(num_kv_tokens, device='cpu')
        attn_weights_collected = []
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, args, kwargs, output):
                # Try to capture attention weights if available
                if isinstance(output, tuple) and len(output) >= 2:
                    attn_w = output[1]  # attention weights [batch, heads, query_len, kv_len]
                    if attn_w is not None:
                        attn_weights_collected.append(attn_w.detach().cpu())
            return hook_fn

        # Register hooks on MHA attention layers only (skip linear_attn)
        for name, module in self.target_model.named_modules():
            if 'self_attn' in name and 'linear_attn' not in name and not any(
                sub in name for sub in ['.q_proj', '.k_proj', '.v_proj', '.o_proj']
            ):
                if hasattr(module, 'forward'):
                    h = module.register_forward_hook(make_hook(len(hooks)), with_kwargs=True)
                    hooks.append(h)

        # Run forward pass with output_attentions=True
        device = next(self.target_model.parameters()).device
        try:
            out = self.target_model(
                draft_tokens.view(1, -1).to(device),
                past_key_values=target_kv,
                use_cache=True,
                output_attentions=True,
            )
            # If model returns attentions directly
            if hasattr(out, 'attentions') and out.attentions is not None:
                for layer_attn in out.attentions:
                    # layer_attn: [batch, heads, query_len, kv_len]
                    if layer_attn is not None:
                        # Sum attention to each KV position from all query positions
                        attn_to_kv = layer_attn[0, :, :, :num_kv_tokens].sum(dim=(0, 1))
                        importance[:len(attn_to_kv)] += attn_to_kv.cpu()
        except Exception as e:
            logger.warning("Could not get attention weights: %s. Using uniform importance.", e)
            importance = torch.ones(num_kv_tokens)
        finally:
            for h in hooks:
                h.remove()
            # Trim KV cache back to original length (MHA layers only)
            kv_layers = get_kv_layer_indices(target_kv)
            for layer_i in kv_layers:
                k, v = get_kv_tensors(target_kv, layer_i)
                if k is not None and k.shape[2] > num_kv_tokens:
                    set_kv_tensors(
                        target_kv, layer_i,
                        k[:, :, :num_kv_tokens, :],
                        v[:, :, :num_kv_tokens, :],
                    )

        return importance

    @staticmethod
    def _gini_coefficient(values: torch.Tensor) -> float:
        """Compute Gini coefficient of a 1D tensor."""
        if values.numel() == 0:
            return 0.0
        sorted_vals = values.sort().values.float()
        n = sorted_vals.numel()
        if sorted_vals.sum() == 0:
            return 0.0
        cumsum = sorted_vals.cumsum(0)
        return (2.0 * (torch.arange(1, n + 1, dtype=torch.float32) * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n).item()

    @staticmethod
    def compute_cumulative_curve(
        sensitivities: torch.Tensor,
        num_points: int = 20,
    ) -> Tuple[List[float], List[float]]:
        """Compute cumulative sensitivity curve: fraction retained vs fraction captured."""
        sorted_sens, _ = sensitivities.sort(descending=True)
        total = sorted_sens.sum().item()
        if total == 0:
            return list(torch.linspace(0, 1, num_points).tolist()), [0.0] * num_points

        fractions = torch.linspace(0, 1, num_points)
        captured = []
        for f in fractions:
            k = max(1, int(f.item() * len(sorted_sens)))
            captured.append(sorted_sens[:k].sum().item() / total)
        return fractions.tolist(), captured


# ---------------------------------------------------------------------------
# Predictor: MTP/Draft Attention × Value Norm
# ---------------------------------------------------------------------------

class AcceptPredictor:
    """Predicts acceptance-critical tokens from MTP head or draft model attention.

    score(i) = Σ_h w_h · a_h(q, k_i) · ||v_i||_2

    where a_h is attention weight from the MTP head (preferred) or draft model,
    ||v_i|| is value norm, and w_h is a per-head weight learned from oracle data.

    In MTP mode, a_h comes from the MTP decoder layer's attention — zero extra
    cost since it is already computed during drafting.
    """

    def __init__(
        self,
        num_heads: int,
        theta_critical: float = 0.8,
        theta_low: float = 0.3,
    ):
        self.num_heads = num_heads
        self.head_weights = torch.ones(num_heads) / num_heads  # uniform init
        self.theta_critical = theta_critical  # above → FP16
        self.theta_low = theta_low            # below → 2-bit/evict
        self._fitted = False

    def fit(
        self,
        draft_attentions: List[torch.Tensor],
        value_norms: List[torch.Tensor],
        oracle_labels: List[torch.Tensor],
    ):
        """Fit head weights via logistic regression on oracle calibration data.

        Args:
            draft_attentions: list of [num_heads, num_kv_tokens] per step
            value_norms: list of [num_kv_tokens] per step
            oracle_labels: list of binary [num_kv_tokens] (1=critical, 0=not)
        """
        # Concatenate all steps
        all_features = []
        all_labels = []
        for attn, vnorm, labels in zip(draft_attentions, value_norms, oracle_labels):
            # Features: per-head attention × value_norm → [num_kv, num_heads]
            features = attn.T * vnorm.unsqueeze(1)  # [kv, heads]
            all_features.append(features)
            all_labels.append(labels)

        X = torch.cat(all_features, dim=0).float()  # [total_tokens, heads]
        y = torch.cat(all_labels, dim=0).float()     # [total_tokens]

        # Simple logistic regression via gradient descent
        w = torch.zeros(self.num_heads, requires_grad=True)
        optimizer = torch.optim.LBFGS([w], lr=1.0, max_iter=50)

        def closure():
            optimizer.zero_grad()
            logits = (X * F.softmax(w, dim=0)).sum(dim=1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.head_weights = F.softmax(w.detach(), dim=0)
        self._fitted = True
        logger.info("AcceptPredictor fitted. Head weights: %s", self.head_weights.tolist())

    def predict_scores(
        self,
        draft_attention: torch.Tensor,
        value_norms: torch.Tensor,
    ) -> torch.Tensor:
        """Predict acceptance-criticality scores for all KV tokens.

        Args:
            draft_attention: [num_heads, num_kv_tokens] — attention from last draft query
            value_norms: [num_kv_tokens] — L2 norm of each token's value vector

        Returns:
            scores: [num_kv_tokens] — higher = more critical
        """
        # score(i) = Σ_h w_h · a_h(q, k_i) · ||v_i||
        weighted_attn = (self.head_weights.unsqueeze(1).to(draft_attention.device) * draft_attention)
        scores = weighted_attn.sum(dim=0) * value_norms
        return scores

    def predict_tags(
        self,
        draft_attention: torch.Tensor,
        value_norms: torch.Tensor,
        critical_fraction: float = 0.2,
    ) -> torch.Tensor:
        """Predict per-token precision tags.

        Returns:
            tags: [num_kv_tokens] — TAG_FP16, TAG_4BIT, TAG_2BIT, or TAG_EVICTED
        """
        scores = self.predict_scores(draft_attention, value_norms)
        n = scores.numel()

        # Adaptive thresholds based on score distribution
        sorted_scores, _ = scores.sort(descending=True)
        n_critical = max(1, int(critical_fraction * n))
        n_moderate = max(1, int(0.3 * n))  # next 30% at 4-bit

        tags = torch.full((n,), TAG_2BIT, dtype=torch.uint8)

        # Top critical_fraction → FP16
        _, top_indices = scores.topk(n_critical)
        tags[top_indices] = TAG_FP16

        # Next 30% → 4-bit
        remaining_mask = tags != TAG_FP16
        remaining_scores = scores.clone()
        remaining_scores[~remaining_mask] = -float('inf')
        _, moderate_indices = remaining_scores.topk(n_moderate)
        tags[moderate_indices] = TAG_4BIT

        # Bottom tokens with very low scores → evict
        evict_threshold = sorted_scores[min(int(0.9 * n), n - 1)]
        tags[scores < evict_threshold] = TAG_EVICTED

        return tags


# ---------------------------------------------------------------------------
# Mixed-Precision KV Cache
# ---------------------------------------------------------------------------

class MixedPrecisionKV:
    """KV cache with per-token precision tags and differential compression."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        seed: int = 42,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.rotation = HadamardRotation(dim=head_dim, seed=seed)
        self.quant_2bit = ScalarQuantizer(bits=2, block_size=128)
        self.quant_4bit = ScalarQuantizer(bits=4, block_size=128)

        # Per-layer storage
        self.tags: List[Optional[torch.Tensor]] = [None] * num_layers
        # FP16 tokens: stored as-is in the HF cache
        # Compressed tokens: stored in separate buffers
        self.compressed_k: List[Dict] = [{} for _ in range(num_layers)]
        self.compressed_v: List[Dict] = [{} for _ in range(num_layers)]

    def compress_kv(
        self,
        kv_cache: Any,
        tags: torch.Tensor,
    ) -> Any:
        """Compress KV cache according to per-token tags.

        Modifies kv_cache in-place: non-critical tokens are quantized,
        evicted tokens are zeroed.  Only operates on MHA layers with
        standard KV tensors (skips linear attention recurrent states).
        """
        kv_layers = get_kv_layer_indices(kv_cache)
        for layer_i in kv_layers:
            k, v = get_kv_tensors(kv_cache, layer_i)
            if k is None:
                continue

            self.tags[layer_i] = tags.clone()

            # Process each precision level
            for tag_val, quantizer in [(TAG_2BIT, self.quant_2bit), (TAG_4BIT, self.quant_4bit)]:
                mask = (tags == tag_val)
                if not mask.any():
                    continue

                indices = mask.nonzero(as_tuple=True)[0]
                for idx in indices:
                    idx_val = idx.item()
                    # Quantize this token's K and V
                    k_tok = k[:, :, idx_val:idx_val+1, :].float()
                    v_tok = v[:, :, idx_val:idx_val+1, :].float()

                    k_rot = self.rotation.rotate(k_tok)
                    v_rot = self.rotation.rotate(v_tok)

                    k_codes, k_s, k_z = quantizer.quantize(k_rot)
                    v_codes, v_s, v_z = quantizer.quantize(v_rot)

                    k_deq = self.rotation.inverse_rotate(
                        quantizer.dequantize(k_codes, k_s, k_z)
                    ).to(k.dtype)
                    v_deq = self.rotation.inverse_rotate(
                        quantizer.dequantize(v_codes, v_s, v_z)
                    ).to(v.dtype)

                    k[:, :, idx_val:idx_val+1, :] = k_deq
                    v[:, :, idx_val:idx_val+1, :] = v_deq

            # Zero out evicted tokens
            evict_mask = (tags == TAG_EVICTED)
            if evict_mask.any():
                indices = evict_mask.nonzero(as_tuple=True)[0]
                for idx in indices:
                    idx_val = idx.item()
                    k[:, :, idx_val:idx_val+1, :] = 0
                    v[:, :, idx_val:idx_val+1, :] = 0

        return kv_cache

    def get_compression_stats(self, tags: torch.Tensor) -> Dict:
        """Return compression statistics."""
        n = tags.numel()
        return {
            'total_tokens': n,
            'fp16_tokens': (tags == TAG_FP16).sum().item(),
            'fp16_fraction': (tags == TAG_FP16).float().mean().item(),
            '4bit_tokens': (tags == TAG_4BIT).sum().item(),
            '2bit_tokens': (tags == TAG_2BIT).sum().item(),
            'evicted_tokens': (tags == TAG_EVICTED).sum().item(),
            'effective_bits': self._effective_bits(tags),
            'compression_ratio': 16.0 / max(self._effective_bits(tags), 0.1),
        }

    @staticmethod
    def _effective_bits(tags: torch.Tensor) -> float:
        n = tags.numel()
        if n == 0:
            return 16.0
        bits = (
            (tags == TAG_FP16).float() * 16
            + (tags == TAG_4BIT).float() * 4
            + (tags == TAG_2BIT).float() * 2
            + (tags == TAG_EVICTED).float() * 0
        )
        return bits.mean().item()
