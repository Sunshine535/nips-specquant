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
    """Result of oracle acceptance sensitivity measurement for one verification step.

    Note: only a subset of tokens are actually perturbed (controlled by
    sample_fraction).  ``sensitivities`` has zeros for unsampled tokens.
    Use ``sample_indices`` to identify which entries are real measurements
    vs zero-filled placeholders.  ``sampled_sensitivities`` gives a dense
    vector of only the measured values (for unbiased metric computation).

    Additional signals for MarginSpec (C2 mechanism validation):
      - logit_tv_sensitivities: |TV(p_full, p_perturbed_i)| — continuous
        signal, not subject to gamma=5 discretization of acceptance count.
      - margin_sensitivities: predicted sensitivity from top-2 logit margin
        × attention × value_norm (closed-form, zero extra forward pass).
    """
    step_idx: int
    num_kv_tokens: int
    # Per-token sensitivity: S_accept(i) = |alpha_full - alpha_perturbed_i|
    # WARNING: zeros at unsampled positions — use sample_indices for correct metrics
    sensitivities: torch.Tensor       # [num_kv_tokens]
    # Dense vector of ONLY sampled sensitivities (no zero padding)
    sampled_sensitivities: torch.Tensor  # [n_sample]
    # Indices of tokens that were actually perturbed
    sample_indices: torch.Tensor       # [n_sample]
    # Per-token attention importance (sum across heads/layers)
    attention_importance: torch.Tensor  # [num_kv_tokens]
    # Full acceptance count (baseline)
    alpha_full: float
    # Gini coefficient computed on SAMPLED tokens only (unbiased)
    gini: float
    # Continuous logit-TV sensitivity (MarginSpec C2 signal, dense over sample_indices)
    sampled_logit_tv: Optional[torch.Tensor] = None  # [n_sample]
    # Margin-sensitivity proxy: score(i) = attention_sum_i * ||v_i|| * margin_factor
    # Computable from single forward pass (no per-token perturbation)
    margin_sensitivities: Optional[torch.Tensor] = None  # [num_kv_tokens]
    # Scalar: top-2 logit margin at this verification step (for diagnostics)
    top2_margin: float = 0.0


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
    each token's KV is quantized to 2-bit.  Uses sampled perturbation for
    efficiency (not all tokens are perturbed).

    Implementation notes:
      - This is a SINGLE-DRAW estimator: one set of coupled uniform random
        variables U is used per measurement.  The resulting sensitivity
        S_accept(i) is a realization, not an expectation over U.  For
        gamma=5, values are multiples of 0.2.  Averaging over multiple U
        draws would give a smoother estimate but is ~num_U_draws times more
        expensive.
      - Quantization uses the FULL BLOCK context (neighboring tokens share
        the same scale/zero), matching the deployed quantizer behavior.
      - Sparsity metrics (Gini, top-k coverage) are computed ONLY on
        sampled tokens to avoid bias from zero-filled unsampled positions.
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
        This is a single-draw estimator — NOT an expectation over U.

        Args:
            draft_probs: per-position SCALAR draft probability for the
                selected token, shape [gamma].  NOT full-vocabulary tensors.
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

        # Trim KV back to pre-draft length (model forward appends draft tokens)
        for layer_i in kv_layers:
            k, v = get_kv_tensors(target_kv, layer_i)
            if k is not None and k.shape[2] > num_kv_tokens:
                set_kv_tensors(
                    target_kv, layer_i,
                    k[:, :, :num_kv_tokens, :],
                    v[:, :, :num_kv_tokens, :],
                )

        # Compute per-position acceptance probabilities
        alpha_full, per_pos_accept = self._compute_acceptance(
            target_next_logits, verify_logits, draft_tokens, draft_probs,
            gamma, temperature, coupled_seeds,
        )

        # Compute baseline softmax for logit-TV signal + top-2 margin (for MarginSpec)
        with torch.no_grad():
            # Use verify_logits[0, 0, :] — first draft position's target prediction
            # (continuous signal, not subject to gamma=5 accept/reject discretization)
            baseline_probs = torch.softmax(verify_logits[0].float(), dim=-1).detach().cpu()
            # Top-2 logit margin at first position (proxy for verifier decision "tightness")
            top2_vals, _ = verify_logits[0, 0, :].float().topk(2)
            top2_margin = float((top2_vals[0] - top2_vals[1]).item())

        # Step 2: Extract attention importance (on pre-draft KV, not extended)
        attention_importance = self._get_attention_importance(
            target_kv, draft_tokens, num_kv_tokens,
        )

        # Step 2.5: Compute margin-sensitivity score (proper per-token Jacobian)
        #   m(i) = a_i × |<margin_dir, o_proj @ v_i>|
        # where margin_dir = lm_head[top1,:] - lm_head[top2,:] is the direction
        # in residual space that encodes the top1-top2 gap, and o_proj @ v_i is
        # per-token's contribution. The inner product is PER-TOKEN DIFFERENT
        # (unlike the previous formula where 1/margin was a per-step constant).
        margin_sensitivities = self._compute_margin_sensitivity_jacobian(
            target_kv, attention_importance, num_kv_tokens,
            verify_logits,
        )

        # Step 3: Sample tokens to perturb (honor sample_fraction)
        n_sample = max(1, int(self.sample_fraction * num_kv_tokens))
        n_sample = min(n_sample, num_kv_tokens)
        sample_indices = torch.randperm(num_kv_tokens, generator=self.rng)[:n_sample]

        # Only store sensitivities for SAMPLED tokens (avoid misleading zeros)
        sampled_sensitivities = torch.zeros(n_sample, device='cpu')
        sampled_logit_tv = torch.zeros(n_sample, device='cpu')

        # Step 4: For each sampled token, perturb using FULL BLOCK context
        for si, idx in enumerate(sample_indices):
            idx_val = idx.item()

            # Determine the block this token belongs to
            block_idx = idx_val // self.quantizer.block_size
            block_start = block_idx * self.quantizer.block_size
            block_end = min(block_start + self.quantizer.block_size, num_kv_tokens)

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
                # Quantize this token's KV using FULL BLOCK context
                # (shared scale/zero from the real block neighbors)
                k_block = k[:, :, block_start:block_end, :].float()
                v_block = v[:, :, block_start:block_end, :].float()
                k_block_rot = self.rotation.rotate(k_block)
                v_block_rot = self.rotation.rotate(v_block)
                k_codes, k_scales, k_zeros = self.quantizer.quantize(k_block_rot)
                v_codes, v_scales, v_zeros = self.quantizer.quantize(v_block_rot)
                k_block_deq = self.rotation.inverse_rotate(
                    self.quantizer.dequantize(k_codes, k_scales, k_zeros)
                ).to(k.dtype)
                v_block_deq = self.rotation.inverse_rotate(
                    self.quantizer.dequantize(v_codes, v_scales, v_zeros)
                ).to(v.dtype)
                # Only replace the TARGET token (not the whole block)
                local_idx = idx_val - block_start
                k[:, :, idx_val:idx_val+1, :] = k_block_deq[:, :, local_idx:local_idx+1, :]
                v[:, :, idx_val:idx_val+1, :] = v_block_deq[:, :, local_idx:local_idx+1, :]

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

            sampled_sensitivities[si] = abs(alpha_full - alpha_perturbed)

            # Logit-TV sensitivity: continuous alternative to accept/reject count
            # TV(p, p') = 0.5 * ||p - p'||_1, averaged across gamma positions
            with torch.no_grad():
                perturbed_probs = torch.softmax(perturbed_logits[0].float(), dim=-1).detach().cpu()
                tv_per_pos = 0.5 * (baseline_probs - perturbed_probs).abs().sum(dim=-1)
                sampled_logit_tv[si] = float(tv_per_pos.mean().item())

            # Restore original KV
            for layer_i in kv_layers:
                k, v = get_kv_tensors(target_kv, layer_i)
                if k is None or layer_i not in orig_kvs:
                    continue
                orig_k, orig_v = orig_kvs[layer_i]
                k[:, :, idx_val:idx_val+1, :] = orig_k
                v[:, :, idx_val:idx_val+1, :] = orig_v

            # Trim extended KV from re-running model
            for layer_i in kv_layers:
                k, v = get_kv_tensors(target_kv, layer_i)
                if k is not None and k.shape[2] > num_kv_tokens:
                    set_kv_tensors(
                        target_kv, layer_i,
                        k[:, :, :num_kv_tokens, :],
                        v[:, :, :num_kv_tokens, :],
                    )

        # Build full sensitivity vector (sampled values at their indices)
        sensitivities = torch.zeros(num_kv_tokens, device='cpu')
        for si, idx in enumerate(sample_indices):
            sensitivities[idx.item()] = sampled_sensitivities[si]

        # Compute sparsity metrics ONLY on sampled tokens (avoid bias from zeros)
        gini = self._gini_coefficient(sampled_sensitivities)

        return SensitivityResult(
            step_idx=0,
            num_kv_tokens=num_kv_tokens,
            sensitivities=sensitivities,
            sampled_sensitivities=sampled_sensitivities,
            sample_indices=sample_indices,
            attention_importance=attention_importance,
            alpha_full=alpha_full,
            gini=gini,
            sampled_logit_tv=sampled_logit_tv,
            margin_sensitivities=margin_sensitivities,
            top2_margin=top2_margin,
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

    def _compute_margin_sensitivity(
        self,
        target_kv: Any,
        attention_importance: torch.Tensor,
        num_kv_tokens: int,
        top2_margin: float,
    ) -> torch.Tensor:
        """Margin-sensitivity proxy (MarginSpec C2):

            m(i) = attention_sum_i × ||v_i||_2 × margin_factor

        where margin_factor = 1 / (top2_margin + eps) captures how "tight"
        the verifier's decision is. Low margin → argmax easily flips →
        sensitivity amplified. High margin → verifier is confident → robust.

        This is closed-form (no extra forward pass) once attention_importance
        and value norms are available.
        """
        # Compute per-token value norm across all MHA layers (average)
        kv_layers = get_kv_layer_indices(target_kv)
        v_norms = torch.zeros(num_kv_tokens, device='cpu')
        n_layers_ok = 0
        for layer_i in kv_layers:
            _, v = get_kv_tensors(target_kv, layer_i)
            if v is None:
                continue
            # v: [batch, num_kv_heads, seq, head_dim]
            seq = min(v.shape[2], num_kv_tokens)
            layer_norms = v[0, :, :seq, :].float().norm(dim=-1).mean(dim=0).cpu()
            v_norms[:seq] += layer_norms
            n_layers_ok += 1
        if n_layers_ok > 0:
            v_norms = v_norms / n_layers_ok

        # Margin factor: amplify when verifier is close to flipping argmax
        eps = 1e-3
        margin_factor = 1.0 / (abs(top2_margin) + eps)

        # Compose: attention × value_norm × margin_factor
        attn_cpu = attention_importance.float().cpu()
        scores = attn_cpu * v_norms * margin_factor
        return scores

    def _compute_margin_sensitivity_jacobian(
        self,
        target_kv: Any,
        attention_importance: torch.Tensor,
        num_kv_tokens: int,
        verify_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Proper per-token margin-sensitivity via closed-form Jacobian.

            m(i) = a_i × |<margin_dir, o_proj @ v_i>|

        where:
            margin_dir = lm_head_top1 - lm_head_top2   [model_dim]
            v_i        = last layer's value vector for token i
            o_proj     = attention output projection of last MHA layer

        This properly differentiates per token — previous version used
        1/margin as a per-step constant, making it rank-equivalent to
        attention×value_norm (the 'margin' dimension provided no info).

        Returns:
            scores: [num_kv_tokens] per-token margin sensitivity
        """
        try:
            with torch.no_grad():
                # 1. Extract top1, top2 at first verification position
                pos0_logits = verify_logits[0, 0, :].float()
                top2_vals, top2_ids = pos0_logits.topk(2)

                # 2. margin_dir in residual stream space
                lm_head = getattr(self.target_model, 'lm_head', None)
                if lm_head is None:
                    lm_head = getattr(self.target_model.model, 'lm_head', None)
                if lm_head is None:
                    return attention_importance.float().cpu()  # fallback
                W_emb = lm_head.weight  # [vocab, model_dim]
                margin_dir = (W_emb[top2_ids[0]] - W_emb[top2_ids[1]]).float()  # [model_dim]

                # 3. Find last MHA layer with o_proj
                try:
                    layers = self.target_model.model.layers
                except AttributeError:
                    return attention_importance.float().cpu()
                last_layer_idx = None
                for i in reversed(range(len(layers))):
                    try:
                        _ = layers[i].self_attn.o_proj.weight
                        last_layer_idx = i
                        break
                    except AttributeError:
                        continue
                if last_layer_idx is None:
                    return attention_importance.float().cpu()

                W_o = layers[last_layer_idx].self_attn.o_proj.weight.float()
                # W_o: [model_dim, num_heads * head_dim]
                model_dim = W_o.shape[0]
                total_head_dim = W_o.shape[1]

                num_heads = self.target_model.config.num_attention_heads
                head_dim = total_head_dim // num_heads

                # 4. Pull v_margin through o_proj: which v_i direction most affects margin?
                # v_margin[h, d] = dot(margin_dir, W_o[:, h*head_dim+d])
                margin_dir_d = margin_dir.to(W_o.device)
                v_margin = (margin_dir_d @ W_o).view(num_heads, head_dim)

                # 5. Get last layer's V cache
                _, v = get_kv_tensors(target_kv, last_layer_idx)
                if v is None:
                    return attention_importance.float().cpu()
                v_last = v[0].float()  # [num_kv_heads, seq, head_dim]
                num_kv_heads = v_last.shape[0]

                # 6. Handle GQA: group v_margin into num_kv_heads
                if num_heads != num_kv_heads and num_kv_heads > 0:
                    group = num_heads // num_kv_heads
                    # Average margin direction within each GQA group
                    v_margin = v_margin.view(num_kv_heads, group, head_dim).mean(dim=1)
                    # Now v_margin: [num_kv_heads, head_dim]

                # 7. Per-token, per-head contribution: <v_margin[h], v_last[h, i, :]>
                # v_last: [num_kv_heads, seq, head_dim], v_margin: [num_kv_heads, head_dim]
                seq = min(v_last.shape[1], num_kv_tokens)
                margin_contrib = (
                    v_last[:, :seq, :] * v_margin.unsqueeze(1)
                ).sum(dim=-1)  # [num_kv_heads, seq]

                # 8. Aggregate per-head (sum of absolute contributions)
                per_token_margin = margin_contrib.abs().sum(dim=0).cpu()  # [seq]

                # 9. Pad to num_kv_tokens
                out = torch.zeros(num_kv_tokens)
                out[:seq] = per_token_margin

                # 10. Weight by attention importance (a_i)
                attn_cpu = attention_importance.float().cpu()[:num_kv_tokens]
                if attn_cpu.numel() < num_kv_tokens:
                    attn_cpu = torch.cat([attn_cpu, torch.zeros(num_kv_tokens - attn_cpu.numel())])
                scores = out * attn_cpu
                return scores
        except Exception as e:
            logger.debug("Margin-sensitivity Jacobian failed (%s), falling back to attention", e)
            return attention_importance.float().cpu()[:num_kv_tokens]

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
        """Compute Gini coefficient of a 1D tensor.

        Uses standard formula: G = (2*sum(i*v_(i))) / (n*sum(v)) - (n+1)/n
        where v_(i) is the i-th value in ascending order.
        Requires values >= 0.
        """
        if values.numel() == 0:
            return 0.0
        sorted_vals = values.sort().values.float()
        n = sorted_vals.numel()
        if sorted_vals.sum() == 0:
            return 0.0
        ranks = torch.arange(1, n + 1, dtype=torch.float32)
        return (2.0 * (ranks * sorted_vals).sum() / (n * sorted_vals.sum()) - (n + 1) / n).item()

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
        """Fit head weights on oracle calibration data.

        Uses softmax-parameterized weighted combination of per-head features.
        Note: the loss is convex in the simplex weights u = softmax(w), but
        NOT convex in the unconstrained parameters w.  LBFGS may converge to
        a local minimum.  In practice this is acceptable because:
          (a) the number of heads is small (32-64) and the landscape is smooth,
          (b) we only need a "good enough" predictor (F1 > 0.75), not the
              global optimum.

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

        # Softmax-parameterized logistic model (non-convex in w)
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
