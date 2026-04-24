"""MARA: Margin-Calibrated Acceptance-Risk Allocation.

Core module implementing the new MAIN METHOD PATH from GPT-5.5 Pro diagnosis:
  1. AcceptanceRiskOracle — collect continuous risk labels by perturbing KV tokens
  2. AcceptanceRiskPredictor — calibrated risk predictor outputting (μ, σ)
  3. RiskBudgetAllocator — greedy precision allocation under budget constraint
  4. MarginUncertaintyGate — adaptive budget based on margin/uncertainty

Key insight: acceptance preservation is a RISK ALLOCATION problem, not a static
token selection problem. A token's importance depends on verifier margin, draft
probability path, compression action, budget, and context.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Precision action costs (relative to FP16 = 1.0)
ACTION_COSTS = {
    "fp16": 1.0,
    "4bit": 0.25,
    "2bit": 0.125,
    "evict": 0.0,
}

ACTION_LIST = ["fp16", "4bit", "2bit", "evict"]


@dataclass
class RiskLabel:
    """Continuous risk label for a single (token, action) pair at a verification step."""

    step_idx: int
    token_idx: int
    action: str
    alpha_full: float
    alpha_perturbed: float
    acceptance_risk: float  # max(0, alpha_full - alpha_perturbed)
    tv_distance: float  # TV(p_full, p_perturbed)
    margin_full: float  # top-2 logit margin at full KV
    margin_perturbed: float  # top-2 logit margin at perturbed KV
    margin_risk: float  # max(0, margin_full - margin_perturbed) if margin < threshold


@dataclass
class RiskLabelSet:
    """Collection of risk labels from oracle measurement."""

    labels: List[RiskLabel] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to feature tensor X and risk target y for predictor training.

        Features are PRE-DECISION only (available before perturbation):
          - alpha_full: current step acceptance rate
          - margin_full: top-2 logit margin (verifier confidence)
          - action_idx: which precision action (0=fp16, 1=4bit, 2=2bit, 3=evict)
          - step_idx: generation step position
          - relative_position: token_idx / step_idx (recency)

        Post-perturbation signals (tv_distance, margin_risk) are LABELS ONLY,
        never used as inference features.
        """
        if not self.labels:
            return torch.empty(0, 5), torch.empty(0)

        X = torch.tensor(
            [
                [
                    l.alpha_full,
                    l.margin_full,
                    ACTION_LIST.index(l.action),
                    l.step_idx,
                    l.token_idx / max(1, l.step_idx),
                ]
                for l in self.labels
            ],
            dtype=torch.float32,
        )
        # Combined risk target: acceptance_risk + weighted TV + margin_risk
        y = torch.tensor(
            [
                l.acceptance_risk + 0.5 * l.tv_distance + 0.3 * l.margin_risk
                for l in self.labels
            ],
            dtype=torch.float32,
        )
        return X, y

    def save(self, path: str):
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": self.metadata,
            "num_labels": len(self.labels),
            "labels": [
                {
                    "step_idx": l.step_idx,
                    "token_idx": l.token_idx,
                    "action": l.action,
                    "alpha_full": l.alpha_full,
                    "alpha_perturbed": l.alpha_perturbed,
                    "acceptance_risk": l.acceptance_risk,
                    "tv_distance": l.tv_distance,
                    "margin_full": l.margin_full,
                    "margin_perturbed": l.margin_perturbed,
                    "margin_risk": l.margin_risk,
                }
                for l in self.labels
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class AcceptanceRiskOracle:
    """Collect continuous risk labels by perturbing KV tokens under different actions.

    Unlike the old AcceptSensitivityOracle which measured binary criticality,
    this oracle measures continuous acceptance degradation per (token, action) pair.
    """

    def __init__(
        self,
        model,
        tokenizer,
        quantizer=None,
        sample_fraction: float = 0.3,
        actions: Optional[List[str]] = None,
        margin_threshold: float = 2.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.quantizer = quantizer
        self.sample_fraction = sample_fraction
        self.actions = actions or ["4bit", "2bit"]
        self.margin_threshold = margin_threshold

    def measure_step_risk(
        self,
        kv_cache,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        coupled_uniforms: torch.Tensor,
        target_logits_full: torch.Tensor,
        step_idx: int,
    ) -> List[RiskLabel]:
        """Measure risk labels for sampled tokens at one verification step.

        Args:
            kv_cache: Current KV cache (full precision)
            draft_tokens: [gamma] draft token ids
            draft_probs: [gamma, vocab] draft probability distributions
            coupled_uniforms: [gamma] pre-sampled uniform RVs for paired measurement
            target_logits_full: [gamma, vocab] target logits with full KV
            step_idx: Current generation step index

        Returns:
            List of RiskLabel for each (sampled_token, action) pair
        """
        device = draft_tokens.device
        gamma = len(draft_tokens)

        # Compute full-KV acceptance and margin
        target_probs_full = F.softmax(target_logits_full, dim=-1)
        alpha_full = self._compute_acceptance(
            target_probs_full, draft_probs, draft_tokens, coupled_uniforms
        )
        margin_full = self._compute_margin(target_logits_full)

        # Determine KV cache length and sample tokens to perturb
        kv_len = self._get_kv_len(kv_cache)
        n_sample = max(1, int(kv_len * self.sample_fraction))
        sample_indices = torch.randperm(kv_len)[:n_sample].sort().values

        labels = []
        for token_idx in sample_indices.tolist():
            for action in self.actions:
                # Perturb this token's KV to the specified precision
                kv_perturbed = self._perturb_token(kv_cache, token_idx, action)

                # Re-run verification with perturbed KV
                with torch.no_grad():
                    target_logits_perturbed = self._recompute_logits(
                        kv_perturbed, draft_tokens
                    )

                target_probs_perturbed = F.softmax(target_logits_perturbed, dim=-1)
                alpha_perturbed = self._compute_acceptance(
                    target_probs_perturbed, draft_probs, draft_tokens, coupled_uniforms
                )
                margin_perturbed = self._compute_margin(target_logits_perturbed)

                # Compute TV distance
                tv = 0.5 * (target_probs_full - target_probs_perturbed).abs().sum(-1).mean().item()

                # Compute risk components
                acceptance_risk = max(0.0, alpha_full - alpha_perturbed)
                margin_risk = 0.0
                if margin_full < self.margin_threshold:
                    margin_risk = max(0.0, margin_full - margin_perturbed)

                labels.append(
                    RiskLabel(
                        step_idx=step_idx,
                        token_idx=token_idx,
                        action=action,
                        alpha_full=alpha_full,
                        alpha_perturbed=alpha_perturbed,
                        acceptance_risk=acceptance_risk,
                        tv_distance=tv,
                        margin_full=margin_full,
                        margin_perturbed=margin_perturbed,
                        margin_risk=margin_risk,
                    )
                )

                # Restore original KV
                self._restore_token(kv_cache, kv_perturbed, token_idx)

        return labels

    def _compute_acceptance(
        self,
        target_probs: torch.Tensor,
        draft_probs: torch.Tensor,
        draft_tokens: torch.Tensor,
        uniforms: torch.Tensor,
    ) -> float:
        """Compute acceptance count using coupled uniforms (rejection sampling)."""
        gamma = len(draft_tokens)
        accepted = 0
        for j in range(gamma):
            t_idx = draft_tokens[j].item()
            p_target = target_probs[j, t_idx].item()
            p_draft = draft_probs[j, t_idx].item()
            if p_draft > 0:
                ratio = min(1.0, p_target / p_draft)
            else:
                ratio = 1.0 if p_target > 0 else 0.0
            if uniforms[j].item() < ratio:
                accepted += 1
            else:
                break
        return accepted / max(1, gamma)

    @staticmethod
    def _compute_margin(logits: torch.Tensor) -> float:
        """Compute mean top-2 logit margin across positions."""
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        top2 = logits.topk(2, dim=-1).values
        margins = (top2[:, 0] - top2[:, 1]).abs()
        return margins.mean().item()

    def _get_kv_len(self, kv_cache) -> int:
        if isinstance(kv_cache, (list, tuple)) and len(kv_cache) > 0:
            layer0 = kv_cache[0]
            if isinstance(layer0, (list, tuple)) and len(layer0) >= 2:
                return layer0[0].shape[2]
        return 0

    def _perturb_token(self, kv_cache, token_idx: int, action: str):
        """Perturb a single token's KV to simulated lower precision.

        Handles HybridCache (Qwen3.5) by only touching MHA layers
        and skipping LinearAttention layers.
        """
        import copy
        from .utils import get_kv_tensors, set_kv_tensors, get_kv_layer_indices

        kv_perturbed = copy.deepcopy(kv_cache)
        mha_layers = get_kv_layer_indices(kv_perturbed)

        for li in mha_layers:
            k, v = get_kv_tensors(kv_perturbed, li)
            if k is None or token_idx >= k.shape[2]:
                continue

            if action == "evict":
                k[:, :, token_idx, :] = 0.0
                v[:, :, token_idx, :] = 0.0
            elif action in ("2bit", "4bit"):
                bits = 2 if action == "2bit" else 4
                n_levels = 2**bits - 1
                for tensor in [k, v]:
                    val = tensor[:, :, token_idx, :]
                    vmin, vmax = val.min(), val.max()
                    if vmax > vmin:
                        normalized = (val - vmin) / (vmax - vmin)
                        quantized = torch.round(normalized * n_levels) / n_levels
                        tensor[:, :, token_idx, :] = quantized * (vmax - vmin) + vmin

            set_kv_tensors(kv_perturbed, li, k, v)

        return kv_perturbed

    def _restore_token(self, kv_original, kv_perturbed, token_idx: int):
        """No-op: we use deep copy, so original is not modified."""
        pass

    def _recompute_logits(self, kv_cache, draft_tokens: torch.Tensor) -> torch.Tensor:
        """Recompute target logits using perturbed KV cache."""
        # This requires running the model with modified past_key_values
        # Implementation depends on model interface
        with torch.no_grad():
            outputs = self.model(
                input_ids=draft_tokens.unsqueeze(0),
                past_key_values=kv_cache,
                use_cache=True,
            )
        return outputs.logits[0]


class AcceptanceRiskPredictor:
    """Calibrated risk predictor: predicts (μ, σ) for acceptance risk.

    Uses a simple 2-layer network that outputs mean and log-variance.
    Trained on oracle risk labels with Huber + ranking + calibration loss.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weights: Optional[Dict[str, torch.Tensor]] = None
        self._build_model()

    def _build_model(self):
        """Initialize simple MLP weights."""
        self.weights = {
            "w1": torch.randn(self.input_dim, self.hidden_dim) * 0.1,
            "b1": torch.zeros(self.hidden_dim),
            "w_mu": torch.randn(self.hidden_dim, 1) * 0.1,
            "b_mu": torch.zeros(1),
            "w_sigma": torch.randn(self.hidden_dim, 1) * 0.1,
            "b_sigma": torch.zeros(1),
        }

    def predict(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict risk mean and uncertainty.

        Args:
            features: [N, input_dim] feature tensor

        Returns:
            mu: [N] predicted risk mean
            sigma: [N] predicted risk uncertainty (std dev)
        """
        h = torch.relu(features @ self.weights["w1"] + self.weights["b1"])
        mu = (h @ self.weights["w_mu"] + self.weights["b_mu"]).squeeze(-1)
        log_sigma = (h @ self.weights["w_sigma"] + self.weights["b_sigma"]).squeeze(-1)
        sigma = torch.exp(log_sigma.clamp(-5, 5))
        return mu, sigma

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.01,
        epochs: int = 100,
        lambda_rank: float = 0.1,
        lambda_cal: float = 0.01,
    ) -> Dict[str, float]:
        """Fit predictor on risk labels.

        Returns dict of final loss components.
        """
        if len(X) == 0:
            logger.warning("No training data for risk predictor")
            return {"loss": float("nan")}

        params = list(self.weights.values())
        for p in params:
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(params, lr=lr)
        log_y = torch.log1p(y)

        best_loss = float("inf")
        for epoch in range(epochs):
            mu, sigma = self.predict(X)

            # Huber loss on log(1 + risk)
            loss_risk = F.huber_loss(mu, log_y, delta=1.0)

            # Pairwise ranking loss (sample pairs)
            n = len(y)
            if n > 1:
                idx_i = torch.randint(0, n, (min(n * 2, 256),))
                idx_j = torch.randint(0, n, (min(n * 2, 256),))
                sign = torch.sign(y[idx_i] - y[idx_j])
                diff = mu[idx_i] - mu[idx_j]
                loss_rank = torch.log1p(torch.exp(-sign * diff)).mean()
            else:
                loss_rank = torch.tensor(0.0)

            # NLL calibration loss
            loss_cal = (
                0.5 * ((log_y - mu) ** 2 / (sigma**2 + 1e-8) + 2 * torch.log(sigma + 1e-8))
            ).mean()

            loss = loss_risk + lambda_rank * loss_rank + lambda_cal * loss_cal

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()

        for p in params:
            p.requires_grad_(False)

        return {
            "loss_total": best_loss,
            "loss_risk": loss_risk.item(),
            "loss_rank": loss_rank.item() if isinstance(loss_rank, torch.Tensor) else 0.0,
            "loss_cal": loss_cal.item(),
        }

    def compute_calibration_metrics(
        self, X: torch.Tensor, y: torch.Tensor, n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute ECE and Spearman correlation on held-out data."""
        from scipy.stats import spearmanr

        mu, sigma = self.predict(X)
        mu_np = mu.detach().numpy()
        y_np = y.numpy()

        # Spearman rank correlation between predicted and actual risk
        if len(y_np) > 2:
            rho, pval = spearmanr(mu_np, y_np)
        else:
            rho, pval = 0.0, 1.0

        # Simple ECE: bin by predicted risk, compare mean predicted vs mean actual
        log_y = np.log1p(y_np)
        bins = np.linspace(mu_np.min() - 1e-8, mu_np.max() + 1e-8, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (mu_np >= bins[i]) & (mu_np < bins[i + 1])
            if mask.sum() > 0:
                ece += mask.sum() * abs(mu_np[mask].mean() - log_y[mask].mean())
        ece /= max(1, len(mu_np))

        return {
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "ece": float(ece),
            "n_samples": len(y_np),
        }

    def save(self, path: str):
        import json
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.weights, path)

    def load(self, path: str):
        self.weights = torch.load(path, weights_only=True)


class RiskBudgetAllocator:
    """Allocate precision actions to KV tokens minimizing risk UCB under budget.

    Solves: min_{a_i} Σ_i [μ_{i,a_i} + β σ_{i,a_i}]
    subject to: Σ_i cost(a_i) ≤ B
    """

    def __init__(self, beta: float = 1.0, budget: float = 0.2):
        self.beta = beta
        self.budget = budget

    def allocate(
        self,
        risk_mu: Dict[str, torch.Tensor],
        risk_sigma: Dict[str, torch.Tensor],
        n_tokens: int,
    ) -> List[str]:
        """Allocate precision actions to tokens.

        Args:
            risk_mu: {action_name: [n_tokens] tensor of predicted risk means}
            risk_sigma: {action_name: [n_tokens] tensor of predicted risk uncertainties}
            n_tokens: Number of KV tokens

        Returns:
            List of action names, one per token
        """
        # Compute UCB score for each (token, action)
        ucb = {}
        for action in ACTION_LIST:
            if action in risk_mu:
                ucb[action] = risk_mu[action] + self.beta * risk_sigma[action]
            else:
                ucb[action] = torch.zeros(n_tokens)

        # Greedy allocation: start with all evicted, upgrade tokens with highest risk-per-bit
        actions = ["evict"] * n_tokens
        total_cost = 0.0
        max_cost = self.budget * n_tokens  # budget is fraction of FP16 cost

        # Compute ALL possible upgrade steps for all tokens
        upgrade_path = [("evict", "2bit"), ("2bit", "4bit"), ("4bit", "fp16")]

        candidates = []
        for token_idx in range(n_tokens):
            for from_action, to_action in upgrade_path:
                cost_delta = ACTION_COSTS[to_action] - ACTION_COSTS[from_action]
                risk_delta = ucb[from_action][token_idx] - ucb[to_action][token_idx]
                efficiency = risk_delta / max(cost_delta, 1e-8)
                candidates.append((efficiency.item(), token_idx, from_action, to_action, cost_delta))

        # Sort by risk-reduction efficiency (descending)
        candidates.sort(key=lambda x: -x[0])

        for efficiency, token_idx, from_action, to_action, cost_delta in candidates:
            if actions[token_idx] != from_action:
                continue
            if total_cost + cost_delta <= max_cost:
                actions[token_idx] = to_action
                total_cost += cost_delta

        logger.debug(
            f"Budget allocation: {sum(1 for a in actions if a == 'fp16')} fp16, "
            f"{sum(1 for a in actions if a == '4bit')} 4bit, "
            f"{sum(1 for a in actions if a == '2bit')} 2bit, "
            f"{sum(1 for a in actions if a == 'evict')} evict, "
            f"cost={total_cost:.2f}/{max_cost:.2f}"
        )
        return actions


@dataclass
class GateDecision:
    """Record of margin/uncertainty gate decision for a verification step."""

    step_idx: int
    margin: float
    mean_uncertainty: float
    margin_gate_active: bool
    uncertainty_gate_active: bool
    budget_base: float
    budget_adjusted: float


class MarginUncertaintyGate:
    """Adaptive budget gate based on verifier margin and risk uncertainty.

    When margin is low or uncertainty is high, increase the FP16 budget
    to reduce risk of acceptance degradation.
    """

    def __init__(
        self,
        budget_base: float = 0.2,
        margin_threshold: float = 2.0,
        uncertainty_threshold: float = 0.5,
        delta_margin: float = 0.1,
        delta_uncertainty: float = 0.05,
        budget_max: float = 0.5,
    ):
        self.budget_base = budget_base
        self.margin_threshold = margin_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.delta_margin = delta_margin
        self.delta_uncertainty = delta_uncertainty
        self.budget_max = budget_max

    def compute_budget(
        self,
        margin: float,
        mean_uncertainty: float,
        step_idx: int,
    ) -> GateDecision:
        """Compute adjusted budget for this verification step."""
        margin_active = margin < self.margin_threshold
        unc_active = mean_uncertainty > self.uncertainty_threshold

        budget = self.budget_base
        if margin_active:
            budget += self.delta_margin
        if unc_active:
            budget += self.delta_uncertainty
        budget = min(budget, self.budget_max)

        return GateDecision(
            step_idx=step_idx,
            margin=margin,
            mean_uncertainty=mean_uncertainty,
            margin_gate_active=margin_active,
            uncertainty_gate_active=unc_active,
            budget_base=self.budget_base,
            budget_adjusted=budget,
        )
