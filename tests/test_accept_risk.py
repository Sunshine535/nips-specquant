"""Tests for MARA: Margin-Calibrated Acceptance-Risk Allocation.

Tests cover:
  1. Risk predictor can overfit toy data
  2. Budget allocator respects constraints
  3. Margin/uncertainty gate activates correctly
  4. Risk labels have correct structure
  5. Coupled uniforms are deterministic
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.accept_risk import (
    ACTION_COSTS,
    AcceptanceRiskPredictor,
    GateDecision,
    MarginUncertaintyGate,
    RiskBudgetAllocator,
    RiskLabel,
    RiskLabelSet,
)
from src.repro import make_calib_eval_split, make_coupled_uniforms, set_global_seed


class TestAcceptanceRiskPredictor:
    def test_predict_shape(self):
        pred = AcceptanceRiskPredictor(input_dim=5, hidden_dim=16)
        X = torch.randn(10, 5)
        mu, sigma = pred.predict(X)
        assert mu.shape == (10,)
        assert sigma.shape == (10,)
        assert (sigma > 0).all(), "Sigma must be positive"

    def test_overfit_toy_data(self):
        """Predictor should be able to overfit a simple pattern."""
        torch.manual_seed(42)
        pred = AcceptanceRiskPredictor(input_dim=5, hidden_dim=32)

        # Create toy data: risk = sum of features (simple linear pattern)
        X = torch.randn(50, 5)
        y = X.sum(dim=1).abs()  # positive risk values

        losses = pred.fit(X, y, lr=0.01, epochs=200)
        assert losses["loss_total"] < 1.0, f"Should overfit toy data, got loss={losses['loss_total']}"

    def test_ranking_preserved(self):
        """After fitting, predicted ranking should roughly match true ranking."""
        torch.manual_seed(42)
        pred = AcceptanceRiskPredictor(input_dim=5, hidden_dim=32)

        X = torch.randn(30, 5)
        y = X[:, 0].abs() + 0.5 * X[:, 1].abs()  # risk depends on features 0,1

        pred.fit(X, y, lr=0.01, epochs=300, lambda_rank=0.5)
        mu, _ = pred.predict(X)

        # Check Spearman correlation is positive
        from scipy.stats import spearmanr
        rho, _ = spearmanr(mu.detach().numpy(), y.numpy())
        assert rho > 0.3, f"Ranking should be partially preserved, got rho={rho}"

    def test_calibration_metrics(self):
        """Calibration metrics should be finite and non-NaN."""
        torch.manual_seed(42)
        pred = AcceptanceRiskPredictor(input_dim=5, hidden_dim=16)

        X = torch.randn(20, 5)
        y = X.sum(dim=1).abs()
        pred.fit(X, y, lr=0.01, epochs=50)

        metrics = pred.compute_calibration_metrics(X, y)
        assert not np.isnan(metrics["spearman_rho"]), "Spearman should not be NaN"
        assert not np.isnan(metrics["ece"]), "ECE should not be NaN"
        assert metrics["n_samples"] == 20

    def test_empty_data(self):
        """Should handle empty training data gracefully."""
        pred = AcceptanceRiskPredictor(input_dim=5, hidden_dim=16)
        X = torch.empty(0, 6)
        y = torch.empty(0)
        losses = pred.fit(X, y)
        assert np.isnan(losses["loss"])


class TestRiskBudgetAllocator:
    def test_budget_respected(self):
        """Allocator must not exceed budget."""
        allocator = RiskBudgetAllocator(beta=1.0, budget=0.2)
        n = 100

        risk_mu = {
            "fp16": torch.zeros(n),
            "4bit": torch.rand(n) * 0.5,
            "2bit": torch.rand(n),
            "evict": torch.rand(n) * 2.0,
        }
        risk_sigma = {a: torch.rand(n) * 0.1 for a in risk_mu}

        actions = allocator.allocate(risk_mu, risk_sigma, n)
        assert len(actions) == n

        total_cost = sum(ACTION_COSTS[a] for a in actions)
        max_cost = 0.2 * n
        assert total_cost <= max_cost + 1e-6, f"Budget exceeded: {total_cost} > {max_cost}"

    def test_high_budget_mostly_fp16(self):
        """With high budget, most tokens should be FP16."""
        allocator = RiskBudgetAllocator(beta=1.0, budget=0.9)
        n = 20

        risk_mu = {
            "fp16": torch.zeros(n),
            "4bit": torch.ones(n) * 0.5,
            "2bit": torch.ones(n),
            "evict": torch.ones(n) * 2.0,
        }
        risk_sigma = {a: torch.ones(n) * 0.01 for a in risk_mu}

        actions = allocator.allocate(risk_mu, risk_sigma, n)
        fp16_count = sum(1 for a in actions if a == "fp16")
        assert fp16_count > n * 0.5, f"With 90% budget, should have many FP16, got {fp16_count}/{n}"

    def test_zero_budget_all_evict(self):
        """With zero budget, all tokens should be evicted."""
        allocator = RiskBudgetAllocator(beta=1.0, budget=0.0)
        n = 10

        risk_mu = {"fp16": torch.zeros(n), "evict": torch.ones(n)}
        risk_sigma = {a: torch.ones(n) * 0.1 for a in risk_mu}

        actions = allocator.allocate(risk_mu, risk_sigma, n)
        assert all(a == "evict" for a in actions)


class TestMarginUncertaintyGate:
    def test_no_gate_activation(self):
        """With high margin and low uncertainty, no gate should activate."""
        gate = MarginUncertaintyGate(
            budget_base=0.2, margin_threshold=2.0, uncertainty_threshold=0.5
        )
        decision = gate.compute_budget(margin=5.0, mean_uncertainty=0.1, step_idx=0)
        assert not decision.margin_gate_active
        assert not decision.uncertainty_gate_active
        assert decision.budget_adjusted == 0.2

    def test_margin_gate_activates(self):
        """Low margin should activate margin gate and increase budget."""
        gate = MarginUncertaintyGate(
            budget_base=0.2,
            margin_threshold=2.0,
            delta_margin=0.1,
        )
        decision = gate.compute_budget(margin=1.0, mean_uncertainty=0.1, step_idx=0)
        assert decision.margin_gate_active
        assert decision.budget_adjusted > 0.2

    def test_uncertainty_gate_activates(self):
        """High uncertainty should activate uncertainty gate."""
        gate = MarginUncertaintyGate(
            budget_base=0.2,
            uncertainty_threshold=0.5,
            delta_uncertainty=0.05,
        )
        decision = gate.compute_budget(margin=5.0, mean_uncertainty=1.0, step_idx=0)
        assert decision.uncertainty_gate_active
        assert decision.budget_adjusted > 0.2

    def test_both_gates_capped(self):
        """Both gates active should not exceed budget_max."""
        gate = MarginUncertaintyGate(
            budget_base=0.2,
            delta_margin=0.2,
            delta_uncertainty=0.2,
            budget_max=0.5,
        )
        decision = gate.compute_budget(margin=0.1, mean_uncertainty=10.0, step_idx=0)
        assert decision.margin_gate_active
        assert decision.uncertainty_gate_active
        assert decision.budget_adjusted <= 0.5


class TestRiskLabelSet:
    def test_to_tensors(self):
        labels = [
            RiskLabel(
                step_idx=0, token_idx=i, action="4bit",
                alpha_full=0.8, alpha_perturbed=0.8 - i * 0.1,
                acceptance_risk=i * 0.1,
                tv_distance=0.01 * i,
                margin_full=3.0, margin_perturbed=2.5,
                margin_risk=0.0,
            )
            for i in range(5)
        ]
        rls = RiskLabelSet(labels=labels)
        X, y = rls.to_tensors()
        assert X.shape == (5, 5), f"Expected (5, 5), got {X.shape}"
        assert y.shape == (5,)
        assert (y >= 0).all()

    def test_empty_set(self):
        rls = RiskLabelSet()
        X, y = rls.to_tensors()
        assert X.shape[0] == 0
        assert y.shape[0] == 0


class TestReproducibility:
    def test_coupled_uniforms_deterministic(self):
        gen1 = set_global_seed(42)
        u1 = make_coupled_uniforms(10, 5, gen1)

        gen2 = set_global_seed(42)
        u2 = make_coupled_uniforms(10, 5, gen2)

        assert torch.allclose(u1, u2), "Same seed must produce same uniforms"

    def test_different_seeds_different(self):
        gen1 = set_global_seed(42)
        u1 = make_coupled_uniforms(10, 5, gen1)

        gen2 = set_global_seed(123)
        u2 = make_coupled_uniforms(10, 5, gen2)

        assert not torch.allclose(u1, u2), "Different seeds must produce different uniforms"

    def test_split_no_overlap(self):
        split = make_calib_eval_split(100, calib_fraction=0.3, seed=42)
        overlap = set(split.calib_indices) & set(split.eval_indices)
        assert len(overlap) == 0, f"Calibration/eval overlap: {overlap}"
        assert len(split.calib_indices) + len(split.eval_indices) == 100

    def test_split_deterministic(self):
        s1 = make_calib_eval_split(100, calib_fraction=0.3, seed=42)
        s2 = make_calib_eval_split(100, calib_fraction=0.3, seed=42)
        assert s1.calib_indices == s2.calib_indices
        assert s1.eval_indices == s2.eval_indices
        assert s1.split_hash == s2.split_hash
