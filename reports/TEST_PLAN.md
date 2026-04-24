# Test Plan

| Test | Purpose | Command | Expected Result | Status |
|------|---------|---------|----------------|--------|
| Risk predictor shape | Predict outputs correct dimensions | `pytest tests/test_accept_risk.py::TestAcceptanceRiskPredictor::test_predict_shape` | Pass | PASS |
| Risk predictor overfit | Can overfit toy data | `pytest tests/test_accept_risk.py::TestAcceptanceRiskPredictor::test_overfit_toy_data` | loss < 1.0 | PASS |
| Risk predictor ranking | Ranking preserved after fit | `pytest tests/test_accept_risk.py::TestAcceptanceRiskPredictor::test_ranking_preserved` | ρ > 0.3 | PASS |
| Calibration metrics | ECE/Spearman finite | `pytest tests/test_accept_risk.py::TestAcceptanceRiskPredictor::test_calibration_metrics` | No NaN | PASS |
| Empty data handling | Graceful empty input | `pytest tests/test_accept_risk.py::TestAcceptanceRiskPredictor::test_empty_data` | Pass | PASS |
| Budget respected | Allocator stays within budget | `pytest tests/test_accept_risk.py::TestRiskBudgetAllocator::test_budget_respected` | cost ≤ budget | PASS |
| High budget FP16 | 90% budget → many FP16 | `pytest tests/test_accept_risk.py::TestRiskBudgetAllocator::test_high_budget_mostly_fp16` | >50% FP16 | PASS |
| Zero budget evict | 0% budget → all evicted | `pytest tests/test_accept_risk.py::TestRiskBudgetAllocator::test_zero_budget_all_evict` | all evict | PASS |
| No gate activation | High margin, low unc → no gate | `pytest tests/test_accept_risk.py::TestMarginUncertaintyGate::test_no_gate_activation` | budget unchanged | PASS |
| Margin gate | Low margin → gate activates | `pytest tests/test_accept_risk.py::TestMarginUncertaintyGate::test_margin_gate_activates` | budget increases | PASS |
| Uncertainty gate | High uncertainty → gate activates | `pytest tests/test_accept_risk.py::TestMarginUncertaintyGate::test_uncertainty_gate_activates` | budget increases | PASS |
| Both gates capped | Both active → budget ≤ max | `pytest tests/test_accept_risk.py::TestMarginUncertaintyGate::test_both_gates_capped` | budget ≤ 0.5 | PASS |
| Risk labels tensor | Labels convert to tensors | `pytest tests/test_accept_risk.py::TestRiskLabelSet::test_to_tensors` | correct shape | PASS |
| Empty labels | Empty set handled | `pytest tests/test_accept_risk.py::TestRiskLabelSet::test_empty_set` | shape (0,) | PASS |
| Coupled uniforms deterministic | Same seed → same uniforms | `pytest tests/test_accept_risk.py::TestReproducibility::test_coupled_uniforms_deterministic` | exact match | PASS |
| Different seeds different | Different seeds → different | `pytest tests/test_accept_risk.py::TestReproducibility::test_different_seeds_different` | not equal | PASS |
| Split no overlap | Calib/eval disjoint | `pytest tests/test_accept_risk.py::TestReproducibility::test_split_no_overlap` | zero overlap | PASS |
| Split deterministic | Same seed → same split | `pytest tests/test_accept_risk.py::TestReproducibility::test_split_deterministic` | exact match | PASS |
| GSM8K answer extraction | Known answers parsed | `pytest tests/test_data_metric_sanity.py::TestGSM8KMetric` | all pass | PASS |
| Split sanity various sizes | No overlap at different N | `pytest tests/test_data_metric_sanity.py::TestSplitSanity` | all pass | PASS |

**Summary**: 25/25 tests PASS
