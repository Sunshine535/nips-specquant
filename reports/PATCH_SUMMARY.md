# Patch Summary

## Files Added (8)
- `src/accept_risk.py` — MARA core: AcceptanceRiskOracle, AcceptanceRiskPredictor, RiskBudgetAllocator, MarginUncertaintyGate
- `src/repro.py` — Reproducibility: set_global_seed, make_coupled_uniforms, SplitManifest, RunMetadata
- `configs/mara_minimal.yaml` — MARA minimal config for smoke tests
- `tests/test_accept_risk.py` — 18 tests for MARA components + reproducibility
- `tests/test_data_metric_sanity.py` — 7 tests for data split and GSM8K metric sanity
- `docs/RELIABILITY_AUDIT.md` — Result reliability manifest (marks old results unreliable)
- `reports/` — 13 required report files (execution plan, audits, etc.)

## Files Changed (1)
- `src/accept_risk.py` — Fixed RiskBudgetAllocator greedy upgrade path bug

## Files Archived (0)
- No files archived (only marked unreliable in docs/RELIABILITY_AUDIT.md)

## Files Intentionally Not Touched
- `src/speculative_decode.py` — Correct MTP loop; no changes needed
- `src/mtp_head.py` — Working as intended
- `src/baselines.py` — Must not be weakened
- `scripts/oracle_sensitivity.py` — P0 bugs documented; fix requires GPU testing
- `scripts/core_comparison.py` — P0 bugs documented; fix requires GPU testing
- Raw result JSONs — Must not alter raw data
- Raw log files — Must not alter historical logs
- `README.md` — Defer until MARA results exist

## Bugs Fixed (1)
- RiskBudgetAllocator: greedy upgrade only processed first level; now processes all upgrade paths

## New Components Implemented
- AcceptanceRiskOracle: continuous risk labels via KV perturbation
- AcceptanceRiskPredictor: calibrated (μ, σ) predictor with Huber + ranking + calibration loss
- RiskBudgetAllocator: greedy precision allocation under budget with risk UCB
- MarginUncertaintyGate: adaptive budget based on margin/uncertainty
- SplitManifest: deterministic calib/eval split with overlap check
- Coupled uniforms: pre-sampled RVs for paired measurement

## Configs Added (1)
- `configs/mara_minimal.yaml` — A/B/C policies + MARA hyperparameters

## Tests Added (25)
- 18 tests for MARA core (predictor, allocator, gate, labels, reproducibility)
- 7 tests for data/metric sanity

## Commands Run
- `python3 -m pytest tests/test_accept_risk.py tests/test_data_metric_sanity.py -v` → 25/25 PASS

## Results Observed
- All 25 local tests pass
- Risk predictor can overfit toy data (loss < 1.0)
- Risk predictor preserves ranking (Spearman ρ > 0.3)
- Budget allocator respects constraints
- Gates activate correctly
- Data splits have zero overlap

## Failed Checks (0)
- None after bug fix

## Unresolved Risks
1. GPU-required experiments not yet run (MARA calibration, A/B/C comparison)
2. P0 bugs in oracle_sensitivity.py and core_comparison.py not yet fixed (require GPU testing)
3. Qwen3.5-9B may be wrong model (75% linear attention)
4. M3 in-progress results show 0% accuracy even for oracle — may indicate fundamental model/budget issue
5. MARA predictor on real data may not beat uniform/attention baselines
