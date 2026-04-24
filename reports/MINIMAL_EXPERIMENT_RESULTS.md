# Minimal Experiment Results

## Local Tests (No GPU Required)

| Experiment | Command | Config | Dataset | Seed | Metric | Result | Expected | Pass/Fail | Interpretation |
|------------|---------|--------|---------|------|--------|--------|----------|-----------|----------------|
| Risk predictor overfit | `pytest tests/test_accept_risk.py` | toy | synthetic | 42 | loss | < 1.0 | < 1.0 | PASS | Predictor architecture can learn |
| Risk predictor ranking | same | toy | synthetic | 42 | Spearman ρ | > 0.3 | > 0.3 | PASS | Ranking signal preserved |
| Calibration metrics | same | toy | synthetic | 42 | ECE/ρ | finite | non-NaN | PASS | Metrics computable |
| Budget constraint | same | toy | synthetic | — | cost ≤ budget | respected | respected | PASS | Allocator correct |
| Gate activation | same | toy | synthetic | — | gate flags | correct | correct | PASS | Gates fire appropriately |
| Data split sanity | `pytest tests/test_data_metric_sanity.py` | — | various | 42 | overlap | 0 | 0 | PASS | No leakage |
| GSM8K metric | same | — | known answers | — | exact match | correct | correct | PASS | Answer extraction works |

## GPU-Required Experiments (Pending Remote Execution)

| Experiment | Command | Config | Dataset | Seed | Status |
|------------|---------|--------|---------|------|--------|
| MTP loading check | `python3 scripts/check_mtp_loading.py --model Qwen/Qwen3.5-9B` | minimal | 1 prompt | 42 | NOT RUN (needs GPU) |
| Corrected oracle smoke | `python3 scripts/oracle_sensitivity.py --num_problems 2 --max_tokens 32` | fixed oracle | GSM8K | 42 | NOT RUN (needs GPU) |
| MARA calibration smoke | `python3 scripts/calibrate_mara.py --config configs/mara_minimal.yaml --num_calib 4` | MARA | GSM8K | 42 | NOT RUN (needs GPU) |
| A/B/C comparison | `python3 scripts/core_comparison.py --policies existing_best_fragment_only,mara_no_gate_or_uncertainty,mara_full` | MARA | GSM8K | 42 | NOT RUN (needs GPU) |

**Note**: GPU experiments require remote server execution. Commands will be provided after code push.
