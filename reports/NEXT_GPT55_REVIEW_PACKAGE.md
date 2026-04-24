# Next GPT-5.5 Pro Review Package

## Summary of Changes

1. **MARA core implemented** (`src/accept_risk.py`): AcceptanceRiskOracle, AcceptanceRiskPredictor, RiskBudgetAllocator, MarginUncertaintyGate — all tested locally (25/25 pass)
2. **Reproducibility utilities** (`src/repro.py`): deterministic seeds, coupled uniforms, split manifests, run metadata
3. **Config** (`configs/mara_minimal.yaml`): A/B/C policy definitions + MARA hyperparameters
4. **Reliability audit** (`docs/RELIABILITY_AUDIT.md`): all old results marked with reliability status
5. **13 report files** created per GPT-5.5 requirements

## Git Diff Summary

- 8 new files (src/, configs/, tests/, docs/)
- 13 new report files (reports/)
- 1 bug fix (RiskBudgetAllocator greedy path)
- 0 existing source files modified
- 0 raw results altered

## Commands Run

```bash
python3 -m pytest tests/test_accept_risk.py tests/test_data_metric_sanity.py -v
# Result: 25/25 PASS (3.44s)
```

## Result Tables

### Local Tests
All 25 tests pass. See reports/TEST_PLAN.md for full table.

### GPU Experiments
NOT YET RUN. Requires remote server. See reports/MINIMAL_EXPERIMENT_RESULTS.md.

## Mechanism Logs
NOT YET AVAILABLE. MARA mechanism logging is implemented in the code but requires GPU execution to produce real data.

## Failed Tests
None after bug fix.

## Unresolved Questions

1. **Does risk signal exist?** — MARA predictor has not been tested on real acceptance-risk labels from GPU experiments
2. **Does MARA beat A and B?** — A/B/C comparison not yet run
3. **Is Qwen3.5-9B viable?** — M3 showing 0% accuracy even for oracle at 20% budget
4. **Do P0 oracle/comparison bugs affect MARA conclusions?** — Bugs documented but not yet fixed in experiment scripts
5. **Can MARA beat SpecAttn/QuantSpec?** — Not yet compared

## Whether New Results Support Original Diagnosis

**Partially supported:**
- MARA core modules work correctly (local tests pass)
- Reliability audit confirms all old results are unreliable (supports diagnosis)
- Budget allocator correctly handles risk-based allocation (mechanism valid in toy setting)

**Not yet testable:**
- Whether calibrated risk actually improves acceptance retention (needs GPU)
- Whether margin/uncertainty gates help (needs real margin data)
- Whether MARA beats uniform/attention baselines (needs head-to-head comparison)

## What GPT-5.5 Pro Should Review Next

1. **`src/accept_risk.py`** — Is the risk predictor architecture sufficient? Is the loss function correct?
2. **`configs/mara_minimal.yaml`** — Are the default hyperparameters reasonable?
3. **`tests/test_accept_risk.py`** — Are the tests comprehensive enough?
4. **M3 results** — When available, does fp16_baseline accuracy > 0%?
5. **P0 bug fixes** — Should these be prioritized before MARA integration?
6. **Model choice** — Should we pivot from Qwen3.5-9B to Qwen3-8B before MARA experiments?
