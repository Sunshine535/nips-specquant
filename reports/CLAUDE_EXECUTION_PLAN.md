# Claude Code Execution Plan

**Date**: 2026-04-24
**GPT-5.5 Diagnosis**: Found at `/home/tarkoy/nips/nips-specquant/GPT55_DIAGNOSIS.md`

## 1. MAIN METHOD PATH

**MARA — Margin-Calibrated Acceptance-Risk Allocation**

Instead of predicting sparse critical KV tokens, estimate calibrated per-token per-action acceptance-risk with uncertainty, then allocate precision budget via risk upper-confidence bound with margin/uncertainty gating.

## 2. Missing Mechanism to Implement

**Calibrated Acceptance-Risk Control**: 
- Continuous risk labels (not binary critical/non-critical)
- Calibrated risk predictor outputting (μ, σ)
- Risk-based budget allocator with UCB
- Margin/uncertainty safety gates

## 3. Current Evidence Supporting Diagnosis

- M0/M1 gates FAILED (hard sparsity not proven)
- M2 predictor F1=0.233 (FAILED)
- M2 margin ρ=0.017 (near zero, margin alone not predictive)
- Attention proxy F1=0.984 (anomalous, likely artifact)
- Old benchmark: higher acceptance → slower throughput
- ThinkCompress: adaptive ≈ uniform

## 4. Current Evidence Contradicting/Weakening Diagnosis

- M2 Spearman ρ(accept,ppl)=-0.305 shows SOME accept-perplexity divergence (but may be artifacted)
- M1 shard1 top20=0.70 suggests moderate sparsity exists in some contexts (but not hard)
- M3 still running — fp16_baseline result pending (if fp16 also 0%, problem is model/budget, not method)
- Qwen3.5-9B is 75% linear attention — may be wrong model for KV studies regardless of method

## 5. Files to Inspect

- [x] `src/acceptspec.py` — current predictor, oracle, mixed precision
- [x] `src/speculative_decode.py` — MTP loop
- [x] `scripts/oracle_sensitivity.py` — P0 kv_len bug
- [x] `scripts/core_comparison.py` — P0 MTP path bug  
- [x] `scripts/triple_divergence.py` — better MTP path
- [x] `configs/default.yaml` — current settings
- [x] `results/acceptspec/*` — all result JSONs
- [x] `GPT55_DIAGNOSIS.md` — full diagnosis

## 6. Files to Edit

| File | Change Type | Reason |
|------|-------------|--------|
| `src/accept_risk.py` | NEW | MARA core module |
| `src/repro.py` | NEW | Seed/split/metadata utilities |
| `scripts/calibrate_mara.py` | NEW | MARA calibration script |
| `configs/mara_minimal.yaml` | NEW | MARA config |
| `tests/test_accept_risk.py` | NEW | MARA tests |
| `tests/test_mtp_invariants.py` | NEW | KV length + MTP path tests |
| `tests/test_data_metric_sanity.py` | NEW | Split/metric sanity |
| `docs/RELIABILITY_AUDIT.md` | NEW | Result reliability manifest |

## 7. Files to Archive

| File | Reason |
|------|--------|
| `src/thinkcompress.py` | Historical negative evidence (adaptive ≈ uniform) |
| Results marked unreliable in RELIABILITY_AUDIT.md | Bug-contaminated, kept in place with annotations |

## 8. Files NOT to Touch

| File | Reason |
|------|--------|
| `src/speculative_decode.py` | Core MTP loop is correct; only add shared helper extraction |
| `src/mtp_head.py` | Working as intended |
| `src/turboquant_kv.py` | Quantization backend, correct |
| `src/baselines.py` | Baselines must not be weakened |
| Raw result JSONs | Must not alter raw data |
| Raw log files | Must not alter historical logs |

## 9. Tests to Run Before/After

| Test | Before | After | Purpose |
|------|--------|-------|---------|
| `pytest tests/` | Run existing | Must still pass | No regressions |
| `pytest tests/test_accept_risk.py` | N/A | Must pass | MARA core works |
| `pytest tests/test_mtp_invariants.py` | N/A | Should expose bugs → then fix | MTP correctness |
| `pytest tests/test_data_metric_sanity.py` | N/A | Must pass | No data leakage |

## 10. Rollback Conditions

- If MARA implementation changes baseline definitions → revert
- If tests require GPU/large models to run → make them mockable
- If existing test suite breaks → fix before proceeding
- If reliability manifest touches raw results → revert to annotation-only

## Execution Order

1. Create `docs/RELIABILITY_AUDIT.md` (Task 1)
2. Create `src/repro.py` — seed/split/metadata (Task 2)
3. Create `tests/test_mtp_invariants.py` — KV/MTP tests (Task 3)
4. Create `src/accept_risk.py` — MARA core (Task 6)
5. Create `configs/mara_minimal.yaml` (Task 7)
6. Create `scripts/calibrate_mara.py` (Task 7)
7. Create `tests/test_accept_risk.py` (Task 6)
8. Create `tests/test_data_metric_sanity.py` (Task 9)
9. Create `reports/KEEP_REWRITE_ARCHIVE_PLAN.md` (Step 4)
10. Run all tests locally
11. Create remaining reports
12. Prepare remote execution commands
