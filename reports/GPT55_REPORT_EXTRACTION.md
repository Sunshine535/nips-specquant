# GPT-5.5 Pro Report Extraction

## Diagnosis File Used

`/home/tarkoy/nips/nips-specquant/GPT55_DIAGNOSIS.md` (1155 lines, ~30K tokens)

## Recommended MAIN METHOD PATH

**MARA — Margin-Calibrated Acceptance-Risk Allocation**

In true MTP speculative decoding, do not predict "which KV tokens are sparse critical." Instead, estimate each KV token's calibrated risk under different precision actions (FP16/4-bit/2-bit/evict), then allocate precision budget via risk upper-confidence bound with margin/uncertainty gating.

Core formula:
```
R_{t,i,a} = expected acceptance loss + distribution shift + low-margin penalty
Allocation: min_{a_i} Σ_i [μ_{t,i,a_i} + β σ_{t,i,a_i}] s.t. Σ_i cost(a_i) ≤ B_t
Budget gate: B_t = B_base + ΔB_low_margin · 1[m_t < τ_m] + ΔB_unc · 1[mean σ_t > τ_σ]
```

## Missing Mechanism

**Calibrated Acceptance-Risk Control**: A calibrated risk model that estimates per-token, per-action acceptance degradation with uncertainty, and a closed-loop budget controller that uses margin/uncertainty to adapt precision allocation per verification step.

## Evidence From Positive Results

- Oracle perturbation direction is correct (directly measuring acceptance loss)
- Old benchmark confirms acceptance and compression strategy change behavior
- MTP implementation in `src/speculative_decode.py` is correct at the core

## Evidence From Negative Results

- M0/M1 gate FAILED: top-20%=27.5%/56%, Gini=0.137/0.12 (thresholds: 80%/0.5)
- M2 predictor F1 = 0.23-0.27 (FAILED, threshold 0.75)
- M2 AttentionProxy F1 = 0.98-0.996 (anomalously high, likely artifact)
- M3 no final results; core comparison broken (target-as-draft P0 bug)
- Old benchmark: specquant_4bit acceptance=0.888 but throughput=17.86 vs AR=39.51 (SLOWER)
- ThinkCompress adaptive ≈ uniform_4bit (generic importance doesn't help)

## Evidence From Unstable Results

- M1 shard0 top20=0.56 vs shard1 top20=0.70 (context-dependent sensitivity)
- M1 aggregate equals shard0 only (merge bug, not true 100-problem average)

## Evidence From Failed Ablations

- Current AcceptPredictor: binary classification with softmax weights, no calibration → F1 collapse
- Margin-only Jacobian heuristic: ρ≈0 with oracle acceptance → not a valid mechanism
- Attention-vs-acceptance divergence: mask artifact makes Spearman ρ≈0 compatible with null

## Why Existing Best Positive Fragment Is Insufficient

Even if some shard shows top20≈0.70, it cannot explain: predictor F1 failure, M3 missing, throughput negative result, aggregation bug, context variability, or why uniform matches adaptive.

## Files to Inspect

- `src/acceptspec.py` — predictor, oracle, mixed precision
- `src/speculative_decode.py` — MTP loop
- `scripts/oracle_sensitivity.py` — P0 kv_len bug
- `scripts/core_comparison.py` — P0 MTP path bug
- `scripts/triple_divergence.py` — better MTP path, merge
- `configs/default.yaml` — current settings
- `results/acceptspec/*` — contaminated results
- `logs/*` — experiment logs

## Files to Edit

- `src/accept_risk.py` — NEW: MARA core module
- `src/utils.py` — add seed/split/metadata utilities
- `scripts/oracle_sensitivity.py` — fix kv_len, matched-support
- `scripts/core_comparison.py` — fix MTP path, add A/B/C policies
- `scripts/calibrate_mara.py` — NEW: MARA calibration script
- `configs/mara_minimal.yaml` — NEW: MARA config
- `tests/test_accept_risk.py` — NEW: MARA tests
- `tests/test_acceptspec_core.py` — NEW: KV invariant tests
- `tests/test_mtp_policy_path.py` — NEW: MTP path tests

## Files to Archive

- `results/acceptspec/oracle_m0.json` — mark as historical/unreliable
- `results/acceptspec/oracle_m1.json` — mark as historical/unreliable (merge contaminated)
- `logs/M3_comparison_*.log` — mark as historical (broken MTP path)
- `src/thinkcompress.py` — archive as historical negative evidence

## Files to Keep

- `src/speculative_decode.py` — core MTP loop (KEEP, minor refactor)
- `src/mtp_head.py` — MTP head (KEEP)
- `src/gpu_auto.py` — model loading (KEEP)
- `src/turboquant_kv.py` — quantization backend (KEEP)
- `src/baselines.py` — comparison baselines (KEEP AS BASELINE)

## Files to Keep Only as Baseline

- `src/baselines.py` — RTN/KIVI/Absmax
- Attention proxy in `src/acceptspec.py`

## Files to Keep Only as Ablation

- Margin-only score
- Existing AcceptPredictor (predicted top-k)
- Oracle top-k ranking

## Suspected Bugs

| Priority | Location | Bug | Evidence |
|----------|----------|-----|----------|
| P0 | `scripts/oracle_sensitivity.py` ~L376-394 | kv_len not advanced after target forward on last_tok | Correct code in speculative_decode.py advances kv_len |
| P0 | `results/acceptspec/oracle_m1.json` | Aggregate copied from shard0, not recomputed | M1 top20 matches shard0=0.56, shard1=0.70 |
| P0 | `scripts/core_comparison.py` run_sd_with_policy | Policy path uses target-as-draft in MTP mode | SpeculativeDecoder aliases draft_model=target_model |
| P0 | `scripts/core_comparison.py` acceptspec_predicted | Builds fake 1-token draft KV, not real context | Features not aligned to actual KV/draft state |
| P1 | `scripts/oracle_sensitivity.py` ~L155-183 | Zero-filled aggregation distorts top-k metrics | sample_indices exist but script uses dense vectors |
| P1 | `src/acceptspec.py` | Attention importance silent fallback to zeros/uniform | Review notes uniform attention fallback |
| P1 | `src/acceptspec.py` MixedPrecisionKV | Simulates perturbation, not actual compressed storage | No real memory/latency reduction |
| P2 | `src/acceptspec.py` AcceptPredictor | Positive-only softmax, no bias, no calibration | M2 F1=0.23 failed badly |

## Required Logging

- Risk labels per token/action (continuous)
- Calibrated μ, σ predictions
- Margin gate activation rate
- Uncertainty gate activation rate
- Budget per step and realized precision histogram
- Position-wise acceptance (excluding position-0)
- Matched-support mask for all ranking metrics
- Model revision, seed, prompt IDs, gamma, temperature

## Required Minimal Experiments

1. Smoke test (unit tests pass)
2. Data split sanity (no calib/eval overlap)
3. Metric sanity (GSM8K known-answer)
4. One-batch risk overfit
5. MTP/checkpoint loading check
6. Corrected oracle reproduction
7. A: Existing Best Fragment Only
8. B: MARA without gate/uncertainty
9. C: Full MARA
10. Small baseline comparison (uniform, attention, KIVI)
11. 3-seed stability
12. Expansion gate (50-100 prompts)

## Required Core Comparison

A vs B vs C at equal KV budget, same prompts/seeds/MTP loop:
- A: existing_best_fragment_only (old AcceptPredictor top-k)
- B: mara_no_gate_or_uncertainty (risk allocation without margin/uncertainty)
- C: mara_full (complete MARA)

## Required Baselines

FP16, vanilla MTP, uniform 4-bit/2-bit, RTN, KIVI, KVQuant, H2O, SnapKV, R-KV, QuantSpec, SpecAttn (where feasible)

## Stop / Continue / Pivot Criteria

**Continue** if: C > A and C > B consistently, risk predictor calibrated, gains persist 3 seeds + GSM8K + MATH

**Stop** if: No predictable acceptance-risk signal; uniform/KIVI beats MARA consistently; C ≈ A/B

**Pivot** if: Risk exists but overhead dominates → quality-only paper; SpecAttn/QuantSpec covers mechanism → abandon novelty claim
