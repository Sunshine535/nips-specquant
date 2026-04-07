# Research Proposal: AcceptSpec — Acceptance-Preserving KV Cache Management for Speculative Decoding of Reasoning Models

## Problem Anchor

- **Bottom-line problem**: Long-CoT reasoning generates thousands of thinking tokens. KV cache consumes 88-100% of inference memory and dominates verification latency. SD and KV compression are applied independently, leaving joint gains on the table.
- **Must-solve bottleneck**: During speculative verification, the verifier loads full KV for ALL tokens, but acceptance depends on only a sparse subset. Compressing non-critical KV reduces verification bandwidth without hurting acceptance.
- **Constraints**: Training-free, NeurIPS 2026, ≤150 GPU-hours on 2×H100, Qwen3-8B primary.
- **Success condition**: (1) Demonstrate acceptance sensitivity sparsity (top-20% tokens → >80% sensitivity), (2) Show acceptance-optimal ≠ perplexity-optimal KV retention, (3) Beat naive EAGLE-3+ThinKV composition by ≥10% latency.

## Method Thesis

Speculative verification's acceptance decision has sparse KV sensitivity: a small fraction of "acceptance-critical" tokens disproportionately influence whether the verifier accepts a drafted block. By identifying these tokens and compressing the rest, we achieve principled KV compression specifically designed for the SD pipeline.

## Dominant Contribution

**Discovery**: Acceptance-critical tokens in speculative decoding are a distinct category from attention-important tokens in standard inference. The right optimization target for KV compression in SD is acceptance rate, not perplexity.

**Contribution hierarchy**:
1. **Primary (conceptual)**: Empirical validation that acceptance sensitivity ≠ perplexity sensitivity for KV in SD — a new finding about the structure of speculative verification
2. **Secondary (algorithmic)**: AcceptPredictor — zero-overhead predictor using draft attention patterns
3. **Tertiary (systems)**: End-to-end integration showing the finding translates to real speedups

**Paper template**: Discovery paper with practical implications (cf. H2O, SnapKV, Attention Sink).

## Core Mechanism

### 1. Acceptance Sensitivity Measurement (Empirical Proposition)

**Empirical Proposition**: For Qwen3-8B on reasoning tasks, acceptance sensitivity is concentrated: top-20% of tokens (ranked by acceptance sensitivity) account for >80% of total sensitivity mass.

**Definition** (block-level, coupled randomness):
- EAGLE-3 proposes γ draft tokens. For position j ∈ {1,...,γ}, accept if U_j < p_target(x_j|x_{<j}, KV) / p_draft(x_j|x_{<j})
- α(KV, U) = expected accepted tokens / γ
- S_accept(i) = |α(KV_full, U) - α(KV_{i→2bit}, U)| averaged over U samples
- Paired measurement with same U samples → clean, low-noise oracle

### 2. AcceptPredictor (exact algorithm)

From the draft model's forward pass, for each token i in KV cache:
```
score(i) = Σ_{h=1}^{H} w_h · a_h(q_draft, k_i) · ||v_i||_2
```
- a_h(q_draft, k_i): draft attention weight (already computed in draft forward pass)
- ||v_i||_2: value norm (pre-stored, one-time cost)
- w_h: per-head weight (logistic regression on 50 calibration examples)
- Cost: ≈0, piggybacks on draft attention computation

**Decision**: score(i) > θ → critical (FP16), θ_low < score(i) ≤ θ → moderate (4-bit), score(i) ≤ θ_low → non-critical (2-bit or evict). Thresholds calibrated offline.

### 3. Mixed-Precision KV Implementation

Per-token 2-bit tag: 00=evicted, 01=2-bit, 10=4-bit, 11=FP16.
Custom attention kernel: reads tags, dispatches to appropriate dequant path.
Extension of ZipCache/KVTuner mixed-precision kernel designs.

### 4. Integration with EAGLE-3

1. EAGLE-3 generates γ draft tokens
2. Extract draft attention weights → compute AcceptPredictor scores (zero extra cost)
3. Compress non-critical KV: Hadamard rotation + scalar quant (TurboQuant)
4. Target model verifies with mixed-precision KV
5. Accepted tokens: KV enters cache at FP16, scored next round
6. Rejected tokens: KV discarded

### 5. Cost Model

```
T_acceptspec = T_draft(γ) + T_score(≈0) + T_compress(N_noncrit) + T_verify(γ, mixed-precision KV)
Speedup when: T_compress(N_noncrit) < T_verify_savings(N_noncrit × bandwidth_reduction)
```

## Key Claims

| Claim | Experiment | Metric | Threshold |
|-------|-----------|--------|-----------|
| C1: Sparsity | Oracle masking sweep | Acceptance at 20% retention | >80% of full |
| C2: Accept ≠ perplexity | Rank correlation | Spearman ρ | <0.7 |
| C3: Accept-targeted beats perplexity-targeted | Same-budget comparison | Accuracy gap | ≥3pp |
| C4: Beats naive composition | E2E benchmark | Latency improvement | ≥10% |
| C5: Predictor accuracy | Oracle vs learned | F1 of critical set | >0.75 |

## Evaluation Structure (oracle-first)

**Phase 1: Oracle Study** — Validate sparsity on 100 GSM8K. ABORT if top-20% < 60% sensitivity.
**Phase 2: Predictor Validation** — Train on 50 examples, test on 50. Fallback if F1 < 0.6.
**Phase 3: Objective Comparison** — Fixed budget, compare retention policies. Core result.
**Phase 4: System Benchmark** — E2E vs baselines with wall-clock profiling.
**Phase 5: Robustness** — τ ∈ {0,0.3,0.6,0.9}, γ ∈ {3,5,7,10}, difficulty strata, 3 seeds.
**Phase 6: Generalization** — Qwen3-14B on GSM8K + MATH-500.

## Baselines
- Vanilla autoregressive
- EAGLE-3 (best SD)
- ThinKV (best reasoning KV compression)
- EAGLE-3 + ThinKV (naive composition)
- QuantSpec (SD + uniform 4-bit)

## Risks and Mitigations
1. **Sparsity doesn't hold** → Phase 1 catches in <10 GPU-hours. Report negative result.
2. **Predictor inaccurate** → Phase 2 catches in <15 GPU-hours. Fallback to SpecAttn-style.
3. **Overhead negates savings** → Lazy batched compression. Measured in Phase 4.

## Compute: ~150 GPU-hours on 2×H100. Timeline: 2-3 weeks.

## Complexity Intentionally Rejected
- Custom segment-level SD engine (use EAGLE-3 off-the-shelf)
- Theoretical acceptance bound (empirical proposition instead)
- Cross-architecture evaluation (focus Qwen3, one-shot 14B check)
- RL-based predictor training (simple logistic regression sufficient)
