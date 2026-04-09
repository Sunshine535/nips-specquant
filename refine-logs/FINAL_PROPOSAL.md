# Research Proposal: AcceptSpec — Acceptance-Preserving KV Cache Management for Speculative Decoding of Reasoning Models

**Version**: 2.0 (Apr 9, 2026 — full re-refine from idea discovery)
**Target**: NeurIPS 2026 (deadline: May 2026)

## Problem Anchor

- **Bottom-line problem**: Long-CoT reasoning generates thousands of thinking tokens. KV cache consumes 88-100% of inference memory and dominates verification latency. SD and KV compression are applied independently, leaving joint gains on the table.
- **Must-solve bottleneck**: During speculative verification, the verifier loads full KV for ALL tokens, but acceptance depends on only a sparse subset. Compressing non-critical KV reduces verification bandwidth without hurting acceptance.
- **Constraints**: Training-free, NeurIPS 2026, ≤150 GPU-hours on available GPUs (auto-adaptive), Qwen3-8B primary (standard MHA).
- **Success condition**: (1) Demonstrate acceptance sensitivity sparsity (top-20% tokens → >80% sensitivity), (2) Show acceptance-optimal ≠ perplexity-optimal ≠ attention-optimal KV retention, (3) Beat naive SD+KV composition by ≥10% latency.

## Method Thesis

Speculative verification's acceptance decision has sparse KV sensitivity: a small fraction of "acceptance-critical" tokens disproportionately influence whether the verifier accepts a drafted block. By identifying these tokens and compressing the rest, we achieve principled KV compression specifically designed for the SD pipeline.

## Dominant Contribution

**Discovery**: Acceptance-critical tokens in speculative decoding are a distinct category from attention-important tokens (used by SmallKV, H2O, SnapKV) and perplexity-sensitive tokens. The right optimization target for KV compression in SD is acceptance rate, not perplexity or attention.

**Contribution hierarchy**:
1. **Primary (conceptual)**: Empirical validation that acceptance sensitivity ≠ perplexity sensitivity ≠ attention sensitivity for KV in SD — a new finding about the structure of speculative verification
2. **Secondary (algorithmic)**: AcceptPredictor — zero-overhead predictor using draft attention patterns
3. **Tertiary (systems)**: End-to-end integration showing the finding translates to real speedups
4. **Quaternary (universality)**: Cross-model validation (Qwen3 + Llama-3.1) demonstrating this is a general property

**Paper template**: Discovery paper with practical implications (cf. H2O, SnapKV, Attention Sink).

## Core Mechanism

### 1. Acceptance Sensitivity Measurement (Empirical Proposition)

**Empirical Proposition**: For reasoning LLMs under speculative decoding, acceptance sensitivity is concentrated: top-20% of tokens (ranked by acceptance sensitivity) account for >80% of total sensitivity mass.

**Definition** (block-level, coupled randomness):
- Draft model proposes γ tokens. For position j ∈ {1,...,γ}, accept if U_j < p_target(x_j|x_{<j}, KV) / p_draft(x_j|x_{<j})
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

### 4. Integration with Speculative Decoding

1. Draft model generates γ tokens (vanilla SD or EAGLE-3)
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
| C2: Accept ≠ perplexity ≠ attention | Rank correlation | Spearman ρ | <0.7 (accept vs perplexity), <0.7 (accept vs attention) |
| C3: Accept-targeted beats perplexity-targeted AND attention-targeted | Same-budget comparison | Accuracy gap | ≥3pp over perplexity, ≥3pp over attention (SmallKV-style) |
| C4: Beats naive composition | E2E benchmark | Latency improvement | ≥10% |
| C5: Predictor accuracy | Oracle vs learned | F1 of critical set | >0.75 |
| C6: Universal property | Cross-model validation | Pattern consistency | Sparsity holds for Qwen3 + Llama |

## Baselines (updated Apr 9)

1. **Vanilla autoregressive** (no SD, no compression)
2. **SD only** (vanilla rejection sampling, γ=5)
3. **EAGLE-3** (tree-based SD, if EAGLE head available for Qwen3)
4. **R-KV** (NeurIPS 2025, redundancy-aware KV compression, GitHub: Zefan-Cai/R-KV)
5. **SmallKV** (NeurIPS 2025 Spotlight, SLM attention-proxy KV compression) — KEY COMPARISON
6. **SD + R-KV** (naive composition)
7. **SD + SmallKV** (naive composition with attention proxy)
8. **QuantSpec** (self-speculative, 4-bit KV+weights)
9. **AcceptSpec (ours)**

Note: ThinKV dropped due to no public code. R-KV (NeurIPS'25, has code) replaces it as the KV compression baseline.

## Evaluation Structure (oracle-first)

**Phase 1: Oracle Study** — Validate sparsity on 100 GSM8K. ABORT if top-20% < 60% sensitivity.
**Phase 2: Triple Divergence** — Show accept ≠ perplexity ≠ attention ranking. Spearman ρ < 0.7 for all pairs.
**Phase 3: Objective Comparison** — Fixed budget, compare all retention policies including SmallKV-style. Core result.
**Phase 4: System Benchmark** — E2E vs all baselines with wall-clock profiling.
**Phase 5: Robustness** — τ ∈ {0,0.3,0.6,0.9}, γ ∈ {3,5,7,10}, difficulty strata, 3 seeds.
**Phase 6: Universality** — Llama-3.1-8B oracle + E2E. Qwen3.5-9B (hybrid MHA subset).

## Models (unified, Apr 9)

**Primary**:
- Draft: Qwen/Qwen3-0.6B (MHA, 0.6B)
- Target: Qwen/Qwen3-8B (MHA, 8B)

**Cross-architecture**:
- Draft: meta-llama/Llama-3.1-8B
- Target: meta-llama/Llama-3.1-70B (or 3.2-3B → 3.1-8B for compute)

**Hybrid architecture check**:
- Draft: Qwen/Qwen3.5-0.8B (GatedDeltaNet hybrid)
- Target: Qwen/Qwen3.5-9B (GatedDeltaNet hybrid, MHA subset only)

## Risks and Mitigations

1. **Sparsity doesn't hold** → Phase 1 catches in <10 GPU-hours. Report as negative result.
2. **Accept ≈ attention (SmallKV wins)** → Phase 2 catches. If ρ > 0.85, pivot to showing WHEN they diverge.
3. **Overhead negates savings** → Lazy batched compression. Measured in Phase 4.
4. **EAGLE-3 head unavailable for Qwen3** → Use vanilla SD as primary, note EAGLE-3 as future work.
5. **Qwen3.5 hybrid breaks assumptions** → Expected: only test MHA layers, note linear-attn layers are separate.

## Compute: ~150 GPU-hours on available GPUs (auto-adaptive). Timeline: 2-3 weeks.

## Complexity Intentionally Rejected
- Custom segment-level SD engine (use vanilla SD + optional EAGLE-3)
- Theoretical acceptance bound (empirical proposition instead)
- RL-based predictor training (simple logistic regression sufficient)
- Qwen3.5 linear attention KV (only MHA layers relevant)

## What Changed from v1.0 (Apr 7)

| Change | Reason |
|--------|--------|
| Added SmallKV as explicit baseline | NeurIPS'25 Spotlight, closest competitor, reviewer will ask |
| Replaced ThinKV with R-KV | ThinKV has no public code; R-KV (NeurIPS'25) has GitHub |
| Unified models to Qwen3 (MHA) primary | Qwen3.5 GatedDeltaNet changes KV dynamics fundamentally |
| Added C6 (universality claim) | Cross-model validation strengthens "discovery" framing |
| Added triple divergence (C2 expanded) | Must show accept ≠ attention (not just ≠ perplexity) to beat SmallKV |
| EAGLE-3 as optional bonus, not required | EAGLE head availability uncertain for Qwen3 |
| Added sparse verification (2512.21911) to related work | Relevant but different mechanism |
