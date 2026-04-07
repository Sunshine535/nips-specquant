# Research Proposal: AcceptSpec — Acceptance-Preserving KV Cache Management for Speculative Decoding of Reasoning Models

## Problem Anchor

- **Bottom-line problem**: Long-chain-of-thought (CoT) reasoning in LLMs generates thousands of thinking tokens. The KV cache for these tokens consumes 88-100% of inference memory and dominates latency. Speculative decoding (SD) reduces latency but doesn't address KV memory. KV compression reduces memory but doesn't leverage SD's draft-verify structure. These two mature techniques are applied independently, leaving joint gains on the table.
- **Must-solve bottleneck**: During speculative verification of reasoning traces, the verifier loads the full KV cache for ALL previous tokens, but its acceptance decision is sensitive to only a sparse subset. Compressing non-critical KV would reduce verification bandwidth without hurting acceptance.
- **Constraints**: Training-free, NeurIPS 2026, ≤150 GPU-hours on 2×H100, Qwen3-8B/14B primary.
- **Success condition**: (1) AcceptSpec beats naive composition of best standalone SD + best standalone KV compression by ≥15% on latency, (2) ≥6x KV compression with <2% accuracy drop, (3) Formal demonstration that acceptance-optimal KV precision ≠ perplexity-optimal precision.

## Method Thesis

Speculative verification's acceptance decision has sparse KV sensitivity: a small fraction of cached tokens (acceptance-critical tokens) disproportionately influence whether the verifier accepts or rejects a drafted block. By formalizing this as an acceptance-preservation objective and predicting the critical set from draft dynamics, we achieve principled KV compression that is specifically designed for the SD pipeline — yielding gains beyond what independent SD + KV compression can achieve.

## Dominant Contribution

A formal framework showing that speculative acceptance has sparse KV dependence, plus a practical system that exploits this for joint latency + memory savings. The key insight is that the right optimization target for KV compression in SD is acceptance rate, not perplexity.

## Core Mechanism

### 1. Acceptance Sensitivity Analysis (Proposition 1)

Define the acceptance sensitivity of token i as:
  S_accept(i) = |α(KV_full) - α(KV_{mask_i})| / α(KV_full)

where α is the acceptance probability and KV_{mask_i} is the cache with token i's KV quantized to low precision.

**Proposition 1** (Sparse Acceptance Sensitivity): For transformer models with softmax attention, the acceptance sensitivity S_accept(i) is bounded by the attention weight the verifier assigns to token i. Specifically, if the verifier's attention to token i across all heads/layers is below threshold τ, then S_accept(i) ≤ f(τ, d, b) where f is a function of head dimension d and quantization bits b.

This means tokens that receive little verifier attention can be safely compressed without affecting acceptance.

### 2. Online Acceptance-Critical Prediction

During draft generation, we collect cheap signals that predict which tokens will be acceptance-critical for the upcoming verification:

- **Draft entropy**: Low-entropy draft tokens indicate high confidence → surrounding KV is less critical
- **Draft-target agreement trend**: Tokens in segments where draft and target historically agree well are less critical
- **Attention concentration**: Tokens in segments where draft's attention is highly concentrated on recent tokens (vs. spread over history) indicate the historical KV is less critical
- **Segment role**: Boundary tokens between thought segments and tokens introducing new information are more likely critical

These signals are available from the draft forward pass at zero additional cost.

### 3. Differential KV Compression Policy

Based on predicted acceptance-criticality:
- **Critical tokens (top-k%, predicted acceptance-sensitive)**: Keep at FP16 or 8-bit
- **Moderate tokens**: Quantize to 4-bit using Hadamard rotation + scalar quantization (TurboQuant-style)
- **Non-critical tokens (bulk of exploration/repetition)**: Quantize to 2-bit or evict entirely
- **Rejected draft segments**: Discard KV entirely (free cleanup)

The fraction f_critical is adaptive: harder problems with more diverse reasoning get larger critical sets.

### 4. Segment-Level Scheduling

Integrated with segment-level SD (SpecCoT-style):
1. Draft model generates a thought-level segment (guided by <think> structure)
2. Before verification: compress non-critical KV from previous segments
3. Verify segment with mixed-precision KV: full precision for critical tokens + compressed for rest
4. Accept: compress accepted segment's KV to long-term format
5. Reject: discard segment's KV entirely
6. Repeat until </think>

### 5. Formal Cost Model

Total latency per segment:
  T_total = T_draft(γ) + T_verify(N_crit × C_full + N_rest × C_compressed) + T_compress

Joint benefit condition: AcceptSpec outperforms naive composition when:
  f_critical < (T_verify_full - T_compress) / (T_verify_full × (1 - C_compressed/C_full))

i.e., when the fraction of critical tokens is small enough that compression savings exceed compression overhead.

## Key Claims

| Claim | Evidence Required |
|-------|-------------------|
| C1: Acceptance sensitivity is sparse — <20% of tokens are acceptance-critical | Empirical measurement on Qwen3-8B/14B across GSM8K/MATH/AIME |
| C2: Acceptance-optimal precision ≠ perplexity-optimal precision | Head-to-head comparison of which tokens each objective selects |
| C3: AcceptSpec beats naive SpecCoT+ThinKV composition by ≥15% | Controlled experiment with best standalone components |
| C4: ≥6x KV compression with <2% accuracy drop | Main result table across 4+ benchmarks |
| C5: Cost model accurately predicts when joint scheduling helps | Predicted vs actual benefit across settings |

## Evaluation Plan

### Primary
1. **Sparsity validation**: Measure S_accept(i) distribution on Qwen3-8B across GSM8K/MATH — confirm <20% tokens are critical
2. **Acceptance vs perplexity**: Compare token rankings by acceptance-sensitivity vs perplexity-sensitivity — show divergence
3. **Main result**: AcceptSpec vs baselines on GSM8K-full, MATH-500, AIME-2024, HumanEval, GPQA-Diamond
4. **Critical ablation**: AcceptSpec vs naive SpecCoT+ThinKV composition (MUST show gap ≥15%)

### Secondary
5. **Oracle study**: Oracle acceptance-critical mask → upper bound
6. **Predictor accuracy**: How well do draft signals predict true acceptance-critical set?
7. **f_critical analysis**: How does critical fraction vary by task difficulty?
8. **Cost model validation**: Predicted vs actual latency across settings

### Baselines
- Vanilla autoregressive
- EAGLE-3 (best SD baseline)
- SpecCoT (segment-level SD)
- ThinKV (thought-aware KV compression)
- QuantSpec (SD + uniform 4-bit KV)
- Naive composition: SpecCoT + ThinKV with default settings

## Compute: ~150 GPU-hours on 2×H100. Timeline: 2-3 weeks.

## Risks and Mitigations

- **Risk**: Acceptance sensitivity is NOT sparse (>50% tokens are critical)
  **Mitigation**: If this happens, the formal contribution still stands as a negative result showing why naive combination doesn't help. Pivot to showing which reasoning patterns have sparse vs dense sensitivity.
- **Risk**: Draft signals are poor predictors of acceptance-criticality
  **Mitigation**: Fall back to post-hoc verification-guided compression (SpecAttn-style) rather than predictive.
- **Risk**: Compression overhead negates savings
  **Mitigation**: Use efficient Hadamard rotation (0.5μs/token) + lazy compression (batch compress every K segments).
