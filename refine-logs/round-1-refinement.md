# Round 1 Refinement

## Problem Anchor
- **Bottom-line problem**: Speculative decoding's verification phase is memory-bandwidth bound. Each verification round requires a full forward pass through the (large) target model's KV cache, creating a "memory wall" that limits end-to-end throughput gains from drafting improvements.
- **Must-solve bottleneck**: The target model's verification forward pass loads the entire KV cache from HBM for attention computation. As sequence length grows, KV cache size scales linearly, and memory bandwidth becomes the dominant bottleneck — not compute. Current quantization approaches (Quasar) use generic low-bit quantization that sacrifices acceptance rates or requires careful per-model calibration.
- **Non-goals**: Improving the draft model or drafting strategy; Training-based quantization; Weight quantization of target model; Changing rejection sampling algorithm.
- **Constraints**: Training-free, data-oblivious, acceptance rate within 2pp of full-precision, ≤200 GPU-hours, NeurIPS 2026.
- **Success condition**: ≥1.5x end-to-end throughput over vanilla spec decode on batch-1 long-context inference, acceptance rate degradation ≤2pp, using principled near-optimal quantization.

## Anchor Check
- **Original bottleneck**: Memory bandwidth during verification forward pass in speculative decoding
- **Why the revised method still addresses it**: The core mechanism (compressed-domain attention in rotated space) directly reduces HBM traffic during verification by 4-5x
- **Reviewer suggestions rejected as drift**: None — reviewer feedback was anchored correctly. All suggestions sharpen the verification focus.

## Simplicity Check
- **Dominant contribution after revision**: Compressed-domain verification attention — compute speculative verification entirely in the rotated-quantized KV space, applying inverse rotation only once per head on the final attention output
- **Components removed or merged**:
  - Removed: Residual QJL (was optional noise; plain rotation + scalar quant is sufficient at 3-4 bits)
  - Removed: Per-layer adaptive bit allocation (becomes appendix stretch result, not core)
  - Simplified: Dense random orthogonal R → structured random-sign Hadamard (kernel-friendly, same theoretical guarantees via Johnson-Lindenstrauss)
  - Removed: MT-Bench, MMLU evaluation (not relevant to long-context verification bottleneck)
- **Reviewer suggestions rejected as unnecessary complexity**: None — all simplification suggestions accepted
- **Why the remaining mechanism is still the smallest adequate route**: One structural change (Hadamard rotation + scalar quantization of KV cache, attention in rotated space) with zero training, zero calibration, and one hyperparameter (bit-width b)

## Changes Made

### 1. Kernel Interface (CRITICAL fix)
- **Reviewer said**: "Inverse-rotate R^T before attention is the wrong interface. Compute attention in rotated space."
- **Action**: Redesigned the core mechanism. Now: q' = Hq (Hadamard-rotate query), k' = Q_b(Hk), v' = Q_b(Hv), attention in rotated space, R^T applied only once to head output.
- **Reasoning**: Computing attention in rotated space means dequantization is just scale+offset lookup (no matrix multiply per token), and inverse rotation is amortized across all sequence positions.
- **Impact**: This is the key insight that makes the method practical. Dequantization cost drops from O(seq_len × d × d) to O(d × d) per head.

### 2. Contribution Focusing (CRITICAL fix)
- **Reviewer said**: "Too many side ideas; novelty risks collapsing to 'apply existing quantizer.'"
- **Action**: Stripped to one mechanism: compressed-domain verification attention with structured Hadamard rotation. Removed QJL and adaptive allocation from main paper. Reframed theory from "direct acceptance bound" to modest "attention error bound → empirical acceptance validation."
- **Reasoning**: The contribution is NOT "apply TurboQuant to KV cache" (that's an application). The contribution IS "speculative verification has a unique property: attention computation can be done entirely in a near-isotropic rotated space, making sub-4-bit KV quantization essentially lossless for verification acceptance."
- **Impact**: Paper story becomes sharp: one observation (verification-specific rotated-space attention), one mechanism (Hadamard + scalar quant), one result (1.5x+ throughput with negligible acceptance loss).

### 3. Narrowed Scope (IMPORTANT fix)
- **Reviewer said**: "One kernel path, fixed 3-bit/4-bit, 1-2 model families."
- **Action**: Fixed to 3-bit and 4-bit experiments on Qwen3.5 family (0.8B→9B, 4B→14B). 2.5-bit is appendix-only. Llama-3 is a supplementary cross-family check, not a core claim.
- **Reasoning**: Qwen3.5 has GQA and hybrid KV cache, which stress-tests the method. One family with multiple size ratios is sufficient for core claims.
- **Impact**: Reduces GPU budget to ~100h and timeline to 2 weeks.

### 4. Evaluation Redesign (IMPORTANT fix)
- **Reviewer said**: "Center on context-length sweeps and verifier microbenchmarks."
- **Action**: Primary evaluation is now:
  - Verifier microbenchmark: HBM traffic (GB), kernel latency (ms), decomposed by quantize/dequant/attention
  - Context-length sweep: acceptance rate & throughput at 1K/2K/4K/8K/16K sequence lengths
  - End-to-end: tokens/sec at batch=1 with γ=5
  - Sanity: GSM8K accuracy, HumanEval pass@1 (generation quality unchanged)
- **Reasoning**: The bottleneck claim is about memory bandwidth; the evaluation must measure memory bandwidth directly.
- **Impact**: Evaluation is now perfectly aligned with the problem anchor.

## Revised Proposal

# Research Proposal: SpecQuant — Compressed-Domain Verification Attention for Speculative Decoding

## Problem Anchor
[Verbatim from above]

## Technical Gap

Speculative decoding splits LLM inference into fast drafting (small model) and accurate verification (large model). Recent advances (EAGLE, Medusa, etc.) have minimized drafting overhead, making **verification the primary bottleneck**.

The verification forward pass is dominated by attention computation over the target model's KV cache. This is memory-bandwidth bound: loading K and V tensors from HBM accounts for >70% of verification latency at long contexts.

**Quasar** (2026.3) applied generic low-bit quantization to verification, achieving 1.28x throughput. But generic quantization (RTN/GPTQ-style) at ≤3 bits causes measurable acceptance rate degradation because quantization error in K and V propagates non-uniformly through softmax attention.

**The missing insight**: Speculative verification has a unique structural property — we only need the *output distribution* of the target model to be preserved (for rejection sampling), not every intermediate activation. If we can ensure the attention output is close, the acceptance rate is preserved. This means we can compute attention entirely in a rotated, near-isotropic space where per-coordinate scalar quantization is provably near-optimal.

## Method Thesis

- **One-sentence thesis**: Speculative verification attention can be computed entirely in a Hadamard-rotated space where sub-4-bit scalar KV quantization achieves near-optimal distortion, reducing verification HBM traffic by 4-5x while maintaining acceptance rates within 1-2pp of full precision.

- **Why smallest adequate intervention**: One fixed linear transform (Hadamard) + one scalar quantizer per coordinate. No training, no calibration data, no learned parameters.

- **Why timely**: TurboQuant (2025) showed data-oblivious rotation makes scalar quantization near-optimal for generic vectors. We observe that speculative decoding's rejection-sampling interface creates the ideal application: the acceptance rate depends only on logit distribution similarity, which is robust to small, balanced attention perturbations — exactly what rotated scalar quantization produces.

## Contribution Focus

- **Dominant contribution**: Compressed-domain verification attention — a training-free method that performs speculative verification in Hadamard-rotated space with 3-bit KV quantization, achieving 1.5x+ throughput with ≤2pp acceptance rate loss.
- **Explicit non-contributions**: New draft models, drafting strategies, weight quantization, learned rotations.

## Proposed Method

### Complexity Budget
- **Frozen**: All model weights (draft and target), rejection sampling logic
- **New trainable components**: None
- **Not used**: QJL residual, adaptive bit allocation, learned rotations, tree-structured drafting

### System Overview

```
Prefix encoding (standard)
    │
    ▼
┌─────────────────────────────────┐
│ KV Cache Compression (once)      │
│ For each layer l, head h:        │
│   k'_{l,h} = Q_b(H · k_{l,h})  │
│   v'_{l,h} = Q_b(H · v_{l,h})  │
│ Store: quantized codes + scales  │
└──────────┬──────────────────────┘
           │
    ┌──────┼──────────────────────┐
    │      │ Speculative Loop     │
    │      ▼                      │
    │  Draft γ tokens (unchanged) │
    │      │                      │
    │      ▼                      │
    │  Compressed-Domain Verify:  │
    │   q' = H · q                │
    │   attn = softmax(q'k'^T/√d) │
    │   out = R_H^T · (attn · v') │
    │      │                      │
    │      ▼                      │
    │  Rejection sampling         │
    │  (unchanged)                │
    └─────────────────────────────┘
```

### Core Mechanism: Compressed-Domain Verification Attention

**Quantization (applied once to stored KV cache):**
1. Hadamard rotation: k' = H · k where H is a d×d normalized Hadamard matrix with random sign flips (fast Walsh-Hadamard transform, O(d log d) per vector)
2. Per-coordinate scalar quantization at b bits: Q_b(k'_i) = round((k'_i - min_i) / Δ_i) where Δ_i = (max_i - min_i) / (2^b - 1), with min_i and max_i computed per block of B=128 tokens
3. Storage: b-bit codes (packed) + fp16 scale and zero-point per block per coordinate

**Verification forward pass:**
1. Rotate the query: q' = H · q (one fast WHT, O(d log d))
2. Attention in rotated space: scores = q' · dequant(k')^T / √d (dequant is scale+offset, fused with matmul)
3. Attention output: o' = softmax(scores) · dequant(v')
4. Inverse rotation (once per head): o = H^T · o' (one fast WHT, O(d log d))

**Effective storage per KV entry:**
- b bits per coordinate for quantized codes
- 32 bits per block per coordinate for scale+zero-point (amortized over B=128 tokens → 0.25 bits/token/coord)
- Total: b + 0.25 bits per coordinate (e.g., 3.25 bits at b=3 vs 16 bits full-precision → **4.9x compression**)

**Why this works for verification specifically:**
- Rejection sampling compares target logits vs draft logits. Small balanced perturbations to attention outputs produce small balanced perturbations to logits, which have bounded effect on acceptance probability.
- Hadamard rotation makes the per-coordinate quantization error nearly independent and identically distributed across coordinates. This means the total attention error concentrates (by CLT-type arguments) and does not exhibit the heavy-tailed errors that generic quantization produces.

### Block Size and Metadata

| Setting | Block Size | Bits/coord | Metadata bits/coord | Effective bits | Compression |
|---------|-----------|------------|---------------------|----------------|-------------|
| 3-bit   | 128       | 3          | 0.25                | 3.25           | 4.9x        |
| 4-bit   | 128       | 4          | 0.25                | 4.25           | 3.8x        |

### Failure Modes and Diagnostics

1. **Acceptance rate drops at 3-bit**: Monitor per-position acceptance; fall back to 4-bit
2. **Dequantization latency dominates**: Profile kernel; fuse dequant with matmul (standard technique in FlashDecoding variants)
3. **GQA head-sharing complicates rotation**: Apply rotation per KV head group; shared heads share rotation

### Novelty and Elegance Argument

**Closest work**: Quasar (2026.3) quantizes verification with generic low-bit methods.

**Exact difference**:
1. **Interface**: Quasar dequantizes before attention; we compute attention in the rotated-quantized space. This avoids O(seq_len × d) dequantization overhead per query position.
2. **Quantization quality**: Hadamard rotation + scalar quantization achieves provably near-optimal distortion; generic RTN does not.
3. **Specificity**: We leverage the speculative decoding property that only the logit distribution must be preserved, not pointwise activations. This justifies why verification is the ideal regime for aggressive KV quantization.

## Claim-Driven Validation Sketch

### Claim 1: ≥1.5x end-to-end throughput with ≤2pp acceptance rate drop
- **Experiment**: Qwen3.5 pairs (0.8B→9B, 4B→14B) at batch=1, γ=5, context lengths 1K/4K/8K
- **Baselines**: vanilla spec decode, Quasar, autoregressive target-only
- **Metrics**: tokens/sec, acceptance rate, per-position acceptance rates
- **Expected**: 1.5-2x throughput at 3-bit, ≤2pp acceptance rate drop

### Claim 2: Near-optimal quantization preserves acceptance where generic quantization fails
- **Experiment**: Bit-width sweep 2/3/4 bits on Qwen3.5 4B→14B
- **Compare**: SpecQuant (Hadamard+scalar) vs RTN vs absmax
- **Metrics**: acceptance rate vs bit-width curve, throughput vs bit-width
- **Expected**: SpecQuant maintains >95% of full-precision acceptance at 3 bits; RTN drops >5pp

### Verifier Microbenchmark
- **Measure**: Verification kernel latency, HBM traffic (GB), breakdown by quantize/attention/dequant
- **Context**: 1K/2K/4K/8K/16K sequence lengths on 14B target model
- **Expected**: HBM traffic reduced 4-5x, kernel latency reduced 1.5-2x (compute stays similar)

### Sanity Checks
- GSM8K accuracy and HumanEval pass@1 with SpecQuant vs vanilla (quality should be statistically identical)

## Compute & Timeline Estimate
- **GPU-hours**: ~100h on H100 (8-GPU)
  - Model pair evaluation (2 pairs × 3 context lengths × 3 methods): ~40h
  - Bit-width sweep: ~20h
  - Verifier microbenchmarks: ~15h
  - Figures + analysis: ~25h
- **Timeline**: 2 weeks from code to results
