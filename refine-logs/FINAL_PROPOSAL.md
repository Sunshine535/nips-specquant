# Research Proposal: SpecQuant — Compressed-Domain Verification Attention for Speculative Decoding

## Problem Anchor

- **Bottom-line problem**: Speculative decoding's verification phase is memory-bandwidth bound. KV cache loading from HBM dominates verification latency at long contexts.
- **Must-solve bottleneck**: Generic low-bit quantization (Quasar) degrades acceptance rates at ≤3 bits. Need principled, data-oblivious quantization that preserves the verification contract.
- **Constraints**: Training-free, data-oblivious, NeurIPS 2026, ≤200 GPU-hours.
- **Success condition**: ≥1.5x end-to-end throughput, ≤3pp acceptance loss at 3-bit, ≤1pp at 4-bit.

## Method Thesis

Speculative verification's rejection-sampling interface induces a total-variation contract on the verifier's logit distribution. Under this contract, Hadamard-rotated sub-4-bit scalar KV quantization is provably near-optimal: it minimizes TV distance for a given bit budget by making quantization error isotropic. This yields 4-5x KV compression with ≤3pp acceptance loss at 3-bit and ≤1pp at 4-bit.

## Dominant Contribution

Compressed-domain verification attention with a per-step acceptance preservation guarantee (Proposition 1), establishing that speculative verification is a uniquely favorable regime for aggressive KV quantization.

## Core Mechanism

### KV Cache Compression (applied once to stored cache)
1. **Hadamard rotation**: k' = FHT_s(k), v' = FHT_s(v) where FHT_s is fast Walsh-Hadamard transform with random sign vector s (shared per KV head group). O(d log d) per vector, ~0.5μs/token at d=128.
2. **b-bit scalar quantization**: Q_b(x_i) = round((x_i - min_i)/Δ_i) with per-block (B=128 tokens) min/max scales.
3. **Packed storage**: b-bit codes contiguous in memory + fp16 scale/zero per block per coordinate.

### Verification Forward Pass (per speculative round)
1. Rotate query: q' = FHT_s(q)
2. Attention in rotated space: scores = q' · deQ(k')^T / √d (dequant fused with matmul)
3. Weighted sum: o' = softmax(scores) · deQ(v')
4. Inverse rotation (once per head): o = FHT_s^{-1}(o')

### Effective Bits
| Setting | Bits/coord | Metadata | Effective | Compression |
|---------|-----------|----------|-----------|-------------|
| 3-bit   | 3         | 0.25     | 3.25      | 4.9x        |
| 4-bit   | 4         | 0.25     | 4.25      | 3.8x        |

## Proposition 1: Acceptance Preservation Bound

For b-bit Hadamard-rotated scalar quantization with block size B, the per-step acceptance rate drop satisfies:

**Δα ≤ ||W_o||_F / τ · [R_V·√d/(2^b·√(12B)) + ||V||_F·R_K·√d/(2^b·√(12B)·τ)]**

where R_K, R_V = dynamic ranges of K, V; d = head dimension; τ = temperature; W_o = output projection; C = 1/(2√3).

| Setting | TV bound | Acceptance drop |
|---------|----------|-----------------|
| 4-bit   | ≤0.005   | ≤1pp           |
| 3-bit   | ≤0.015   | ≤3pp           |
| 2-bit   | ≤0.06    | ≤12pp (not viable) |

## Systems Integration
- **KV write path**: FHT on cache write after attention projection
- **Verification kernel**: Fused dequant-dot (packed codes → scale+offset → matmul)
- **GQA**: Per-KV-head rotation, shared across query heads
- **Paged KV**: b-bit pages with block scale headers
- **Break-even**: ~110 tokens at 3-bit, d=128

## Evaluation Plan

### Primary
1. End-to-end throughput + acceptance: Qwen3.5 (0.8B→9B, 4B→14B), batch=1, γ=5, ctx 1K/4K/8K vs vanilla/Quasar/FP8/autoregressive
2. Bit-width sweep: 2/3/4, SpecQuant vs RTN vs absmax
3. Theory validation: empirical TV vs predicted bound, tightness ratio
4. Verifier microbenchmark: HBM traffic, kernel latency at 1K-16K

### Secondary
5. Per-layer sensitivity analysis (3-bit)
6. Long-context acceptance drift (1K-16K)
7. Failure envelope (2-bit)

### Appendix
8. Llama-3.1 cross-family check
9. GSM8K/HumanEval sanity
10. Mixed-precision variant

## Compute: ~100 GPU-hours H100. Timeline: 2 weeks.
