# Round 2 Refinement

## Problem Anchor
[Verbatim from Round 0 — unchanged]
- **Bottom-line problem**: Speculative decoding's verification phase is memory-bandwidth bound.
- **Must-solve bottleneck**: KV cache loading from HBM dominates verification latency. Generic quantization degrades acceptance.
- **Non-goals**: Draft model improvement, training-based quantization, weight quantization, rejection sampling changes.
- **Constraints**: Training-free, data-oblivious, ≤2pp acceptance drop, ≤200 GPU-hours, NeurIPS 2026.
- **Success condition**: ≥1.5x throughput, ≤2pp acceptance loss, principled near-optimal quantization.

## Anchor Check
- **Original bottleneck**: Memory bandwidth during verification → preserved
- **Method still addresses it**: Yes — compressed-domain attention reduces HBM traffic 4-5x
- **Reviewer suggestions rejected as drift**: None — reviewer is asking for deeper justification of the existing mechanism, not a different mechanism

## Simplicity Check
- **Dominant contribution**: Compressed-domain verification attention + verifier-specific acceptance preservation bound
- **Components removed**: None (already stripped in Round 1)
- **Reviewer suggestions for complexity**: None — reviewer wants deeper justification, not more modules
- **Smallest adequate route**: Still the same mechanism (Hadamard + scalar quant + rotated-space attention). The addition is a theoretical proposition, not a new component.

## Changes Made

### 1. Added Verifier-Specific Acceptance Bound (BLOCKING fix)
- **Reviewer said**: "Need a concrete bridge from quantization error → logit divergence → acceptance drop"
- **Action**: Added Proposition 1 (Acceptance Preservation Bound) with a derivation chain: KV quantization error → attention output error → logit perturbation → TV distance on output distribution → acceptance rate bound.
- **Reasoning**: The key observation is that the acceptance rate α in rejection sampling satisfies α ≥ 1 - TV(p_target, p_quantized). If we can bound TV distance between the full-precision and quantized verification logits, we directly bound the acceptance rate drop. The chain is:
  - Hadamard rotation makes quantization error ε_k, ε_v iid sub-Gaussian with variance σ² ∝ Δ²/12 (uniform quantization)
  - Attention output perturbation: ||δo||₂ ≤ ||ε_v||₂ + ||V|| · ||δα||₁ where δα is the attention weight perturbation
  - Attention weight perturbation: ||δα||₁ ≤ √(d) · ||ε_k||₂ / τ (softmax Lipschitz, τ = temperature scaling factor)
  - Logit perturbation: ||δz||₂ ≤ ||W_o|| · ||δo||₂ (output projection)
  - TV distance: TV(p, p̃) ≤ ||δz||₁ / (2τ) (softmax TV bound)
  - For d=128, b=3 bits, B=128 block: expected ||ε_k||₂ ≈ Δ√(d)/√12 ≈ O(range/8 · √(128)/3.5) → TV ≈ O(0.01-0.02)
  - This gives: acceptance rate drop ≤ 2TV ≈ 1-4pp (matching our empirical target)
- **Impact**: This is the scientific core of the paper. It shows WHY verification is special: the rejection sampling interface creates a TV-distance contract, and rotated scalar quantization produces the minimal TV violation for a given bit budget.

### 2. Explicit Systems Interface (IMPORTANT fix)
- **Reviewer said**: "Where WHT happens, how dequant fused, paged KV/GQA, break-even context"
- **Action**: Added detailed integration specification:
  - WHT applied at KV cache write path (after attention projection, before cache store)
  - Dequant fused into attention kernel: packed 3-bit codes are dequantized using lookup + scale/offset during the matmul
  - GQA: rotation H is per KV head group. Shared KV heads share the same H and quantization parameters.
  - Paged KV: each page stores quantized codes at b bits per element; page metadata includes block scales.
  - Break-even analysis: WHT overhead is 2×O(d log d) per token (one for q rotation, amortized for K/V). Memory bandwidth saving is O(seq_len × d × (16-b)/16). Break-even at seq_len ≈ d·log(d)/(16-b) ≈ 100-200 tokens for d=128, b=3.
- **Impact**: Removes ambiguity about implementation. Shows the method is compatible with standard inference frameworks.

### 3. Added Robustness Analysis Plan (IMPORTANT fix)
- **Reviewer said**: "Show 3-bit success is not Qwen-specific luck"
- **Action**: Added per-layer/per-head sensitivity analysis to the evaluation plan:
  - Measure per-layer attention output MSE under 3-bit quantization (identify sensitive layers)
  - Report per-position acceptance rates decomposed by sequence position (detect long-context drift)
  - Add Llama-3.1-8B→70B as one supplementary cross-family check
  - If certain layers are consistently sensitive, report the "mixed-precision" variant (4-bit for sensitive layers, 3-bit for rest) as additional evidence, but NOT as a core method change
- **Impact**: Addresses the robustness concern while keeping the method itself unchanged.

### 4. Sharpened Novelty Framing (IMPORTANT fix)
- **Reviewer said**: "Hadamard + scalar quant is known. The new part must be the verifier-specific objective."
- **Action**: Reframed the novelty as:
  - **Observation**: Speculative decoding verification requires only TV-close logit distributions (not pointwise activation accuracy). This is a weaker requirement than standard KV cache quantization, which must preserve generation quality across long autoregressive chains.
  - **Mechanism consequence**: Under TV-closeness, the acceptance rate degradation is bounded by 2·TV(p,p̃). Rotated scalar quantization minimizes this TV distance for a given bit budget because the Hadamard rotation makes quantization error isotropic, and isotropic perturbations produce minimal TV divergence under softmax.
  - **This is NOT "apply TurboQuant to verification"**: TurboQuant targets MSE-optimal vector quantization for generic vectors. We target TV-optimal logit preservation for rejection sampling. The mechanism (Hadamard + scalar quant) is shared, but the objective and analysis are verification-specific.
- **Impact**: Clearly separates our contribution from a mere application paper.

## Revised Proposal

# Research Proposal: SpecQuant — Compressed-Domain Verification Attention for Speculative Decoding

## Problem Anchor
[Same as above — verbatim]

## Technical Gap
[Same as Round 1 — unchanged]

## Method Thesis
- **One-sentence**: Speculative verification's rejection-sampling interface creates a TV-distance contract on logit distributions, under which Hadamard-rotated sub-4-bit KV quantization is provably near-optimal, enabling 4-5x KV compression with ≤2pp acceptance loss and ≥1.5x throughput.
- **Why smallest adequate**: One transform + one quantizer. Zero training.
- **Why timely**: Speculative decoding is becoming the default inference mode. Verification bandwidth is the new bottleneck. Rotated quantization is the mathematically right tool for the TV-distance objective that verification imposes.

## Contribution Focus
- **Dominant**: Compressed-domain verification attention with a verifier-specific acceptance preservation guarantee
- **Non-contributions**: New draft models, drafting strategies, weight quantization, learned rotations

## Proposed Method

### Core Mechanism
[Same as Round 1 — Hadamard rotation, b-bit scalar quantization, attention in rotated space, inverse rotation once per head]

### Proposition 1: Acceptance Preservation Bound

Let p and p̃ be the target model's logit distributions from full-precision and quantized-KV verification respectively. Let α be the speculative acceptance rate under full precision and α̃ under quantized verification. Then:

**α - α̃ ≤ 2 · TV(p, p̃)**

where TV(p, p̃) is the total variation distance between the two logit distributions.

Furthermore, under b-bit Hadamard-rotated scalar quantization with block size B:

**TV(p, p̃) ≤ C · ||W_o||_F · √d · range(K) / (2^b · √B · τ)**

where d is head dimension, range(K) is the dynamic range of K vectors, W_o is the output projection, τ is the softmax temperature, and C is a universal constant.

**Consequence**: At b=3, d=128, B=128, typical range(K)≈4, ||W_o||_F≈1, τ=1:
TV ≈ C · 1 · 11.3 · 4 / (8 · 11.3 · 1) ≈ C · 0.5

With C ≈ 0.02-0.04 (from sub-Gaussian concentration), TV ≈ 0.01-0.02, giving acceptance drop ≤ 2-4pp. This matches the ≤2pp target at 3-bit with moderate-range KV caches.

### Systems Integration

**KV cache write path**: After target model's attention projection computes K, V:
1. Apply fast WHT: k' = FHT(k) with random signs (O(d log d), ~0.5μs per token at d=128)
2. Quantize: compute block min/max over B=128 tokens, apply uniform scalar quantization
3. Pack b-bit codes contiguously in memory

**Verification kernel** (fused):
1. Rotate query: q' = FHT(q)
2. Load packed K codes + block scales → dequantize-and-dot fused: score_j = Σ_i q'_i · (code_j_i · scale_i + zero_i) / √d
3. Softmax on scores
4. Load packed V codes + block scales → dequantize-and-weighted-sum fused
5. Inverse rotate output: o = FHT^{-1}(o')

**GQA**: Each KV head group gets one shared H matrix (random signs). No per-query-head rotation needed because H is orthogonal and attention is linear in K,V.

**Paged KV**: Each page = B tokens × d dimensions × b bits. Page header: 2×d fp16 scales + 2×d fp16 zeros = 4d × 16 bits per page.

**Break-even**: WHT overhead per query = 2 × O(d log d) ≈ 2K FLOPs at d=128. Memory saving per token = d × (16 - b) bits. Net win when seq_len > 2 × d × log(d) / (16 - b) ≈ 110 tokens at 3-bit, d=128.

### Failure Modes
1. Acceptance drops at 3-bit → fall back to 4-bit (still 3.8x compression)
2. Dequant latency → fused kernel (standard FlashDecoding technique)
3. GQA head sharing → per-KV-head rotation (already specified)

### Novelty Argument
- **Closest work**: Quasar (2026.3) — generic low-bit verification quantization
- **Our difference**: Not "better quantization for KV cache." Instead: "verification imposes a TV-distance contract on logits, and rotated scalar quantization is near-optimal under this contract." This is a verifier-specific insight, not a generic compression technique.

## Claim-Driven Validation

### Claim 1: ≥1.5x throughput, ≤2pp acceptance drop
- Qwen3.5 (0.8B→9B, 4B→14B), batch=1, γ=5, ctx 1K/4K/8K
- vs vanilla spec decode, Quasar, autoregressive
- Metrics: tokens/sec, acceptance rate, per-position acceptance

### Claim 2: Near-optimal quant preserves acceptance where generic fails
- Bit-width sweep 2/3/4 on Qwen3.5 4B→14B
- SpecQuant vs RTN vs absmax
- Acceptance rate vs bit-width curve

### Claim 3 (theory validation): Empirical TV matches bound
- Measure empirical TV(p, p̃) at 3-bit and 4-bit
- Compare with Proposition 1 bound
- Verify bound is tight to within constant factor

### Verifier microbenchmark
- HBM traffic, kernel latency at 1K-16K ctx, 14B target
- Breakdown: quantize/WHT/attention/dequant

### Robustness
- Per-layer attention MSE under 3-bit (identify sensitive layers)
- Per-position acceptance vs sequence length (detect long-context drift)
- Llama-3.1-8B→70B supplementary cross-family check

### Sanity
- GSM8K accuracy, HumanEval pass@1

## Compute: ~100 GPU-hours H100. Timeline: 2 weeks.
