# Research Proposal: SpecQuant — TurboQuant-Accelerated Verification for Speculative Decoding

## Problem Anchor

- **Bottom-line problem**: Speculative decoding's verification phase is memory-bandwidth bound. Each verification round requires a full forward pass through the (large) target model's KV cache, creating a "memory wall" that limits end-to-end throughput gains from drafting improvements.

- **Must-solve bottleneck**: The target model's verification forward pass loads the entire KV cache from HBM for attention computation. As sequence length grows, KV cache size scales linearly, and memory bandwidth becomes the dominant bottleneck — not compute. Current quantization approaches (Quasar) use generic low-bit quantization that sacrifices acceptance rates or requires careful per-model calibration.

- **Non-goals**:
  - Improving the draft model or drafting strategy (orthogonal direction)
  - Training-based quantization methods requiring retraining
  - Weight quantization of the target model (we focus on KV cache only)
  - Changing the rejection sampling algorithm itself

- **Constraints**:
  - Training-free: method must work without any model retraining or calibration data
  - Data-oblivious: must work across different input distributions without tuning
  - Must maintain acceptance rate within 2% of full-precision baseline
  - Must demonstrate on standard LLM families (Qwen3.5, Llama-3)
  - Compute budget: ≤200 GPU-hours on H100/A100 for all experiments
  - Target venue: NeurIPS 2026

- **Success condition**: ≥1.5x end-to-end throughput improvement over vanilla speculative decoding on batch-1 long-context inference, with acceptance rate degradation ≤2pp, using a principled near-optimal quantization scheme for the verification KV cache.

## Technical Gap

### Why current methods fail

1. **Vanilla speculative decoding** (Leviathan et al. 2023): Verification requires full-precision KV cache attention. Memory bandwidth is 2x (K+V) × seq_len × d_model × num_layers × sizeof(fp16) per verification round. For a 14B model at 4k context, this is ~2GB per round.

2. **Quasar** (Huang & Wen, 2026.3): Applies generic low-bit quantization to verification. Achieves 1.28x throughput but uses simple RTN/GPTQ-style quantization without theoretical optimality guarantees. The quantization error accumulates across layers, causing acceptance rate drops at very low bit-widths (≤3 bits).

3. **QuantSpec** (2025.2): Focuses on self-speculative decoding with quantized KV cache, but uses hierarchical 4-bit quantization — still relatively high bit-width with limited memory savings during verification.

### Why naive fixes are insufficient

- **Simply reducing bit-width** below 4 bits with standard quantization causes significant acceptance rate drops because the KV cache quantization error distorts the target model's logit distribution.
- **Per-channel calibration** (as in GPTQ/AWQ) requires calibration datasets and is not data-oblivious.
- **Larger draft models** don't solve the verification bottleneck — they make drafting slower while verification remains the same.

### Smallest adequate intervention

Apply TurboQuant's mathematically principled, data-oblivious quantization specifically to the verification-phase KV cache. TurboQuant's random rotation + optimal scalar quantization achieves near-Shannon-limit distortion at 2.5-3.5 bits per channel, which is far better than generic quantization at the same bit-width.

## Method Thesis

- **One-sentence thesis**: By replacing the verification-phase KV cache with TurboQuant's near-optimal data-oblivious quantization, we reduce memory bandwidth by 4-6x during verification while maintaining speculative acceptance rates within 2pp of full-precision, yielding ≥1.5x end-to-end throughput improvement without any training or calibration.

- **Why this is the smallest adequate intervention**: We change exactly one component — how the KV cache is stored and loaded during the verification forward pass. The draft model, rejection sampling, and all other speculative decoding machinery remain untouched.

- **Why this route is timely in the foundation-model era**: (1) TurboQuant was just published (2025.4) with strong theoretical guarantees but has not been applied to speculative decoding. (2) Speculative decoding is becoming the default inference strategy for large LLMs, making verification optimization increasingly important. (3) The memory wall problem worsens with longer contexts (100K+ tokens), making KV cache compression critical.

## Contribution Focus

- **Dominant contribution**: A principled, training-free method to accelerate speculative decoding verification by applying near-optimal KV cache quantization (TurboQuant), with theoretical analysis of how quantization error affects acceptance rates.
- **Optional supporting contribution**: Empirical characterization of the "quantization-acceptance tradeoff curve" across bit-widths and model families, establishing when and why near-optimal quantization preserves acceptance rates while generic quantization fails.
- **Explicit non-contributions**: We do not propose new draft models, new drafting strategies, or new rejection sampling algorithms. We do not retrain any models.

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: Target model and draft model weights are completely frozen. Standard speculative decoding pipeline (draft → verify → reject/accept) is reused.
- **New trainable components**: None. TurboQuant is entirely training-free.
- **Tempting additions intentionally not used**: Weight quantization of target model (orthogonal, adds complexity), learned rotation matrices (breaks data-oblivious guarantee), tree-structured drafting (orthogonal optimization).

### System Overview

```
Input prompt
    │
    ▼
┌─────────────┐
│ Draft Model  │ ── generate γ draft tokens (standard, no change)
│ (small LLM)  │
└──────┬──────┘
       │ draft tokens + draft probabilities
       ▼
┌──────────────────────────────────────────┐
│ Target Model Verification (MODIFIED)      │
│                                           │
│  1. Compute attention with QUANTIZED      │
│     KV cache via TurboQuant:              │
│     - Random rotation R applied to K,V    │
│     - Scalar quantize each coordinate     │
│       at b bits (2.5-3.5 bits/channel)    │
│     - Store compressed KV in HBM          │
│                                           │
│  2. Verification forward pass loads       │
│     compressed KV → dequantize on-the-fly │
│     → standard attention computation      │
│                                           │
│  Output: target logits for each draft pos │
└──────┬──────────────────────────────────┘
       │ target logits
       ▼
┌─────────────────┐
│ Rejection Sample │ ── accept/reject draft tokens (standard, no change)
└──────┬──────────┘
       │ accepted tokens
       ▼
   Output tokens
```

### Core Mechanism

- **Input**: KV tensors from the target model's forward pass (shape: [batch, heads, seq_len, head_dim])
- **Output**: Quantized KV cache with 4-6x memory reduction
- **Process**:
  1. **Rotation**: Apply a fixed random orthogonal matrix R to each head_dim-dimensional K and V vector. This transforms the potentially non-uniform distribution into a near-isotropic one.
  2. **Scalar Quantization**: Quantize each rotated coordinate independently using an optimal scalar quantizer (Lloyd-Max style) at the target bit-width. For 3 bits/channel, this gives 8 levels per coordinate.
  3. **Residual QJL** (optional, for inner product preservation): Apply a 1-bit Quantized Johnson-Lindenstrauss transform on the quantization residual for unbiased inner product estimation.
  4. **Storage**: Store the quantized codes + rotation matrix R (shared across all positions) + scale/offset per channel.
  5. **Dequantization**: During attention, dequantize on-the-fly: apply inverse rotation R^T to the reconstructed vectors before computing attention scores.

- **Why this is the main novelty**: TurboQuant provides a theoretically grounded quantization with near-Shannon-limit distortion (within 2.7x of information-theoretic lower bound). When applied to KV cache in verification, this mathematical guarantee translates to bounded attention score error, which we can relate to bounded logit divergence and thus bounded acceptance rate degradation.

### Modern Primitive Usage

- **Which primitive**: TurboQuant (data-oblivious vector quantization via random rotation + scalar quantization)
- **Exact role**: Compresses the KV cache stored in HBM during the verification phase, reducing memory bandwidth for attention computation
- **Why more natural than alternatives**: TurboQuant is data-oblivious (no calibration needed), has mathematical optimality guarantees, and its random rotation mechanism naturally handles the varying statistical properties of KV vectors across different layers and attention heads

### Integration into Speculative Decoding Pipeline

1. **Where it attaches**: Between the target model's attention layer input and the KV cache storage
2. **What is frozen**: All model weights (draft and target)
3. **What is modified**: Only the KV cache storage/retrieval mechanism in the target model
4. **Inference order**: Prefix encoding (full precision) → TurboQuant compression of KV cache → Speculative drafting → Verification with compressed KV cache → Rejection sampling
5. **Key detail**: The prefix KV cache is quantized once after encoding. During each verification round, the new KV entries (from draft tokens being verified) are also quantized before storage.

### Training Plan

No training required. TurboQuant is entirely algorithmic:
- Random rotation matrix R is sampled once per model instance (or per layer/head if needed)
- Scalar quantizer levels are computed from the theoretical optimal quantizer for the induced Beta distribution (closed-form, no data needed)
- The only "hyperparameter" is the target bit-width b (default: 3.0 bits/channel)

### Failure Modes and Diagnostics

- **Failure mode 1**: Acceptance rate drops significantly at low bit-widths
  - **Detection**: Monitor per-position acceptance rates; compare with full-precision baseline
  - **Mitigation**: Increase bit-width to 3.5 or 4 bits; apply residual QJL for critical layers

- **Failure mode 2**: Dequantization overhead cancels out memory bandwidth savings
  - **Detection**: Profile wall-clock time breakdown (quantize + dequantize vs memory load)
  - **Mitigation**: Fuse rotation + dequantization into a custom CUDA kernel; use half-precision rotation

- **Failure mode 3**: Long-context scenarios amplify quantization error cumulatively
  - **Detection**: Measure acceptance rate as function of sequence length
  - **Mitigation**: Apply per-layer adaptive bit-width (higher bits for bottom layers which have higher impact on output)

### Novelty and Elegance Argument

**Closest work**: Quasar (Huang & Wen, 2026.3) applies generic low-bit quantization to verification.

**Exact difference**:
1. Quasar uses RTN-style quantization without optimality guarantees; we use TurboQuant with near-Shannon-limit distortion
2. Quasar requires model-specific tuning of quantization parameters; TurboQuant is data-oblivious
3. We provide theoretical analysis connecting KV quantization error → attention score error → logit divergence → acceptance rate bound, which Quasar lacks
4. We can push to lower bit-widths (2.5 bits) with maintained acceptance rates where Quasar fails

**Why focused**: One mechanism (TurboQuant on verification KV cache), one theoretical contribution (quantization-acceptance tradeoff bound), one empirical validation (throughput + acceptance rate across models and tasks).

## Claim-Driven Validation Sketch

### Claim 1: TurboQuant verification achieves ≥1.5x throughput with ≤2pp acceptance rate drop

- **Minimal experiment**: Compare vanilla speculative decoding vs SpecQuant on Qwen3.5-{0.8B→9B, 4B→14B} pairs across GSM8K, HumanEval, MT-Bench
- **Baselines**: (a) vanilla spec decode (full precision), (b) Quasar (generic quantization), (c) QuantSpec (4-bit hierarchical)
- **Metric**: end-to-end throughput (tokens/sec), acceptance rate, generation quality (pass@1 for code, accuracy for math)
- **Expected evidence**: SpecQuant at 3 bits achieves 1.5-2x throughput vs vanilla, acceptance rate within 2pp, matching Quasar's quality with better throughput

### Claim 2: Near-optimal quantization preserves acceptance rates where generic quantization fails

- **Minimal experiment**: Sweep bit-widths from 2 to 4 bits, compare acceptance rate curves for TurboQuant vs RTN vs GPTQ-style quantization
- **Baselines**: RTN (round-to-nearest), GPTQ-style (calibrated), TurboQuant (ours)
- **Metric**: acceptance rate vs bit-width curve, throughput vs bit-width curve
- **Expected evidence**: TurboQuant maintains >90% acceptance rate at 3 bits; RTN drops below 80%; GPTQ is close but requires calibration

## Experiment Handoff Inputs

- **Must-prove claims**: (1) throughput gain ≥1.5x with ≤2pp acceptance rate drop, (2) superiority over generic quantization at ≤3 bits
- **Must-run ablations**: bit-width sweep (2.0-4.0), with/without residual QJL, per-layer vs global bit allocation
- **Critical datasets/metrics**: GSM8K accuracy, HumanEval pass@1, acceptance rate, tokens/sec
- **Highest-risk assumptions**: (1) dequantization overhead is small relative to bandwidth savings, (2) TurboQuant's near-isotropy assumption holds for real KV cache distributions

## Compute & Timeline Estimate

- **Estimated GPU-hours**: ~150 GPU-hours on H100 (8-GPU node)
  - Model download + setup: ~5h
  - Main throughput experiments (4 model pairs × 4 benchmarks × 3 methods): ~80h
  - Bit-width sweep + ablations: ~40h
  - Paper figures + analysis: ~25h
- **Data / annotation cost**: None (all benchmarks are public)
- **Timeline**: 2-3 weeks from code implementation to results
