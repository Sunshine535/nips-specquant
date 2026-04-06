# Experiment Plan: ThinkCompress

**Problem**: CoT reasoning models generate 7-100x more thinking tokens than answer tokens. Thinking KV cache occupies 88-100% of inference memory, reaching 55GB for Qwen3-72B on complex tasks — exceeding single-GPU capacity. No existing work addresses compression specifically for the thinking phase.

**Method Thesis**: Thinking tokens have heterogeneous utility — "conclusion tokens" that answer generation heavily attends to must be preserved, while "process tokens" (exploration, self-correction, intermediate steps) can be aggressively compressed or evicted. ThinkCompress exploits this structure via attention-importance scoring + adaptive per-token bit-width assignment during streaming generation.

**Date**: 2026-04-06

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|----------------------------|---------------|
| C1: ThinkCompress achieves 4-8x thinking KV compression with <2% answer quality loss | Core value proposition — enables long CoT on memory-constrained GPUs | GSM8K/MATH accuracy within 2% of FP16 at 4-8x compression across Qwen3-8B/14B | B1, B2 |
| C2: Importance-based adaptive compression outperforms uniform compression at same ratio | Justifies the method's complexity over naive quantization | ≥5% accuracy gap between ThinkCompress and uniform-quant at aggressive (8x) compression | B3 |
| Anti-claim: "The gain is just from token eviction (H2O/StreamingLLM), not adaptive quantization" | Must show quantization of retained tokens matters | Ablation: eviction-only vs evict+quantize, showing the combination is needed | B4 |

## Paper Storyline

**Main paper must prove:**
1. Thinking tokens dominate KV memory (empirical characterization across models/tasks)
2. Attention importance is highly skewed during answer generation → natural compression target
3. ThinkCompress achieves high compression with minimal quality loss
4. Adaptive > uniform at same compression ratio

**Appendix can support:**
- Per-layer analysis of thinking vs answer attention patterns
- Theoretical TV bound for mixed-precision KV
- Extended model/dataset results
- Wall-clock latency measurements

**Experiments intentionally cut:**
- Speculative decoding integration (orthogonal, adds complexity without strengthening core claim)
- Training-time fine-tuning for compression-awareness (we claim training-free)
- Multi-turn dialogue (focus on single-turn reasoning where thinking is most prominent)

## Experiment Blocks

### Block 1: Thinking KV Characterization (Motivation)
- **Claim tested**: Thinking tokens dominate KV memory and have heterogeneous attention utility
- **Why**: Establishes the problem and motivates the method
- **Dataset**: GSM8K (50 samples), MATH (50 samples), AIME 2024 (30 samples)
- **Models**: Qwen3-8B, Qwen3-14B
- **Metrics**: (a) thinking/answer token ratio, (b) attention entropy from answer→thinking positions, (c) per-token importance distribution (Gini coefficient)
- **Setup**: Generate CoT with thinking mode, capture attention weights during answer generation, compute per-thinking-token cumulative attention score
- **Success criterion**: >80% of thinking tokens receive <20% of total attention weight (high skew)
- **Failure interpretation**: If attention is uniform, adaptive compression has no advantage over uniform
- **Table/figure target**: Figure 1 (attention heatmap), Table 1 (token ratios), Figure 2 (importance distribution)
- **Priority**: MUST-RUN

### Block 2: Main Result — Accuracy vs Compression
- **Claim tested**: C1 — ThinkCompress achieves 4-8x compression with <2% quality loss
- **Why**: Core paper result
- **Dataset**: GSM8K (full 1319 test), MATH-500, HumanEval (164), GPQA-Diamond (198)
- **Models**: Qwen3-8B (primary), Qwen3-14B (generalization)
- **Compared systems**: (a) FP16 baseline (no compression), (b) ThinkCompress at 2x/4x/6x/8x compression, (c) Uniform quantization (all tokens same bits) at same ratios, (d) H2O eviction at same memory budget, (e) StreamingLLM (sliding window) at same budget
- **Metrics**: Task accuracy (exact-match for GSM8K/MATH, pass@1 for HumanEval, accuracy for GPQA), KV memory usage (bytes), peak GPU memory
- **Setup**: Qwen3-8B on cuda:0, importance scoring on cuda:1. Generate thinking with budget constraint. 3 seeds for stochastic methods.
- **Success criterion**: <2% accuracy drop at 4x, <5% at 8x. ThinkCompress > all baselines at every compression level.
- **Failure interpretation**: If uniform quantization matches ThinkCompress, the importance scoring adds no value
- **Table/figure target**: Table 2 (main result), Figure 3 (accuracy vs compression Pareto curve)
- **Priority**: MUST-RUN

### Block 3: Novelty Isolation — Adaptive vs Uniform
- **Claim tested**: C2 — Importance-based assignment matters
- **Why**: Proves the method's core contribution beyond naive quantization
- **Dataset**: GSM8K, MATH-500
- **Compared systems**: (a) ThinkCompress (adaptive per-token bits), (b) Uniform-quant (all thinking tokens at same bit-width), (c) Random assignment (same bit distribution but randomly assigned), (d) Oracle (assign bits using future attention from answer — upper bound)
- **Metrics**: Accuracy, per-token bit-width distribution
- **Setup**: Fix total compression ratio, vary allocation strategy
- **Success criterion**: ThinkCompress within 80% of oracle, >5% better than uniform/random
- **Failure interpretation**: If random matches adaptive, importance scoring is not useful
- **Table/figure target**: Table 3 (ablation), Figure 4 (bit allocation visualization)
- **Priority**: MUST-RUN

### Block 4: Simplicity Check — Eviction vs Quantization vs Both
- **Claim tested**: Anti-claim ruled out — both eviction and quantization contribute
- **Why**: Reviewer will ask "why not just evict tokens?"
- **Dataset**: GSM8K, MATH-500
- **Compared systems**: (a) Eviction-only (drop low-importance, keep rest FP16), (b) Quantization-only (keep all, quantize by importance), (c) ThinkCompress (evict + quantize), (d) FP16 baseline
- **Metrics**: Accuracy at 4x and 8x compression
- **Success criterion**: Combined > either component alone
- **Failure interpretation**: If eviction-only matches, quantization is unnecessary complexity
- **Table/figure target**: Table 4 (component ablation)
- **Priority**: MUST-RUN

### Block 5: Practical Impact — Memory & Throughput
- **Claim tested**: ThinkCompress enables previously-impossible inference scenarios
- **Why**: Demonstrates practical value beyond accuracy numbers
- **Dataset**: AIME 2024 (long reasoning), custom long-CoT prompts
- **Models**: Qwen3-14B (TP=2), Qwen3-32B (TP=2) if feasible
- **Metrics**: Peak GPU memory, maximum feasible thinking length before OOM, tokens/second
- **Setup**: Increase thinking token budget until OOM, compare FP16 vs ThinkCompress
- **Success criterion**: ThinkCompress sustains 2-3x longer thinking sequences before OOM
- **Failure interpretation**: If memory savings don't translate to practical capability, the method is academic-only
- **Table/figure target**: Figure 5 (memory vs thinking length), Table 5 (max feasible length)
- **Priority**: MUST-RUN

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M0: Sanity | Verify importance scoring works, pipeline runs end-to-end | R001-R003 | Attention importance IS skewed (Gini > 0.6) | 2 GPU-hr | Low — we already saw 88% thinking tokens |
| M1: Characterization | Full Block 1 — establish motivation with data | R004-R006 | Clear skew across models/tasks | 8 GPU-hr | Low |
| M2: Core baselines | FP16 + uniform quant + H2O on GSM8K | R007-R012 | Baselines are reasonable, our metrics are correct | 15 GPU-hr | Medium — need to implement H2O correctly |
| M3: Main method | ThinkCompress on all datasets/compression levels | R013-R020 | <2% loss at 4x on GSM8K | 30 GPU-hr | Medium — importance scoring quality |
| M4: Ablations | Blocks 3-4 decisive experiments | R021-R028 | Adaptive > uniform by ≥5% | 20 GPU-hr | High — this is the key differentiator |
| M5: Scale & polish | Block 5 + figures + extended results | R029-R035 | Memory savings demonstrated on 14B/32B | 25 GPU-hr | Medium — large model logistics |

**Total estimated: ~100 GPU-hours on 2×H100**

## Compute and Data Budget
- Total estimated GPU-hours: 100 (fits in ~2 days on 2×H100)
- Data preparation: GSM8K, MATH, HumanEval, GPQA from HuggingFace (no prep needed)
- AIME 2024: scrape or use existing collection (30 problems)
- Human evaluation: Not needed (automated metrics sufficient)
- Biggest bottleneck: Qwen3-32B inference on 2×H100 (tight memory)

## Risks and Mitigations
- **Risk**: Attention importance is NOT skewed (uniform across thinking tokens)
  **Mitigation**: We already observed 88% thinking tokens for "17*23" — skew is expected. If surprisingly uniform, pivot to temporal decay (recent tokens more important).
- **Risk**: Importance scoring overhead is too high (slows generation)
  **Mitigation**: Use exponential moving average of attention scores (O(1) per token, no extra forward pass). Can also use a lightweight proxy (attention sink pattern).
- **Risk**: H2O/StreamingLLM already achieves similar results
  **Mitigation**: H2O is task-agnostic eviction; ThinkCompress is phase-aware (thinking vs answer). Show the phase-awareness matters.
- **Risk**: Qwen3-32B doesn't fit on 2×H100 with TP=2
  **Mitigation**: Use Qwen3-14B as primary, 32B as stretch goal with aggressive compression.

## Final Checklist
- [x] Main paper tables are covered (Tables 1-5, Figures 1-5)
- [x] Novelty is isolated (Block 3: adaptive vs uniform vs random vs oracle)
- [x] Simplicity is defended (Block 4: eviction vs quant vs both)
- [x] Frontier contribution justified (thinking-phase-aware compression is unique to 2025+ CoT models)
- [x] Nice-to-have separated from must-run (Block 5 scale experiments are last)
