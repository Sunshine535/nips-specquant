# Experiment Plan: AcceptSpec v2.0

**Problem**: SD verification loads full KV for all tokens but acceptance depends on sparse subset. Compressing non-critical KV reduces bandwidth without hurting acceptance.

**Method Thesis**: Acceptance-critical tokens ≠ attention-important tokens ≠ perplexity-sensitive tokens. The right KV compression objective for SD is acceptance rate.

**Date**: 2026-04-09 (v2.0, re-planned from idea discovery)

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|----------------------------|---------------|
| C1: Top-20% tokens capture >80% acceptance sensitivity | Core premise — if false, no paper | Oracle masking sweep on Qwen3-8B × 100 GSM8K problems | B1 |
| C2: Accept-ranked ≠ perplexity-ranked ≠ attention-ranked | Justifies new objective vs SmallKV | Triple Spearman ρ < 0.7 between all pairs | B2 |
| C3: Accept-targeted retention > perplexity AND attention at same memory | Core result — proves the objective matters | ≥3pp accuracy gap at 20% KV retention over both | B3 |
| C4: AcceptSpec beats naive SD+R-KV | Systems value — proves joint design helps | ≥10% latency improvement at same accuracy | B4 |
| C5: Predictor recovers oracle | Practical value — predictor must be accurate | F1 > 0.75 against oracle critical set | B2 |
| C6: Universality across models | Discovery paper must show generality | Pattern holds for Llama-3.1 | B6 |
| Anti-claim: "The gain is just from KV compression, SD doesn't matter" | Must show SD-specific benefit | AcceptSpec w/ SD vs AcceptSpec w/o SD | B3 |

## Paper Storyline

**Main paper must prove:**
1. Acceptance sensitivity IS sparse (empirical, not assumed) — B1
2. Accept ≠ perplexity ≠ attention — B2 (the conceptual contribution, SmallKV differentiator)
3. This difference translates to accuracy gains at same memory — B3 (the practical contribution)
4. End-to-end system wins — B4

**Appendix:**
- Per-layer/per-head acceptance sensitivity distribution
- Robustness sweeps (temperature, γ, difficulty)
- Llama-3.1 + Qwen3.5 generalization
- Cost model validation

## Experiment Blocks

### Block 1: Oracle Acceptance Sensitivity Study (MUST-RUN, FIRST)
- **Claim**: C1 — sparsity exists
- **Models**: Qwen3-8B (target), Qwen3-0.6B (draft)
- **Dataset**: GSM8K (100 problems)
- **Method**: For each verification step, compute S_accept(i) for all tokens by perturbing each to 2-bit with coupled randomness. Aggregate distribution.
- **Metrics**: (a) Cumulative sensitivity curve, (b) Gini coefficient, (c) Top-20% coverage
- **Success**: Top-20% tokens → >80% sensitivity. Gini > 0.6.
- **Failure**: If top-20% < 60% → ABORT project
- **Table/Figure**: Figure 1 (cumulative sensitivity curve), Figure 2 (per-layer heatmap)
- **Cost**: ~10 GPU-hours
- **Priority**: MUST-RUN FIRST (gate)

### Block 2: Triple Divergence + Predictor Validation
- **Claims**: C2 (divergence) + C5 (predictor)
- **Models**: Qwen3-8B
- **Dataset**: GSM8K (100 problems, split 50 train / 50 test)
- **Method**:
  (a) Compute acceptance-sensitivity ranking, perplexity-sensitivity ranking, AND attention-importance ranking for same tokens. Measure pairwise Spearman ρ.
  (b) Train AcceptPredictor on 50 examples, test on 50.
  (c) Compare predictor vs oracle critical sets.
  (d) Compare AcceptPredictor vs SmallKV-style attention proxy (same predictor framework, different features).
- **Metrics**: (a) 3 pairwise Spearman ρ, (b) F1/precision/recall, (c) rank displacement analysis
- **Success**: All pairwise ρ < 0.7, F1 > 0.75. AcceptPredictor F1 > attention-proxy F1.
- **Failure**: ρ(accept, attention) > 0.85 → SmallKV approach is nearly as good → contribution weakens
- **Table/Figure**: Figure 3 (3-way scatter), Table 1 (predictor metrics + SmallKV comparison)
- **Cost**: ~15 GPU-hours
- **Priority**: MUST-RUN

### Block 3: Objective Comparison — Core Result
- **Claims**: C3 (accept > perplexity > attention) + Anti-claim
- **Models**: Qwen3-8B
- **Dataset**: GSM8K-full (1319), MATH-500
- **Compared systems** (all at SAME memory budget = 20% KV retained):
  (a) Oracle acceptance-ranked retention (upper bound)
  (b) AcceptSpec predicted retention
  (c) Perplexity-ranked retention
  (d) Attention-ranked retention (SmallKV/H2O-style)
  (e) R-KV (redundancy + importance joint scoring)
  (f) Random retention
  (g) FP16 baseline (no compression)
  (h) AcceptSpec w/o SD (just compression, no speculative decoding) — for anti-claim
- **Metrics**: Task accuracy (exact-match), acceptance rate (for SD variants), KV memory
- **Success**: AcceptSpec ≥3pp > perplexity-ranked AND ≥3pp > attention-ranked.
- **Failure**: If attention-ranked matches → SmallKV approach works → paper weakens
- **Table/Figure**: Table 2 (main comparison), Figure 4 (accuracy vs KV budget Pareto)
- **Cost**: ~40 GPU-hours
- **Priority**: MUST-RUN (core paper result)

### Block 4: End-to-End System Benchmark
- **Claim**: C4 (beats naive composition)
- **Models**: Qwen3-8B
- **Dataset**: GSM8K-full (1319), MATH-500
- **Compared systems**:
  (a) Vanilla autoregressive
  (b) Vanilla SD (rejection sampling, γ=5)
  (c) R-KV only (no SD)
  (d) SmallKV only (no additional SD)
  (e) SD + R-KV (naive composition)
  (f) SD + SmallKV (naive composition with attention proxy)
  (g) QuantSpec (self-speculative, 4-bit)
  (h) AcceptSpec (ours)
  (i) EAGLE-3 + AcceptSpec (if EAGLE head available)
- **Metrics**: Tokens/sec, wall-clock latency, KV memory (peak + average), task accuracy, acceptance rate
- **Profiling**: Per-component breakdown: draft time, score time, compress time, verify time
- **Success**: AcceptSpec ≥10% faster than SD + R-KV at same accuracy
- **Table/Figure**: Table 3 (main benchmark), Figure 5 (latency breakdown), Figure 6 (memory timeline)
- **Cost**: ~35 GPU-hours
- **Priority**: MUST-RUN

### Block 5: Robustness Sweeps
- **Claims**: Robustness of C1-C3
- **Models**: Qwen3-8B
- **Dataset**: GSM8K (200 samples)
- **Sweeps**:
  (a) Temperature: τ ∈ {0.0, 0.3, 0.6, 0.9}
  (b) Draft length: γ ∈ {3, 5, 7, 10}
  (c) Difficulty strata: easy (1-step) / medium / hard (multi-step)
  (d) KV budget: {10%, 20%, 30%, 50%} retained
- **Metrics**: Sparsity pattern stability, accuracy, acceptance rate
- **Success**: Patterns hold across settings
- **Table/Figure**: Table 4 (robustness grid), Figure 7 (sparsity vs temperature/γ)
- **Cost**: ~20 GPU-hours
- **Priority**: SHOULD-RUN

### Block 6: Universality Check
- **Claims**: C6 — method transfers across model families
- **Models**: Llama-3.1-8B (or Llama-3.2-3B → 3.1-8B for compute), optionally Qwen3.5-9B (MHA layers)
- **Dataset**: GSM8K-full, MATH-500
- **Method**: Run B1 (oracle, 50 problems) + B4 (E2E benchmark) on Llama
- **Success**: Similar sparsity pattern, similar relative gains
- **Table/Figure**: Table 5 (cross-model results)
- **Cost**: ~30 GPU-hours
- **Priority**: SHOULD-RUN (strengthens discovery framing)

## Run Order and Milestones

| Milestone | Goal | Blocks | Decision Gate | Cost | Risk |
|-----------|------|--------|---------------|------|------|
| M0: Oracle sanity | Verify sparsity on 10 problems | B1 (partial) | Gini > 0.5 | 2 GPU-hr | LOW |
| M1: Full oracle | Complete sparsity study | B1 (full) | Top-20% > 80% sensitivity | 10 GPU-hr | MEDIUM |
| M2: Triple divergence | Accept ≠ perplexity ≠ attention | B2 | All pairwise ρ < 0.7 | 15 GPU-hr | MEDIUM |
| M3: Core comparison | Accept-targeted > both alternatives | B3 | ≥3pp gap over both | 40 GPU-hr | HIGH |
| M4: System benchmark | E2E wins over naive composition | B4 | ≥10% over SD+R-KV | 35 GPU-hr | MEDIUM |
| M5: Robustness + universality | Patterns hold | B5 + B6 | Consistent | 50 GPU-hr | LOW |

**Total: ~150 GPU-hours** (within budget, auto-adaptive GPU allocation)

## Compute and Data Budget
- Total GPU-hours: ~150 (auto-adaptive: 1/2/4+ GPU configs via gpu_auto.py)
- Data: GSM8K, MATH-500 from HuggingFace (no prep)
- External baselines: R-KV (GitHub: Zefan-Cai/R-KV), SmallKV (implement attention-proxy variant)
- Calibration: 50 GSM8K problems for AcceptPredictor training

## Risks and Mitigations
- **Risk**: Oracle study shows diffuse sensitivity → ABORT at M1 (<10 GPU-hours wasted)
- **Risk**: Accept ≈ attention (SmallKV wins) → Caught at M2. Pivot to showing WHEN they diverge (per reasoning phase)
- **Risk**: Compression overhead eats speedup → Profiling in B4 identifies bottleneck. Use lazy batched compression
- **Risk**: R-KV integration fails → Fall back to H2O/SnapKV (simpler baselines with known implementations)
- **Risk**: EAGLE-3 head unavailable for Qwen3 → Use vanilla SD, note as limitation

## Final Checklist
- [x] Oracle-first design (abort gate at M1)
- [x] Claim-to-experiment 1:1 mapping
- [x] Naive composition as explicit baseline (SD+R-KV, SD+SmallKV)
- [x] Anti-claim addressed (B3: w/ vs w/o SD)
- [x] SmallKV comparison (attention vs acceptance)
- [x] Robustness sweeps included
- [x] Cross-model universality (Llama + optional Qwen3.5)
- [x] Within 150 GPU-hour budget
- [x] Discovery paper positioning
- [x] Auto-adaptive multi-GPU support
