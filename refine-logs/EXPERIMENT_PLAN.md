# Experiment Plan: AcceptSpec

**Problem**: SD verification loads full KV for all tokens but acceptance depends on sparse subset. Compressing non-critical KV reduces bandwidth without hurting acceptance.

**Method Thesis**: Acceptance-critical tokens ≠ attention-important tokens. The right KV compression objective for SD is acceptance rate, not perplexity.

**Date**: 2026-04-07

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|----------------------------|---------------|
| C1: Top-20% tokens capture >80% acceptance sensitivity | Core premise — if false, no paper | Oracle masking sweep on Qwen3-8B × 100 GSM8K problems | B1 |
| C2: Acceptance-ranked ≠ perplexity-ranked | Justifies new objective — if identical to perplexity, no contribution | Spearman ρ < 0.7 between rankings | B2 |
| C3: Acceptance-targeted retention > perplexity-targeted at same memory | Core result — proves the objective matters | ≥3pp accuracy gap at 20% KV retention | B3 |
| C4: AcceptSpec beats naive EAGLE-3+ThinKV | Systems value — proves joint design helps | ≥10% latency improvement at same accuracy | B4 |
| C5: Predictor recovers oracle | Practical value — predictor must be accurate | F1 > 0.75 against oracle critical set | B2 |
| Anti-claim: "The gain is just from KV compression, SD doesn't matter" | Must show SD-specific benefit | Ablation: AcceptSpec w/ SD vs AcceptSpec w/o SD (just compression) | B3 |

## Paper Storyline

**Main paper must prove:**
1. Acceptance sensitivity IS sparse (empirical, not assumed) — B1
2. Acceptance-ranked ≠ perplexity-ranked — B2 (the conceptual contribution)
3. This difference translates to accuracy gains at same memory — B3 (the practical contribution)
4. End-to-end system wins — B4

**Appendix:**
- Per-layer/per-head acceptance sensitivity distribution
- Robustness sweeps (temperature, γ, difficulty)
- Qwen3-14B generalization
- Cost model validation

## Experiment Blocks

### Block 1: Oracle Acceptance Sensitivity Study (MUST-RUN, FIRST)
- **Claim**: C1 — sparsity exists
- **Models**: Qwen3-8B
- **Dataset**: GSM8K (100 problems)
- **Method**: For each verification step, compute S_accept(i) for all tokens by perturbing each to 2-bit with coupled randomness. Aggregate distribution.
- **Metrics**: (a) Cumulative sensitivity curve (% tokens retained vs % sensitivity captured), (b) Gini coefficient, (c) Top-20% coverage
- **Success**: Top-20% tokens → >80% sensitivity. Gini > 0.6.
- **Failure**: If top-20% < 60% → ABORT project
- **Table/Figure**: Figure 1 (cumulative sensitivity curve), Figure 2 (per-layer heatmap)
- **Cost**: ~10 GPU-hours (100 problems × N tokens × perturbation, but can batch)
- **Priority**: MUST-RUN FIRST (gate)

### Block 2: Objective Divergence + Predictor Validation
- **Claims**: C2 (divergence) + C5 (predictor)
- **Models**: Qwen3-8B
- **Dataset**: GSM8K (100 problems, split 50 train / 50 test)
- **Method**:
  (a) Compute acceptance-sensitivity ranking and perplexity-sensitivity ranking for same tokens. Measure Spearman ρ.
  (b) Train AcceptPredictor (logistic regression on draft attention × value norm) on 50 calibration examples. Test F1 on remaining 50.
  (c) Compare predictor vs oracle critical sets
- **Metrics**: (a) Spearman ρ, (b) F1/precision/recall, (c) rank displacement analysis
- **Success**: ρ < 0.7, F1 > 0.75
- **Failure**: ρ > 0.85 → accept and perplexity are basically the same → contribution weakens significantly
- **Table/Figure**: Figure 3 (scatter: acceptance rank vs perplexity rank), Table 1 (predictor metrics)
- **Cost**: ~15 GPU-hours
- **Priority**: MUST-RUN

### Block 3: Objective Comparison — Core Result
- **Claims**: C3 (accept > perplexity) + Anti-claim
- **Models**: Qwen3-8B
- **Dataset**: GSM8K-full (1319), MATH-500
- **Compared systems** (all at SAME memory budget = 20% KV retained):
  (a) Oracle acceptance-ranked retention (upper bound)
  (b) AcceptSpec predicted retention
  (c) Perplexity-ranked retention
  (d) Attention-ranked retention (H2O-style)
  (e) Random retention
  (f) FP16 baseline (no compression)
  (g) AcceptSpec w/o SD (just compression, no speculative decoding) — for anti-claim
- **Metrics**: Task accuracy (exact-match), acceptance rate (for SD variants), KV memory
- **Success**: AcceptSpec ≥3pp > perplexity-ranked. AcceptSpec w/ SD > AcceptSpec w/o SD.
- **Failure**: If perplexity-ranked matches → objective doesn't matter → paper weakens
- **Table/Figure**: Table 2 (main comparison), Figure 4 (accuracy vs KV budget Pareto)
- **Cost**: ~40 GPU-hours
- **Priority**: MUST-RUN (core paper result)

### Block 4: End-to-End System Benchmark
- **Claim**: C4 (beats naive composition)
- **Models**: Qwen3-8B
- **Dataset**: GSM8K-full (1319), MATH-500, AIME-2024 (30)
- **Compared systems**:
  (a) Vanilla autoregressive
  (b) EAGLE-3 (SD only, no KV compression)
  (c) ThinKV (KV compression only, no SD)
  (d) QuantSpec (SD + uniform 4-bit)
  (e) EAGLE-3 + ThinKV (naive composition)
  (f) AcceptSpec (ours)
- **Metrics**: Tokens/sec, wall-clock latency, KV memory (peak + average), task accuracy, acceptance rate
- **Profiling**: Per-component breakdown: draft time, score time, compress time, verify time, reconstruct time
- **Success**: AcceptSpec ≥10% faster than naive composition at same accuracy
- **Failure**: If naive composition matches → joint scheduling doesn't help → just use them separately
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

### Block 6: Generalization Check
- **Claims**: Method transfers across model size
- **Models**: Qwen3-14B
- **Dataset**: GSM8K-full, MATH-500
- **Method**: Run B1 (oracle, 50 problems) + B4 (E2E benchmark) on 14B
- **Success**: Similar sparsity pattern, similar relative gains
- **Table/Figure**: Table 5 (14B results)
- **Cost**: ~25 GPU-hours
- **Priority**: SHOULD-RUN

## Run Order and Milestones

| Milestone | Goal | Blocks | Decision Gate | Cost | Risk |
|-----------|------|--------|---------------|------|------|
| M0: Oracle sanity | Verify sparsity on 10 problems | B1 (partial) | Gini > 0.5 | 2 GPU-hr | LOW |
| M1: Full oracle | Complete sparsity study | B1 (full) | Top-20% > 80% sensitivity | 10 GPU-hr | MEDIUM — core assumption |
| M2: Divergence | Accept ≠ perplexity | B2 | ρ < 0.7 | 15 GPU-hr | MEDIUM — conceptual contribution |
| M3: Core comparison | Accept-targeted > perplexity-targeted | B3 | ≥3pp gap | 40 GPU-hr | HIGH — key differentiator |
| M4: System benchmark | E2E wins | B4 | ≥10% over naive | 35 GPU-hr | MEDIUM — systems complexity |
| M5: Robustness + generalization | Patterns hold | B5 + B6 | Consistent | 45 GPU-hr | LOW |

**Total: ~147 GPU-hours** (within 150 budget)

## Compute and Data Budget
- Total GPU-hours: ~147 on 2×H100
- Data: GSM8K, MATH-500, AIME-2024 from HuggingFace (no prep)
- Calibration: 50 GSM8K problems for AcceptPredictor training
- Biggest bottleneck: B1 oracle study (O(N) forward passes per verification step)

## Risks and Mitigations
- **Risk**: Oracle study shows diffuse sensitivity (no sparsity)
  **Mitigation**: ABORT at M1 (<10 GPU-hours wasted). Report as negative result.
- **Risk**: Accept ≈ perplexity ranking (ρ > 0.85)
  **Mitigation**: Caught at M2. Pivot to showing WHEN they diverge (per reasoning phase).
- **Risk**: Compression overhead eats speedup
  **Mitigation**: Profiling in B4 identifies bottleneck. Use lazy batched compression.
- **Risk**: EAGLE-3 integration breaks KV assumptions
  **Mitigation**: AcceptSpec only touches KV between steps; no EAGLE-3 modifications.

## Final Checklist
- [x] Oracle-first design (abort gate at M1)
- [x] Claim-to-experiment 1:1 mapping
- [x] Naive composition as explicit baseline
- [x] Anti-claim addressed (B3: w/ vs w/o SD)
- [x] Robustness sweeps included
- [x] Within 150 GPU-hour budget
- [x] Discovery paper positioning (not algorithm paper)
