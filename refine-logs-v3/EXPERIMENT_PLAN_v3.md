# Experiment Plan v3: MarginSpec

**Date**: 2026-04-19
**Target submission**: NeurIPS 2026 (deadline May 2026, ~14 days)
**Compute budget**: 150 GPU-hours on 8×A800-80GB

## Claim Map

| Claim | Blocks | Must-run? | GPU-hrs | Gate |
|-------|--------|-----------|---------|------|
| **C1**: Rank divergence (≥3 families, ≥2 draft) | B1, B2, B3 | MUST | 50 | ρ ≤ 0.2 on ≥3 families |
| **C2**: Margin-sensitivity mechanism | B4 | MUST | 20 | F1 > 0.75 vs oracle |
| **C3**: Separation theorem | B5 (paper) | SHOULD | 0 (theory) | Clean statement |
| **C4**: Systems integration | B6, B7 | MUST | 40 | ≥ 2pp gain drop-in |
| **C5**: Oracle future-attn fails | B8 | MUST | 15 | F1 gap ≥ 10pp |
| **Robustness**: α sweep | B9 | MUST | 15 | C1 holds at α > 0.5 |

Total: **~140 GPU-hours** (buffer 10)

## Blocks

### B1: Qwen3.5-MTP reproduce (recycled from AcceptSpec)
- **Purpose**: Reproduce C1 on known-working stack (existing `oracle_sensitivity.py` + `triple_divergence.py`)
- **Setup**: Qwen3.5-9B + native MTP, GSM8K 100 probs × 128 tokens × 25 samples
- **Output**: Spearman ρ matrix, margin-sensitivity F1
- **GPU-hrs**: 10 (already completed once, re-run with fixes)
- **Fallback**: Already have partial data (ρ=-0.18/-0.001 from first run)

### B2: Llama-3.1-8B + EAGLE-3
- **Purpose**: Cross-family C1. Critical oral gate.
- **Setup**: Llama-3.1-8B-Instruct + EAGLE-3 draft head (HuggingFace: nm-testing/EAGLE-3-Llama-3.1-8B or retrain). GSM8K 100 + MATH-500 50.
- **Expected α**: 0.5-0.7 (EAGLE-3 much stronger than naive MTP)
- **Output**: same metrics as B1. Critically defeats "low-α artifact" attack.
- **GPU-hrs**: 20
- **Risk**: EAGLE-3 Llama head availability. Mitigation: retrain small EAGLE head in 4 GPU-hours if needed.

### B3: DeepSeek-R1-Distill-Qwen-7B + Medusa
- **Purpose**: Third family (reasoning-tuned), different draft style
- **Setup**: R1-Distill-7B + Medusa heads. GSM8K 100 + AIME24 30.
- **Output**: same metrics.
- **GPU-hrs**: 20

### B4: Margin-sensitivity mechanism validation
- **Purpose**: Derive and validate margin-sens score as the correct proxy (C2)
- **Method**: For each verification step, compute
  - oracle sensitivity S(i) (perturbation-based, same as AcceptSpec)
  - margin-sens m(i) (closed-form)
  - attention sum a(i)
  - ppl sensitivity p(i)
- **Output**: F1 score of each proxy against binarized oracle (top-20%). Scatter plots.
- **GPU-hrs**: 20 (runs alongside B1-B3, reuses their oracle data)
- **Gate**: margin-sens F1 > 0.75, > attention F1 + 10pp

### B5: Separation theorem (no GPU, paper work)
- **Purpose**: Prove Ω(log n) lower bound for attention-only ranking
- **Scope**: Under moderate-entropy verifier + bounded top-2 margin assumption
- **Construction**: Dyadic verifier states where attention-top-k misses log-factor acceptance
- **Output**: Theorem statement + proof in appendix
- **Duration**: 2-3 days of paper work in parallel with experiments

### B6: Systems integration — SpecPV drop-in
- **Purpose**: C4 systems validation
- **Setup**: SpecPV code (arXiv:2512.02337) with scoring function swapped to margin-sens
- **Dataset**: GSM8K-full (1319), MATH-500
- **Metrics**: wall-clock throughput, accuracy, acceptance rate, KV memory
- **Output**: Table: SpecPV-native vs SpecPV+margin-sens at same budget.
- **GPU-hrs**: 15
- **Gate**: ≥ 2pp accuracy or acceptance gain at same budget

### B7: Systems integration — SparseSpec drop-in
- **Purpose**: Second systems validation
- **Setup**: SparseSpec PillarAttn scoring → margin-sens
- **GPU-hrs**: 15
- **Gate**: ≥ 2pp gain

### B8: Oracle future-attention falsification (C5)
- **Purpose**: Show even the ICLR'26 Expected Attention oracle fails C1
- **Method**: Implement Expected Attention prediction from their paper (2-layer MLP on current-step hidden states). Apply as scoring function. Measure rank correlation with oracle acceptance.
- **Output**: Spearman ρ(future-attn, accept) ≤ 0.3 (claim) vs ρ(margin-sens, accept) > 0.7
- **GPU-hrs**: 15

### B9: α-robustness sweep
- **Purpose**: Defeat "weak MTP artifact" reviewer attack
- **Method**: Tune MTP temperature / draft head sampling to achieve α ∈ {0.3, 0.5, 0.7}. Reproduce C1 at each α.
- **GPU-hrs**: 15
- **Gate**: ρ bound holds at α = 0.5+

## Run Order

| Day | Milestone | Blocks | GPU-hrs |
|-----|-----------|--------|---------|
| 1-2 | M1: Recycle + reproduce | B1 fully clean | 10 |
| 3-5 | M2: Cross-family C1 | B2 + B3 + B9 | 55 |
| 6-7 | M3: Mechanism C2 + Oracle future-attn C5 | B4 + B8 (parallel) | 35 |
| 8-10 | M4: Systems C4 | B6 + B7 | 30 |
| 11-12 | M5: Theorem C3 + paper draft | B5 + writing | 0 (paper) |
| 13-14 | M6: Polish + submit | — | 10 (rerun failures) |

## Decision Gates

**Hard stops (ABORT and pivot if fail)**:
- **Day 5 gate**: ≥ 2 families show ρ ≤ 0.2 in B1-B3. If only Qwen3.5 shows it, fall back to AcceptSpec v2 story (strong paper, not oral).
- **Day 7 gate**: margin-sens F1 > attention F1 by ≥ 5pp. If not, mechanism story collapses; fall back to pure empirical discovery paper.

**Soft stops**:
- C4 systems fails → reframe as "principle paper, systems work left for future"
- C5 Expected Attention beats margin-sens → drop C5, weaken paper

## Writing Plan

- **Intro**: lead with C1 result, motivate "attention is wrong proxy"
- **Mechanism section**: margin-sensitivity derivation (compact, 1.5 pages)
- **Empirical section**: cross-family C1 + C2 + α-robustness (4 pages)
- **Theory section**: C3 theorem + proof sketch (appendix full proof)
- **Systems section**: C4 drop-in gains (1.5 pages)
- **Falsification**: C5 Expected Attention fails (0.5 pages)
- **Related work**: position against ThinKV, PM-KVQ, ChanMix, Expected Attention, HSD, SpecPV, SparseSpec, SmallKV, R-KV

## Baselines (must compare)

1. FP16 (upper bound)
2. **SmallKV** (NeurIPS'25 — attention proxy)
3. **SpecPV** (Dec 2025 — partial KV)
4. **SparseSpec** (Dec 2025 — PillarAttn)
5. **ThinKV** (ICLR 2026 Oral — thought-segment precision)
6. **PM-KVQ** (ICLR 2026 Poster — mixed precision)
7. **Expected Attention** (ICLR 2026 — future-attn proxy)
8. **R-KV** (NeurIPS 2025 — redundancy)
9. **H2O** (NeurIPS 2023 — classic attention eviction)
10. **MarginSpec (ours)**

## Artifacts to Track

- `results/marginspec/oracle/` — B1-B3 raw sensitivity data
- `results/marginspec/mechanism/` — B4 F1 scores per proxy
- `results/marginspec/systems/` — B6-B7 wall-clock + accuracy
- `results/marginspec/falsification/` — B8 Expected Attention comparison
- `results/marginspec/robustness/` — B9 α sweep
- `paper/` — LaTeX draft

## Interaction with AcceptSpec v2 (in-flight)

AcceptSpec v2 (the current running experiments) IS B1 with a different framing. The M2/M3 re-run currently fixing bugs on the server will directly produce **B1** data. No need to restart. Just need to:
1. Add margin-sensitivity score computation to the same pipeline (cheap, no extra forward pass)
2. Continue with B2/B3 on different models

**Concrete next action**: modify existing `triple_divergence.py` to also log margin-sensitivity per step, then B1 data doubles as B4 data.
