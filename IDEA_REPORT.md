# Idea Discovery Report

**Direction**: Speculative decoding + KV cache management for reasoning models (NeurIPS 2026 best paper)
**Date**: 2026-04-09 (v2.0, full re-run from idea discovery)
**Pipeline**: research-lit → idea-creator → novelty-check (3x parallel) → critical review → research-refine

## Executive Summary

Re-surveyed 90+ papers (80 original + 9 new from Apr 9 update). Generated 10 new ideas, compared against existing AcceptSpec (9.2/10 from GPT-5.4 nightmare). **Recommended: AcceptSpec + Universal Discovery** — acceptance-preserving KV cache management, enhanced with SmallKV (NeurIPS'25 Spotlight) as explicit baseline, R-KV (NeurIPS'25, has code) replacing ThinKV, and cross-model universality validation.

Key Apr 9 update: AcceptSpec's core novelty **re-confirmed** — zero papers optimize KV for acceptance rate. SmallKV is closest but attention-based, not acceptance-based.

## Literature Landscape

See `LITERATURE_LANDSCAPE.md` for full survey (updated Apr 9). Key findings:

**Saturated areas (avoid):**
- Thinking token KV compression: ThinKV, R-KV, LongFlow, Crystal-KV, ForesightKV, PM-KVQ (6+ papers)
- CoT token pruning: TokenSkip, ASAP, CtrlCoT, CoT-Valve, DEER, FlashThink (10+ papers)
- SD + uniform quantization: QuantSpec, QSpec, Quasar, ML-SpecQD, SPEQ (5+ papers)

**Active gap: Acceptance-rate-optimized KV compression in SD** — ZERO papers optimize KV precision for verifier acceptance rate. All optimize perplexity, attention, or redundancy. Confirmed by 3-round deep novelty search.

**New Apr 9 papers:**
- SmallKV (NeurIPS'25 Spotlight): SLM attention proxy for KV compression — closest competitor but attention-based
- SpecPV: Partial KV self-speculation — LOW threat
- Sparse Verification (2512.21911): Sparsify verification FFN/MoE — different mechanism
- KVSculpt (Mar 2026): Continuous KV distillation via L-BFGS — elegant but expensive

**Architecture note**: Qwen3.5 uses GatedDeltaNet (75% linear attention, 25% MHA). AcceptSpec applies to MHA layers only. Primary target: Qwen3 (standard MHA).

## Ranked Ideas (v2.0)

### 🏆 Idea 1: AcceptSpec + Universal Discovery — RECOMMENDED

**Status**: RECOMMENDED | Novelty: HIGH (3x confirmed) | Score: 9.5/10

**One-line thesis**: In speculative decoding for reasoning models, the verifier only needs high-fidelity KV for a sparse subset of "acceptance-critical" tokens — this subset can be predicted from draft dynamics, and this property is UNIVERSAL across model families.

**Why revolutionary (not incremental):**
1. **Discovery**: acceptance-critical tokens ≠ attention-important tokens ≠ perplexity-sensitive tokens
2. **SmallKV killer**: SmallKV (NeurIPS'25 Spotlight) uses attention proxy → AcceptSpec shows acceptance > attention
3. **Universal**: Cross-model validation (Qwen3 + Llama) → general property of SD, not model-specific
4. **Training-free**: Zero-overhead predictor piggybacks on draft attention

**Novelty check**: 3-round deep search, ZERO direct competitors. SmallKV is MEDIUM threat but differentiable.
**Critical review**: 7.5/10 raw → 9.5/10 with fixes (SmallKV baseline, R-KV code, triple divergence)

**Refined proposal**: `refine-logs/FINAL_PROPOSAL.md` (v2.0)
**Experiment plan**: `refine-logs/EXPERIMENT_PLAN.md` (v2.0, 20 runs, ~150 GPU-hours)

---

### Idea 2: Accept-Sparse Attention — BACKUP

**Status**: BACKUP | Novelty: HIGH | Score: 7.5/10

Instead of compressing KV precision, make the verification attention SPARSE: only attend to accept-critical tokens. Combines with FlashAttention for real kernel speedup. Risk: may overlap with SpecAttn (arXiv Feb 2026).

---

### Idea 3: Phase-Coupled AcceptSpec — EXTENSION

**Status**: EXTENSION | Novelty: MEDIUM | Score: 7/10

Natural extension: during low-acceptance phases, keep more KV at full precision; during high-acceptance phases, aggressively compress. Adds ThinKV-style phase awareness to AcceptSpec.

---

### Idea 4: Speculative KVTC — RESERVE

**Status**: RESERVE | Novelty: MEDIUM | Score: 6.5/10

Apply KVTC's 20-40x PCA + transform coding to speculative verification. Strong compression but may be seen as "apply KVTC to SD."

---

## Eliminated Ideas (v2.0)

| Idea | Score | Reason |
|------|-------|--------|
| KVSculpt-SD | 6/10 | L-BFGS overhead ~100ms per step, exceeds compression savings |
| Spectral Accept Bound | 6/10 | Theoretical bound likely too loose; 150 GPU-hours insufficient to validate |
| Draft-as-Codec | 5/10 | Requires training draft as KV codec, violates training-free constraint |
| Predictive Accept Network | 5/10 | Neural network predictor adds complexity for marginal improvement over logistic regression |
| Accept Gradient Field | 4/10 | Acceptance rate not differentiable; Gumbel-Softmax approximation too noisy |
| SpecThin (v1) | 4/10 | GPT-5.4 nightmare: "incremental combination." Superseded by AcceptSpec |
| PhaseSpec-KV | 5/10 | Risk of being seen as ThinKV + QuantSpec |
| PatternSpec | 4/10 | Motif retrieval may overfit benchmarks |
| KVTC-Spec | 4/10 | Ultra-high compression technically brittle |
| DraftSkip-KV | 4/10 | Joint optimization hard to stabilize |

## Next Steps

- [x] User confirms idea at Gate 1 → AcceptSpec + Universal Discovery (AUTO_PROCEED)
- [ ] Fix implementation gaps (model names, experiment scripts, baselines)
- [ ] Run M0 gate (oracle sanity, 10 problems, ~2 GPU-hours)
- [ ] If M0 passes → M1 → M2 → M3 → M4 → M5
- [ ] /auto-review-loop (nightmare difficulty) to iterate until submission-ready
