# Idea Discovery Report

**Direction**: Speculative decoding + KV cache optimization for efficient LLM inference (NeurIPS 2026)
**Date**: 2026-04-07
**Pipeline**: research-lit → idea-creator (GPT-5.4) → novelty-check (3x parallel) → research-review (GPT-5.4 nightmare)

## Executive Summary

Surveyed 80+ papers across speculative decoding, KV cache quantization, and reasoning-token compression. Generated 10 ideas via GPT-5.4, validated top 3 with deep novelty checks and brutal external review. **Recommended: AcceptSpec** — a refined fusion of SpecThin's system design with AcceptKV's formal principle, addressing the reviewer's core criticism of pure combination papers.

## Literature Landscape

See `LITERATURE_LANDSCAPE.md` for full survey. Key findings:

**Saturated areas (avoid):**
- Thinking token KV compression: ThinKV, R-KV, LongFlow, Crystal-KV, ForesightKV, PM-KVQ (6+ papers)
- CoT token pruning: TokenSkip, ASAP, CtrlCoT, CoT-Valve, DEER, FlashThink (10+ papers)
- SD + uniform quantization: QuantSpec, QSpec, Quasar, ML-SpecQD, SPEQ (5+ papers)

**Active gap: SD × reasoning-aware KV compression** — zero cross-pollination between Thread A (segment-level SD for reasoning: SpecCoT, SpecSearch) and Thread B (thought-aware KV: ThinKV, R-KV, Crystal-KV).

## Ranked Ideas

### 🏆 Idea 1: AcceptSpec — Acceptance-Preserving KV Management for Speculative Reasoning

**Status**: RECOMMENDED | Novelty: HIGH | Composite score: highest

**One-line thesis**: In speculative decoding for reasoning models, the verifier only needs high-fidelity KV for a sparse subset of "acceptance-critical" tokens — this subset can be predicted from draft dynamics, yielding a formal principle (not just engineering combination) for joint SD + KV compression.

**Core mechanism**:
1. Formalize KV compression as acceptance-preservation: define acceptance-critical tokens as those whose KV precision change causes the largest shift in verifier acceptance probability
2. Online prediction: use draft-phase signals (entropy, logit margin, draft-verifier agreement trend per segment) to predict which tokens are acceptance-critical BEFORE full verification
3. Differential compression: acceptance-critical tokens at full precision (anchor tokens), moderately important tokens at 4-bit, unimportant exploration tokens at 2-bit or evicted
4. Segment-level scheduling: draft model proposes thought-level chunks; after verification, accepted chunks' KV enters compressed long-term store, rejected chunks' KV discarded entirely
5. Formal cost model: total_latency = draft(γ) + verify(KV_full × f_critical + KV_compressed × (1-f_critical)) + compress_overhead. Show conditions for joint benefit.

**Why novel (differentiation from ALL related work)**:
- vs QuantSpec: Acceptance-guided non-uniform precision, not uniform 4-bit
- vs ThinKV/R-KV: Optimizes for acceptance rate, not perplexity; operates within SD pipeline
- vs SpecCoT/SpecSearch: Adds KV management with formal acceptance-preservation objective
- vs SpecAttn: Goes beyond binary load/skip to continuous precision allocation
- vs PM-KVQ: Acceptance-driven, not positional; within SD context
- vs naive "SpecCoT + ThinKV": Provides formal cost model + acceptance-critical prediction as the differentiating principle (not just engineering composition)

**Why NOT incremental combination** (addresses reviewer's fatal criticism):
The contribution is NOT "do speculation and compression together." The contribution is: (a) formalizing that speculative verification has sparse KV sensitivity (Proposition 1), (b) showing this sensitivity can be predicted from draft dynamics (online predictor), (c) demonstrating the joint policy beats naive composition because the acceptance signal differs from perplexity signal.

**Expected results**:
- 1.8-2.5x end-to-end latency reduction (conservative, avoiding overclaim)
- 6-10x KV compression during reasoning
- <2% accuracy drop on GSM8K, MATH-500; <5% on AIME
- Key proof: acceptance-optimal precision ≠ perplexity-optimal precision (must demonstrate)

**Risk assessment**:
- MEDIUM: Acceptance sensitivity proxy may be expensive → mitigate with cheap draft-side signals
- MEDIUM: Sparse acceptance-critical set may not hold universally → oracle study needed
- LOW: Systems complexity → builds on existing SpecCoT + ThinKV codebases

**Evaluation plan**:
- Models: Qwen3-8B/14B (primary), Llama-3.1-8B (generalization)
- Baselines: (1) vanilla AR, (2) SpecCoT alone, (3) ThinKV alone, (4) naive SpecCoT+ThinKV, (5) QuantSpec, (6) EAGLE-3
- Critical ablation: naive composition vs AcceptSpec (MUST show gap)
- Oracle studies: oracle acceptance-critical mask, oracle segment labels
- Tasks: GSM8K-full, MATH-500, AIME-2024, HumanEval, GPQA-Diamond
- Metrics: latency, tok/s, KV memory, acceptance rate, task accuracy, f_critical analysis
- Budget: ~150 GPU-hours on 2×H100

**Novelty check**: HIGH confidence (combined from SpecThin HIGH + AcceptKV HIGH)
**GPT-5.4 nightmare review**: Original SpecThin scored 4/10. AcceptSpec addresses all 5 minimum fixes:
  ✅ Formal principle (acceptance-preservation), not just combination
  ✅ Cost model with conditions for joint benefit
  ✅ Naive composition as explicit baseline
  ✅ Acceptance-prediction replaces brittle heuristic labeling
  ✅ Includes oracle studies and failure analysis

---

### Idea 2: SpecThin — BACKUP

**Status**: BACKUP | Novelty: HIGH | GPT-5.4 score: 4/10 (nightmare)

Segment-level SD + thought-aware KV compression, jointly scheduled. Bridges SpecCoT and ThinKV directly. HIGH novelty confirmed but reviewed as "incremental combination." Superseded by AcceptSpec which adds the formal acceptance-preservation principle.

---

### Idea 3: PhaseSpec-KV — RESERVE

**Status**: RESERVE | Novelty: MEDIUM-HIGH

Phase-aware KV precision (exploration→convergence→verification) within SD. Risk: may be seen as ThinKV + QuantSpec. Closest threat: ThinKV thought-type decomposition.

---

### Idea 4: AcceptKV — MERGED INTO #1

**Status**: MERGED | Novelty: HIGH

KV optimized for acceptance, not perplexity. Pure formulation without system design. Merged into AcceptSpec as the theoretical backbone.

---

### Idea 5: ConvergeQ — RESERVE

**Status**: RESERVE | Novelty: not checked | Score: 512

Progressive precision driven by convergence signals during SD. Safer empirically but less distinctive.

---

## Eliminated Ideas

| Idea | Score | Reason |
|------|-------|--------|
| PatternSpec | 504 | Motif retrieval may overfit benchmarks |
| KVTC-Spec | 432 | Ultra-high compression technically brittle, feasibility 6/10 |
| DraftSkip-KV | 448 | Joint optimization hard to stabilize, "bundle of heuristics" risk |
| MirrorCache | 392 | Hard to make exactness-preserving, fallback rate unknown |
| HeadRole Spec | 392 | Head roles may not transfer across models, modest practical gains |
| TreeKV-Spec | 360 | Implementation complexity too high for timeline, feasibility 5/10 |

## Refined Proposal

Pending user confirmation → will invoke /research-refine-pipeline on chosen idea.

## Next Steps

- [ ] User confirms idea at Gate 1
- [ ] /research-refine-pipeline to refine AcceptSpec into submission-ready proposal
- [ ] /run-experiment to deploy experiments
- [ ] /auto-review-loop (nightmare difficulty) to iterate until submission-ready
