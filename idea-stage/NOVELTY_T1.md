# Novelty Check: Idea T1 — Acceptance-Distortion Theorem + Joint Draft-Compressor Co-training

**Date**: 2026-04-19
**Reviewer models**: WebSearch/WebFetch survey + GPT-5.4 xhigh cross-check

## Verdict: PARTIAL (confidence: MEDIUM-HIGH)

Not an exact duplicate as of Apr 2026. The theorem (claim A) is easy for reviewers to dismiss as a Lipschitz corollary; joint co-training (B) is novel in detail but reads as "obvious combination"; the token-level rank-divergence finding (C) is the strongest genuinely-novel piece.

---

## Claim-by-Claim Novelty

| Claim | Novelty | Closest prior | Risk |
|-------|---------|---------------|------|
| (A) TV(p_accept_full, p_accept_comp) <= f(B, entropy, L_softmax) | LOW-MEDIUM | LK Losses (arXiv:2602.23881), Yin et al. NeurIPS'24 (arXiv:2411.00841), VSD (arXiv:2602.05774) — use alpha=1-TV(p,q) as known identity but WITHOUT compression budget B | Reviewer: "Just KV-error -> logit perturb -> softmax Lipschitz -> TV gap — mathematically routine" |
| (B) Joint draft + KV compressor objective | MEDIUM | QuantSpec (arXiv:2502.10424), MagicDec (arXiv:2408.11049), SmallKV (arXiv:2508.02751), Spec-Meets-Quant (arXiv:2505.22179), SpecExtend — all combine SD+KV-compression but none jointly optimize both with a shared loss | Reviewer: "Trivial extension; show non-separability not just combination" |
| (C) Token-level rank divergence: accept vs attention vs perplexity (rho~=0) | HIGH | None found. SmallKV argues attention-eviction misses some saliency; Expected Attention (arXiv:2510.00636) predicts future attention; Acceptance Dynamics (arXiv:2604.14682) shows only TASK-level entropy-acceptance weak correlation (rho in [-0.20,-0.15]) — not token-level KV ranking | LOW if result reproduces across models/tasks |
| (D) >=5x joint vs 2x independent KV compression | N/A | Depends on baseline fairness; not a novelty claim | Requires strong composed-baseline |

---

## Closest Prior Work (top 3)

1. **LK Losses** (arXiv:2602.23881, Feb 2026): Directly optimizes draft via acceptance-rate-aligned losses using alpha=1-TV(p,q). Has NO KV compression term, analysis purely for draft training. **Differentiation**: Our theorem introduces compression budget B as a variable in the bound; LK treats B as fixed/absent.

2. **Variational Speculative Decoding** (arXiv:2602.05774, Feb 2026): ELBO training for draft to maximize marginal target-acceptance; proves lower bound on expected acceptance length. NO KV compression, only draft training. **Differentiation**: We jointly optimize the KV compressor; VSD's ELBO does not include a compressor term. (Note: arXiv ID may be unstable — verify before citing.)

3. **SpecExtend / QuantSpec / MagicDec** (ICLR'25 / ICML'25): Systems-level SD+KV-compression combinations. None provides an acceptance-distortion theorem, none performs joint training with a shared objective, none studies token-level rank divergence. **Differentiation**: Theorem + joint objective + rank-divergence empirical finding are all absent.

Secondary threats surveyed (none a duplicate): Online Speculative Decoding (arXiv:2310.07177), When Drafts Evolve (arXiv:2603.12617), Learning-to-Draft RL (arXiv:2603.01639), Aurora (arXiv:2602.06932), Cactus (arXiv:2604.04987), Pivot-Aware SD (arXiv:2511.00351 — pivot = OUTPUT tokens, not KV), Adaptive KV-Cache (arXiv:2509.03136, ICLR'26), Expected Attention (arXiv:2510.00636), Acceptance Dynamics (arXiv:2604.14682), Speculative-Speculative Decoding (ICLR'26, arXiv:2603.03251), A Theoretical Perspective for SD (arXiv:2411.00841, NeurIPS'24), KV Cache Transform Coding (ICLR'26).

---

## Claims NOT Published Before

- Joint objective coupling draft distribution q AND KV compressor C under a single surrogate that provably bounds acceptance loss.
- Token-level empirical finding: acceptance-sensitivity ranking has rho~=0 Spearman with attention-importance ranking AND with perplexity-sensitivity ranking (if it reproduces).
- Non-separability / suboptimality-of-independent-composition as a formal result.
- >=5x vs 2x compression-at-matched-acceptance benchmark on reasoning models.

## Claims AT RISK from Concurrent Preprints

- The TV bound via softmax Lipschitz is derivable-in-one-afternoon from LK Losses + softmax 1/2-Lipschitz (arXiv:2510.23012). A Feb-Apr 2026 workshop paper could appear any day. HIGH risk unless paired with a matching lower bound or tightness analysis.
- Any paper combining VSD/LK-style acceptance-aware draft training with a trainable KV mask would preempt claim (B).
- SpecExtend already does cross-model attention-guided draft-side KV eviction — closest system-level threat to "acceptance-aware KV management."

---

## Confidence Rationale: MEDIUM-HIGH

- HIGH confidence no exact duplicate exists as of 2026-04-19 after 10+ searches across arXiv/OpenReview/ICLR'25-'26/NeurIPS'24-'25/ICML'25.
- MEDIUM on concurrent-preprint risk given rapid pace (5+ SD papers in Feb-Apr 2026 alone).
- HIGH that reviewers will attack the theorem as a corollary unless paired with a matching lower bound, a formal non-separability result, an induced optimal budget-allocation rule, OR strong predictive accuracy on real acceptance curves.

---

## Positioning Recommendation

Do NOT lead with "we proved a TV bound with B inserted" — weak novelty. Lead with:
1. **Empirical discovery** (C): acceptance defines a genuinely different KV saliency axis from attention/perplexity — motivate WHY existing KV compressors are misaligned with SD.
2. **Joint framework** (B): show compressor+draft co-design unlocks a regime unreachable by any independent composition of strong components (SmallKV, R-KV, QuantSpec + VSD/LK/EAGLE-3).
3. **Theorem** (A) as SUPPORTING apparatus — include matching lower bound or induced budget-allocation rule, otherwise demote to appendix.

Key defenses to prepare: (i) non-separability proof or tight lower bound, (ii) baselines = strong-KV + strong-draft independent composition, not weak strawmen, (iii) rank-divergence robustness across models (Qwen3.5-9B/27B), tasks (GSM8K/MATH-500), and compressor families (quantization/sparsification/eviction).

---

## Sources

- [Variational Speculative Decoding (2602.05774)](https://arxiv.org/abs/2602.05774)
- [LK Losses (2602.23881)](https://arxiv.org/abs/2602.23881)
- [A Theoretical Perspective for SD (2411.00841)](https://arxiv.org/abs/2411.00841)
- [QuantSpec (2502.10424)](https://arxiv.org/abs/2502.10424)
- [MagicDec (2408.11049)](https://arxiv.org/abs/2408.11049)
- [SmallKV (2508.02751)](https://arxiv.org/abs/2508.02751)
- [Spec-Meets-Quant (2505.22179)](https://arxiv.org/abs/2505.22179)
- [Pivot-Aware SD (2511.00351)](https://arxiv.org/abs/2511.00351)
- [Cactus (2604.04987)](https://arxiv.org/abs/2604.04987)
- [Adaptive KV-Cache Compression ICLR'26 (2509.03136)](https://arxiv.org/abs/2509.03136)
- [Expected Attention (2510.00636)](https://arxiv.org/abs/2510.00636)
- [Acceptance Dynamics (2604.14682)](https://arxiv.org/abs/2604.14682)
- [Softmax 1/2-Lipschitz (2510.23012)](https://arxiv.org/abs/2510.23012)
- [When Drafts Evolve (2603.12617)](https://arxiv.org/abs/2603.12617)
- [Learning to Draft RL (2603.01639)](https://arxiv.org/abs/2603.01639)
- [Online Speculative Decoding (2310.07177)](https://arxiv.org/abs/2310.07177)
- [Speculative Speculative Decoding ICLR'26 (2603.03251)](https://arxiv.org/abs/2603.03251)
