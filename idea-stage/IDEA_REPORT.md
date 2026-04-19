# Idea Discovery Report (v3)

**Direction**: NeurIPS 2026 best-paper-level ideas in LLM inference efficiency (no historical duplicates)
**Date**: 2026-04-19
**Pipeline**: research-lit (2 parallel agents) → idea-creator → novelty-check (3 parallel agents) → research-review (Codex GPT-5.4 xhigh)

## Executive Summary

Merging two landscape scans (general LLM inference + reasoning-specific) produced **5 candidate seeds**, filtered to **3 for deep novelty check**. Codex adversarial review scored the top candidate at **6.4/10 (strong accept, not yet oral)** and recommended reframing.

**Recommended direction: RecipeT1' — "Acceptance is Margin-Sensitive, Not Attention-Sensitive"**

This lifts the paper from "yet another mixed-precision KV heuristic" (crowded, see ThinKV ICLR'26 Oral) to **a new mechanistic law** + its systems consequence. The user's existing C2 rank divergence finding (ρ≈0 across 23K tokens) is the keystone.

## Literature Landscape

See `LANDSCAPE_v3.md` and `LANDSCAPE_REASONING_v3.md` for full detail.

**Critical competitors (ICLR 2026, all published Jan 26, 2026)**:
- **ThinKV** (Oral) — thought-adaptive precision, <5% KV, 5.8× throughput
- **PM-KVQ** (Poster) — mixed-precision KV for long-CoT
- **ChanMix** (Poster) — channel-mixed precision
- **Expected Attention** (Submission) — estimates future attention for eviction
- **HSD** (Oral) — verification-aware SD

**Top recent SD-for-reasoning**:
- **SpecBranch** (ICLR 2026 Poster, arXiv:2506.01979) — branch-aware SD (ghost-scoops T2)
- **SpecPV** (arXiv:2512.02337) — self-SD partial KV
- **Lookahead Reasoning** (arXiv:2506.19830) — step-parallel SD

## Ranked Ideas

### 🏆 T1' (Refined): **"Margin-Sensitive Acceptance: A New Mechanistic Law for Verifier-Aware KV Compression"**

**Status**: RECOMMENDED | Novelty: HIGH (empirical keystone + new mechanism) | Codex review: 6.4/10 → targetable to 8+ with reframing

**Thesis reframe (key change from original AcceptSpec)**:
> Attention mass is the WRONG proxy for verifier-side KV importance. Token-level acceptance-criticality tracks the verifier's **top-2 logit margin** (how easily its argmax flips under KV perturbation), not how much attention it receives. We prove this is a new empirical law across model families and draft mechanisms, derive a separation theorem (attention-top-k is at best O(1/log n) approximation to optimal acceptance-aware selection), and demonstrate a training-free margin-sensitivity score that plugs into existing KV compression systems for additive gains.

**Core contributions**:
1. **C1 (empirical law)**: Token-level rank divergence across 3+ model families (Qwen3.5-MTP, Llama-3.1 + EAGLE, DeepSeek-R1-distill + Medusa) and 2+ draft styles. Show ρ≈0 is not a weak-MTP artifact (also holds at α > 0.5).
2. **C2 (mechanism)**: Acceptance-criticality = verifier's Jacobian alignment with top-2 logit margin. Derive from rejection sampling theory + local linearization. Validate empirically (margin-sensitivity score F1 > 0.75 on oracle labels).
3. **C3 (separation theorem)**: Attention-only ranking incurs Ω(log n) acceptance loss vs optimal, under moderate-entropy verifier assumption. Constructive: dyadic adversarial verifier states.
4. **C4 (systems)**: Margin-sensitivity score as drop-in replacement in SpecPV / SparseSpec / SmallKV — additive gains (2-3pp over their native scoring).
5. **C5 (falsification)**: Oracle FUTURE attention (Expected Attention ICLR'26) still fails to capture acceptance-critical tokens — proving even the strongest attention-proxy is insufficient.

**Pilot evidence in hand**:
- C2 rank divergence: 100 GSM8K × 23,068 tokens on Qwen3.5-9B-MTP: ρ(accept, ppl) = -0.179 (p<10⁻²⁰), ρ(accept, attn) = -0.001 (p=0.86). **Strongly supports the law.**
- Weaknesses: α = 0.33 (undertrained MTP suspected); need to reproduce at higher α.

**Reviewer attacks + defenses** (from Codex review):
1. "Artifact of weak MTP" → reproduce at α > 0.5 on Llama-EAGLE/DeepSeek-Medusa
2. "Incremental vs ThinKV" → reframe around mechanism law, not heuristic
3. "Theorem contrived" → weaken to clean separation/impossibility under moderate-entropy

**Expected paper score**: 7-8/10 (strong accept, borderline oral) if C1+C2+C5 are robustly established. **Oral-grade** if C5 (oracle future attention still fails) is demonstrated cleanly.

**Refined proposal + experiment plan**: `refine-logs-v3/FINAL_PROPOSAL_v3.md` + `refine-logs-v3/EXPERIMENT_PLAN_v3.md` (to be generated next).

---

### Idea 2 (Backup): **T3 — ContraKV: Contrastive Backtrack KV for Self-Correcting CoT**

**Status**: BACKUP | Novelty: NOVEL (0.75 confidence) | Primary risk: Contextual Drag (2602.04288) contradicts premise

**Thesis**: When reasoning LLMs backtrack ("Wait, that's wrong"), keep the failed branch's KV at 2-bit as **contrastive anti-memory** rather than evict. Precision level acts as "contrast strength". Hypothesis: reduces repeated-mistake rate and raises downstream acceptance.

**Why backup not primary**: Contextual Drag shows failed attempts in context cause 10-20% accuracy drops at normal precision. ContraKV requires showing 2-bit precision flips this sign — unproven. Needs a pilot (2 GPU-days) before committing.

**Pilot plan**: Sweep precision ∈ {2, 3, 4, 8, FP16} for failed branches on AIME24 with repeat-mistake rate and acceptance rate as DVs. If 2-bit materially beats both evict and retain-FP16 → green light.

---

### Idea 3 (Demoted): **T2 — BPSD: Branch-Predicting Speculative Draft**

**Status**: DEMOTED — SpecBranch (ICLR 2026 Poster, 2506.01979) already owns "SD + branch prediction"

**If pursued, must reframe as**: "Reasoning-trace-supervised backtrack-aware drafting for MTP self-speculation" with SpecBranch + Lookahead Reasoning as mandatory baselines and AIME/HARP evaluation. Still PARTIAL novelty.

## Eliminated Ideas

- **Seed A** (Acceptance-Distortion Theorem alone) — too easy corollary of softmax Lipschitz + known TV-acceptance identity
- **Seed E** (Verifier-in-the-Loop Draft Distillation) — merged into T1', since joint training only makes sense with C1-C2 as foundation
- **Seed B** (PhaseSpec) — risks being framed as "ThinKV + accept labels"
- **Seed C** (Asymmetric Draft-Target KV) — interesting but incremental without C1-C2
- **Seed D** (Cross-Layer Acceptance Propagation) — elegant but too much theoretical work for 2-week window

## Next Steps

1. **Generate refined proposal** (`refine-logs-v3/FINAL_PROPOSAL_v3.md`) — freeze problem anchor, thesis, and 5 core claims
2. **Generate experiment plan** (`refine-logs-v3/EXPERIMENT_PLAN_v3.md`) — prioritize cross-family reproduction (the main oral gate)
3. **In parallel**: wait for current AcceptSpec M2/M3 re-run to complete (already fixing remaining bugs) — those results feed directly into C1 validation
4. **Before committing to T1'**: verify α > 0.5 is achievable on Llama-3.1 + EAGLE-3 (2-hour sanity pilot)

## Timeline (2 weeks to NeurIPS 2026 deadline)

- Week 1: Cross-family reproduction of C1 (3 model families × 2 draft styles), falsification of future-attention (C5)
- Week 1.5: Mechanism experiments (C2 margin-sensitivity score derivation + oracle F1)
- Week 2: Systems integration (C4: plug into SpecPV/SparseSpec/SmallKV), writing

**Minimum viable oral bar**: C1 + C2 + C5 on 2+ families. **Stretch**: + C3 (theorem) + C4 (systems additive gains).
