# Research Proposal v3: MarginSpec

## "Acceptance is Margin-Sensitive, Not Attention-Sensitive: A New Mechanistic Law for Verifier-Aware KV Compression"

**Version**: 3.0 (Apr 19, 2026 — refined from AcceptSpec v2 after Codex GPT-5.4 xhigh review)
**Target**: NeurIPS 2026 (deadline: May 2026)
**Codex current score**: 6.4/10 → target 8+ (borderline oral)

## Problem Anchor

- **Bottom-line problem**: Every KV compression method for speculative decoding selects tokens by attention mass (SmallKV, SpecPV, SparseSpec, H2O, SnapKV, Expected Attention) or perplexity (KV-Sketch family). This is the WRONG proxy for verifier-side preservation. We prove empirically and theoretically that acceptance-criticality is near-orthogonal to both, and identify the correct underlying quantity: the verifier's **top-2 logit margin sensitivity**.
- **Must-solve bottleneck**: In reasoning LLMs with long CoT (thousands of tokens), verifier-side KV compression compounds with speculative decoding to form the dominant inference bottleneck. Using the wrong importance signal leaves 2-3pp acceptance on the table, which compounds to 20-40% throughput loss over long traces.
- **Constraints**: Training-free (no draft/target retraining), NeurIPS 2026 timeline (2 weeks), ≤200 GPU-hours on available A800-80GB × 8.
- **Success condition**:
  1. **C1 (empirical law)**: Spearman ρ(accept, attn), ρ(accept, ppl) ≤ 0.2 across ≥3 model families and ≥2 draft mechanisms
  2. **C2 (mechanism)**: Margin-sensitivity score achieves F1 > 0.75 against oracle acceptance labels (vs <0.5 for attention proxy)
  3. **C5 (falsification)**: Oracle future-attention (from Expected Attention ICLR'26) also fails C1 bound
  4. **C4 (systems)**: Drop-in replacement in SpecPV/SparseSpec yields ≥2pp acceptance gain at same budget

## Method Thesis

Token-level acceptance sensitivity = verifier Jacobian alignment with the top-2 logit margin direction. This quantity can be computed online from draft attention + output layer norm + a cheap margin probe, with zero training. It subsumes attention-importance as a special case (when margins are tight and uniform) but diverges sharply in the typical reasoning regime.

## Dominant Contribution

A **mechanistic law**: acceptance is margin-sensitive, not attention-sensitive. This is a new finding about the structure of speculative verification — not a heuristic improvement. The consequence is a correction to half a dozen concurrent papers (SmallKV, SpecPV, SparseSpec, Expected Attention, SmallKV) that all use the wrong proxy.

**Contribution hierarchy**:

1. **Primary (scientific)**: Empirical law C1 + mechanism C2, reproduced across ≥3 model families and ≥2 draft mechanisms. This is the oral-grade piece.
2. **Secondary (theoretical)**: Separation theorem (C3) — attention-only ranking is at best O(1/log n) approximation to acceptance-optimal.
3. **Tertiary (falsification)**: Even oracle future-attention (C5) fails — strongest attention-proxy insufficient.
4. **Quaternary (systems)**: Margin-sensitivity score as drop-in replacement in existing systems for additive gains.

**Paper template**: Discovery paper with mechanism explanation and falsification of competing proxies (cf. Thought Anchors, Attention Sink, but for verifier-side KV). Strong empirical findings + clean theoretical statement.

## Core Mechanism

### 1. Margin-Sensitivity Score (training-free, online-computable)

For each KV token i at verification step:
```
m(i) = ∂(verifier_top2_logit_margin) / ∂(KV_i)  [Jacobian]
     ≈ ||V_i||_2 × α_h(q, k_i) × λ_margin(h)
```
where λ_margin(h) is head-specific margin alignment (computed once per layer from pilot calibration).

Key difference from attention: λ_margin can be **negative** for heads that smooth predictions, and varies per head by orders of magnitude. Attention sum ignores this.

### 2. Mixed-Precision Schedule

Budget B of FP16 tokens, rest at 2-bit (Hadamard + scalar quant from TurboQuant stack):
- Sort tokens by |m(i)|
- Top B% → FP16
- Remaining → 2-bit with Hadamard rotation

No training, no draft modification. Plug into existing SD pipeline.

### 3. Systems Integration (C4)

Drop-in replacement for the scoring function in:
- **SpecPV** (which uses attention magnitude for block selection)
- **SparseSpec** (PillarAttn attention-top-k)
- **SmallKV** (SLM attention proxy)

Measure the **composite gain**: their native score → margin-sensitivity score, at same KV budget.

## Key Claims

| Claim | Experiment | Metric | Threshold |
|-------|-----------|--------|-----------|
| **C1** Rank divergence across families | Cross-family oracle | Spearman ρ(accept, attn) and ρ(accept, ppl) | ≤ 0.2 on ≥3 families |
| **C2** Margin-sensitivity predicts acceptance | Oracle labeling on held-out | F1 score vs oracle | > 0.75 |
| **C3** Attention has Ω(log n) lower bound | Theorem + synthetic verifier | Formal proof + demo | — |
| **C4** Systems additive gain | SpecPV/SparseSpec swap | Acceptance gain at same budget | ≥ 2pp |
| **C5** Oracle future-attn fails | Expected-Attention comparison | F1 gap vs margin-sens | ≥ 10pp |
| **Robustness** α > 0.5 reproduces C1 | Strong-draft setup | Same ρ bound | Same threshold |

## Evaluation Plan

### Primary (the oral gate)
1. **Cross-family C1**: Reproduce rank divergence on Qwen3.5-9B-MTP, Llama-3.1-8B + EAGLE-3, DeepSeek-R1-Distill-Qwen-7B + Medusa. 100 GSM8K + 100 MATH-500 problems each.
2. **C2 mechanism**: Derive margin-sensitivity score, validate F1 on held-out oracle labels. Compare against attention-proxy, ppl-proxy, SmallKV-score, SpecPV-score.
3. **C5 falsification**: Reproduce Expected Attention's oracle future-attention prediction, apply as scoring function, measure F1 vs margin-sens.

### Secondary
4. **C3 theorem**: Prove under moderate-entropy verifier + bounded top-2 margin. Write up cleanly in appendix.
5. **C4 systems**: Integrate margin-sens into SpecPV / SparseSpec code; measure end-to-end acceptance + latency.
6. **Robustness**: α sweep (0.3, 0.5, 0.7) to defeat "weak MTP artifact" attack.

### Baselines (mandatory)
- ThinKV (ICLR 2026 Oral) — same-spirit paper, different proxy
- PM-KVQ (ICLR 2026 Poster) — mixed-precision
- ChanMix (ICLR 2026 Poster) — channel-mixed
- Expected Attention (ICLR 2026) — future-attention proxy (our falsification target C5)
- SmallKV (NeurIPS 2025) — attention proxy
- SpecPV (arXiv:2512.02337) — partial KV verification
- SparseSpec (arXiv:2512.01278) — PillarAttn
- R-KV, H2O (classic baselines)

## Compute: ~150 GPU-hours on 8×A800-80GB. Timeline: 14 days.

## Risks and Mitigations

1. **Risk: C1 is weak-MTP artifact** → Mitigation: reproduce on Llama-3.1-EAGLE (known stronger α). Kill project if even 1 family shows ρ > 0.4.
2. **Risk: margin-sens F1 ≈ attention F1** (mechanism doesn't pan out) → Fall back to "AcceptSpec mixed-precision" story (weaker, borderline accept only).
3. **Risk: systems integration flat** (C4 fails) → Report as "principle holds, systems work left for future" — C1+C2+C5 alone is still a strong paper.
4. **Risk: ThinKV/HSD ghost-scoops at submission** → C1's empirical law is unscooped per novelty check (Apr 19); monitor arXiv weekly until deadline.

## Complexity Intentionally Rejected
- Theoretical acceptance-distortion bound (Codex: too easy corollary, will be attacked)
- Joint draft-compressor training (Codex: separation theorem is harder to prove than joint is to implement — reframe to discover vs engineer)
- Qwen3.5 linear-attention layers (irrelevant, only MHA layers matter)
- RL-based predictor (logistic/linear margin probe is sufficient)

## What Changed from v2 (AcceptSpec)

| Change | Reason (Codex review + novelty check) |
|--------|---------|
| Title: "Margin-Sensitive Acceptance" not "AcceptSpec" | Center on mechanism, not method |
| Lead with C1 discovery, not mixed-precision method | Mechanism is oral-grade; method is incremental |
| Add Oracle Future-Attention falsification (C5) | Counters Expected Attention ICLR'26 |
| Cross-family (Qwen + Llama + DeepSeek) | Defeats "weak-MTP artifact" attack |
| Add ≥2 draft styles (MTP + EAGLE/Medusa) | Same defense |
| Margin-sensitivity score replaces attention×value_norm | Principled, matches mechanism |
| Separation theorem (Ω(log n)) replaces distortion bound | Less contrived, more defensible |
| Systems integration as DROP-IN for SpecPV/SparseSpec | Additive gains, not replacement story |
| Report wall-clock latency, not just acceptance | Avoids "per-token mixed precision kills kernel" attack |
