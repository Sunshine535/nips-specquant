# LANDSCAPE v3: NeurIPS-2026-Bar Ideas at the SD × KV × Reasoning Intersection

*Scope: Jan 2025 – Apr 2026. Focus: speculative decoding (SD), KV cache management, reasoning LLMs, verification-aware methods. Bar: best-paper / oral / spotlight level.*

---

## 1. Recent Top-Venue Papers (2025 spotlights / orals / ICLR'26 / NeurIPS'25)

**Speculative decoding (draft-verify frontier)**
- **SuffixDecoding** (NeurIPS'25 Spotlight) — model-free draft via suffix trees; 5.3× on agentic loops.
- **Mirror Speculative Decoding** (2510.13161) — parallel draft/target with early-exit across heterogeneous accelerators.
- **Speculative Speculative Decoding** (ICLR'26, aL1Wnml9Ef) — parallelizes the drafting/verify sequential dependence.
- **EAGLE-3** (2503.01840, NeurIPS'25) — training-time-test; removes feature-prediction constraint; scales with data.
- **DEER** (2512.15176) — diffusion drafter + AR verifier; 32-token acceptance length vs EAGLE-3's ~10.
- **Variational SD** (2602.05774) — retrains draft for *sequence* acceptance not token likelihood.
- **ConfSpec** (2602.18447) — confidence-gated step-level escalation; 2.24× on reasoning.
- **SpecGuard** (2604.15244) — step-level verify via attention-grounding + log-prob (no KV angle).
- **Lookahead Reasoning** (2506.19830) — step-level SD replacing >50% of DeepSeek-R1 steps.
- **SpecReason** (2504.07891) — exploits reasoning tolerance; small model drafts intermediate steps.

**KV cache for reasoning (the crowded lane)**
- **TurboQuant** (ICLR'26 oral, Google) — Hadamard + online scalar quant; 5–7× with near-optimal distortion.
- **KVTC** (ICLR'26, 1cef9774...) — PCA + adaptive quant + entropy code; 20× on R1-Qwen2.5 / AIME25 / GSM8K.
- **RLKV** (2510.08525, HGdg76iVo6) — RL-gated per-head full-vs-compressed; reward = answer correctness.
- **ForesightKV** (2602.03203) — training-based eviction via future-attention prediction (Golden + GRPO) on AIME24/25.
- **ThinKV** (2510.01290) — thought-segment classes R/T/E (100–300 tok); 5.8× throughput.
- **SmallKV** (NeurIPS'25, 2508.02751) — small-model attention alignment compensates marginal-token V-cache.
- **R-KV** (2505.24133) — redundancy-aware KV compression (key AcceptSpec baseline).
- **SideQuest** (2602.22603) — LRM self-reasons (aux task) about its own KV usefulness.
- **Hold Onto That Thought** (2512.12008, NeurIPS'25) — diagnostic: H2O/SnapKV dominate; low budgets *lengthen* traces.
- **Expected Attention** (2510.00636), **LookaheadKV** (2603.10899), **SpecKV** (Galim'26), **LAQ** — draft/surrogate tokens to estimate future-query importance.

**Joint SD × KV quant**
- **QuantSpec** (2502.10424) — hierarchical INT4/INT8 KV for self-SD; >90% accept, 2.5× (direct baseline).
- **SpecPV** (2512.02337) — self-SD *partial KV verification* (sink+retrieval+local+buffer blocks) + periodic full refresh. **Close competitor.**
- **SparseSpec** (2512.01278) — PillarAttn sparse attn in draft + dynamic KV; self-speculation.
- **SD-Sparse-Verification** (2512.21911) — sparsifies attn+FFN+MoE during verify; inter-draft/inter-layer reuse.
- **SD meets Quantization** (2505.22179) — W4A16 hierarchical draft/target; "quant has minimal impact on accepted length".

---

## 2. Saturated Directions (do NOT enter)

| Direction | Why saturated |
|---|---|
| Attention-score / H2O-style eviction for reasoning | ≥10 papers 2025-2026 (H2O, SnapKV, PyramidKV, R-KV, SmallKV, ForesightKV, Expected Attention). Hold-Onto-That-Thought shows H2O/SnapKV already near-optimal within this family. |
| Draft/surrogate-response for future importance | LAQ → SpecKV → LookaheadKV → ForesightKV; the whole "use a proxy to predict what query will attend to" idea is mined out. |
| Uniform KV quant for SD (INT4 + hierarchical) | QuantSpec + "SD meets Quant" occupy this. Acceptance-rate preservation via hierarchy + INT8 refresh is a known trick. |
| Step-level SD for reasoning | Lookahead Reasoning, SpecReason, ConfSpec, SpecGuard — 4 concurrent papers in 6 months. |
| Thought-segment compression | ThinKV (R/T/E), SideQuest (self-reasoning), DynTS — crowded. |
| Training a better EAGLE-style draft | EAGLE-1/2/3, HASS, FastEagle, Variational SD, Calibrated SD — fully worked. |
| Test-time compute budget scheduling | Plan-and-Budget, FastTTS, 30B-token benchmark — not best-paper territory anymore. |

---

## 3. White Spaces (real, novelty-defensible gaps)

**W1. Verifier-facing formulation is untouched.** All KV compression optimizes a proxy (attn mass, ppl, answer acc). None optimize SD acceptance probability under compressed KV. ForesightKV uses final-answer reward but is not SD-in-the-loop; RLKV is standard decoding only.

**W2. Triple rank-divergence unclaimed.** No paper establishes ρ≈0 across (accept, attn, ppl) — the user has this C2 result already.

**W3. Per-token mixed precision by acceptance-sensitivity.** MiKV (2402.18096) and "Don't Waste Bits" (2604.04722) do importance-aware mixed precision but use ppl, not SD acceptance.

**W4. Asymmetric draft-target KV.** MagicDec notes draft-KV compression raises acceptance; QuantSpec treats both symmetrically. SpecPV's block partial KV is uniform over selection, not sensitivity-graded.

**W5. CoT temporal dynamics of acceptance.** Hold-Onto-That-Thought shows aggressive eviction lengthens traces; nobody models how acceptance sensitivity evolves along a trace.

**W6. Theoretical acceptance-distortion bound.** No analogue to TurboQuant's distortion bound exists for acceptance under KV budget B.

**W7. Cross-layer acceptance propagation.** All methods treat layers independently; nobody traces acceptance-degradation propagation through the verifier stack.

**W8. Hardware co-design for acceptance-aware per-token precision.** TriAttention/SPEQ/NVFP4-MTP all fix precision; dynamic 2/4/8-bit dispatch driven by an acceptance predictor on Blackwell FP4 is unclaimed.

---

## 4. Concurrent Threats (Feb–Apr 2026 preprints to track)

| Paper | Overlap risk | Mitigation |
|---|---|---|
| **SpecPV** (2512.02337) | *Highest.* Partial KV in self-SD verification with sink/retrieval/local/buffer. BUT uses q·k magnitude (attention-style) for retrieval blocks, not acceptance sensitivity. No ρ-divergence result. | Emphasize: (a) acceptance-sensitivity ≠ attention importance (ρ≈0); (b) AcceptSpec does mixed-precision not binary drop; (c) predictor is trainable. |
| **SparseSpec** (2512.01278) | PillarAttn + dynamic KV in self-SD. Token selection is attention-magnitude based. | Same as SpecPV — the claim "acceptance ≠ attention" falsifies their sparsity rule. |
| **ForesightKV** (2602.03203) | Closest reasoning-aware eviction; GRPO on final answer. But NOT in-the-loop with SD; measures end-task reward. | Show acceptance-rate signal gives *finer* gradient than answer correctness — more samples per token, not sparse reward. |
| **RLKV / "Which Heads Matter"** (2510.08525) | Head-level RL with correctness reward. | Operate at *token×layer* granularity; they are head-level only; combine or differentiate. |
| **SD meets Quantization** (2505.22179) | Observes "quant has minimal impact on accepted length" — *threat to AcceptSpec's motivation*. | Counter: they use uniform quant with intermediate model, never measure per-token acceptance sensitivity. Reasoning traces (100K+ tokens) have different statistics than their short-context benchmarks. |
| **ThinKV** (2510.01290) | Thought-adaptive precision; not SD-aware but close to spirit. | Token-level acceptance-sensitivity is strictly finer than thought-segment R/T/E classes. |
| **SideQuest** (2602.22603) | LRM self-reasons about compression. No SD integration. | Ours gives a bounded, measurable objective (acceptance) vs their LRM-as-oracle. |

**Action**: cite ALL of the above as related work on first draft; differentiate on the (accept vs attn vs ppl) ρ≈0 result as the empirical anchor.

---

## 5. Five Candidate Idea Seeds (NeurIPS-bar)

### Seed A — **Acceptance-Distortion Frontier** *(theoretical anchor for AcceptSpec)*
Prove a distortion bound: expected-acceptance-loss ≤ f(KV compression budget B, target's entropy profile). Analogous to TurboQuant's information-theoretic distortion but with acceptance probability as the loss. Pair with an empirical Pareto frontier on Qwen3-MTP. **Why NeurIPS**: combines a clean theorem with a new empirical finding (C2 divergence); no prior analogue.

### Seed B — **PhaseSpec: CoT-Phase-Conditional Acceptance Policies**
Cluster reasoning traces into phases (exploration / verification / answer-emission) via hidden-state trajectory. Fit a per-phase acceptance-sensitivity predictor. Schedule KV budget along the trace. **Why different from ThinKV**: ThinKV uses attention-sparsity classes — we use learned-acceptance phases. Hold-Onto-That-Thought's "aggressive eviction lengthens traces" becomes a corollary we explicitly control for.

### Seed C — **Asymmetric Draft-Target KV** *(directly extends W4)*
Keep draft's KV in high precision *only* on acceptance-critical tokens (identified by oracle); aggressively compress target's KV on acceptance-*insensitive* tokens. This inverts the QuantSpec symmetry. Combined memory ≤ 50% of QuantSpec; acceptance ≥ QuantSpec.

### Seed D — **Cross-Layer Acceptance Propagation Analysis + AdaLayer-KV**
Measure, per layer, how much a KV perturbation at layer l degrades acceptance at the output. Derive a closed-form propagation factor (linearized Jacobian through the verifier stack). Allocate bits per (token, layer) by marginal acceptance sensitivity. Novel because all prior per-layer allocations (PyramidKV etc.) use attention-entropy heuristics, not acceptance-gradient signals.

### Seed E — **Verifier-in-the-Loop Distillation for Draft Model**
Retrain draft model not to match target's logits (EAGLE-3) and not for sequence acceptance (Variational SD) but to *preserve acceptance under the target's compressed-KV regime*. I.e., joint optimization where the draft is aware of the KV compressor it will be paired with. Output: a draft+compressor pair with 2–3× higher acceptance than EAGLE-3 + R-KV stacked independently. Cleanly beats "independent SD + independent KV" (the exact AcceptSpec thesis).

**Ranked priority**: Seed A (defensible anchor) + Seed E (biggest numerical win) > Seed C (practical) > Seed D (elegant, risky) > Seed B (incremental).

---

*Word count: ~1480.*

*Sources cited inline as arxiv IDs and OpenReview IDs above.*
