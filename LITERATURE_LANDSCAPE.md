# Literature Landscape: Speculative Decoding + KV Cache Quantization (Apr 2026, updated Apr 9)

## A. Speculative Decoding + Quantization (Direct Intersection)

| Paper | Venue | Method | Result |
|-------|-------|--------|--------|
| QuantSpec (Apple) | ICML 2025 | Self-spec with hierarchical 4-bit KV+weights | 2.5x speedup, >90% acceptance |
| QSpec | EMNLP 2025 | Complementary quant: W4A4 draft + W4A16 verify | 1.64x throughput |
| Quasar | arXiv Mar 2026 | W8A8 quantized verification stage | 1.28x throughput |
| ML-SpecQD | arXiv Mar 2025 | Multi-level MXFP4 quantized drafts | 2.72x speedup |
| SPEQ | arXiv Oct 2025 | Bit-sharing FP format, extract 4-bit draft from FP16 | 2.07x, 0.976 acceptance |
| SpecAttn | arXiv Feb 2026 | Verification-guided sparse attention + self-spec | 2.81x over AR |
| MagicDec | ICLR 2025 | Sparse KV for drafting at large batch | 2.51x speedup |
| SD Meets Quant | arXiv May 2025 | Systematic compatibility study + hierarchical framework | 2.78x for 4-bit 70B |

## B. KV Cache Compression for Reasoning Models (ThinkCompress competitors)

| Paper | Venue | Method | Result |
|-------|-------|--------|--------|
| **ThinKV** (NVIDIA) | arXiv Oct 2025 | Thought-type decomposition (Reasoning/Execution/Transition) + hybrid quant-eviction | <5% KV cache, 5.8x throughput |
| **R-KV** | NeurIPS 2025 | Joint importance + redundancy scoring | 10% KV → ~100% accuracy, 6.6x throughput |
| **LongFlow** (ByteDance) | arXiv Mar 2026 | Long-output KV compression with fused kernel | 11.8x throughput, 80% compression |
| **Crystal-KV** | arXiv Jan 2026 | Answer-first principle: SlipKV vs CrystalKV | SOTA compression for CoT |
| **ForesightKV** | arXiv Feb 2026 | MDP formulation + GRPO for eviction | Beats all at 50% budget |
| **LazyEviction** | arXiv Jun 2025 | Token Importance Recurrence discovery | 50-70% reduction |
| **PM-KVQ** (Tsinghua) | Under review 2025 | Progressive mixed-precision for CoT | 8% better than SOTA at same budget |
| **SideQuest** | arXiv Feb 2026 | LRM self-referential cache management | Novel self-reasoning about KV |

## C. CoT Token Pruning / Compression (Output-side)

TokenSkip (EMNLP 2025), Step Entropy, ASAP, CTS, CtrlCoT, DRP, CoT-Valve, L1, DEER, FlashThink, RCPD, Elastic Reasoning — all mature.

## D. KV Cache Quantization Foundations

KIVI (ICML 2024), KVQuant (NeurIPS 2024), QuaRot, CQ (NeurIPS 2024), TurboQuant, KVTC (ICLR 2026, 20x compression), AQUA-KV (ICML 2025, 2-bit), QServe, KVLinC, ZipCache, GEAR, QJL, MiniCache, KVTuner (ICML 2025), ShadowKV (ICML 2025)

## A2. NEW Papers Found (Apr 9 Update)

| Paper | Venue | Method | Result | Novelty Threat to AcceptSpec |
|-------|-------|--------|--------|------------------------------|
| **SmallKV** | NeurIPS 2025 Spotlight | SLM attention to compensate LLM KV compression + SD integration | 1.75-2.56x throughput | **MEDIUM**: Uses small model for KV importance but attention-based, NOT acceptance-based. AcceptSpec's differentiator (accept ≠ attn) still holds. |
| **SpecPV** | arXiv Dec 2025 | Partial KV states for self-speculative verification | Faster long-context gen | **LOW**: Self-speculation, no draft model, no acceptance-aware compression |
| **SparseSpec** | arXiv Dec 2025 | PillarAttn sparse self-speculation | 2.13x throughput | **NONE**: Sparse attention for drafting, not KV compression |
| **LongSpec** | arXiv Feb 2025 | Constant-size KV draft for long context SD | Improved long-context SD | **NONE**: Draft-side KV, not verifier-side |
| **P-EAGLE** | AWS 2026 | Parallel EAGLE-3 drafting in single pass | 1.69x over EAGLE-3 | **NONE**: Drafting optimization only |
| **TransKV** | TechRxiv 2025 | Transactional KV caching for paged SD memory | Correct rollback | **NONE**: Memory management, not compression |
| **KV Cache for Sampling** | arXiv Jan 2026 | KV cache itself as reasoning substrate | Novel reasoning approach | **NONE**: Different problem entirely |
| **Beyond Tokens** | arXiv Feb 2026 | Semantic-aware SD via internal states | Semantic-level speculation | **NONE**: Different SD paradigm |
| **NVFP4** | NVIDIA 2026 | FP4 KV with dequant to FP8 before attention | Production KV quant | **LOW**: Uniform precision, our method is adaptive |

## A3. Architecture Note: Qwen3.5 GatedDeltaNet

Qwen3.5 uses hybrid attention: ~75% GatedDeltaNet (linear, fixed-size recurrent state) + ~25% full softmax attention. This means traditional KV cache exists ONLY for the 25% full-attention layers. AcceptSpec's per-token KV sensitivity assumptions apply to MHA layers only. **Implication**: Use Qwen3 (standard MHA) as primary, Qwen3.5 for generalization check on the MHA subset.

## E. Key Frontier Papers

| Paper | Why Critical |
|-------|-------------|
| KVTC (NVIDIA, ICLR 2026) | PCA + DP + entropy coding → 20-40x compression. Game-changer. |
| SpecCoT (EMNLP 2025) | Segment-level speculation for reasoning. 48-66% latency reduction. |
| SSD (ICLR 2026) | Speculates-on-speculation. 2x over optimized SD. |
| KaVa (ICLR 2026) | Latent reasoning via compressed KV-cache distillation. |
| MoBiQuant (2026) | Token-level bit-width routing. |

## F. CONFIRMED GAPS (NeurIPS 2026 opportunities, re-validated Apr 9)

1. **🎯 Acceptance-rate-optimized KV compression in SD**: NO paper optimizes KV precision for verifier acceptance rate (all optimize perplexity or attention). Search for "acceptance-critical tokens" / "acceptance-preserving KV" returns ZERO results. **AcceptSpec's core novelty is confirmed.**
2. **SD + reasoning KV compression**: SmallKV (NeurIPS'25) uses SLM attention as proxy but still attention-based, not acceptance-based. Gap remains for acceptance-driven approach.
3. **Segment-level SD + KV compression**: SpecCoT does segment-level speculation but no KV compression; ThinKV does KV compression but no speculation
4. **Joint output pruning + KV compression**: TokenSkip (output) + ThinKV (KV) are disjoint; no joint optimization
5. **KVTC-level compression in SD pipeline**: 20x transform coding unexplored for speculative decoding
6. **Thinking-aware draft model**: No draft model exploits repetitive reasoning patterns for higher acceptance
7. **Safety of compressed reasoning KV**: Only 1 paper (Pitfalls) on failure modes
8. **Dynamic precision across reasoning phases**: Exploration→convergence→verification get uniform treatment
9. **NEW: Hybrid attention (GatedDeltaNet) KV compression for SD**: Qwen3.5's hybrid arch has no SD+KV work yet
