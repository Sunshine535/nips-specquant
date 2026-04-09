# Query Pack (auto-generated 2026-04-09)

## Project Direction
AcceptSpec: acceptance-preserving KV cache management for speculative decoding of reasoning models. Core finding: acceptance-critical tokens ≠ attention-important ≠ perplexity-sensitive. NeurIPS 2026. Qwen3-8B primary.

## Top Gaps
G1: No work optimizes KV for acceptance rate (all use perplexity/attention) — CORE GAP
G2: SD + reasoning-aware KV: zero cross-pollination between SD and thought-aware KV methods
G3: SmallKV (NeurIPS'25) uses attention proxy, never compared against acceptance-based importance
G4: No cross-model validation of acceptance sensitivity sparsity
G5: GatedDeltaNet hybrid attention + SD KV compression unexplored

## Paper Clusters
- **SD+Quant**: QuantSpec, QSpec, Quasar, ML-SpecQD, SPEQ — uniform quantization in SD, 2-2.8x speedup
- **Reasoning KV**: ThinKV, R-KV, LongFlow, Crystal-KV, ForesightKV — thought-aware KV compression, 5-10x
- **Frontier SD**: EAGLE-3, SSD, SpecCoT, P-EAGLE — tree/segment-level SD, 3-6.5x
- **KV Theory**: KVTC, TurboQuant, KVSculpt — compression foundations, 6-40x

## Failed Ideas
- SpecThin (4/10): "incremental combination" of SD+KV, no formal principle
- PhaseSpec-KV: risk of being ThinKV+QuantSpec
- Draft-as-Codec: needs training, violates constraint
- KVSculpt-SD: L-BFGS overhead kills online use

## Active Idea
AcceptSpec + Universal Discovery (9.5/10): acceptance-preserving KV management, cross-model universality, SmallKV comparison
