# Current Result Audit

**Date**: 2026-04-24
**Auditor**: Claude Code (Opus 4.6)

## Result Table

| Result | File | Dataset | Config | Seed | Metric | Value | Compared Against | Supports GPT-5.5 Diagnosis? | Notes |
|--------|------|---------|--------|------|--------|-------|------------------|-----------------------------|-------|
| M0 Oracle | oracle_m0.json | GSM8K (9 problems) | Qwen3.5-9B, γ=5, T=0.0 | Partial | Gini | 0.137 | Threshold 0.5 | YES (negative) | Gate FAIL; only 9/10 problems due to shard loss |
| M0 Oracle | oracle_m0.json | GSM8K | same | Partial | top20_coverage | 0.275 | Threshold 0.8 | YES (negative) | Hard sparsity not supported |
| M1 Oracle | oracle_m1.json | GSM8K (aggregate) | Qwen3.5-9B, γ=5, T=0.0 | Partial | Gini | 0.120 | Threshold 0.5 | YES (negative) | Gate FAIL; aggregate contaminated (=shard0) |
| M1 Oracle | oracle_m1.json | GSM8K (aggregate) | same | Partial | top20_coverage | 0.560 | Threshold 0.8 | YES (negative) | Below threshold; shard1 has 0.70 |
| M1 Shard0 | oracle_m1_shard0.json | GSM8K (shard) | same | Partial | top20_coverage | 0.560 | M1 aggregate | YES (negative) | Aggregate equals shard0 exactly |
| M1 Shard1 | oracle_m1_shard1.json | GSM8K (shard) | same | Partial | top20_coverage | 0.702 | Shard0 | YES (unstable) | Context-dependent sensitivity |
| M2 Divergence | triple_divergence.json | GSM8K (12 problems) | Qwen3.5-9B MTP | Partial | ρ(accept,ppl) | -0.305 | Threshold 0.7 | PARTIAL | Pass (ρ<0.7) but mask artifact risk |
| M2 Divergence | triple_divergence.json | same | same | Partial | ρ(accept,attn) | 0.005 | Threshold 0.7 | UNCERTAIN | Pass but compatible with sparse-vs-dense null artifact |
| M2 Predictor | triple_divergence.json | same | same | Partial | AcceptPredictor F1 | 0.233 | Threshold 0.75 | YES (negative) | Gate FAIL; predictor not viable |
| M2 Attention | triple_divergence.json | same | same | Partial | AttentionProxy F1 | 0.984 | AcceptPredictor | YES (anomalous) | Likely label/mask artifact |
| M2 Margin | triple_divergence.json | same | same | Partial | ρ(accept,margin) | 0.017 | Meaningful signal | YES (negative) | Margin alone not predictive |
| M2 LogitTV | triple_divergence.json | same | same | Partial | ρ(accept,logit_tv) | 0.008 | Meaningful signal | YES (negative) | Logit-TV near zero |
| M3 GSM8K | IN PROGRESS | GSM8K (50, QUICK) | Qwen3.5-9B, budget=0.2, max_tokens=512 | Partial | oracle_accept accuracy | 0.0% (2 shards done) | Baselines | YES (negative) | Even oracle gets 0% at 20% budget |
| M3 GSM8K | IN PROGRESS | GSM8K | same | same | fp16_baseline accuracy | PENDING | Full precision | PENDING | Critical: if fp16=0%, model can't solve |
| Old Bench | benchmark_Qwen2.5*.json | Synthetic | Qwen2.5-7B/14B | Partial | throughput | specquant 17.86 vs AR 39.51 | AR baseline | YES (negative) | Higher acceptance → SLOWER |
| ThinkCompress | thinkcompress_gsm8k*.json | GSM8K (50) | Qwen3-8B | Missing | accuracy | All variants ≈ 0.9 | Uniform 4-bit | YES (negative) | Adaptive not better than uniform |

## Variant Existence Check

**A. Existing Best Positive Fragment Only**: NO — no clean isolated run of old AcceptPredictor top-k policy exists with reliable MTP path

**B. New MAIN METHOD Without New Mechanism**: NO — MARA not yet implemented

**C. Full New MAIN METHOD**: NO — MARA not yet implemented

All three variants are MISSING and must be implemented.

## Result-Based Execution Decision

**Decision: FIX BUG FIRST + ADD LOGGING FIRST**

**Reason**: 
1. P0 bugs in oracle (kv_len) and core_comparison (target-as-draft MTP) contaminate all existing results
2. No matched-support evaluation exists — Spearman ρ values may be artifacts
3. M3 in-progress results show 0% accuracy even for oracle, suggesting either model/budget issue or remaining bugs
4. MARA cannot be meaningfully tested until measurement pipeline is fixed
5. The GPT-5.5 diagnosis's MARA path is evidence-backed and should be implemented, but only after P0 bugs are fixed
