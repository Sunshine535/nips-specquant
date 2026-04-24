# Reliability Audit

**Date**: 2026-04-24
**Purpose**: Mark each existing result's reliability status. No raw data is modified.

## Result Reliability Status

| Result File | Status | Known Bugs | Paper-Claimable? | Notes |
|-------------|--------|------------|------------------|-------|
| `results/acceptspec/oracle_m0.json` | UNRELIABLE | P0 kv_len bug; only 9/10 problems | NO | Historical negative evidence only |
| `results/acceptspec/oracle_m0_shard*.json` | UNRELIABLE | P0 kv_len bug | NO | Raw shard data, same bug |
| `results/acceptspec/oracle_m1.json` | UNRELIABLE | P0 kv_len bug; aggregate = shard0 copy | NO | Merge contaminated |
| `results/acceptspec/oracle_m1_shard*.json` | UNRELIABLE | P0 kv_len bug; zero-filled aggregation | NO | Raw shard data, same bug |
| `results/acceptspec/divergence/triple_divergence.json` | PARTIALLY RELIABLE | Mask artifact risk; matched-support not verified | NO (metrics) | Spearman ρ may be artifact; negative predictor F1 is genuine |
| `results/acceptspec/divergence/triple_divergence_shard*.json` | PARTIALLY RELIABLE | Same as above | NO | Individual shard data |
| `results/acceptspec/.markers/M0_done` | STALE | Pre-fix marker | N/A | Does not indicate reliable results |
| `results/acceptspec/.markers/M2_done` | STALE | Pre-fix marker | N/A | Same |
| `results/benchmark/benchmark_Qwen2.5*.json` | HISTORICAL | Different model/setup | NO | Useful negative: acceptance ≠ speed |
| `results/gsm8k/thinkcompress*.json` | HISTORICAL | Different method/setup | NO | Useful negative: adaptive ≈ uniform |
| `results/microbenchmark/microbenchmark*.json` | HISTORICAL | Old setup | NO | Reference only |
| `logs/M0_oracle_sanity.log` | REFERENCE | Documents P0 bug era | N/A | Keep for debugging history |
| `logs/M1_oracle_full.log` | REFERENCE | Documents P0 bug era | N/A | Keep for debugging history |
| `logs/M2_divergence.log` | REFERENCE | Documents M2 failures | N/A | Keep for debugging history |
| `logs/M3_comparison_gsm8k.log` | IN PROGRESS | M3 running on remote server | PENDING | Using fixed code (max_tokens=512, real draft_probs) |

## What These Results Teach Us (Historical Value)

1. **M0/M1**: Hard sparsity (top-20% → 80%) is NOT reliably demonstrated. Signal is moderate at best (56-70%), even with bugs that may inflate or deflate it.
2. **M2**: AcceptPredictor F1=0.23 is a genuine failure. Current binary classifier cannot identify acceptance-critical tokens.
3. **M2**: AttentionProxy F1=0.98 is anomalously high — likely label/mask artifact, not evidence that attention works.
4. **M2**: Margin correlations near zero — margin alone is not a predictor of acceptance sensitivity.
5. **Old benchmark**: Higher acceptance rate does NOT imply faster throughput.
6. **ThinkCompress**: Generic adaptive importance scoring does not beat uniform quantization.

## Rules for Using These Results

- DO NOT cite any value from UNRELIABLE results as paper evidence
- DO cite the negative findings as motivation for MARA pivot
- DO preserve all raw files for reproducibility and historical record
- DO NOT delete any result file
- DO run new experiments with fixed code before making any claims
