# Claim Update Log

No paper claims have been updated in this session. Claims will only be updated after GPU experiments produce evidence.

## Current Claim Status

| Claim | Old Text | New Text | Evidence | Status |
|-------|----------|----------|----------|--------|
| Sparse acceptance-critical tokens | "top-20% → >80% sensitivity" | — | M0/M1 FAIL (27.5%, 56%) | UNSUPPORTED — do not claim |
| Accept ≠ perplexity ≠ attention | "Spearman ρ < 0.7" | — | M2 ρ values may be artifact | UNCERTAIN — needs matched-support rerun |
| Predictor identifies critical tokens | "F1 > 0.75" | — | M2 F1 = 0.233 | CONTRADICTED — old predictor fails |
| Mixed precision preserves acceptance | "≥3pp gap" | — | M3 incomplete, 0% accuracy so far | UNSUPPORTED — needs results |
| Method improves throughput | "≥10% latency win" | — | Old benchmark: acceptance → SLOWER | CONTRADICTED — simulation only |
| Attention is wrong objective | Broad claim | — | Too broad per reviewer | UNSAFE — needs narrowing |
| Zero papers optimize KV for acceptance | IDEA_REPORT claim | — | SpecAttn, QuantSpec exist | UNSAFE — must cite and differentiate |

## Rules

- No claims will be strengthened until A/B/C comparison shows C > A and C > B
- Negative results from M0/M1/M2 will be preserved and cited as motivation
- "SOTA" will not be claimed until official baselines are reproduced
