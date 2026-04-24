# Core Comparison

## Status: NOT YET EXECUTABLE

The A/B/C core comparison requires GPU execution on the remote server. MARA core modules are implemented and tested locally. The comparison structure is defined below.

## Planned Comparison

| Variant | Config | Dataset | Seeds | Metric Mean | Std | Compared To | Result | Interpretation |
|---------|--------|---------|-------|-------------|-----|-------------|--------|----------------|
| A: existing_best_fragment_only | mara_minimal.yaml | GSM8K | 42,123,456 | PENDING | — | baselines | — | Old AcceptPredictor top-k |
| B: mara_no_gate_or_uncertainty | mara_minimal.yaml | GSM8K | 42,123,456 | PENDING | — | A | — | MARA risk allocation only |
| C: mara_full | mara_minimal.yaml | GSM8K | 42,123,456 | PENDING | — | A, B | — | Full MARA with gate |
| fp16_baseline | mara_minimal.yaml | GSM8K | 42 | PENDING | — | all | — | Upper bound |
| uniform_4bit | mara_minimal.yaml | GSM8K | 42 | PENDING | — | C | — | Strong simple baseline |
| attention_h2o | mara_minimal.yaml | GSM8K | 42 | PENDING | — | C | — | SmallKV-style baseline |
| oracle_risk | mara_minimal.yaml | GSM8K | 42 | PENDING | — | C | — | Upper bound for risk |

## Interpretation Rules

1. **C > A and C > B consistently** → New mechanism adds real value. Proceed.
2. **C ≈ A** → MARA may only reuse old fragment. Diagnose.
3. **C ≈ B** → Gate/uncertainty not contributing. Check mechanism logs.
4. **C < A** → New method hurts. Check implementation.
5. **Unstable across seeds** → Do not claim success. Add stability analysis.
6. **C wins only on GSM8K** → Narrow claim. Test MATH.

## Metrics

- Primary: acceptance retention at equal KV budget (20%)
- Secondary: downstream task accuracy (GSM8K exact match)
- Diagnostic: margin gate activation rate, uncertainty gate rate, risk Spearman ρ, ECE
