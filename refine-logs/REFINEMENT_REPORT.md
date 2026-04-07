# Refinement Report: AcceptSpec

## Starting Point
- Direction: speculative decoding + KV cache quantization (NeurIPS 2026)
- Initial idea: SpecThin (segment-level SD + thought-aware KV compression)
- Initial review: 4/10 — "incremental combination"

## Refinement Journey

### Iteration 1: SpecThin → AcceptSpec
- **Problem**: Pure combination of SpecCoT + ThinKV lacks a formal principle
- **Fix**: Introduced acceptance-preservation objective as the differentiating insight
- **Outcome**: Reviewer acknowledged "plausible core intuition"

### Iteration 2: Tighten method
- **Problem**: Proposition 1 hand-wavy, predictor underspecified, scope too broad
- **Fixes**: (a) Empirical proposition not theorem, (b) Exact AcceptPredictor formula, (c) Cut to 1 model + 3 tasks
- **Outcome**: 7.6/10, "weak accept"

### Iteration 3: Final polish
- **Problem**: α(KV) definition ambiguous, novelty concern, thin benchmark
- **Fixes**: (a) Block-level coupled-randomness definition, (b) "Discovery paper" framing, (c) Robustness sweeps
- **Outcome**: 8.1/10, "solid accept"

## What Was Rejected
- Custom segment-level SD engine (too complex, use EAGLE-3)
- Theoretical acceptance bound (fragile, use empirical)
- Cross-architecture evaluation (use Qwen3 only)
- RL-based predictor (overkill, logistic regression sufficient)
- HumanEval/GPQA benchmarks (focus on reasoning)

## Key Design Decisions
1. Oracle-first validation: invest <10 GPU-hours before committing
2. Discovery > algorithm: position as finding about SD structure
3. AcceptPredictor piggybacks on draft attention: zero overhead
4. EAGLE-3 off-the-shelf: don't re-invent SD
