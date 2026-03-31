# Round 3 Review (GPT-5.4)

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 9.4/10 |
| Method Specificity | 8.9/10 |
| Contribution Quality | 8.8/10 |
| Frontier Leverage | 8.9/10 |
| Feasibility | 9.1/10 |
| Validation Focus | 8.6/10 |
| Venue Readiness | 8.5/10 |
| **Overall** | **8.9/10** |

## Verdict: REVISE (high-end, close to READY)

## Remaining Items (all IMPORTANT, no blocking)
1. Headline mismatch: thesis says ≤2pp but 3-bit bound gives 2-4pp → sharpen bound or soften headline
2. Proposition 1 needs full explicitness: what C depends on, per-step vs sequence-level, V quantization
3. Claim 3 must be quantitative: predicted TV vs observed, predicted acceptance vs measured
4. Failure envelope: very long context, 2-bit collapse, most sensitive layers

## Simplification
- Put verifier contract in first paragraph, keep it throughout
- One proposition + one corollary; derivation in appendix
- GSM8K/HumanEval to appendix

## Modernization
- Report accepted tokens/sec and end-to-end speculative speedup
- Compare FP8 KV, fused 4-bit RTN, uncompressed optimized verifier
- Show paged-attention serving integration path
