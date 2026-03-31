# Round 1 Review (GPT-5.4)

## Scores

| Dimension | Score |
|-----------|-------|
| Problem Fidelity | 8/10 |
| Method Specificity | 5/10 |
| Contribution Quality | 6/10 |
| Frontier Leverage | 8/10 |
| Feasibility | 6/10 |
| Validation Focus | 6/10 |
| Venue Readiness | 6/10 |
| **Overall** | **6.4/10** |

## Verdict: REVISE

## Key Issues

### CRITICAL: Method Specificity (5/10)
- "Inverse-rotate R^T before attention" is the wrong interface — may erase speedup
- Must define one exact verifier kernel: q' = Rq, store k' = Q(Rk), v' = Q(Rv), compute attention in rotated space, apply R^T only once to final head output
- Specify block size, scale/zero-point policy, total effective bits including metadata

### CRITICAL: Contribution Quality (6/10)
- Too many side ideas: TurboQuant transplant, residual QJL, adaptive bit allocation, fused CUDA, direct acceptance theory
- Reads as contribution sprawl; novelty risks collapsing to "apply existing quantizer to KV cache"
- Cut: QJL and adaptive allocation to appendix/remove. Reframe theory to logit-perturbation bound + empirical.

### IMPORTANT: Feasibility (6/10)
- Narrow scope: one kernel, one rotation, fixed 3-bit and 4-bit, 1-2 model families
- Treat 2.5-bit and adaptive allocation as stretch results

### IMPORTANT: Validation Focus (6/10)
- Center on context-length sweeps and verifier microbenchmarks (HBM traffic, kernel latency, tokens/s)
- Keep one reasoning + one code task as sanity; drop MT-Bench

### IMPORTANT: Venue Readiness (6/10)
- Sharpen story: verification is a special KV-access regime where compressed-domain attention is the right mechanism
- Keep theory modest, systems story concrete

## Simplification Opportunities
1. Remove residual QJL from main method
2. Drop adaptive bit allocation from core proposal
3. Replace dense random orthogonal with structured random-sign Hadamard (simpler, kernel-friendly)
4. Stick with 3-bit/4-bit, not 3.5-bit

## Modernization Opportunities
1. Leverage FlashAttention/FlashDecoding-style compressed-domain kernel
2. Align with hardware-native low-bit paths
3. Make verifier-only interface explicit (foundation-model-era systems angle)

## Drift Warning
- Mostly anchored. Drift if becomes generic KV-cache quant paper vs spec-verification paper.
- Drift if learned rotations, calibration, or draft model changes enter main story.

<details>
<summary>Raw Response</summary>

[Full GPT-5.4 response saved above]

</details>
