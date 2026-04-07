# Pipeline Summary

**Problem**: SD verification loads full KV for all tokens but acceptance depends on sparse subset.
**Final Method Thesis**: Acceptance-critical tokens in SD are distinct from attention-important tokens; optimizing KV for acceptance rate yields better compression than optimizing for perplexity.
**Final Verdict**: READY (8.1/10 after 3 rounds GPT-5.4 nightmare review)
**Date**: 2026-04-07

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/round-1-review.md`, `refine-logs/round-2-review.md`
- Score history: `refine-logs/score-history.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- Dominant contribution: Discovery that acceptance sensitivity ≠ perplexity sensitivity for KV in SD
- Supporting contribution: AcceptPredictor (zero-overhead predictor from draft attention)
- Systems contribution: E2E integration with EAGLE-3 showing joint latency + memory wins
- Explicitly rejected complexity: Custom SD engine, theoretical bounds, RL predictor, cross-architecture eval

## Must-Prove Claims
- C1: Top-20% tokens capture >80% acceptance sensitivity (oracle study)
- C2: Spearman ρ < 0.7 between acceptance-ranked and perplexity-ranked tokens
- C3: ≥3pp accuracy gap between acceptance-targeted and perplexity-targeted retention at same budget
- C4: ≥10% latency improvement over naive EAGLE-3+ThinKV composition
- C5: Predictor F1 > 0.75 against oracle

## First Runs to Launch
1. R001: Oracle sanity check on 10 GSM8K problems (~2 GPU-hours)
2. R002: Full oracle study on 100 GSM8K problems (~10 GPU-hours)
3. R003: Acceptance vs perplexity ranking divergence (~15 GPU-hours)

## Main Risks
- Sparsity doesn't hold: ABORT at M1 (<10 GPU-hours wasted)
- Accept ≈ perplexity: Caught at M2. Pivot to phase-conditional divergence.
- Compression overhead: Profiling in B4 identifies bottleneck.

## Next Action
- Proceed to Stage 2: Implementation (AcceptSpec core + oracle measurement + predictor)
- Then `/run-experiment` for M0-M1 oracle gate
