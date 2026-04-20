# Reviewer Memory

*Persistent across rounds. Nightmare difficulty — reviewer reads repo directly.*

## Round 1 — 2026-04-21 — Score 2.5/10, verdict "not ready"

### Confirmed suspicions (verified by reviewer reading raw files)
1. **Aggregate merge broken**: `oracle_m1.json` inherits shard 0 values (scripts/parallel_run.sh:135,:144)
2. **spearman_rho = NaN** in oracle_m1.json:43 — quoted ρ numbers don't exist in raw files
3. **Mask artifact**: `(a>0)|(b>0)` Spearman on zero-padded sparse vectors is biased (reviewer independently sanity-checked)
4. **Jacobian "mechanism" oversold** — last-layer V-side residual heuristic, not true Jacobian

### Patterns flagged
- Author overclaims from partial data
- Rescue stories via renaming/refactoring
- Proof audit was itself a downgrade (bound → proxy)
- "MarginSpec is story rescue after C2 failed once"

## Round 2 — 2026-04-21 — Score 4/10 (up from 2.5)

### Previous suspicions status
- ✅ (1) merge now recomputes from per_problem (commit 13f7cb8) — VERIFIED by reviewer in code
- ✅ (2) matched_spearman_correlations added to output schema — VERIFIED
- ✅ (3) `pairwise_spearman_matched` function added — VERIFIED
- ⚠️ (4) Jacobian still there, reviewer unchanged opinion (oversold) — to drop from paper
- ❌ NOT YET VERIFIED: corrected M2 rerun hasn't happened — reviewer explicitly notes "no corrected M2 matched-support numbers in the repo"

### New suspicions added Round 2
- 6. **M3 oracle_accept fabricates draft_probs** (core_comparison.py:818-877) — "good enough for triage, not paper-quality oracle"
- 7. **M3 shard merge broken** (parallel_run.sh:144 vs core_comparison.py:1176) — only handles top-level per_problem, not per_policy.*.per_problem
- 8. **Qwen3.5-9B is wrong primary model** — mostly GatedDeltaNet (linear attention); LITERATURE_LANDSCAPE.md:53 itself says Qwen3 should be primary for KV studies

### Concurrent-work threats reviewer identified
- ForesightKV (2602.03203) — biggest threat to downstream-accuracy-KV paper
- SpecAttn (2602.07223) — closest to verifier-side KV importance in SD
- SideQuest (2602.22603), Acceptance Dynamics (2604.14682), Quasar (2603.01399)
- **No paper does acceptance-ranked verifier-side KV importance yet** → still defensible

### Track for Round 3
- Whether 2 new M3 blockers get fixed (oracle_accept real draft_probs + per_policy merge)
- Whether M3 Gate A passes on Qwen3.5-9B (oracle_accept - attention_h2o ≥ 1pp on 64 problems)
- Whether pivot to Qwen3-8B + Qwen3-0.6B happens if Gate A fails
- Whether corrected M2 matched-support numbers materialize
- Whether author stops MarginSpec mechanism chase if M3 passes

### Reviewer final line
> "The only result that matters for the paper decision is: oracle_accept vs attention_h2o downstream accuracy at 20% KV budget."
