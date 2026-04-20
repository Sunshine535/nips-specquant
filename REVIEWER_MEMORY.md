# Reviewer Memory

*Persistent across rounds. Nightmare difficulty — reviewer reads repo directly.*

## Round 1 — 2026-04-21 — Score 2.5/10, verdict "not ready"

### Confirmed suspicions (verified by reviewer reading raw files)
1. **Aggregate merge is broken**: `oracle_m1.json` inherits shard 0 values instead of recomputing — top20=0.5600 = shard 0 only (shard 1 had 0.7018). Reviewer cites: scripts/parallel_run.sh:135, :144.
2. **spearman_rho saved as NaN** in oracle_m1.json at line :43. The ρ numbers used in proposals are NOT from raw files.
3. **Mask artifact**: Spearman over `(a>0)|(b>0)` of zero-padded sparse vectors is biased. Reviewer independently sanity-checked: the "ρ≈0 / ρ=-0.15" pattern is compatible with the artifact alone.
4. **Jacobian "mechanism" is oversold**: It's a last-layer V-side heuristic, not the true Jacobian of acceptance w.r.t. KV.

### Unresolved concerns (carry to Round 2)
- Whether corrected ρ(accept,attn) and ρ(accept,ppl) survive after matched-sample recomputation
- Whether any α > 0.5 setup reproduces corrected C1 (defeats "weak MTP" attack)
- Whether margin heuristic beats attention/ppl on DENSE oracle evaluation (20 problems, 100% perturbation)
- Whether GQA attention extraction still silently falls back to uniform (triple_divergence.py:562)

### Patterns reviewer flagged to track
- Author tends to overclaim from partial data (aggregate from shard 0)
- Author tends to rescue stories by renaming/refactoring rather than fixing measurement
- Author's proof audit was itself a downgrade (bound → proxy) not a strengthening — the same pattern may repeat here
- "MarginSpec is story rescue after C2 failed once, not a robust mechanism result"

### Reviewer verdict in one line
> Spend 48 hours fixing the measurement pipeline. Rerun corrected C1. If it survives on matched samples and α > 0.5, pivot to lean C1-only discovery paper. If it collapses, kill this line.
