# Auto Review Loop — MarginSpec / AcceptSpec

**Project**: NeurIPS 2026, "Acceptance is Margin-Sensitive, Not Attention-Sensitive"
**Start**: 2026-04-21
**Difficulty**: nightmare
**Reviewer**: `mcp__codex__codex` with read-only sandbox (closest oracle-pro equivalent available)
**Max rounds**: 4

---

## Round 1 — 2026-04-21

### Assessment

- **Score**: **2.5 / 10**
- **Verdict**: NOT READY
- **Codex threadId**: `019dabcb-87dc-7d80-876e-31180b22c645`

### Top 3 reviewer findings (all verified by reading raw files)

1. **C1 is not currently trustworthy.** Downstream stats use zero-padded partial labels even though `SensitivityResult` warns not to (src/acceptspec.py:44), then run Spearman on `(a>0)|(b>0)` mask. Reviewer sanity-checked: sparse-vs-dense null scores give ρ≈0, and sparse-vs-sparse independent scores give strongly negative ρ. **The observed "0 / -0.15" pattern is compatible with the mask artifact ALONE**.

2. **Merged aggregate is broken.** `oracle_m1.json`'s top20_coverage = 0.5600 and total_tokens_measured EXACTLY match shard 0 only (shard 1 has top20 = 0.7018). The merge script never recomputes top-20 coverage because `per_problem` has no `top20_coverage` field. **The "100-problem aggregate" is actually just shard 0.** Also: `spearman_rho = NaN` in the saved file — the ρ≈0 numbers we've been citing don't exist in raw results; they're only in the review header.

3. **Jacobian mechanism is oversold.** `_compute_margin_sensitivity_jacobian` is a last-layer V-side residual heuristic, not the Jacobian of acceptance w.r.t. KV. Uses first-position top-2 logits, last-layer o_proj and V only, averages GQA groups, then multiplies by aggregate attention importance.

### Other verified issues
- `oracle_m0.json` config says 10 problems, merged file contains 9 (one shard lost)
- `_extract_draft_features` silently falls back to uniform attention on GQA mismatch
- "Attention is the wrong objective" is too broad vs Expected Attention + ThinKV + PM-KVQ

### Recommendation

**Third direction** (neither continue MarginSpec mechanism nor do full C1-only paper yet):
- Spend **48 hours fixing the measurement pipeline**
- Rerun corrected C1 on matched supports
- If corrected C1 survives → lean C1-only discovery paper
- If it collapses → kill this line

### Round 2 experiments to prioritize

| Priority | Experiment | GPU-hrs | Gate |
|----------|-----------|---------|------|
| 1 | Corrected Qwen/MTP rerun (save sample_indices end-to-end, compute ρ on matched supports only) | 10-12 | ρ survives mask fix |
| 2 | Dense mechanism micro-study (100% token perturbation on 20 problems, same sampled set for all proxies) | 10-15 | margin beats attention/ppl on dense eval |
| 3 | High-α control on stronger drafter (Llama-EAGLE or Medusa, 50 problems) | 20-30 | ρ survives at α > 0.5 |
| 4 (conditional) | Second-family C1 reproduction | 20-25 | only if 1-3 pass |

### Do NOT spend on (per reviewer)
- Theorem (C3)
- Oracle future-attention falsification (C5)
- Systems drop-in integration (C4)

...until C1 is verified on matched supports and at α > 0.5.

### Reviewer Raw Response

<details>
<summary>Click to expand (Score 2.5/10, verdict "not ready")</summary>

This is below weak-accept right now. The only remotely paper-worthy nugget is C1, but the local artifacts do not validate it cleanly. The proof audit is not saving you either; it mostly documents that the prior theorem was downgraded to a heuristic TV proxy in PROOF_AUDIT.md:34.

**Verified from the saved file**: oracle_m1.json reports top20_coverage = 0.5600, mean_gini = 0.1201, and gate.passed = false.

**Unverified/misleading**: that consolidated oracle_m1.json aggregate is not a true 100-problem aggregate. Its top20_coverage and total_tokens_measured exactly match shard 0, which are identical to oracle_m1_shard0.json, while shard 1 has a different top20_coverage = 0.7018. The merge script starts from shards[0] and never recomputes top-20 coverage because per_problem has no top20_coverage field.

**Unverified**: oracle_m1.json does not confirm ρ(accept,attn)≈0 or ρ(accept,ppl)≈-0.15. Its only stored correlation fields are spearman_rho = NaN and spearman_pval = NaN. The quoted rho values only appear in the review header at review-stage/AUTO_REVIEW.md:11.

**Unverified**: even the saved 56% top-20 number is not a clean oracle coverage estimate. SensitivityResult explicitly says sensitivities is zero-filled outside sample_indices, but oracle_sensitivity.py concatenates those zero-filled vectors and sorts them globally.

**Hygiene issue**: oracle_m0.json config says 10 problems, but the merged file contains 9.

**Top 3 weaknesses**:
1. C1 is not currently trustworthy (mask artifact compatible with observation).
2. Predictor/F1 path is likely artifact-prone under GQA (_extract_draft_features allocates num_heads = num_kv_heads but reads with query-head count; silently falls back to uniform).
3. The Jacobian mechanism is oversold — last-layer V-side heuristic, not true Jacobian.

**Recommendation**: Third direction. Don't continue MarginSpec mechanism/theorem. Spend 48 hours fixing the measurement pipeline and rerunning corrected C1. If corrected C1 survives on matched samples and one higher-acceptance setup, pivot to lean C1-only discovery paper. If it collapses, kill this line.

MarginSpec reads like story rescue after C2 failed once, not like a robust mechanism result. "Attention is the wrong objective" too broad: Expected Attention already argues for future/global output effect (arXiv:2510.00636), ThinKV/PM-KVQ occupy richer importance narratives (arXiv:2510.01290, arXiv:2505.18610). Defensible claim is narrower: current attention-style proxies may fail for verifier acceptance on corrected evaluation.

**Round 2 experiments** (prioritized above).

**Memory update**:
- Track whether corrected ρ(accept,attn) and ρ(accept,ppl) survive after matched-sample recomputation
- Track whether merged aggregates are recomputed from raw step-level data rather than inherited from shard 0
- Track whether GQA attention extraction still falls back to uniform
- Track whether any α > 0.5 setup reproduces corrected C1
- Track whether the margin heuristic beats attention/ppl on dense oracle evaluation after GQA/K-path fixes

</details>

### Status
- **Next action**: Phase C — implement fixes before Round 2.
- Priority 0: fix merge script (use per-step raw data, not shard[0])
- Priority 1: save sample_indices end-to-end, correlate on matched supports only
- Priority 2: add dense-oracle subset mode (100% perturbation on small subset)
- Priority 3: fix GQA extract_draft_features to not silently fallback
- Continuing to Round 2 after fixes + partial rerun
