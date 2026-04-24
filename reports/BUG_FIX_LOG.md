# Bug Fix Log

## Bug Fix: RiskBudgetAllocator greedy upgrade path

**Files changed**: `src/accept_risk.py`
**Reason**: Allocator only generated upgrade candidates from the initial state (evict), missing multi-step upgrades (evict→2bit→4bit→fp16).
**Evidence**: test_high_budget_mostly_fp16 failed — 0 FP16 tokens even at 90% budget.
**Change**: Generate ALL (token, from_action, to_action) candidates for all upgrade steps, not just from current state.
**Verification command**: `python3 -m pytest tests/test_accept_risk.py -v`
**Before**: 0/20 FP16 at 90% budget
**After**: All 25 tests pass, high budget correctly allocates FP16
**Remaining risk**: None for this specific bug.

## Known P0 Bugs (Documented, Not Yet Fixed in This Session)

These bugs are identified in GPT55_DIAGNOSIS.md and documented in RELIABILITY_AUDIT.md. They exist in the experiment scripts that run on the remote GPU server, not in the MARA core:

1. **oracle_sensitivity.py kv_len**: Not advanced after target forward on last_tok
2. **core_comparison.py MTP path**: Policy path uses target-as-draft in MTP mode
3. **core_comparison.py predictor features**: Builds fake 1-token draft KV
4. **M1 aggregate merge**: Copied from shard0, not recomputed

These require GPU-side testing and will be addressed when MARA integration is deployed.
