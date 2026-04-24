# Keep / Rewrite / Archive Plan

| Item | Path | Current Role | Evidence | Action | Reason | Risk |
|------|------|--------------|----------|--------|--------|------|
| MTP Decoder | `src/speculative_decode.py` | True MTP generation | Core correct path | KEEP | Reuse tested loop | Low |
| MTP Head | `src/mtp_head.py` | MTP implementation | Working | KEEP | Required | Low |
| Model Loader | `src/gpu_auto.py` | Load models/MTP | Working | KEEP | Required | Low |
| TurboQuant | `src/turboquant_kv.py` | Quantization backend | Action simulation | KEEP | MARA action backend | Low |
| Utils | `src/utils.py` | KV cache helpers | Working | KEEP | Required | Low |
| AcceptSpec Core | `src/acceptspec.py` | Oracle, predictor, mixed precision | P0/P1 bugs, F1=0.23 | KEEP + SUPPLEMENT | Keep as Variant A ablation; MARA supplements | Medium |
| Baselines | `src/baselines.py` | RTN/KIVI/Absmax | Needed for comparison | KEEP AS BASELINE | Must not weaken | Low |
| ThinkCompress | `src/thinkcompress.py` | Old adaptive compression | Adaptive ≈ uniform (negative) | KEEP AS HISTORICAL | Archive candidate | Low |
| Linear Attn Quantizer | `src/linear_attn_quantizer.py` | Qwen3.5 linear attn | Not relevant to MHA | FREEZE | Not main path | Low |
| Quantized Verifier | `src/quantized_verifier.py` | Monkey-patched attention | Simulation only | FREEZE | May be needed | Low |
| MARA Core | `src/accept_risk.py` | NEW: risk oracle, predictor, allocator, gate | GPT-5.5 main method | NEW | Core missing mechanism | Medium |
| Repro Utils | `src/repro.py` | NEW: seeds, splits, metadata | GPT-5.5 Task 2 | NEW | Reproducibility | Low |
| Oracle Sensitivity | `scripts/oracle_sensitivity.py` | M0/M1 measurement | P0 kv_len bug | KEEP (fix later) | Fix required for MARA labels | High |
| Core Comparison | `scripts/core_comparison.py` | M3 policy comparison | P0 MTP path bug | KEEP (fix later) | Add MARA policies | High |
| Triple Divergence | `scripts/triple_divergence.py` | M2 diagnostics | Failed but useful MTP path | KEEP | Reference for diagnostics | Low |
| MARA Config | `configs/mara_minimal.yaml` | NEW: MARA settings | GPT-5.5 Task 7 | NEW | Required | Low |
| M0/M1 Results | `results/acceptspec/oracle_m*.json` | Historical oracle outputs | Bug-contaminated | KEEP + MARK UNRELIABLE | In docs/RELIABILITY_AUDIT.md | Low |
| M2 Results | `results/acceptspec/divergence/` | Historical divergence | Artifact-prone | KEEP + MARK UNRELIABLE | Historical negative evidence | Low |
| Old Benchmarks | `results/benchmark/` | Historical throughput data | acceptance ≠ speed | KEEP AS HISTORICAL | Useful negative evidence | Low |
| README | `README.md` | Project pitch | Claims ahead of evidence | DEFER REWRITE | After MARA results | Medium |
| IDEA_REPORT | `IDEA_REPORT.md` | Idea discovery | "Zero papers" claim unsafe | DEFER REWRITE | After novelty recheck | Medium |
| Reliability Audit | `docs/RELIABILITY_AUDIT.md` | NEW: result status manifest | GPT-5.5 Task 1 | NEW | Prevents false claims | Low |
| Tests | `tests/test_accept_risk.py` | NEW: MARA tests | 25/25 passing | NEW | Verification | Low |
| Tests | `tests/test_data_metric_sanity.py` | NEW: data/metric sanity | All passing | NEW | Prevents leakage | Low |
