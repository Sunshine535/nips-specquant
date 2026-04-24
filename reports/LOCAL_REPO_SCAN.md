# Local Repository Scan

**Date**: 2026-04-24
**Scanned by**: Claude Code (Opus 4.6)

## Top-Level Directory Map

```
nips-specquant/
├── configs/          # Configuration files
├── idea-stage/       # Idea discovery documents
├── logs/             # Experiment logs (M0-M3, old benchmarks)
├── refine-logs/      # Method refinement logs (v1/v2)
├── refine-logs-v3/   # MarginSpec proposal (v3)
├── reports/          # Execution reports (this directory)
├── research-wiki/    # Research knowledge base
├── results/          # Experiment results (M0/M1/M2, benchmarks)
├── review-stage/     # Auto-review loop documents
├── scripts/          # Experiment scripts
├── src/              # Core source modules
├── tests/            # Test suite
├── GPT55_DIAGNOSIS.md
├── README.md
├── CLAUDE.md
├── IDEA_REPORT.md
├── LITERATURE_LANDSCAPE.md
├── REVIEWER_MEMORY.md
├── REVIEW_STATE.json
├── PROOF_AUDIT.md
├── PROOF_SKELETON.md
├── PROOF_CHECK_STATE.json
├── AUTO_REVIEW.md
├── requirements.txt
├── setup.sh
└── run.sh
```

## Component Table

| Component | Path | Purpose | Importance | Notes |
|----------|------|---------|------------|-------|
| MTP Speculative Decoder | `src/speculative_decode.py` | True MTP generation loop | High | Core correct path; P0 bug in policy comparison scripts |
| MTP Head | `src/mtp_head.py` | Qwen3.5 MTP head loading/drafting | High | Required for self-speculation |
| AcceptSpec Core | `src/acceptspec.py` | Oracle sensitivity, predictor, mixed precision KV | High | P0/P1 bugs; predictor F1 failed; needs MARA rewrite |
| GPU/Model Loader | `src/gpu_auto.py` | Load models with MTP, GPU assignment | Medium | Uses eager attention; needs metadata logging |
| TurboQuant | `src/turboquant_kv.py` | Hadamard rotation + scalar quantization | Medium | Action simulation backend for MARA |
| Baselines | `src/baselines.py` | RTN/KIVI/Absmax quantization baselines | High | Keep as baselines; verify fairness |
| Quantized Verifier | `src/quantized_verifier.py` | Monkey-patched attention with quantized KV | Medium | Scale bug fixed; simulation only |
| ThinkCompress | `src/thinkcompress.py` | Old adaptive compression (ImportanceScorer) | Low | Archive as historical negative evidence |
| Linear Attn Quantizer | `src/linear_attn_quantizer.py` | Qwen3.5 linear attention quantization | Low | Not relevant to MHA KV studies |
| Utils | `src/utils.py` | KV cache helpers, save_results | Medium | Needs split/seed/metadata additions |
| Oracle Sensitivity | `scripts/oracle_sensitivity.py` | M0/M1 acceptance sensitivity measurement | High | P0 kv_len bug; aggregation issues; REWRITE |
| Core Comparison | `scripts/core_comparison.py` | M3 policy comparison (8 strategies) | High | P0 not true MTP; predictor features invalid; REWRITE |
| Triple Divergence | `scripts/triple_divergence.py` | M2 accept/attention/perplexity divergence | High | Failed but has better MTP path; MERGE |
| E2E Benchmark | `scripts/e2e_benchmark.py` | End-to-end latency benchmark | Medium | Downstream of fixed measurement |
| Parallel Runner | `scripts/parallel_run.sh` | Multi-GPU parallel shard launcher | Medium | Merge logic rewritten; HF race condition |
| Experiment Orchestrator | `scripts/run_all_experiments.sh` | M0-M5 pipeline | Medium | Uses torch_patch.py wrapper |
| Torch Patch | `scripts/torch_patch.py` | Monkey-patch torch.compile for fla | Low | Workaround for GatedDeltaNet |
| Figure Generator | `scripts/generate_figures.py` | Plots and tables | Medium | Placeholder fallback risk |
| TV Distance Eval | `scripts/eval_tv_distance.py` | Distribution shift metrics | Medium | Some placeholder approximations |
| Downstream Eval | `scripts/eval_downstream.py` | Task metrics (GSM8K accuracy) | Medium | Needs metric sanity verification |
| Config | `configs/default.yaml` | Model/dataset/baseline settings | High | Needs MARA config additions |
| Test Suite | `tests/test_specquant.py` | Unit tests for older modules | Medium | Missing AcceptSpec/MTP/oracle tests |
| M0/M1 Results | `results/acceptspec/oracle_m*.json` | Oracle sensitivity outputs | High (evidence) | Bug-contaminated; mark historical |
| M2 Results | `results/acceptspec/divergence/` | Triple divergence shards | Medium | Failed; artifact-prone |
| Old Benchmarks | `results/benchmark/` | Qwen2.5 dual-model benchmarks | Low | Historical negative: acceptance ≠ speed |
| ThinkCompress Results | `results/gsm8k/` | Old adaptive compression results | Low | Shows heuristic not better than uniform |
| README | `README.md` | Project pitch and claims | High | Claims ahead of evidence; needs softening |
| IDEA_REPORT | `IDEA_REPORT.md` | Idea discovery report | Medium | "Zero papers" claim unsafe |
| FINAL_PROPOSAL_v3 | `refine-logs-v3/FINAL_PROPOSAL_v3.md` | MarginSpec proposal | Medium | Unsupported claims; FREEZE |
| AUTO_REVIEW | `review-stage/AUTO_REVIEW.md` | Internal review (Rounds 1-2) | Very High | Aligns with GPT-5.5 diagnosis |
| REVIEWER_MEMORY | `REVIEWER_MEMORY.md` | Persistent reviewer suspicions | High | Tracks known issues |
| GPT55 Diagnosis | `GPT55_DIAGNOSIS.md` | GPT-5.5 Pro external audit | Very High | Defines MARA main method path |
