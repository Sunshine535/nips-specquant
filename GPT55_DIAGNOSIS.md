我先给出审查边界：仓库**公开可读**，但我这里没有成功用 `git clone` 完整克隆到本地；我通过 GitHub 页面与 raw 文件读取了根目录、关键源码、configs、logs、results、review-stage/refine-stage 文档，并下载审计了主要 Python 源码。下面结论来自**静态代码审计 + 已有日志/JSON/README/评审文档**，没有声称我重新跑过大模型实验。

核心结论先放前面：**当前项目不能继续把“稀疏 acceptance-critical KV token”或“margin-law”作为已证实主线。现有正面信号最多是线索，当前更强的证据是：测量管线、MTP 比较路径、aggregation、predictor 和 claim-code-result 对齐都存在 P0/P1 问题。唯一推荐的新主线是把项目重构为 *MARA: Margin-Calibrated Acceptance-Risk Allocation*：在真实 MTP speculative decoding 中，估计每个 KV token 在不同精度下对 verifier 接受决策的**校准风险**，再用 margin/uncertainty gate 做闭环预算分配。**

---

# 0. Repository Readability Check

仓库可访问，根目录明确显示 `configs/`、`src/`、`scripts/`、`logs/`、`results/`、`refine-logs/`、`refine-logs-v3/`、`review-stage/`、`requirements.txt`、`run.sh`、`setup.sh` 等。README 把项目定义为 **AcceptSpec — Acceptance-Preserving KV Cache Management for Speculative Decoding**，并把核心发现写成 “sparse KV sensitivity; acceptance rate not perplexity”。仓库也包含多个实验 gate：M0/M1/M2/C1-C5，但 README 中 C1-C5 仍是 TODO。([GitHub][1])

| Item                          |                          Found? | Location                                                                                  | Notes                                                                                           |
| ----------------------------- | ------------------------------: | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Public repository             |                             Yes | GitHub root                                                                               | 页面可读；我未成功 `git clone`，但可用 web/raw 审查。                                                           |
| README                        |                             Yes | `README.md`                                                                               | 明确项目 thesis、quick commands、M0/M1/M2 gates；部分 commands 与当前脚本接口不完全一致。                             |
| Paper draft                   | Not found as conventional draft | No `paper.tex`/`paper.pdf` found in inspected tree                                        | 有 `IDEA_REPORT.md`、`FINAL_PROPOSAL_v3.md`、review notes，但未见正式论文草稿。                               |
| Method notes                  |                             Yes | `IDEA_REPORT.md`, `refine-logs-v3/FINAL_PROPOSAL_v3.md`, `AUTO_REVIEW.md`                 | 包含 AcceptSpec、MarginSpec、review critique。([GitHub][2])                                          |
| Training / experiment scripts |                             Yes | `scripts/`                                                                                | `oracle_sensitivity.py`, `core_comparison.py`, `triple_divergence.py`, `e2e_benchmark.py`, etc. |
| Evaluation scripts            |                             Yes | `scripts/eval_downstream.py`, `scripts/eval_tv_distance.py`, `scripts/core_comparison.py` | 部分脚本有 placeholder / stale risk。                                                                 |
| Configs                       |                             Yes | `configs/default.yaml`                                                                    | Contains model/dataset/baseline/spec_decode/acceptspec settings.                                |
| Logs                          |                             Yes | `logs/`, `refine-logs/`, `refine-logs-v3/`                                                | M0/M1/M2/M3/full benchmark logs present.                                                        |
| Results                       |                             Yes | `results/acceptspec`, `results/benchmark`, `results/gsm8k`, etc.                          | M0/M1 JSONs and older benchmarks present.                                                       |
| Baselines                     |                             Yes | `src/baselines.py`, `scripts/core_comparison.py`, config baseline list                    | RTN/KIVI/Absmax/H2O/SnapKV/R-KV-like policies appear.                                           |
| Failed experiment notes       |                             Yes | `review-stage/AUTO_REVIEW.md`, `refine-logs/*`, logs                                      | Review explicitly marks several claims untrustworthy.                                           |
| Ablations                     |                       Partially | `scripts/run_ablations.py`, `core_comparison.py`, logs                                    | Many planned; not all have completed reliable outputs.                                          |
| Requirements / env            |                             Yes | `requirements.txt`, `setup.sh`                                                            | No lockfile / exact CUDA env freeze seen.                                                       |

## Missing / needed uploads

| Missing Item                              | Why Needed                                                                                   | What I Should Upload                                                         |
| ----------------------------------------- | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Formal paper draft, if outside repo       | Needed to audit abstract/intro/method/experiment claims exactly.                             | `paper.tex`, `main.pdf`, figures, appendix.                                  |
| Exact commands for each reported table    | Needed to classify results as reproducible vs stale.                                         | Shell history, launch scripts, Slurm logs, WandB links.                      |
| Exact checkpoints / model revisions       | Needed to verify target/draft/MTP compatibility.                                             | HF revision hashes, local checkpoint metadata.                               |
| Raw per-step oracle samples for old M0/M1 | Needed to recompute correct matched-support aggregates.                                      | JSONL / pickle containing `sample_indices`, sampled risks, per-step context. |
| Predictor weights                         | `core_comparison.py` expects `predictor_weights.pt`; missing weights imply uniform fallback. | `predictor_weights.pt` plus training command.                                |
| WandB/TensorBoard exports, if any         | Needed for loss curves, instability, seed variance.                                          | Run exports / CSV.                                                           |
| Full failed ablation outputs              | Needed to distinguish true negative from broken run.                                         | All ablation JSONs/logs, not only summaries.                                 |
| Official baseline reproduction logs       | Needed for fair comparison and novelty claims.                                               | KIVI/KVQuant/SnapKV/H2O/R-KV/QuantSpec reproduction logs.                    |

---

# 1. Repository Map

| Component                 | Path                                        |                                                                            Purpose |                     Importance | Notes                                                                                         |
| ------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------: | -----------------------------: | --------------------------------------------------------------------------------------------- |
| Project thesis / commands | `README.md`                                 |                               Presents AcceptSpec, M0/M1/M2 gates, quick commands. |                           High | Claims are ahead of reliable evidence; C1-C5 are TODO.([GitHub][3])                           |
| Config                    | `configs/default.yaml`                      |                                  Main model/dataset/spec_decode/baseline settings. |                           High | Defines Qwen3.5 MTP, GSM8K/MATH, seeds, budgets, baselines.                                   |
| Core AcceptSpec logic     | `src/acceptspec.py`                         |                                 Oracle sensitivity, predictor, mixed precision KV. |                           High | Central method code; predictor and compression path need rewrite.                             |
| Speculative decoding      | `src/speculative_decode.py`                 |                                           MTP / draft-target speculative decoding. |                           High | Correct MTP loop exists here; other scripts partially duplicate it incorrectly.               |
| MTP head                  | `src/mtp_head.py`                           |                                                               Loads/uses MTP head. |                           High | Needed for true self-speculation.                                                             |
| GPU/model loading         | `src/gpu_auto.py`                           |                                                       Loads Qwen/Llama/MTP models. |                         Medium | Uses eager attention path to collect attentions.                                              |
| Baselines                 | `src/baselines.py`                          |                                     RTN/KIVI/Absmax/H2O/SnapKV/R-KV-like policies. |                           High | Needed, but must verify fairness / official-code parity.                                      |
| Oracle experiment         | `scripts/oracle_sensitivity.py`             |                                          M0/M1 acceptance-sensitivity measurement. |                           High | P0 KV-length bug and aggregation problems.                                                    |
| Triple divergence         | `scripts/triple_divergence.py`              |                   M2: acceptance vs attention/perplexity divergence, predictor F1. |                           High | More faithful MTP path than oracle script, but M2 failed.                                     |
| Core comparison           | `scripts/core_comparison.py`                |                                             Compare policies under same KV budget. |                           High | P0: not true MTP in policy path; predictor features invalid.                                  |
| E2E benchmark             | `scripts/e2e_benchmark.py`                  |                                                              End-to-end benchmark. |                         Medium | Must be downstream of fixed measurement path.                                                 |
| Downstream eval           | `scripts/eval_downstream.py`                |                                                                      Task metrics. |                         Medium | Need metric sanity and split control.                                                         |
| TV-distance eval          | `scripts/eval_tv_distance.py`               |                                                   Distribution shift / TV metrics. |                         Medium | Some placeholder approximations need audit.                                                   |
| Figure generation         | `scripts/generate_figures.py`               |                                                                      Plots/tables. |                         Medium | Needs to avoid placeholder data contaminating claims.                                         |
| Tests                     | `tests/test_specquant.py`                   |                                                      Unit tests for older modules. |                         Medium | Missing AcceptSpec/MTP/oracle/core-comparison tests.                                          |
| Current results           | `results/acceptspec/*`                      |                                                   M0/M1 oracle outputs and shards. | High as evidence, low as proof | M1 aggregate appears contaminated by shard merge issue.                                       |
| Old benchmark             | `results/benchmark/*`                       |                                                  Qwen2.5 dual-model old benchmark. |                         Medium | Useful negative signal: acceptance ≠ speed.                                                   |
| GSM8K old result          | `results/gsm8k/*`                           |                                                             ThinkCompress results. |                         Medium | Shows old adaptive importance not better than uniform.                                        |
| Review critique           | `review-stage/AUTO_REVIEW.md`               | Internal review; identifies untrustworthy C1, broken aggregate, oversold Jacobian. |                      Very high | Aligns with code audit; should not be ignored.([GitHub][4])                                   |
| Experiment tracker        | `refine-logs/EXPERIMENT_TRACKER.md`         |                                                                 Planned R001-R020. |                         Medium | Shows many experiments remain TODO.([GitHub][5])                                              |
| v3 proposal               | `refine-logs-v3/FINAL_PROPOSAL_v3.md`       |                                                                  MarginSpec pivot. |                           High | Strong claims: margin-sensitive, expected F1/gains; not supported by current M2.([GitHub][6]) |
| Literature notes          | `LITERATURE_LANDSCAPE.md`, `IDEA_REPORT.md` |                                                           Related work / ideation. |                         Medium | Contains strong “gap” claims that are now novelty-risky.                                      |

## What the repo is currently trying to solve

The repo tries to reduce KV-cache memory/compute cost during speculative decoding, especially MTP/self-speculative decoding, by preserving the target verifier’s acceptance behavior rather than optimizing generic perplexity or attention preservation. The core intended claim is that only a sparse subset of KV tokens is “acceptance-critical,” so those tokens deserve higher precision or preservation while others can be quantized/evicted.

## Current method

The current method family has three overlapping versions:

1. **AcceptSpec**: measure/learn acceptance-sensitive KV tokens; allocate mixed precision by predicted criticality.
2. **MarginSpec**: pivot from attention sensitivity to top-2 logit margin / acceptance margin as a mechanism.
3. **Older SpecQuant / ThinkCompress / TurboQuant** routes: quantized verifier, adaptive KV compression, linear attention quantization.

## Current core assumptions

| Assumption                                                               | Current Evidence Status                                                                                                 |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Acceptance-critical KV tokens are sparse.                                | Not supported by M0/M1 as currently logged; M0/M1 gates fail.                                                           |
| Acceptance sensitivity differs from attention/perplexity.                | Plausible objective difference, but current M2 predictor evidence fails and attention proxy results are artifact-prone. |
| Margin/logit features predict acceptance sensitivity.                    | Not established; v3 proposal claims this, but M2 logs show weak/failed signal.                                          |
| Mixed precision can preserve acceptance better than uniform compression. | Not yet reliably shown under real MTP and fair baselines.                                                               |
| Acceptance-rate preservation implies speed/quality win.                  | Contradicted by old benchmark: higher acceptance did not yield higher throughput.([GitHub][7])                          |

## Main entry points

| Question                        | Current Answer                                                                                                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Main training/calibration entry | No clean single training entry. Candidate: `scripts/triple_divergence.py` predictor path; future should be new `scripts/calibrate_mara.py`.                              |
| Main evaluation entry           | Intended: `scripts/core_comparison.py`; currently invalid for MTP policy comparison.                                                                                     |
| Data processing                 | Dataset loading inside scripts; GSM8K/MATH test split often sampled directly.                                                                                            |
| Model core                      | `src/speculative_decode.py`, `src/mtp_head.py`, `src/gpu_auto.py`.                                                                                                       |
| Loss/objective                  | No clean acceptance-risk objective; current predictor uses heuristic feature weighting in `src/acceptspec.py`.                                                           |
| Baselines                       | `src/baselines.py`, `scripts/core_comparison.py`, config baseline list.                                                                                                  |
| Results/logs                    | `results/acceptspec`, `logs`, `refine-logs`, `refine-logs-v3`.                                                                                                           |
| Paper claim source              | README, IDEA_REPORT, FINAL_PROPOSAL_v3, review-stage docs.                                                                                                               |
| Mainline files                  | `src/acceptspec.py`, `src/speculative_decode.py`, `scripts/oracle_sensitivity.py`, `scripts/triple_divergence.py`, `scripts/core_comparison.py`, `configs/default.yaml`. |
| Historical/dead-code candidates | `src/thinkcompress.py`, old Qwen2.5 benchmark route, older SpecQuant/TurboQuant modules unless kept as baselines.                                                        |
| Files affecting conclusions     | All metric/eval scripts, oracle, core comparison, aggregation, figure scripts, baseline code.                                                                            |

---

# 2. Result Reliability Audit

Important: I classify “Verified” only if result, config, seed, command, checkpoint, split, and metric are all aligned. None of the major positive results currently meet that bar.

| Result ID       | Result Name                     | Dataset                       | Metric                                   |                                Claimed Value |                                                                   Logged Value | Config          | Seed                        | Command                                      | Checkpoint             | Status                | Reliability                                  | Issue                                                                                                                    |
| --------------- | ------------------------------- | ----------------------------- | ---------------------------------------- | -------------------------------------------: | -----------------------------------------------------------------------------: | --------------- | --------------------------- | -------------------------------------------- | ---------------------- | --------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| R-M0            | Oracle sensitivity M0           | GSM8K test sample             | top-k mass / Gini / Spearman             |      M0 gate: Gini > 0.5; sparse sensitivity | top20 0.2751, mean_gini 0.1371, Spearman NaN, `passed=false`, `num_problems=9` | Partial         | Partial                     | README command exists but interface mismatch | Missing exact revision | Possibly Contaminated | low / unusable as proof                      | P0 oracle KV-length bug; aggregation zero-fill; problem count mismatch.([GitHub][8])                                     |
| R-M1            | Oracle sensitivity M1 aggregate | GSM8K test sample             | top-k mass / Gini                        |                         top20 > 0.8 expected |                                 top20 0.5600, mean_gini 0.1201, `passed=false` | Partial         | Partial                     | Not exact                                    | Missing                | Possibly Contaminated | unusable                                     | Consolidated aggregate equals shard0 while shard1 differs; not reliable.([GitHub][9])                                    |
| R-M1-s0         | Oracle M1 shard0                | GSM8K                         | top-k mass                               |                                            — |                                                                   top20 0.5600 | Partial         | Partial                     | Missing                                      | Missing                | Partially Verified    | low                                          | Raw shard exists, but oracle bug persists.([GitHub][10])                                                                 |
| R-M1-s1         | Oracle M1 shard1                | GSM8K                         | top-k mass                               |                                            — |                                                                   top20 0.7018 | Partial         | Partial                     | Missing                                      | Missing                | Partially Verified    | low                                          | Heterogeneity useful, but not proof.([GitHub][11])                                                                       |
| R-M2            | Triple divergence / predictor   | GSM8K shards                  | Predictor F1, attention F1, correlations | F1 > 0.75, acceptance ≠ attention/perplexity |           AcceptPredictor F1 ≈ 0.23–0.27; AttentionProxy ≈ 0.98–0.996; M2 FAIL | Partial         | Partial                     | Parallel log                                 | Missing                | Partially Verified    | medium as negative, low as mechanistic proof | Logs include filelock failures and likely label/mask artifacts; still invalidates current predictor claim.([GitHub][12]) |
| R-M3-GSM8K      | Core comparison GSM8K           | GSM8K                         | policy comparison                        |                     Full comparison expected |                                        Log shows startup only, no final result | Partial         | Partial                     | Present-ish                                  | Missing                | Missing Log           | unusable                                     | No final metrics; core comparison path itself has P0 MTP bug.([GitHub][13])                                              |
| R-M3-MATH       | Core comparison MATH500         | MATH                          | policy comparison                        |                     Full comparison expected |                                        Log shows startup only, no final result | Partial         | Partial                     | Present-ish                                  | Missing                | Missing Log           | unusable                                     | Same as above.([GitHub][14])                                                                                             |
| R-old-bench     | Old Qwen2.5 benchmark           | Synthetic / benchmark prompts | throughput, acceptance                   |                             SpecQuant useful |        specquant_4bit acc 0.888 but throughput 17.86 vs AR 39.51 and RTN 31.99 | Present in JSON | Missing seeds beyond trials | Missing exact command                        | Missing                | Partially Verified    | medium-low                                   | Useful negative: acceptance not enough for speed.([GitHub][7])                                                           |
| R-thinkcompress | ThinkCompress GSM8K             | GSM8K 50                      | accuracy / compression                   |                 Adaptive preserves reasoning |                           FP16 0.9; TC 2x–16x 0.9; uniform_4bit 0.9 and faster | Present         | Missing                     | Missing                                      | Missing                | Partially Verified    | low-medium                                   | Suggests old adaptive heuristic not better than simple uniform.([GitHub][15])                                            |
| R-README-C1-C5  | README future claims            | —                             | C1-C5                                    |                             C1-C5 proof plan |                                                                    Marked TODO | README          | —                           | —                                            | —                      | Missing Log           | unusable as evidence                         | Claims must not be paper results yet.([GitHub][3])                                                                       |
| R-v3            | MarginSpec proposal claims      | —                             | rho, F1, gains                           |                  rho≤0.2, F1>0.75, ≥2pp gain |                                                                  Proposal only | Proposal        | —                           | —                                            | —                      | Not Reproducible      | unusable                                     | Not evidence; contradicted by M2 directionally.([GitHub][6])                                                             |

## Reliability decision

**Confirmed evidence, high confidence:** files exist; M0/M1/M2 logs/JSON contain failed or unreliable gates; review notes explicitly flag C1/aggregate/Jacobian problems.
**Likely evidence, medium confidence:** current sparse/top-k claim is not supported; current predictor is not usable.
**Hypothesis, medium-low confidence:** after fixing oracle/core comparison, some acceptance-risk signal may still survive, but not as hard sparsity.
**Insufficient evidence:** any SOTA, speedup, NeurIPS-ready, or stable positive claim.

---

# 3. Code Correctness Audit

## Suspected Bug Table

| Priority | File                                             | Function/Class                            | Code Region                                       | Suspicion                                                                                                                                                  | Evidence                                                                                                          | How to Verify                                                                                           | Proposed Fix for Claude Code                                                                                   | Expected Effect                                                                  | Confidence  |
| -------- | ------------------------------------------------ | ----------------------------------------- | ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------- |
| P0       | `scripts/oracle_sensitivity.py`                  | `run_instrumented_sd`                     | accepted-token resync block around lines ~376–394 | `kv_len` is not advanced after running target model on `last_tok`; oracle loop uses stale length.                                                          | Correct implementation in `src/speculative_decode.py` advances `kv_len = new_kv_len + 1`; oracle script does not. | Add assertion that `target_kv[0][0].shape[2] == kv_len` after each target forward; run 2-problem smoke. | After target forward on `last_tok`, set `kv_len = new_kv_len + 1`; centralize with shared MTP loop.            | Old M0/M1 sensitivity numbers may shift; old aggregates cannot be used as proof. | high        |
| P0       | `results/acceptspec/oracle_m1.json` / merge path | aggregation                               | aggregate fields                                  | Consolidated M1 aggregate appears copied from shard0, not recomputed from all shards.                                                                      | M1 aggregate top20/total tokens match shard0, while shard1 has different top20/total.([GitHub][9])                | Write validator comparing consolidated aggregate vs recomputed per-problem; fail if mismatch.           | Mark old file unreliable; create merge script that recomputes from raw matched-support samples.                | Prevents false positive/negative aggregate claims.                               | high        |
| P0       | `scripts/core_comparison.py`                     | `run_sd_with_policy`                      | policy decoding path                              | In MTP mode it uses `decoder.draft_model`; but `SpeculativeDecoder` sets `draft_model=target_model`, so policy path drafts with full target, not MTP head. | `src/speculative_decode.py` MTP mode aliases draft_model to target_model; proper MTP drafting uses `mtp_head`.    | Unit test: in MTP mode assert policy path calls MTP head; monkeypatch target forward counter.           | Refactor shared `mtp_draft_verify_step` and use it in all scripts.                                             | Any policy-comparison result before fix is unusable.                             | high        |
| P0       | `scripts/core_comparison.py`                     | `acceptspec_predicted` score fn           | score function lines ~861–873                     | Builds one-token fake draft KV from `_dec.draft_model(draft_tokens[:1])`; not actual context, target-as-draft in MTP mode.                                 | Feature context not aligned to current KV/draft state.                                                            | Log shapes/context lengths; compare features under same step from real MTP vs fake.                     | Remove path; replace with calibrated risk features from actual verification step.                              | Current predictor comparisons invalid.                                           | high        |
| P1       | `scripts/oracle_sensitivity.py`                  | aggregation                               | aggregate top-k lines ~155–183                    | Uses zero-filled full vectors and masks `all_sens > 0`; sampled zero-risk tokens disappear, unsampled zeros distort top-k.                                 | `AcceptSensitivityOracle` returns `sample_indices` and sampled values, but script aggregates dense vectors.       | Recompute matched-support top-k using only sampled support and compare.                                 | Store per-step `sample_indices`, `sampled_sensitivities`; aggregate matched support only.                      | Sparse claims may weaken or change.                                              | high        |
| P1       | `src/acceptspec.py`, `src/speculative_decode.py` | `_compute_acceptance`, rejection sampling | random draws                                      | Some acceptance/rejection randomness uses global `torch.rand` without per-run generator or logged uniforms.                                                | Oracle has coupled seeds only partially; main decoder uses global RNG.                                            | Repeat same seed twice; check identical accept sequence.                                                | Pass explicit `torch.Generator` or pre-sampled uniforms through oracle/eval.                                   | Multi-seed reliability and paired comparisons improve.                           | medium-high |
| P1       | `src/acceptspec.py`                              | attention importance fallback             | attention extraction                              | If attentions missing or MHA absent, can return zeros/uniform-like importance silently.                                                                    | Review notes mention uniform attention fallback and C1 untrustworthy.([GitHub][4])                                | Log attention coverage per layer/step; assert nonzero support.                                          | Add `attention_valid` metadata and exclude invalid steps from attention comparison.                            | Prevents fake attention-vs-acceptance claims.                                    | medium-high |
| P1       | `src/acceptspec.py`                              | `MixedPrecisionKV.compress_kv`            | mixed precision path                              | Quantizes/dequantizes in place; does not provide actual compressed storage/kernel.                                                                         | Buffers exist but memory/latency not actually implemented.                                                        | Compare allocated memory before/after; inspect cache dtype/shape.                                       | Separate “quality perturbation simulation” from “systems compression”; block speed/memory claims.              | Prevents overclaiming speed/memory.                                              | high        |
| P1       | Dataset loading in scripts                       | `load_gsm8k`, `load_math`                 | data functions                                    | Calibration and evaluation often sample from official `test` split directly.                                                                               | `oracle_sensitivity.py`, `core_comparison.py`, `triple_divergence.py` use test split sample.                      | Add split manifest; verify no overlap between calibration and eval IDs.                                 | Create deterministic `calib/test` split from train/validation where possible; reserve official test for final. | Avoid leakage / reviewer objection.                                              | medium      |
| P2       | `src/acceptspec.py`                              | `AcceptPredictor.fit/predict_tags`        | predictor                                         | Positive-only softmax head weights, no bias, fixed thresholds; cannot calibrate risk or uncertainty.                                                       | M2 F1 failed badly.([GitHub][12])                                                                                 | One-batch overfit and calibration curve.                                                                | Replace with calibrated continuous risk predictor plus uncertainty.                                            | Turns failed classifier into proper mechanism.                                   | high        |
| P2       | `tests/test_specquant.py`                        | tests                                     | test suite                                        | No tests for AcceptSpec oracle, MTP invariants, core comparison.                                                                                           | Grep/local audit.                                                                                                 | `pytest --collect-only`; inspect coverage.                                                              | Add `tests/test_acceptspec_core.py`, `tests/test_mtp_policy_path.py`.                                          | Prevents future silent regressions.                                              | high        |
| P2       | `scripts/generate_figures.py`                    | plotting                                  | figure fallback                                   | May use placeholder data when result files absent.                                                                                                         | Local code audit.                                                                                                 | Run figure generation with missing files; assert failure not fallback.                                  | Require explicit `--allow_placeholder`; default fail.                                                          | Prevents fake tables/figures.                                                    | medium      |
| P2       | `scripts/eval_tv_distance.py`                    | TV eval                                   | feature approximation                             | Some norm terms appear placeholder/hard-coded.                                                                                                             | Local code audit.                                                                                                 | Add unit test vs explicit logits on tiny model.                                                         | Mark as diagnostic only until verified.                                                                        | Avoids unsupported mechanism claims.                                             | medium      |

---

# 4. Claim-Code-Result Matrix

| Claim                                                          | Source File                        | Implementation File                                  | Result Evidence                                                  | Status                     | Problem                                                             | Confidence |
| -------------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------- | ---------- |
| KV sensitivity for acceptance is sparse.                       | `README.md`, `IDEA_REPORT.md`      | `src/acceptspec.py`, `scripts/oracle_sensitivity.py` | M0/M1 fail gates; M1 aggregate contaminated.                     | Contradicted / Unclear     | Sparse top-k not supported by reliable evidence.                    | high       |
| Acceptance rate, not perplexity, is the right target.          | README, IDEA_REPORT                | Oracle and triple divergence scripts                 | Conceptually plausible; M2 pipeline failed and artifacts exist.  | Partially Supported        | Needs corrected matched-support, real MTP evidence.                 | medium     |
| Attention is the wrong objective.                              | README / review docs               | `triple_divergence.py`                               | M2 attention proxy bizarrely high; review says C1 untrustworthy. | Unsupported                | Claim too broad; current evidence artifact-prone.                   | high       |
| Margin explains acceptance-criticality.                        | `FINAL_PROPOSAL_v3.md`             | margin heuristic in `src/acceptspec.py`              | M2 correlations near zero/failed.                                | Unsupported                | Margin should become a gate/control, not thesis.                    | high       |
| Predictor can identify acceptance-critical tokens.             | `src/acceptspec.py`, v3 proposal   | `AcceptPredictor`                                    | F1 ≈ 0.23–0.27; gate FAIL.                                       | Contradicted               | Current predictor not usable.                                       | high       |
| Mixed precision KV preserves acceptance better than baselines. | README / method                    | `MixedPrecisionKV`, `core_comparison.py`             | No reliable M3 final result; core comparison broken.             | Unsupported                | Need real MTP comparison.                                           | high       |
| Method improves throughput/memory.                             | README / old benchmark implication | `MixedPrecisionKV`, old benchmark scripts            | Old benchmark: specquant slower despite high acceptance.         | Contradicted / Unsupported | Current compression is simulation; no kernel/storage proof.         | high       |
| Results are ready for main experiments.                        | README status                      | scripts/results                                      | Review says 2.5/10, C1 untrustworthy, aggregate broken.          | Contradicted               | Internal review is more credible than README optimism.([GitHub][4]) | high       |
| Qwen3.5 MTP self-speculation is the intended setting.          | config/README                      | `src/speculative_decode.py`, `src/mtp_head.py`       | Code exists, but comparison scripts misuse it.                   | Partially Supported        | Core implementation exists; experiment scripts need refactor.       | high       |
| C1-C5 can be claimed.                                          | README                             | not complete                                         | README marks TODO.                                               | Not Testable               | Do not put into paper as results.                                   | high       |

---

# 5. Phenomenon Ledger

| ID   | Observation                                                                                            | Type                    | Where Found                 | Setting           | Metric                | Compared To              | Reliability           | What It Suggests                                                                  | What It Rules Out                                     | Confidence                         |
| ---- | ------------------------------------------------------------------------------------------------------ | ----------------------- | --------------------------- | ----------------- | --------------------- | ------------------------ | --------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------- | ---------------------------------- |
| PH1  | M0 gate failed: top20 ≈ 0.275, mean Gini ≈ 0.137, Spearman NaN, only 9 problems.                       | Negative                | `oracle_m0.json`            | Qwen3.5-9B, GSM8K | top-k/Gini            | Sparse hypothesis        | low                   | If signal exists, it is not globally hard-sparse under this pipeline.             | Strong “top20 explains most sensitivity” claim.       | high as negative, low quantitative |
| PH2  | M1 aggregate top20 ≈ 0.56, gate failed.                                                                | Negative / contaminated | `oracle_m1.json`            | Qwen3.5-9B, GSM8K | top-k/Gini            | M1 gate                  | unusable quantitative | No clean positive result.                                                         | Claim that M1 proves sparsity.                        | high                               |
| PH3  | M1 shard0 top20 ≈ 0.56 vs shard1 ≈ 0.70.                                                               | Unstable / mixed        | shard JSONs                 | shards            | top20                 | across shards            | low-medium            | Sensitivity likely context/dataset dependent; needs adaptive per-step allocation. | One global threshold/policy.                          | medium                             |
| PH4  | Consolidated M1 aggregate equals shard0-like values.                                                   | Anomalous               | M1 aggregate vs shards      | merge             | aggregate             | shards                   | high                  | Result aggregation is broken/stale.                                               | Using merged JSON as evidence.                        | high                               |
| PH5  | M2 predictor F1 ≈ 0.23–0.27; AttentionProxy ≈ 0.98–0.996; M2 FAIL.                                     | Negative / anomalous    | `M2_divergence.log`         | parallel shards   | F1/corr               | predictor/attention      | medium                | Current labels/features likely broken; current predictor not viable.              | Predictor claim; “attention wrong” measured this way. | high                               |
| PH6  | M2 filelock/dataset cache failures occurred in parallel run.                                           | Anomalous               | M2 log                      | 8 GPU parallel    | runtime               | —                        | high                  | Experiment infrastructure unstable.                                               | Treating M2 as clean final.                           | high                               |
| PH7  | M3 comparison logs show startup only, no final metrics.                                                | Missing                 | M3 logs                     | GSM8K/MATH        | policy metrics        | baselines                | high                  | No reliable final comparison exists.                                              | Any M3-based superiority claim.                       | high                               |
| PH8  | Core comparison policy path is not true MTP.                                                           | Bug-derived             | code audit                  | MTP               | policy acceptance     | real MTP                 | high                  | Existing comparison must be refactored before any method ranking.                 | Any old MTP policy result.                            | high                               |
| PH9  | Old benchmark: higher acceptance did not yield speedup; specquant_4bit slower than AR and RTN.         | Negative                | old JSON                    | Qwen2.5 7B/14B    | throughput/acceptance | AR/RTN/KIVI              | medium-low            | Systems overhead and real compression matter; acceptance alone insufficient.      | “Acceptance preservation implies speedup.”            | medium-high                        |
| PH10 | ThinkCompress adaptive variants and uniform_4bit have same GSM8K accuracy; uniform faster.             | Negative                | old GSM8K result            | Qwen3-8B          | accuracy/time         | uniform                  | low-medium            | Heuristic importance may not beat simple uniform.                                 | Generic adaptive compression claim.                   | medium                             |
| PH11 | README/proposals contain strong positive claims while tracker/review says TODO/fail.                   | Anomalous               | README, tracker, review     | project docs      | claim status          | logs                     | high                  | Paper narrative has run ahead of evidence.                                        | Submission-ready claim.                               | high                               |
| PH12 | MTP first token may be exact target token in current generation.                                       | Mechanistic             | `src/speculative_decode.py` | MTP               | position acceptance   | draft quality            | high                  | Acceptance metrics must report per-position excluding position 0.                 | Inflated aggregate acceptance claim.                  | medium-high                        |
| PH13 | MixedPrecisionKV simulates perturbation, not actual memory-compressed cache.                           | Mechanistic             | `src/acceptspec.py`         | compression       | memory/latency        | true systems compression | high                  | Separate quality proxy from systems claim.                                        | Memory/latency improvement claim.                     | high                               |
| PH14 | Review-stage already identifies zero-padded labels/mask artifact, broken aggregate, oversold Jacobian. | Negative / diagnostic   | `AUTO_REVIEW.md`            | review            | qualitative           | claims                   | high                  | Internal critique aligns with code audit; should guide pivot.                     | Ignoring negative evidence.                           | high                               |

---

# 6. Design Constraints

| Constraint ID | Derived From Observation | Constraint Type    | Meaning                                                     | Implication for New Method                                                                | Confidence  |
| ------------- | ------------------------ | ------------------ | ----------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ----------- |
| C1            | PH1, PH2                 | Must Avoid         | Do not assume a globally sparse top-k set.                  | Use continuous risk allocation, not hard top20 criticality.                               | high        |
| C2            | PH3                      | Must Stabilize     | Sensitivity is context/shard dependent.                     | Use per-step adaptive controller with margin/uncertainty.                                 | medium      |
| C3            | PH4                      | Must Fix           | Aggregation cannot be trusted.                              | Save matched-support raw samples and recompute all aggregates.                            | high        |
| C4            | PH5                      | Must Fix           | Current predictor is not a valid mechanism.                 | Replace classification-F1 predictor with calibrated continuous risk model.                | high        |
| C5            | PH6                      | Must Control       | Parallel/cache nondeterminism can corrupt evidence.         | Deterministic split, locked cache, run metadata.                                          | high        |
| C6            | PH7, PH8                 | Must Fix           | Policy comparison must use true MTP.                        | Centralize MTP draft/verify step and assert path correctness.                             | high        |
| C7            | PH9                      | Must Explain       | Acceptance rate alone does not imply throughput.            | Report quality proxy separately from real memory/kernel speed.                            | high        |
| C8            | PH10                     | Must Differentiate | Generic importance heuristics may not help.                 | Show gain over uniform, RTN/KIVI/attention/redundancy policies.                           | medium      |
| C9            | PH11                     | Must Not Claim     | Paper claims must be weaker until verified.                 | Delete/soften “sparse,” “Jacobian law,” “ready,” “SOTA.”                                  | high        |
| C10           | PH12                     | Must Control       | Position-0 exact-target effect can inflate acceptance.      | Log acceptance by draft position and no-position-0 metrics.                               | medium-high |
| C11           | PH13                     | Must Not Claim     | Current mixed precision is simulation.                      | Claim “acceptance-quality preservation” only until real storage/kernel exists.            | high        |
| C12           | PH14                     | Must Preserve      | Negative evidence is method discovery signal.               | Archive negative logs as historical evidence, not hide them.                              | high        |
| C13           | Related work             | Must Differentiate | Verification-guided KV / adaptive quantization are crowded. | Novelty must be acceptance-risk calibration + closed-loop verifier budget under real MTP. | high        |

---

# 7. Negative-to-Insight Analysis

| Negative Observation                               | Failed Assumption                                                  | Why the Assumption Failed                                                                    | What Mechanism Is Missing                                        | New Design Requirement                                                     |
| -------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| M0/M1 sparse gates failed.                         | Acceptance-critical tokens form a small, stable set.               | Sensitivity is spread, noisy, sampled, and context dependent; aggregation is also flawed.    | Continuous risk estimation under budget, not binary criticality. | Predict expected acceptance loss per precision action.                     |
| M1 shard heterogeneity.                            | One global threshold works across prompts.                         | Prompt difficulty, margin, draft quality, and step position likely change sensitivity.       | Per-step adaptive controller.                                    | Budget must depend on margin/uncertainty/context.                          |
| M2 predictor F1 collapse.                          | Attention × value norm / simple features classify critical tokens. | Labels/masks may be artifacted; predictor has weak expressivity and no calibration.          | Calibrated continuous risk model with uncertainty.               | Train on risk magnitude/ranking, not only binary F1.                       |
| AttentionProxy extremely high in M2.               | Attention-vs-acceptance divergence was cleanly measured.           | Likely label construction/mask artifact; review explicitly warns.                            | Matched-support evaluation and validity logging.                 | Evaluate all policies on same sampled tokens and valid attention coverage. |
| M3 missing / core comparison broken.               | Current comparison can validate method.                            | It drafts with target model rather than real MTP path.                                       | Shared, tested MTP draft/verify primitive.                       | All policy comparisons must call the same MTP loop.                        |
| Old benchmark slower despite high acceptance.      | Preserve acceptance → speedup.                                     | Compression overhead and lack of real compressed kernels dominate.                           | Systems-aware separation of quality proxy and runtime.           | First prove acceptance-quality; only then implement real compressed cache. |
| ThinkCompress uniform matches adaptive.            | Generic importance scoring is enough.                              | Reasoning accuracy may be insensitive at small sample; heuristic not targeted to acceptance. | Task-specific acceptance-risk objective.                         | Compare against uniform as strong baseline.                                |
| v3 margin claims unsupported.                      | Margin alone is the law.                                           | Current correlations failed; heuristic is not true Jacobian.                                 | Margin as control/gating variable, not sole predictor.           | Use margin to modulate risk/fallback, not as overclaimed theory.           |
| Review says C1 untrustworthy and aggregate broken. | Existing positive story can be salvaged by wording.                | Method evidence pipeline itself is unreliable.                                               | Evidence-first reconstruction.                                   | Freeze claims until code-level validation passes.                          |

---

# 8. Method Synthesis Table

| Evidence Fragment              | Source in Repo                                       | What It Reveals                                                 | Generalized Principle                               | Use in New Method?        | How to Transform It                                        |
| ------------------------------ | ---------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------- | ------------------------- | ---------------------------------------------------------- |
| Acceptance perturbation oracle | `src/acceptspec.py`, `scripts/oracle_sensitivity.py` | Directly measuring acceptance loss is the right kind of signal. | Optimize verifier decision risk, not perplexity.    | Yes                       | Fix MTP/KV bugs; produce continuous risk labels.           |
| Sparse top-k result            | M0/M1                                                | Hard sparsity not proven; gates fail.                           | Sensitivity may be budget-distributed.              | No as thesis              | Convert to budgeted risk allocation, not top-k claim.      |
| Shard variability              | M1 shards                                            | Context dependence matters.                                     | Need adaptive per-step policy.                      | Yes                       | Add margin/uncertainty gate and budget controller.         |
| Margin proposal                | v3 proposal                                          | Margin is intuitively tied to accept/reject fragility.          | Use margin as safety/control variable.              | Yes, transformed          | Do not claim “margin law”; use as gate and feature.        |
| Predictor failure              | M2                                                   | Simple classifier/feature mix is insufficient.                  | Need calibration, ranking, uncertainty.             | Yes, as negative evidence | Replace with calibrated risk predictor.                    |
| Attention proxy anomaly        | M2/review                                            | Label/mask support matters.                                     | Matched-support evaluation is mandatory.            | Yes                       | Add support-aware metric logging.                          |
| MTP implementation             | `src/speculative_decode.py`, `src/mtp_head.py`       | True MTP path exists.                                           | Reuse tested generation core.                       | Yes                       | Centralize and reuse in oracle/comparison.                 |
| Mixed precision simulation     | `MixedPrecisionKV`                                   | Quality perturbation can be simulated.                          | Separate algorithmic risk from systems compression. | Yes, limited              | Use as simulation backend; do not claim speed.             |
| Old baselines                  | `src/baselines.py`                                   | Strong simple baselines exist.                                  | Uniform/KIVI/RTN must be beaten.                    | Yes                       | Keep as baselines/ablations, not main method.              |
| ThinkCompress route            | `src/thinkcompress.py`                               | Generic reasoning compression not enough.                       | Avoid broad adaptive-compression story.             | Historical only           | Archive or baseline, not main path.                        |
| Internal review critique       | `AUTO_REVIEW.md`                                     | Main risks already identified.                                  | Design must answer reviewer objections.             | Yes                       | Build reliability manifest and negative-evidence appendix. |

---

# 9. Missing Mechanism Diagnosis

1. **Missing Mechanism Name:**
   **Calibrated Acceptance-Risk Control**

2. **One-Sentence Diagnosis:**
   当前方法试图找“稀疏 critical token”，但真正缺少的是一个能在每个 verifier step 上估计“KV 扰动会让接受决策损失多少”的**校准风险模型**，并用 margin/uncertainty 对 precision budget 做闭环控制。

3. **Evidence From Positive Results:**
   有用的正面线索不是“top-k 已经成功”，而是代码中 oracle 直接测 acceptance loss 的方向是对的；old benchmark 也显示 acceptance 和压缩策略确实会改变行为。

4. **Evidence From Negative Results:**
   M0/M1 gate 失败，说明 hard sparse/top20 假设不可靠；M2 F1 失败说明当前 predictor 不能表达真实机制；M3 缺失/代码 bug 说明 comparison 还不能证明任何主方法。

5. **Evidence From Unstable Results:**
   M1 shard0 vs shard1 差异说明 sensitivity 与 prompt/step/draft/context 有关，不能靠一个全局阈值。

6. **Evidence From Failed Ablations:**
   当前 predictor、attention-vs-acceptance divergence、margin-only narrative 都没有通过可靠验证；这些失败说明需要 calibration/ranking/uncertainty，而不是继续堆 heuristic features。

7. **Why Existing Method Cannot Solve It:**
   现有 `AcceptPredictor` 是硬阈值/正权重/无校准的分类器；`MixedPrecisionKV` 是 perturbation simulation；`core_comparison.py` 甚至没有正确调用 MTP policy path。它没有风险建模，也没有闭环预算控制。

8. **Why Simple Tuning Cannot Solve It:**
   这不是调 `top_k`、`theta_critical` 或 `quant_bits` 的问题；如果 label、MTP path、aggregation 和 objective 都不对，tuning 会优化一个污染指标。

9. **Why Existing Best Positive Fragment Is Insufficient:**
   即使某个 fragment 在某个 shard 上看起来 top20 更高，它也不能解释 M2 predictor 失败、M3 缺失、throughput 负结果、aggregation bug 和 context variability。

10. **What New Mechanism Must Do:**
    对每个 token/action 估计 acceptance-risk；输出校准均值和不确定性；在 fixed KV budget 下选择 precision/eviction；当 margin 低或 uncertainty 高时自动 fallback / 提高精度；完整记录 matched-support risk、position-wise acceptance、budget、overhead。

11. **Confidence:**
    **medium** for diagnosis that current method lacks calibrated risk control; **medium-low** that MARA will empirically win before verification.

---

# 10. New MAIN METHOD PATH

## New MAIN METHOD PATH

1. **Method Name Placeholder:**
   **MARA — Margin-Calibrated Acceptance-Risk Allocation**

2. **One-Sentence Core Idea:**
   在真实 MTP speculative decoding 中，不预测“哪些 KV token 是稀疏 critical”，而是在线估计每个 KV token 在不同精度/eviction action 下对 verifier 接受决策的校准风险，并用 margin/uncertainty gate 做闭环预算分配。

3. **Core Missing Mechanism It Adds:**
   Calibrated acceptance-risk estimation + uncertainty-aware budget controller.

4. **What Phenomena It Explains:**

   * M0/M1 fail：风险可能不是 hard sparse，而是分布式预算问题。
   * shard 差异：风险依赖 prompt/step/margin。
   * M2 predictor fail：二分类 critical-token predictor 不够，需要连续风险和校准。
   * old benchmark speed fail：acceptance-risk 是质量目标，不自动等于 systems speed。
   * margin proposal partially useful：margin 是 gate，不是单独定律。

5. **What Negative Results It Fixes:**
   它不再要求 top20 > 80%；不使用当前 F1 predictor 作为主证据；不在 broken MTP comparison 上排名；不把 compression simulation 当 runtime speed。

6. **What Existing Positive Signals It Generalizes:**
   保留 acceptance oracle 的思想，把“扰动后 acceptance 下降”泛化为 per-action expected risk；保留 margin/logit-TV 作为 features/control，而不是 claim。

7. **Why Existing Best Path Is Not Enough:**
   旧路径最多保留一个 heuristic ranker；MARA 增加了风险校准、uncertainty、step-conditioned budget、matched-support evaluation 和真实 MTP path。

8. **Core Mechanism:**
   For each verifier step `t`, token `i`, precision action `a ∈ {fp16, 4bit, 2bit, evict}`, estimate
   `R_{t,i,a} = expected acceptance loss + distribution shift + low-margin penalty`,
   then solve a budgeted allocation with risk upper confidence bound
   `R_hat = μ_phi(x) + β σ_phi(x)`.

9. **New Objective / Loss:**
   Train/calibrate predictor on oracle-sampled risk labels:

   `L_total = L_risk + λ_rank L_rank + λ_cal L_cal + λ_budget L_budget`

   where:

   * `L_risk`: Huber/MSE on `log(1 + observed_risk)`;
   * `L_rank`: pairwise ranking loss for high-risk vs low-risk sampled tokens;
   * `L_cal`: uncertainty / ECE calibration loss;
   * `L_budget`: optional penalty when allocation violates target budget.

10. **New Architecture or Module:**
    Add `src/accept_risk.py` with:

    * `AcceptanceRiskOracle`
    * `AcceptanceRiskDataset`
    * `AcceptanceRiskPredictor`
    * `RiskBudgetAllocator`
    * `MarginUncertaintyGate`

11. **New Training Procedure:**

    * Fix MTP oracle path.
    * Sample calibration prompts from non-final split.
    * For each step, perturb sampled KV tokens under multiple actions with coupled random uniforms.
    * Fit calibrated continuous risk predictor.
    * Validate on held-out prompts and seeds.

12. **New Evaluation Protocol:**
    Mandatory A/B/C comparison:

    * **A. Existing Best Positive Fragment Only**
    * **B. New MAIN METHOD Without New Mechanism**
    * **C. Full MARA**

    Also compare uniform, RTN, KIVI, Absmax, attention/H2O/SnapKV/R-KV-like policies, FP16, vanilla MTP.

13. **What Existing Components It Reuses:**
    `src/speculative_decode.py`, `src/mtp_head.py`, model loading, quantization primitives, baseline scaffolding, acceptance oracle idea.

14. **What Existing Components It Deletes:**
    No silent deletion. Remove from main path: current `AcceptPredictor` critical-token classifier and old sparse/top-k claim.

15. **What Existing Components It Rewrites:**
    `scripts/oracle_sensitivity.py`, `scripts/core_comparison.py`, `src/acceptspec.py` predictor/mixed-precision interface, result aggregation, config.

16. **What Existing Components It Keeps Only as Ablation:**
    attention proxy, margin-only score, oracle top-k, existing predicted top-k, uniform thresholds.

17. **What Existing Components It Keeps Only as Baseline:**
    RTN/KIVI/Absmax/H2O/SnapKV/R-KV-like policies, old SpecQuant/TurboQuant/ThinkCompress where relevant.

18. **Why This Is Not Merely the Existing Best Path:**
    Existing path asks “which tokens are critical?” MARA asks “what is the calibrated risk of each compression action under current verifier margin and uncertainty?” That changes the label, model, allocation, logging, and falsification tests.

19. **Why This Could Produce Real Positive Results:**
    It aligns objective with acceptance decision, avoids brittle top-k sparsity, adapts across context, and can preserve acceptance under tight budgets where uniform/attention policies waste precision on low-risk tokens.

20. **Why This Is Mechanism-Level Different from Prior Work:**
    Its claim is not generic KV compression or generic speculative decoding; it is **verifier acceptance-risk calibrated precision allocation** under true MTP verification. This must be experimentally distinguished from QuantSpec, SpecAttn, KIVI, KVQuant, H2O/SnapKV/CAKE/R-KV.

21. **Main Risk:**
    After fixing measurement, acceptance-risk may not be predictable beyond simple uniform/recency baselines, or systems overhead may dominate.

22. **Minimal Falsification Experiment:**
    On 8–16 calibration/eval prompts, if Full MARA does not beat both A and B on acceptance retention at equal simulated KV budget, and risk predictor cannot beat uniform/attention on matched-support risk ranking, stop or pivot.

23. **Confidence:**
    **medium-low as positive method**, **high as necessary pivot away from current sparse/top-k claim**.

---

# 11. Formal Method Description

## 11.1 Problem Setup

Given a target verifier model `M`, an MTP draft mechanism, prefix KV cache `C_t`, and draft block `y_{t:t+γ-1}`, speculative decoding accepts a prefix of draft tokens according to verifier probabilities. Compression action `a_i` applied to KV token `i` changes verifier logits and therefore acceptance.

Goal: choose actions `a_i ∈ A = {fp16, 4bit, 2bit, evict}` under budget `B_t` to minimize expected acceptance degradation.

## 11.2 Existing Method Failure

The existing method implicitly assumes a sparse set of critical tokens and ranks them by heuristic sensitivity/predictor scores. Current results do not validate that assumption; the predictor failed, the comparison path is broken, and aggregation is unreliable.

## 11.3 New Insight

Acceptance preservation is a **risk allocation** problem, not a static token selection problem. A token is important only relative to:

* current verifier margin;
* draft probability path;
* compression action;
* budget;
* uncertainty of the risk estimator;
* step/dataset context.

## 11.4 Method Overview

MARA has four parts:

1. **Risk oracle:** collect sampled labels by perturbing KV tokens/actions and measuring acceptance/logit/margin loss.
2. **Calibrated risk predictor:** predict risk mean and uncertainty from cheap features.
3. **Budget allocator:** choose precision actions minimizing risk upper bound under budget.
4. **Safety gate:** if margin is low or risk uncertainty is high, allocate more FP16 / fallback.

## 11.5 Objective

For sampled token/action pairs `(t, i, a)`:

`r_{t,i,a} = w_α · max(0, α_full_t - α^{a}_{t,i}) + w_tv · TV(p_full_t, p^{a}_{t,i}) + w_m · 1[m_t < τ_m] · max(0, m_full_t - m^{a}_{t,i})`

Predictor:

`(μ_{t,i,a}, σ_{t,i,a}) = g_φ(x_{t,i,a})`

Training loss:

`L_total = Huber(log(1 + r), μ) + λ_rank · log(1 + exp(-(μ_i - μ_j) · sign(r_i - r_j))) + λ_unc · NLL(r | μ, σ) + λ_ece · ECE(μ, r)`

Allocation:

`min_{a_i ∈ A} Σ_i [ μ_{t,i,a_i} + β σ_{t,i,a_i} ]`

subject to:

`Σ_i cost(a_i) ≤ B_t`

with gate:

`B_t = B_base + ΔB_low_margin · 1[m_t < τ_m] + ΔB_unc · 1[mean σ_t > τ_σ]`

## 11.6 Algorithm

**Algorithm: MARA — Margin-Calibrated Acceptance-Risk Allocation**

**Input:**
Target model `M`, MTP head `H_mtp`, prompt set `D_calib`, eval prompts `D_eval`, budget `B`, draft length `γ`, action set `A`, coupled random uniforms `U`.

**Output:**
Compressed/quantized KV policy `π_MARA`, logs of acceptance, risk calibration, budget, and ablations.

**Steps:**

1. For each calibration prompt, run true MTP speculative decoding with full KV.
2. At selected verification steps, sample KV tokens and actions.
3. For each sampled `(token, action)`, perturb KV, rerun verifier with same draft and same uniforms, compute `r_{t,i,a}`.
4. Train/calibrate `g_φ(x) → (μ, σ)` using continuous risk, ranking, and calibration losses.
5. During eval, compute features for all tokens, predict risk upper bounds, solve budgeted precision allocation.
6. Apply selected precision actions, verify draft tokens, log acceptance, position-wise acceptance, margin, risk calibration, budget, and runtime proxy.
7. Compare against A/B/C and official baselines.

## 11.7 Variables from existing code

| Variable / Concept      | Existing Source                                | Use in MARA                      |
| ----------------------- | ---------------------------------------------- | -------------------------------- |
| MTP generation state    | `src/speculative_decode.py`, `src/mtp_head.py` | True draft/verify loop           |
| Acceptance computation  | `src/acceptspec.py`                            | Rewritten as risk oracle         |
| Attention importance    | `src/acceptspec.py`                            | Optional feature/baseline only   |
| Margin/logit-TV         | `src/acceptspec.py`, `triple_divergence.py`    | Feature and gate, not sole claim |
| Quantization primitives | `src/turboquant_kv.py`, `src/baselines.py`     | Action simulation                |
| Baseline policies       | `src/baselines.py`, `core_comparison.py`       | Fair comparison                  |

## 11.8 New variables needed

| New Variable              | Purpose                                                                  |
| ------------------------- | ------------------------------------------------------------------------ |
| `risk_label`              | Continuous acceptance/logit/margin degradation label.                    |
| `risk_mu`, `risk_sigma`   | Calibrated risk prediction.                                              |
| `risk_ucb`                | Conservative allocation score.                                           |
| `margin_gate_active`      | Whether low-margin fallback activated.                                   |
| `uncertainty_gate_active` | Whether uncertainty fallback activated.                                  |
| `matched_support_mask`    | Ensures all ranking metrics use sampled/evaluable tokens only.           |
| `position_acceptance`     | Acceptance by draft position, excluding exact-target position if needed. |
| `budget_realized_bits`    | Actual simulated precision budget.                                       |

## 11.9 Required logging

* Prompt ID / split / seed / model revision.
* Draft position-wise acceptance.
* `sample_indices`, action, risk labels.
* Matched-support Spearman / PR-AUC / ECE.
* Margin distribution and gate activation rate.
* Budget per step and realized precision histogram.
* A/B/C comparison table.
* Baseline fairness metadata.
* Runtime proxy separated from true memory/kernel speed.

## 11.10 Required ablations

1. Existing best positive fragment only.
2. MARA without risk calibration.
3. MARA without margin gate.
4. MARA without uncertainty term.
5. Attention-only.
6. Margin-only.
7. Uniform 2/4-bit.
8. Oracle risk upper bound.
9. Position-0 excluded metric.
10. Matched-support vs old zero-filled aggregate.

---

# 12. Related Work and Novelty Risk

Speculative decoding itself is well-established: Leviathan et al. and Chen et al. introduced draft-and-verify schemes that preserve the target distribution while accelerating generation; Medusa adds multiple decoding heads and tree verification; EAGLE extrapolates features for faster drafting.([arXiv][16])

KV compression is also crowded: KIVI proposes tuning-free 2-bit KV quantization; H2O keeps heavy-hitter tokens based on attention; SnapKV compresses clustered important positions; CAKE and R-KV add adaptive/layer/redundancy-aware cache strategies; KVQuant adds per-channel/pre-RoPE/non-uniform quantization.([GitHub][17])

The closest novelty threats are **QuantSpec** and **SpecAttn**. QuantSpec combines self-speculative decoding with weight/KV quantization and hierarchical KV cache; SpecAttn explicitly uses verification-guided sparse attention and identifies critical KV entries from verification. These make the repo’s older “zero papers optimize KV for acceptance” claim unsafe.([arXiv][18])

| Paper                                                                       | Year / Venue         | Code                   | Mechanism                                                                 | Why Close                                | Difference from New MAIN METHOD                                                                                                             | Novelty Risk       | Required Differentiation Experiment                                                                      |
| --------------------------------------------------------------------------- | -------------------- | ---------------------- | ------------------------------------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------- |
| Leviathan et al., Fast Inference from Transformers via Speculative Decoding | 2023 / ICML-ish line | Available in ecosystem | Draft model proposes, target verifies.                                    | Base decoding setting.                   | MARA optimizes KV precision during verification, not draft algorithm.                                                                       | medium             | Compare vanilla speculative/MTP at same model/draft.                                                     |
| Chen et al., Accelerating LLM Decoding with Speculative Sampling            | 2023                 | Some impls             | Draft continuation + target verification.                                 | Same acceptance process.                 | MARA targets acceptance-risk under KV compression.                                                                                          | medium             | Show distribution/acceptance preservation under compression.                                             |
| Medusa                                                                      | 2024                 | Yes                    | Multiple decoding heads + tree attention.                                 | Alternative self-speculation.            | MARA can be applied to verifier KV; not a multi-head draft method.                                                                          | medium             | Compare or combine with Medusa-like baseline if scope expands.                                           |
| EAGLE                                                                       | 2024/2025            | Yes                    | Feature extrapolation draft.                                              | Strong speculative baseline.             | MARA controls verifier cache precision.                                                                                                     | medium             | Report against EAGLE where feasible.                                                                     |
| KIVI                                                                        | 2024                 | Yes                    | 2-bit KV quantization, key/value axis choices.                            | Strong KV quant baseline.                | MARA is acceptance-risk adaptive rather than fixed quantization.                                                                            | high baseline risk | Same model/data/budget comparison.                                                                       |
| KVQuant                                                                     | 2024 NeurIPS         | Yes                    | Outlier-aware, pre-RoPE, non-uniform KV quantization.                     | Strong quantization baseline.            | MARA optimizes acceptance-risk objective; can use KVQuant as action backend.                                                                | high               | Compare at same effective bits and memory.                                                               |
| H2O                                                                         | 2023/2024            | Yes                    | Attention heavy hitters.                                                  | Token importance cache selection.        | MARA uses verifier acceptance-risk, not attention mass.                                                                                     | medium             | Matched-support attention vs MARA risk ranking.                                                          |
| SnapKV                                                                      | 2024 NeurIPS         | Yes                    | Observation-window clustered KV compression.                              | Strong KV eviction baseline.             | MARA is speculative-verifier-specific.                                                                                                      | medium             | Long-context + reasoning + MTP comparison.                                                               |
| CAKE                                                                        | 2025 ICLR            | Possibly               | Cascading adaptive layer-wise KV eviction.                                | Adaptive cache policy.                   | MARA action score is acceptance-risk calibrated.                                                                                            | medium-high        | Compare layer/adaptive policy under same budget.                                                         |
| R-KV                                                                        | 2025                 | Yes                    | Reasoning KV importance + redundancy.                                     | Very close for reasoning KV compression. | MARA is speculative acceptance-risk, not final-answer reasoning importance.                                                                 | high               | GSM8K/MATH reasoning comparison, same memory budget.                                                     |
| QuantSpec                                                                   | 2025/2026 preprint   | Likely                 | Self-spec decoding with quantized weights/KV, hierarchical KV, FP buffer. | Very close: quantization + self-spec.    | MARA’s novelty must be calibrated per-token/action acceptance-risk allocation, not generic quantized self-spec.                             | very high          | Reproduce QuantSpec or compare official numbers/code; ablate risk calibration vs FP buffer/hierarchy.    |
| SpecAttn                                                                    | 2026 preprint        | Unknown                | Verification-guided sparse attention, critical KV from verification.      | Extremely close mechanism-wise.          | MARA controls verifier-side precision/quantization with calibrated risk and uncertainty; SpecAttn is sparse attention/loading for drafting. | very high          | Same MTP setting: SpecAttn-style critical loading vs MARA risk allocation; show calibration/gate effect. |
| Don’t Waste Bits                                                            | 2026                 | Unknown                | Adaptive precision policy.                                                | Close adaptive quantization mechanism.   | MARA’s target is speculative verifier acceptance-risk.                                                                                      | high               | Compare adaptive precision objective vs MARA objective.                                                  |

## Novelty-risk decision

* **Cannot claim:** “first to optimize KV for speculative acceptance” or “attention is wrong” broadly.
* **Can claim if verified:** “We formulate verifier-side KV compression as calibrated acceptance-risk allocation and show that risk calibration + margin/uncertainty gating improves acceptance retention over uniform, attention, margin-only, and existing repo fragments under true MTP.”
* **Must include:** QuantSpec, SpecAttn, KIVI, KVQuant, SnapKV/H2O, R-KV/CAKE-style baselines or clearly scoped reproductions.

---

# 13. Keep / Delete / Rewrite / Archive Plan

| Item                             | Type       | File / Directory / Claim / Experiment                 | Current Role             | Problem Under New MAIN PATH                 | Action                                              | Reason                                          |
| -------------------------------- | ---------- | ----------------------------------------------------- | ------------------------ | ------------------------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| MTP decoder                      | Code       | `src/speculative_decode.py`                           | True MTP generation      | Needs shared step API/tests                 | KEEP / REWRITE locally                              | Core correct path exists; centralize.           |
| MTP head                         | Code       | `src/mtp_head.py`                                     | MTP implementation       | Needs tests only                            | KEEP                                                | Required.                                       |
| Model loader                     | Code       | `src/gpu_auto.py`                                     | Load models/MTP          | Needs metadata logging                      | KEEP                                                | Useful.                                         |
| Oracle sensitivity script        | Script     | `scripts/oracle_sensitivity.py`                       | M0/M1                    | P0 KV bug, aggregation issues               | REWRITE                                             | Convert to MARA risk-label collector.           |
| Core comparison                  | Script     | `scripts/core_comparison.py`                          | M3 policy comparison     | P0 not true MTP; invalid predictor features | REWRITE                                             | Must become fair A/B/C evaluator.               |
| Triple divergence                | Script     | `scripts/triple_divergence.py`                        | M2 diagnostics           | Failed but has better MTP path              | MERGE INTO NEW METHOD                               | Reuse matched-support diagnostics.              |
| Current `AcceptPredictor`        | Code       | `src/acceptspec.py`                                   | Critical-token predictor | F1 failed; not calibrated                   | REWRITE                                             | Replace with `AcceptanceRiskPredictor`.         |
| Margin heuristic                 | Code/claim | `src/acceptspec.py`, v3 docs                          | Proposed mechanism       | Not supported as law                        | KEEP ONLY AS ABLATION                               | Use as feature/gate.                            |
| Attention proxy                  | Code       | `src/acceptspec.py`, `triple_divergence.py`           | Baseline / contrast      | Artifacts possible                          | KEEP ONLY AS BASELINE                               | Needs validity logging.                         |
| MixedPrecisionKV simulation      | Code       | `src/acceptspec.py`                                   | Compression simulation   | Not actual memory compression               | REWRITE / KEEP AS SIMULATION                        | Separate quality proxy from systems claim.      |
| Baselines                        | Code       | `src/baselines.py`                                    | Comparators              | Need official parity                        | KEEP ONLY AS BASELINE                               | Essential for fairness.                         |
| ThinkCompress                    | Code       | `src/thinkcompress.py`                                | Old adaptive compression | Not main mechanism                          | ARCHIVE / KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE | Avoid confusing narrative.                      |
| TurboQuant / quant primitives    | Code       | `src/turboquant_kv.py`                                | Quant backend            | May be useful                               | KEEP                                                | Use as action backend.                          |
| M0/M1 JSON                       | Results    | `results/acceptspec/oracle_m0.json`, `oracle_m1.json` | Claimed evidence         | Bug/merge contaminated                      | ARCHIVE                                             | Historical negative/unreliable evidence only.   |
| M2 logs                          | Logs       | `logs/M2_divergence.log`                              | Predictor evidence       | Failed, artifacts                           | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE           | Motivates rewrite.                              |
| M3 logs                          | Logs       | `logs/M3_*`                                           | Comparison evidence      | No final result                             | ARCHIVE                                             | Not usable.                                     |
| Old benchmark JSON               | Results    | `results/benchmark/*`                                 | Old positive/negative    | Different setup, not final                  | KEEP ONLY AS HISTORICAL NEGATIVE EVIDENCE           | Shows acceptance ≠ speed.                       |
| README sparse claims             | Claim      | README                                                | Project pitch            | Too strong                                  | REWRITE                                             | Replace with reliability status and MARA plan.  |
| IDEA_REPORT “zero papers” claim  | Claim      | `IDEA_REPORT.md`                                      | Novelty framing          | False/unsafe after QuantSpec/SpecAttn       | DELETE / REWRITE                                    | High novelty risk.                              |
| FINAL_PROPOSAL margin-law claims | Claim      | `refine-logs-v3/FINAL_PROPOSAL_v3.md`                 | v3 thesis                | Unsupported                                 | FREEZE                                              | Keep as historical proposal, not current paper. |
| Experiment tracker TODOs         | Docs       | `refine-logs/EXPERIMENT_TRACKER.md`                   | Plans                    | Useful                                      | KEEP                                                | Track completion honestly.                      |
| Figure script fallback           | Script     | `scripts/generate_figures.py`                         | Plots                    | Placeholder risk                            | REWRITE                                             | Fail closed unless explicit placeholder flag.   |
| Test suite                       | Tests      | `tests/test_specquant.py`                             | Existing tests           | Missing core tests                          | KEEP + EXTEND                                       | Add AcceptSpec/MTP tests.                       |

---

# 14. Claude Code Implementation Plan

## Task 1: Quarantine unreliable artifacts and create reliability manifest

**Purpose:** Stop old contaminated results/claims from guiding method choice.
**Which Phenomenon / Constraint It Addresses:** PH4, PH11, C3, C9.
**Why It Supports New MAIN METHOD PATH:** MARA must start from clean evidence, not old sparse claims.
**Files to Inspect:** `README.md`, `IDEA_REPORT.md`, `refine-logs-v3/FINAL_PROPOSAL_v3.md`, `review-stage/AUTO_REVIEW.md`, `results/acceptspec/*`, `logs/*`.
**Files to Edit:** Add `docs/RELIABILITY_AUDIT.md`; add `results/acceptspec/README_reliability.md`; lightly edit README status block.
**Files to Delete / Archive:** Do not delete; move or mark old M0/M1/M2/M3 outputs as unreliable/historical.
**Functions / Classes:** None.
**Exact Change:** Add manifest listing each result, status, known bugs, and “not for paper claim” flag.
**Do Not Change:** Do not alter raw JSON/log values.
**Verification Command:**
`python - <<'PY'\nfrom pathlib import Path\nassert Path('docs/RELIABILITY_AUDIT.md').exists()\nassert 'oracle_m1' in Path('docs/RELIABILITY_AUDIT.md').read_text()\nPY`
**Expected Result:** Manifest exists and names all contaminated artifacts.
**Failure Means:** Old results may still be accidentally cited.
**Rollback Condition:** If manifest edits touch raw results, revert.
**Priority:** P0.
**Confidence:** high.

## Task 2: Add deterministic metadata, split, and seed utilities

**Purpose:** Prevent leakage/nondeterminism.
**Addresses:** PH6, C5.
**Files to Inspect:** `src/utils.py`, all dataset loaders in `scripts/`.
**Files to Edit:** `src/repro.py` or `src/utils.py`; scripts using datasets.
**Exact Change:** Add `set_global_seed`, `make_split_manifest`, `save_run_metadata`, explicit `torch.Generator`.
**Do Not Change:** Dataset content or metrics.
**Verification Command:**
`pytest -q tests/test_repro.py`
**Expected Result:** Same seed produces same prompt IDs and random uniforms.
**Failure Means:** Multi-seed comparison remains invalid.
**Rollback Condition:** If dataset order changes without manifest.
**Priority:** P0.
**Confidence:** high.

## Task 3: Add MTP and oracle invariant tests

**Purpose:** Catch KV-length and target-as-draft bugs.
**Addresses:** PH8, C6.
**Files to Inspect:** `src/speculative_decode.py`, `scripts/oracle_sensitivity.py`, `scripts/core_comparison.py`.
**Files to Edit:** `tests/test_acceptspec_core.py`, `tests/test_mtp_policy_path.py`.
**Exact Change:** Add mocked cache tests asserting `kv_len == past_key_values length`; in MTP mode assert policy path calls `mtp_head`, not target model as draft.
**Do Not Change:** Production logic in this task.
**Verification Command:**
`pytest -q tests/test_acceptspec_core.py tests/test_mtp_policy_path.py`
**Expected Result:** Tests initially expose failures before Task 4/5.
**Failure Means:** Tests are not exercising real bug.
**Rollback Condition:** If tests require GPU/large models.
**Priority:** P0.
**Confidence:** high.

## Task 4: Fix `oracle_sensitivity.py` and matched-support aggregation

**Purpose:** Repair M0/M1 measurement.
**Addresses:** PH1, PH2, PH4, C1, C3.
**Files to Edit:** `scripts/oracle_sensitivity.py`.
**Exact Change:**

* Fix `kv_len` resync to `new_kv_len + 1`.
* Pass coupled uniforms / generator through all acceptance calls.
* Save `sample_indices`, `sampled_sensitivities`, `sampled_logit_tv`, action metadata.
* Add matched-support aggregation; remove zero-filled top-k from main metric.
  **Do Not Change:** Do not tune thresholds to pass M0/M1.
  **Verification Command:**
  `python scripts/oracle_sensitivity.py --model Qwen/Qwen3.5-9B --num_problems 2 --max_tokens 32 --gamma 5 --temperature 0.0 --sample_fraction 1.0 --output results/debug/oracle_fixed_smoke.json`
  **Expected Result:** Smoke completes; metadata includes support masks; no KV invariant failure.
  **Failure Means:** MTP/oracle loop still inconsistent.
  **Rollback Condition:** If fixed script changes acceptance semantics without documenting.
  **Priority:** P0.
  **Confidence:** high.

## Task 5: Refactor true MTP draft/verify into shared helper

**Purpose:** Ensure all experiments compare same decoding process.
**Addresses:** PH8, C6.
**Files to Edit:** Add `src/mtp_loop.py`; update `src/speculative_decode.py`, `scripts/core_comparison.py`, `scripts/oracle_sensitivity.py`, optionally `scripts/triple_divergence.py`.
**Exact Change:** Expose a tested function that produces draft tokens/probs, verifier logits, accepted count, updated KV.
**Do Not Change:** Model weights, dataset, metric definitions.
**Verification Command:**
`pytest -q tests/test_mtp_policy_path.py`
**Expected Result:** MTP policy path no longer calls target model as draft.
**Failure Means:** Core comparisons remain unusable.
**Rollback Condition:** If vanilla decoder outputs change on no-compression baseline beyond expected numerical tolerance.
**Priority:** P0.
**Confidence:** high.

## Task 6: Implement MARA risk core

**Purpose:** Add new missing mechanism.
**Addresses:** C1-C4, C8.
**Files to Edit:** New `src/accept_risk.py`; possibly slim `src/acceptspec.py`.
**Functions / Classes:** `AcceptanceRiskOracle`, `AcceptanceRiskPredictor`, `RiskBudgetAllocator`, `MarginUncertaintyGate`.
**Exact Change:** Implement continuous risk labels, simple calibrated predictor, greedy budget allocation by risk-per-bit, margin/uncertainty fallback.
**Do Not Change:** Baseline implementations.
**Verification Command:**
`pytest -q tests/test_accept_risk.py`
**Expected Result:** Predictor can overfit toy risk labels; allocator respects budget.
**Failure Means:** New mechanism not even locally testable.
**Rollback Condition:** If MARA code entangles with baselines or changes metrics.
**Priority:** P0.
**Confidence:** medium.

## Task 7: Add MARA calibration script and config

**Purpose:** Create minimal reproducible path.
**Files to Edit:** `scripts/calibrate_mara.py`, `configs/mara_minimal.yaml`.
**Exact Change:** Script collects oracle risk samples, fits predictor, saves model + calibration report.
**Do Not Change:** Existing default config until smoke passes.
**Verification Command:**
`python scripts/calibrate_mara.py --config configs/mara_minimal.yaml --num_calib 4 --num_eval 4 --sample_fraction 1.0 --output results/mara/calib_smoke.json`
**Expected Result:** JSON includes risk AUC/Spearman/ECE, not NaN.
**Failure Means:** Risk signal or instrumentation broken.
**Rollback Condition:** If it silently falls back to placeholder labels.
**Priority:** P1.
**Confidence:** medium.

## Task 8: Add A/B/C comparison in core comparison

**Purpose:** Prove MARA is not old positive fragment.
**Files to Edit:** `scripts/core_comparison.py`, `configs/mara_minimal.yaml`.
**Exact Change:** Add policy names:

* `existing_best_fragment_only`
* `mara_no_gate_or_uncertainty`
* `mara_full`

All must share true MTP loop and same budget.
**Verification Command:**
`python scripts/core_comparison.py --config configs/mara_minimal.yaml --dataset gsm8k --num_problems 4 --policies existing_best_fragment_only,mara_no_gate_or_uncertainty,mara_full --output_dir results/mara/smoke`
**Expected Result:** Produces comparable table with same prompts/seeds/budget.
**Failure Means:** No valid main-method test.
**Rollback Condition:** If policy changes baseline definitions.
**Priority:** P1.
**Confidence:** medium.

## Task 9: Add metric/data/checkpoint sanity tests

**Purpose:** Prevent train/test leakage and stale checkpoint use.
**Files to Edit:** `tests/test_data_metric_sanity.py`, eval scripts.
**Exact Change:** GSM8K/MATH metric known-answer tests; split-overlap test; checkpoint metadata check.
**Verification Command:**
`pytest -q tests/test_data_metric_sanity.py`
**Expected Result:** All sanity tests pass.
**Failure Means:** Experiment metrics cannot be trusted.
**Priority:** P1.
**Confidence:** high.

## Task 10: Run minimal experiment queue before full benchmark

**Purpose:** Gate expansion.
**Files to Edit:** `scripts/run_mara_minimal_suite.sh` or `scripts/run_mara_minimal_suite.py`.
**Exact Change:** Sequentially run smoke, oracle, calibration, A/B/C, multi-seed small comparison.
**Verification Command:**
`bash scripts/run_mara_minimal_suite.sh`
**Expected Result:** Logs saved; failures stop execution.
**Failure Means:** Do not run full benchmark.
**Priority:** P1.
**Confidence:** medium.

## Task 11: Only after minimal pass, update paper/README claims

**Purpose:** Preserve academic integrity.
**Files to Edit:** README, paper draft if present, docs.
**Exact Change:** Replace sparse/top-k/margin-law claims with evidence-backed MARA thesis.
**Do Not Change:** Do not claim SOTA unless official baselines reproduced.
**Verification Command:**
`python scripts/check_claims_against_results.py` if created; otherwise manual checklist.
**Expected Result:** Every claim maps to a result row.
**Failure Means:** Paper narrative remains overclaimed.
**Priority:** P2.
**Confidence:** high.

---

# 15. Minimal Verification Experiments

| Priority | Experiment                               | Hypothesis                                                | Command                                                                                                                                         | Config           | Dataset          | Seeds    | Metric                              | Success Criterion                           | Failure Interpretation                                              |
| -------: | ---------------------------------------- | --------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | ---------------- | -------- | ----------------------------------- | ------------------------------------------- | ------------------------------------------------------------------- |
|        1 | Smoke test                               | Code imports and tests run.                               | `pytest -q`                                                                                                                                     | —                | —                | —        | pass/fail                           | All non-GPU unit tests pass.                | Repo not safe to modify.                                            |
|        2 | Data sanity check                        | Calibration/eval IDs do not overlap.                      | `pytest -q tests/test_data_metric_sanity.py`                                                                                                    | split manifest   | GSM8K/MATH       | 42       | overlap count                       | zero overlap                                | Leakage risk; stop.                                                 |
|        3 | Metric sanity check                      | GSM8K/MATH metrics parse known answers.                   | same as above                                                                                                                                   | —                | toy examples     | —        | exact match                         | known examples pass                         | Reported accuracy invalid.                                          |
|        4 | One-batch overfit risk predictor         | MARA predictor can fit toy labels.                        | `pytest -q tests/test_accept_risk.py`                                                                                                           | toy              | synthetic        | 42       | loss ↓                              | near-zero toy loss                          | Predictor implementation broken.                                    |
|        5 | Checkpoint/MTP loading check             | True MTP path loads and logs revisions.                   | `python scripts/check_mtp_loading.py --model Qwen/Qwen3.5-9B`                                                                                   | minimal          | one prompt       | 42       | metadata                            | model/MTP revisions saved                   | Cannot compare runs.                                                |
|        6 | Reproduce corrected current negative     | After fixes, sparse top-k still not assumed.              | `python scripts/oracle_sensitivity.py --model Qwen/Qwen3.5-9B --num_problems 8 --sample_fraction 1.0 --output results/mara/oracle_fixed_8.json` | fixed oracle     | GSM8K calib      | 42       | matched-support top-k/Gini          | Valid non-NaN metrics                       | If sparse suddenly passes, inspect old bug effect; still not final. |
|        7 | Reproduce current best positive fragment | Old best fragment can be run as A.                        | `python scripts/core_comparison.py --config configs/mara_minimal.yaml --policies existing_best_fragment_only --num_problems 8`                  | MARA minimal     | GSM8K eval       | 42       | acceptance retention                | Produces valid A row                        | Cannot prove new method adds value.                                 |
|        8 | New mechanism activation check           | MARA gate activates on low-margin/high-uncertainty steps. | `python scripts/calibrate_mara.py --config configs/mara_minimal.yaml --num_calib 8 --num_eval 8`                                                | MARA             | GSM8K calib/eval | 42       | gate rate, ECE                      | Nonzero gate rate, finite ECE               | Mechanism inactive or uncalibrated.                                 |
|        9 | New MAIN METHOD minimal test             | Full MARA improves acceptance retention at equal budget.  | `python scripts/core_comparison.py --config configs/mara_minimal.yaml --policies mara_full --num_problems 8`                                    | MARA             | GSM8K eval       | 42       | accepted tokens / TV / exact answer | Better than uniform or tied with lower risk | If no gain, inspect risk signal.                                    |
|       10 | Key ablation: remove new mechanism       | Without risk/gate, performance drops.                     | `... --policies mara_no_gate_or_uncertainty,mara_full`                                                                                          | MARA             | GSM8K            | 42       | Δ acceptance                        | full > no-gate                              | If tied, gate may be unnecessary.                                   |
|       11 | A: Existing Best Positive Fragment Only  | Establish old-fragment baseline.                          | `... --policies existing_best_fragment_only`                                                                                                    | MARA             | GSM8K            | 42       | acceptance/TV                       | valid row                                   | If A wins, MARA not justified.                                      |
|       12 | B: New MAIN METHOD Without New Mechanism | Control for refactor/overhead.                            | `... --policies mara_no_gate_or_uncertainty`                                                                                                    | MARA             | GSM8K            | 42       | acceptance/TV                       | between A and C                             | If B=C, new mechanism not causal.                                   |
|       13 | C: Full New MAIN METHOD                  | Full method.                                              | `... --policies mara_full`                                                                                                                      | MARA             | GSM8K            | 42       | acceptance/TV/task                  | C > A and C > B                             | If not, stop or pivot.                                              |
|       14 | Small baseline comparison                | Beat simple strong baselines.                             | `... --policies fp16,uniform_4bit,rtn,kivi,attention,mara_full`                                                                                 | MARA             | GSM8K            | 42       | acceptance/task                     | MARA improves retention at same budget      | If uniform wins, method weak.                                       |
|       15 | Multi-seed stability                     | Gain is not seed cherry-pick.                             | same with `--seeds 42,123,456`                                                                                                                  | MARA             | GSM8K            | 3 seeds  | mean±std                            | positive mean, no single-seed dependence    | If high variance, need stabilization.                               |
|       16 | Expansion gate                           | Small result scales to 50–100 prompts.                    | `... --num_problems 50` then 100                                                                                                                | MARA             | GSM8K/MATH       | 3 seeds  | CI                                  | effect persists                             | If disappears, overfit to small calibration.                        |
|       17 | Official baseline reproduction           | Fair comparison.                                          | baseline-specific official commands                                                                                                             | official configs | same             | official | reported metric                     | within tolerance of paper or explain gap    | Cannot claim superiority.                                           |
|       18 | Unified environment comparison           | Avoid env mismatch.                                       | same script all policies                                                                                                                        | one env          | GSM8K/MATH       | 3        | same metrics                        | all rows same metadata                      | Mixed env invalidates table.                                        |
|       19 | Robustness/generalization                | Not dataset-specific.                                     | `... --dataset math` plus gamma/temp variants                                                                                                   | MARA             | MATH/GSM8K       | 3        | acceptance/task                     | no catastrophic collapse                    | If only GSM8K works, narrow claim.                                  |
|       20 | Statistical significance / CI            | Effect not noise.                                         | aggregation script with bootstrap                                                                                                               | MARA             | full eval        | 3+       | CI, paired test                     | CI excludes zero for primary metric         | If not significant, claim exploratory only.                         |

---

# 16. Baseline and SOTA Plan

| Baseline                             | Why Required                             | Official Code                           | Dataset    | Metric                            | Reproduction Requirement                 | Fairness Risk                         |
| ------------------------------------ | ---------------------------------------- | --------------------------------------- | ---------- | --------------------------------- | ---------------------------------------- | ------------------------------------- |
| FP16 autoregressive target           | Absolute quality/speed reference.        | HF generation                           | GSM8K/MATH | accuracy, tokens/s                | Same model/revision.                     | Different max tokens / prompt format. |
| Vanilla MTP speculative decoding     | Direct no-compression baseline.          | Repo MTP path                           | GSM8K/MATH | acceptance, accuracy, speed proxy | Same MTP head, gamma.                    | Position-0 inflation.                 |
| Uniform 4-bit / 2-bit KV             | Simplest strong compression baseline.    | Repo/simple                             | same       | acceptance/task                   | Same budget.                             | Budget mismatch.                      |
| RTN / Absmax                         | Basic quantization baselines.            | Repo                                    | same       | acceptance/task                   | Same calibration/eval split.             | Weak implementation.                  |
| KIVI                                 | Strong KV quantization.                  | Official code available.([GitHub][17])  | same       | task/memory                       | Reproduce official or use faithful port. | Axis/bit mismatch.                    |
| KVQuant                              | Strong quantization SOTA.                | Official/available.([arXiv][19])        | same       | task/memory                       | Same model if feasible.                  | Kernel/backend unfairness.            |
| H2O                                  | Attention heavy-hitter baseline.         | Official/available.([OpenReview][20])   | same       | task/memory                       | Same cache budget.                       | Attention extraction validity.        |
| SnapKV                               | Strong KV compression baseline.          | Official/available.([OpenReview][21])   | same       | task/memory                       | Same context length/window.              | Long-context vs reasoning mismatch.   |
| CAKE                                 | Adaptive KV eviction.                    | Check official.([OpenReview][22])       | same       | task/memory                       | Same budget, layer policy.               | Different long-context setting.       |
| R-KV                                 | Reasoning-specific KV compression.       | GitHub available.([GitHub][23])         | GSM8K/MATH | reasoning accuracy/memory         | Official reproduction if possible.       | Very close; must be fair.             |
| QuantSpec                            | Closest self-spec quantization baseline. | Check code.([arXiv][18])                | same       | acceptance/speed/memory           | Official or faithful.                    | If omitted, novelty challenged.       |
| SpecAttn                             | Closest verification-guided baseline.    | Check availability.([ResearchGate][24]) | same       | acceptance/speed                  | Official or reimplementation.            | Very high novelty risk.               |
| Existing Best Positive Fragment Only | Internal ablation.                       | Repo                                    | same       | acceptance/task                   | Fixed code path after refactor.          | Cherry-picking if not locked.         |
| MARA without new mechanism           | Causal ablation.                         | New repo                                | same       | acceptance/task                   | Same calibration budget.                 | Must not accidentally include gate.   |
| Oracle upper bound                   | Shows headroom.                          | New oracle                              | same       | risk/task                         | Same sampled support.                    | Not deployable; mark upper bound.     |

Paper-reported numbers may be used only as supplementary context if official reproduction is infeasible; they cannot be the only comparison.

---

# 17. Paper Thesis Reconstruction

1. **New Paper Thesis:**
   Speculative decoding KV compression should be formulated as **calibrated verifier acceptance-risk allocation**, not as generic attention preservation or static sparse critical-token selection.

2. **Main Technical Contribution:**
   MARA: a risk oracle, calibrated risk predictor, and margin/uncertainty-aware budget allocator for verifier-side KV precision under MTP speculative decoding.

3. **Main Empirical Claim:**
   If experiments pass: MARA preserves speculative acceptance and downstream accuracy better than uniform, attention-based, margin-only, and existing internal fragments at the same simulated KV budget under a corrected true-MTP evaluation.

4. **What Previous Failures Taught Us:**
   Sparse top-k and margin-law claims are brittle; predictor F1 classification is not enough; aggregation and MTP path correctness dominate conclusions; acceptance does not automatically imply speed.

5. **What We Should Not Claim:**

   * “First acceptance-aware KV compression.”
   * “Attention is wrong” broadly.
   * “Sparse top20 captures sensitivity.”
   * “Jacobian/margin law proven.”
   * “SOTA speedup/memory reduction.”
   * “NeurIPS ready.”

6. **What We Can Claim If Experiments Pass:**

   * Calibrated acceptance-risk is a better allocation signal than uniform/attention/margin-only in corrected MTP.
   * Margin/uncertainty gates improve stability.
   * Old sparse results were insufficient; continuous risk allocation explains mixed outcomes better.

7. **Required Baselines:**
   FP16, vanilla MTP, uniform, RTN/Absmax/KIVI/KVQuant, H2O/SnapKV, R-KV/CAKE, QuantSpec, SpecAttn where feasible.

8. **Required Ablations:**
   A/B/C, no margin gate, no uncertainty, no risk calibration, attention-only, margin-only, oracle upper bound, position-0 excluded.

9. **Required Robustness Tests:**
   GSM8K/MATH, gamma variants, temperature variants, model size if compute allows, multi-seed, calibration size sensitivity.

10. **Reviewer Likely Objections:**
    “This is just QuantSpec/SpecAttn/KVQuant plus heuristics”; “baselines weak”; “old results failed”; “MARA adds overhead”; “no real compression kernel.”

11. **How New MAIN METHOD Answers Them:**
    By making risk calibration measurable, comparing directly to closest mechanisms, separating simulation from systems, and showing C > A/B under matched budget.

12. **What Would Make This NeurIPS-Strong:**
    Clean corrected pipeline, official baselines, statistically stable gains, mechanism ablations, honest negative results, and a clear distinction from QuantSpec/SpecAttn.

13. **What Would Make This Rejected:**
    Overclaiming novelty, hiding failed M0/M2, using broken core comparison, weak baselines, no official reproduction, no real evidence that calibration matters.

14. **What Would Be Required for Oral-Level Strength:**
    Strong gains across multiple model families/draft mechanisms, robust theory/analysis of acceptance-risk calibration, and real memory/speed implementation.

15. **What Would Be Required for Best-Paper-Level Strength:**
    A broadly reusable theory/tool for verifier-risk-aware compression that transfers across speculative decoding families and ships with strong kernels, official baselines, and reproducible benchmark suite. This is far beyond current evidence.

---

# 18. Reviewer Risk Assessment

| Risk                            | Why Reviewer May Object                              | Evidence Needed                              | How New MAIN METHOD Addresses It                                                    | Remaining Weakness                       |
| ------------------------------- | ---------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------------------------------------- | ---------------------------------------- |
| Novelty risk                    | QuantSpec/SpecAttn already close.                    | Direct related-work comparison.              | MARA claims calibrated acceptance-risk allocation, not generic quantized self-spec. | Still high.                              |
| Incremental risk                | Could look like heuristic feature scoring.           | A/B/C causal ablations.                      | Risk calibration + uncertainty + gate must drive gains.                             | If C≈A/B, incremental.                   |
| Baseline weakness risk          | KIVI/KVQuant/R-KV/SnapKV/QuantSpec may be omitted.   | Official reproductions.                      | Baseline plan includes them.                                                        | Compute heavy.                           |
| Reproducibility risk            | Current logs have missing commands/bugs.             | Manifest, exact configs, seeds, checkpoints. | Tasks 1–5 fix this.                                                                 | Old results remain unusable.             |
| Cherry-picking risk             | Positive fragments could be selected after failures. | Predeclared minimal queue and gates.         | A/B/C and stop criteria.                                                            | Must actually follow plan.               |
| Negative result hiding risk     | Repo contains failed M0/M2.                          | Negative-evidence appendix.                  | Archive, cite, explain failures.                                                    | Paper may become harder to sell.         |
| Overclaiming risk               | README/proposals too strong.                         | Claim-code-result matrix.                    | Rewrite claims after evidence.                                                      | Requires discipline.                     |
| Unclear mechanism risk          | Current margin/Jacobian story weak.                  | Calibration curves, gate activation logs.    | Mechanism is risk calibration.                                                      | If risk not predictable, fails.          |
| Ablation insufficiency risk     | Need show new mechanism, not refactor.               | A/B/C, no gate, no uncertainty.              | Included.                                                                           | Must be statistically powered.           |
| Dataset limitation risk         | GSM8K-only not enough.                               | MATH and robustness tests.                   | Minimal expansion gate.                                                             | More tasks may be needed.                |
| Compute unfairness risk         | Different kernels/models/budgets.                    | Unified environment metadata.                | Budget/metadata logging.                                                            | Official baselines hard.                 |
| Implementation reliability risk | P0 bugs found.                                       | Unit tests and invariants.                   | Tasks 3–5.                                                                          | More hidden bugs possible.               |
| Related work omission risk      | Rapidly evolving area.                               | Up-to-date citations.                        | Related work table includes closest threats.                                        | Need final paper update near submission. |

---

# 19. Final Decision

## 1. One-Sentence Verdict

当前项目应停止把 AcceptSpec/MarginSpec 写成“稀疏 critical KV”或“margin law”论文，唯一推荐主线是 **MARA: Margin-Calibrated Acceptance-Risk Allocation**，即在真实 MTP speculative decoding 下做校准 acceptance-risk 估计和 margin/uncertainty 控制的 KV precision allocation。

## 2. Current Most Likely Root Cause

当前失败最可能来自组合原因：

* **code/evaluation bug:** oracle KV length、core comparison MTP path、aggregation；
* **method assumption failure:** hard sparse/top-k criticality 不成立或未证实；
* **baseline mismatch:** 还没有可靠官方/统一环境比较；
* **missing mechanism:** 缺少 calibrated acceptance-risk + uncertainty/budget control；
* **novelty issue:** QuantSpec/SpecAttn/R-KV 等让旧 gap claim 很危险；
* **insufficient evidence:** 没有可支持 SOTA/ready 的结果。

## 3. Why This Is Not Just the Existing Best Path

Existing best fragment 仍是在问“哪些 token 排名最高”。MARA 改成问“在当前 verifier margin 和不确定性下，每个 token 的每个压缩 action 会带来多少 acceptance-risk，以及在预算内该如何分配精度”。这改变了 objective、label、模型、allocator、logging、ablation 和 novelty framing。

## 4. Phenomena Explained

MARA 解释：

* M0/M1 sparse gate 失败：风险不是 hard sparse；
* M1 shard 差异：需要 context/step adaptive；
* M2 predictor F1 失败：二分类 heuristic predictor 不够；
* M3 不可靠：必须先修真实 MTP comparison；
* old benchmark slower：acceptance preservation 不是系统 speedup；
* margin proposal：margin 可做 gate，但不能单独当理论。

## 5. Mechanism Missing in Current Method

当前缺少的是：**校准的 acceptance-risk 模型和闭环预算控制机制**。

## 6. New Mechanism

MARA 新增：

* continuous per-token/per-action risk label；
* calibrated risk predictor `μ, σ`；
* risk upper-confidence allocation；
* low-margin / high-uncertainty fallback；
* matched-support evaluation；
* A/B/C causal ablation。

## 7. What to Delete / Archive / Rewrite

* **ARCHIVE:** old M0/M1 aggregate, M2/M3 stale logs, old benchmark as historical evidence.
* **REWRITE:** `oracle_sensitivity.py`, `core_comparison.py`, `AcceptPredictor`, aggregation, README claims.
* **KEEP:** MTP decoder/head, model loader, quant primitives, baselines.
* **KEEP ONLY AS ABLATION:** attention proxy, margin-only, existing predicted criticality.
* **KEEP ONLY AS BASELINE:** uniform, RTN, KIVI, Absmax, H2O/SnapKV/R-KV-like policies.

## 8. First Five Claude Code Tasks

1. Create reliability manifest and quarantine old results.
2. Add deterministic split/seed/metadata utilities.
3. Add MTP/oracle invariant tests.
4. Fix oracle KV length and matched-support aggregation.
5. Refactor true MTP draft/verify into shared helper and repair core comparison.

## 9. Minimal Experiments

The minimal queue is:

1. smoke/unit tests;
2. data/metric sanity;
3. one-batch risk overfit;
4. checkpoint/MTP loading;
5. corrected oracle negative reproduction;
6. A existing best fragment;
7. B MARA without new mechanism;
8. C Full MARA;
9. small strong baselines;
10. 3-seed stability;
11. expansion gate to 50–100 prompts;
12. official baseline reproduction before any SOTA claim.

## 10. Continue / Stop / Pivot Criteria

**Continue** if:

* Full MARA beats A and B at equal budget on matched-support acceptance-risk and downstream metric;
* risk predictor has nontrivial calibration/ranking;
* gains persist over 3 seeds and at least GSM8K + MATH small eval.

**Stop** if:

* corrected oracle shows no predictable acceptance-risk signal;
* uniform/KIVI/attention baselines beat MARA consistently;
* C ≈ A/B, meaning new mechanism is not causal.

**Pivot** if:

* risk signal exists but systems overhead dominates: pivot to quality-only paper or kernel/backend work;
* risk signal exists only for specific draft positions/prompts: narrow to uncertainty-triggered fallback method;
* SpecAttn/QuantSpec already covers mechanism: pivot to explicit differentiation or abandon novelty claim.

## 11. NeurIPS-Level Gap

Needed:

* fixed code and reproducible results;
* official baseline comparisons;
* multi-seed confidence intervals;
* direct QuantSpec/SpecAttn/KVQuant/R-KV comparison;
* mechanism ablations proving calibration/gate causal;
* honest negative-result discussion;
* no systems speed claim unless real compressed storage/kernel exists.

## 12. Oral / Best Paper Gap

For oral-level: broad model/task coverage, robust theory of acceptance-risk calibration, and strong official baselines.
For best-paper-level: a general, reusable verifier-risk compression framework plus real kernels and cross-decoding-family transfer. Current repo is far from that.

## 13. Confidence

**Overall confidence: medium.**

* **High confidence:** current sparse/top-k/margin-law claims are not reliable enough.
* **High confidence:** P0/P1 code/result reliability issues must be fixed before any claim.
* **Medium confidence:** calibrated acceptance-risk control is the right missing mechanism.
* **Medium-low confidence:** MARA will produce strong positive results; it is an evidence-backed hypothesis, not proven.

---

# 20. Final Claude Code Instruction

```text
Claude Code, execute the following plan.

You must implement the New MAIN METHOD PATH defined in the GPT-5.5 Pro diagnosis report:

MARA — Margin-Calibrated Acceptance-Risk Allocation.

Do not invent a different method.
Do not optimize for superficial positive results.
Do not weaken baselines.
Do not delete negative results silently.
Do not change metrics or datasets unless explicitly instructed.
Do not rewrite unrelated files.
Do not proceed to full benchmark until minimal tests pass.

Your tasks are:

1. Create a reliability manifest.
   - Add docs/RELIABILITY_AUDIT.md.
   - Mark old M0/M1/M2/M3 results as historical/unreliable where appropriate.
   - Do not edit raw JSON/log values.

2. Add deterministic seed, split, and run-metadata utilities.
   - Ensure calibration/eval/test prompt IDs are saved.
   - Ensure random uniforms used in acceptance comparison are reproducible and logged.

3. Add tests for AcceptSpec/MTP correctness.
   - Add KV-length invariant tests.
   - Add tests proving MTP policy comparison uses the MTP head, not the target model as draft.
   - Add matched-support aggregation tests.

4. Fix scripts/oracle_sensitivity.py.
   - Fix the kv_len update bug after target resync.
   - Save sample_indices and sampled risk labels.
   - Replace zero-filled aggregation with matched-support aggregation.
   - Pass explicit coupled random uniforms/generators.
   - Save metadata: model revision, seed, prompt IDs, gamma, temperature, sample fraction.

5. Refactor true MTP draft/verify logic into a shared helper.
   - Add src/mtp_loop.py or equivalent.
   - Update src/speculative_decode.py, scripts/oracle_sensitivity.py, and scripts/core_comparison.py to use the same tested step logic.
   - Do not change model weights, datasets, or metric definitions.

6. Implement MARA core.
   - Add src/accept_risk.py with:
     * AcceptanceRiskOracle
     * AcceptanceRiskPredictor
     * RiskBudgetAllocator
     * MarginUncertaintyGate
   - Risk labels must be continuous acceptance/logit/margin degradation labels.
   - Predictor must output calibrated risk mean and uncertainty.
   - Allocator must respect a fixed KV budget.
   - Margin/uncertainty gate must be logged.

7. Add MARA scripts and config.
   - Add scripts/calibrate_mara.py.
   - Add configs/mara_minimal.yaml.
   - The calibration script must collect risk samples, fit the predictor, and output calibration metrics.

8. Add A/B/C comparison.
   - Update scripts/core_comparison.py to support:
     A. existing_best_fragment_only
     B. mara_no_gate_or_uncertainty
     C. mara_full
   - All three must use the same prompts, seeds, true MTP loop, metric definitions, and KV budget.

9. Add sanity checks.
   - Data split overlap check.
   - GSM8K/MATH metric known-answer tests.
   - Checkpoint/model revision logging.
   - Position-wise acceptance logging, including position-0-excluded metrics.

10. Run the minimal verification suite.
   - Run unit tests.
   - Run 2–4 prompt smoke tests.
   - Run small MARA calibration.
   - Run A/B/C comparison on a small prompt set.
   - Run 3-seed small comparison only after smoke tests pass.

For every task:
- make the smallest necessary change;
- show the diff;
- run the specified verification command;
- save logs;
- report failures;
- stop if verification fails;
- do not proceed to full benchmark until minimal tests pass.

At the end, output:
- files changed;
- files archived or marked historical;
- configs added;
- commands run;
- logs;
- result table;
- failed checks;
- unresolved issues;
- whether Full New MAIN METHOD beats:
  A. Existing Best Positive Fragment Only,
  B. New MAIN METHOD Without New Mechanism,
  C. Full New MAIN METHOD.

Stop conditions:
- If corrected oracle cannot produce valid matched-support risk labels, stop.
- If MTP policy path still uses target model as draft, stop.
- If Full MARA does not beat both A and B in the minimal equal-budget test, do not expand to full benchmark.
- If official/simple baselines beat MARA consistently, report failure rather than tuning until positive.
```

[1]: https://github.com/Sunshine535/nips-specquant "GitHub - Sunshine535/nips-specquant · GitHub"
[2]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/IDEA_REPORT.md "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/README.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/review-stage/AUTO_REVIEW.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/refine-logs/EXPERIMENT_TRACKER.md "raw.githubusercontent.com"
[6]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/refine-logs-v3/FINAL_PROPOSAL_v3.md "raw.githubusercontent.com"
[7]: https://github.com/Sunshine535/nips-specquant/raw/refs/heads/master/results/benchmark/benchmark_Qwen2.5-7B_14B_20260406_040339.json "raw.githubusercontent.com"
[8]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/results/acceptspec/oracle_m0.json "raw.githubusercontent.com"
[9]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/results/acceptspec/oracle_m1.json "raw.githubusercontent.com"
[10]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/results/acceptspec/oracle_m1_shard0.json "raw.githubusercontent.com"
[11]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/results/acceptspec/oracle_m1_shard1.json "raw.githubusercontent.com"
[12]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/logs/M2_divergence.log "raw.githubusercontent.com"
[13]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/logs/M3_comparison_gsm8k.log "raw.githubusercontent.com"
[14]: https://raw.githubusercontent.com/Sunshine535/nips-specquant/master/logs/M3_comparison_math500.log "raw.githubusercontent.com"
[15]: https://github.com/Sunshine535/nips-specquant/raw/refs/heads/master/results/gsm8k/thinkcompress_gsm8k_50_20260407_012143.json "raw.githubusercontent.com"
[16]: https://arxiv.org/abs/2211.17192 "https://arxiv.org/abs/2211.17192"
[17]: https://github.com/jy-yuan/KIVI/blob/main/README.md "https://github.com/jy-yuan/KIVI/blob/main/README.md"
[18]: https://arxiv.org/html/2502.10424v1 "https://arxiv.org/html/2502.10424v1"
[19]: https://arxiv.org/abs/2401.18079 "https://arxiv.org/abs/2401.18079"
[20]: https://openreview.net/forum?id=RkRrPp7GKO "https://openreview.net/forum?id=RkRrPp7GKO"
[21]: https://openreview.net/forum?id=poE54GOq2l "https://openreview.net/forum?id=poE54GOq2l"
[22]: https://openreview.net/forum?id=EQgEMAD4kv "https://openreview.net/forum?id=EQgEMAD4kv"
[23]: https://github.com/zefan-cai/r-kv "https://github.com/zefan-cai/r-kv"
[24]: https://www.researchgate.net/publication/400603323_SpecAttn_Co-Designing_Sparse_Attention_with_Self-Speculative_Decoding "https://www.researchgate.net/publication/400603323_SpecAttn_Co-Designing_Sparse_Attention_with_Self-Speculative_Decoding"
