# Experiment Tracker: ThinkCompress

| Run ID | Milestone | Purpose | System / Variant | Dataset | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|---------|---------|----------|--------|-------|
| R001 | M0 | Sanity: importance scoring on 1 prompt | ThinkCompress | GSM8K (1) | attention_gini, importance_dist | MUST | TODO | Quick validation |
| R002 | M0 | Sanity: end-to-end pipeline Qwen3-8B | ThinkCompress 4x | GSM8K (10) | accuracy, memory | MUST | TODO | |
| R003 | M0 | Sanity: verify uniform quant baseline | Uniform-4bit | GSM8K (10) | accuracy | MUST | TODO | |
| R004 | M1 | Characterization: Qwen3-8B token ratios | FP16 analysis | GSM8K+MATH+AIME (130) | think_ratio, attn_entropy, gini | MUST | TODO | Block 1 |
| R005 | M1 | Characterization: Qwen3-14B | FP16 analysis | GSM8K+MATH (100) | think_ratio, attn_entropy, gini | MUST | TODO | Block 1 |
| R006 | M1 | Characterization: per-layer attention patterns | FP16 hooks | GSM8K (20) | per_layer_importance | NICE | TODO | Appendix |
| R007 | M2 | Baseline: FP16 on GSM8K full | FP16 | GSM8K (1319) | accuracy | MUST | TODO | |
| R008 | M2 | Baseline: FP16 on MATH-500 | FP16 | MATH-500 | accuracy | MUST | TODO | |
| R009 | M2 | Baseline: Uniform quant 4x/8x | Uniform 4bit/2bit | GSM8K (1319) | accuracy | MUST | TODO | |
| R010 | M2 | Baseline: H2O eviction | H2O | GSM8K (1319) | accuracy, memory | MUST | TODO | |
| R011 | M2 | Baseline: StreamingLLM | StreamingLLM | GSM8K (1319) | accuracy, memory | MUST | TODO | |
| R012 | M2 | Baseline: FP16 on HumanEval/GPQA | FP16 | HumanEval+GPQA | accuracy | MUST | TODO | |
| R013 | M3 | Main: ThinkCompress 2x on GSM8K | TC-2x | GSM8K (1319) | accuracy, memory | MUST | TODO | |
| R014 | M3 | Main: ThinkCompress 4x on GSM8K | TC-4x | GSM8K (1319) | accuracy, memory | MUST | TODO | Key result |
| R015 | M3 | Main: ThinkCompress 8x on GSM8K | TC-8x | GSM8K (1319) | accuracy, memory | MUST | TODO | |
| R016 | M3 | Main: ThinkCompress on MATH-500 | TC-2x/4x/8x | MATH-500 | accuracy, memory | MUST | TODO | |
| R017 | M3 | Main: ThinkCompress on HumanEval | TC-4x | HumanEval (164) | pass@1 | MUST | TODO | |
| R018 | M3 | Main: ThinkCompress on GPQA | TC-4x | GPQA-Diamond | accuracy | MUST | TODO | |
| R019 | M3 | Main: Qwen3-14B generalization | TC-4x | GSM8K+MATH | accuracy | MUST | TODO | |
| R020 | M3 | Main: Pareto curve sweep | TC-1.5x to 10x | GSM8K | accuracy vs compression | MUST | TODO | Figure 3 |
| R021 | M4 | Ablation: adaptive vs uniform at 4x | TC vs Uniform | GSM8K+MATH | accuracy gap | MUST | TODO | Block 3 |
| R022 | M4 | Ablation: adaptive vs uniform at 8x | TC vs Uniform | GSM8K+MATH | accuracy gap | MUST | TODO | Block 3 |
| R023 | M4 | Ablation: random bit assignment | Random-4x | GSM8K | accuracy | MUST | TODO | Block 3 |
| R024 | M4 | Ablation: oracle upper bound | Oracle-4x | GSM8K (100) | accuracy | MUST | TODO | Block 3 |
| R025 | M4 | Ablation: eviction-only at 4x | Evict-4x | GSM8K+MATH | accuracy | MUST | TODO | Block 4 |
| R026 | M4 | Ablation: quant-only at 4x | Quant-4x | GSM8K+MATH | accuracy | MUST | TODO | Block 4 |
| R027 | M4 | Ablation: combined at 4x/8x | TC-4x/8x | GSM8K+MATH | accuracy | MUST | TODO | Block 4 |
| R028 | M4 | Ablation: importance scoring method | EMA vs cumulative vs sink | GSM8K | accuracy | NICE | TODO | |
| R029 | M5 | Scale: Qwen3-14B TP=2 memory | TC-4x | AIME | peak_memory, max_think_len | MUST | TODO | Block 5 |
| R030 | M5 | Scale: Qwen3-32B TP=2 feasibility | TC-4x | AIME (5) | peak_memory, OOM_threshold | NICE | TODO | Block 5 |
| R031 | M5 | Polish: attention heatmap figure | FP16 | GSM8K (3) | visualization | MUST | TODO | Figure 1 |
| R032 | M5 | Polish: importance distribution figure | FP16 | GSM8K (50) | visualization | MUST | TODO | Figure 2 |
| R033 | M5 | Polish: bit allocation visualization | TC-4x | GSM8K (3) | visualization | MUST | TODO | Figure 4 |
| R034 | M5 | Polish: memory vs thinking length | TC-4x/8x | Synthetic | memory_curve | MUST | TODO | Figure 5 |
| R035 | M5 | Extended: DeepSeek-R1-distill if avail | TC-4x | GSM8K | accuracy | NICE | TODO | Cross-model |
