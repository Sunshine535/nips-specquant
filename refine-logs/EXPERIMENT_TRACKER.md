# Experiment Tracker: AcceptSpec v2.0

| Run ID | Milestone | Block | Purpose | System | Dataset | Metrics | Priority | Status | Notes |
|--------|-----------|-------|---------|--------|---------|---------|----------|--------|-------|
| R001 | M0 | B1 | Sanity: oracle sensitivity on 10 problems | Oracle | GSM8K (10) | gini, top20_coverage | MUST | TODO | Quick gate |
| R002 | M1 | B1 | Full oracle: 100 problems | Oracle | GSM8K (100) | cumulative_curve, gini, per_layer_heatmap | MUST | TODO | ABORT if top20 < 60% |
| R003 | M2 | B2 | Triple divergence: accept vs perplexity vs attention ranking | Oracle+Perplexity+Attention | GSM8K (100) | 3x spearman_rho | MUST | TODO | SmallKV differentiator |
| R004 | M2 | B2 | Predictor training: 50 calibration examples | AcceptPredictor | GSM8K (50 train) | — | MUST | TODO | Logistic regression |
| R005 | M2 | B2 | Predictor test + SmallKV comparison: 50 test examples | AcceptPredictor vs AttentionProxy | GSM8K (50 test) | F1, precision, recall | MUST | TODO | Must F1 > 0.75 |
| R006 | M3 | B3 | Core comparison: GSM8K full, 20% budget, 8 retention policies | All policies | GSM8K (1319) | accuracy, acceptance_rate | MUST | TODO | Main paper result |
| R007 | M3 | B3 | Core comparison: MATH-500, 20% budget | All policies | MATH-500 | accuracy | MUST | TODO | Second dataset |
| R008 | M3 | B3 | Anti-claim: AcceptSpec w/ vs w/o SD | AcceptSpec variants | GSM8K (1319) | accuracy gap | MUST | TODO | SD-specific benefit |
| R009 | M3 | B3 | Budget sweep: 10%/20%/30%/50% KV retained | AcceptSpec vs perplexity vs attention | GSM8K (500) | Pareto curve | MUST | TODO | Figure 4 |
| R010 | M4 | B4 | E2E benchmark: GSM8K, 9 systems | All baselines | GSM8K (1319) | tok/s, latency, memory, accuracy | MUST | TODO | Table 3 |
| R011 | M4 | B4 | E2E benchmark: MATH-500 | All baselines | MATH-500 | tok/s, latency, memory, accuracy | MUST | TODO | Table 3 |
| R012 | M4 | B4 | Profiling: per-component latency breakdown | AcceptSpec | GSM8K (50) | draft/score/compress/verify times | MUST | TODO | Figure 5 |
| R013 | M5 | B5 | Robustness: temperature sweep | AcceptSpec | GSM8K (200) | sparsity, accuracy per τ | SHOULD | TODO | Table 4 |
| R014 | M5 | B5 | Robustness: draft length sweep | AcceptSpec | GSM8K (200) | sparsity, accuracy per γ | SHOULD | TODO | Table 4 |
| R015 | M5 | B5 | Robustness: difficulty strata | AcceptSpec | GSM8K (200) | sparsity, accuracy per stratum | SHOULD | TODO | Figure 7 |
| R016 | M5 | B5 | Robustness: KV budget sweep | AcceptSpec | GSM8K (200) | accuracy per budget | SHOULD | TODO | Table 4 |
| R017 | M5 | B6 | Universality: Llama-3.1-8B oracle | Oracle | GSM8K (50) | gini, top20_coverage | SHOULD | TODO | Cross-model |
| R018 | M5 | B6 | Universality: Llama-3.1-8B E2E | All baselines | GSM8K (500) | tok/s, accuracy | SHOULD | TODO | Table 5 |
| R019 | M5 | B6 | Universality: Qwen3.5-9B oracle (MHA layers only) | Oracle | GSM8K (50) | gini, top20_coverage | NICE | TODO | Hybrid arch |
| R020 | M5 | B6 | Universality: Qwen3.5-9B E2E | All baselines | GSM8K (500) | tok/s, accuracy | NICE | TODO | Hybrid arch |
