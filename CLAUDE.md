# Project: nips-specquant

## Project goal

**AcceptSpec: Acceptance-Preserving KV Cache Management for Speculative Decoding of Reasoning Models** — 发现投机解码中验证器的接受率对 KV cache 中的 token 具有稀疏敏感性（acceptance-critical tokens），且这种敏感性与 attention 重要性和 perplexity 敏感性显著不同。利用这一发现实现面向接受率（而非 perplexity/attention）优化的 KV 压缩，在推理模型长链思考场景下实现超越独立 SD + KV 压缩的联合增益。

## Key models

**Primary (Qwen3.5 hybrid, GatedDeltaNet + MHA, AcceptSpec targets MHA layers):**
- Qwen/Qwen3.5-0.8B (draft) + Qwen/Qwen3.5-9B (target)

**Scale-up:**
- Qwen/Qwen3.5-0.8B (draft) + Qwen/Qwen3.5-27B (target)

**Cross-architecture (standard MHA):**
- meta-llama/Llama-3.2-3B (draft) + meta-llama/Llama-3.1-8B (target)

## Key datasets

- GSM8K — 数学推理评测 (primary)
- MATH-500 — 数学推理评测 (secondary)

## Baselines

- SmallKV (NeurIPS'25 Spotlight) — SLM attention-proxy KV compression (KEY COMPARISON)
- R-KV (NeurIPS'25, GitHub: Zefan-Cai/R-KV) — redundancy-aware KV compression
- QuantSpec — self-speculative 4-bit KV+weights

## Repo map

- `src/` — 核心模块
  - `acceptspec.py` — **AcceptSpec 核心**: AcceptSensitivityOracle, AcceptPredictor, MixedPrecisionKV
  - `speculative_decode.py` — 投机解码引擎
  - `turboquant_kv.py` — TurboQuant KV cache 量化原语（Hadamard rotation + scalar quant）
  - `quantized_verifier.py` — 量化验证器（monkey-patched attention forward）
  - `baselines.py` — RTN/KIVI/Absmax 量化 baseline
  - `gpu_auto.py` — GPU 自动检测与模型分配
  - `linear_attn_quantizer.py` — Qwen3.5 线性注意力量化
  - `thinkcompress.py` — ThinkCompress（旧方向，ImportanceScorer/AdaptiveBitAllocator 可复用）
  - `utils.py` — 工具函数
- `scripts/` — 实验脚本 (对齐 EXPERIMENT_PLAN.md)
  - `oracle_sensitivity.py` — B1: Oracle 接受率敏感性研究（M0/M1 gate）
  - `triple_divergence.py` — B2: 三重散度 + predictor 验证
  - `core_comparison.py` — B3: 8 种保留策略对比
  - `e2e_benchmark.py` — B4: 9 系统端到端 benchmark
  - `run_all_experiments.sh` — M0-M5 全流程编排
- `configs/` — 配置文件 (default.yaml)
- `results/` — 实验输出
- `refine-logs/` — 方法精炼日志
  - `FINAL_PROPOSAL.md` — AcceptSpec v2.0 proposal
  - `EXPERIMENT_PLAN.md` — 实验计划 v2.0 (20 runs, ~150 GPU-hours)
  - `EXPERIMENT_TRACKER.md` — 实验跟踪
- `IDEA_REPORT.md` — Idea discovery 报告 v2.0
- `LITERATURE_LANDSCAPE.md` — 文献综述 (90+ papers)

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

# Full pipeline
bash run.sh

# Quick test (fewer problems)
QUICK=1 bash run.sh

# Resume from specific milestone
FROM_MILESTONE=2 bash run.sh

# Force re-run
FORCE_RERUN=1 bash run.sh

# Background
nohup bash run.sh > run.log 2>&1 &
```

## Environment

- Python 3.10, PyTorch 2.10 (CUDA 12.8)
- 关键依赖: transformers, accelerate, datasets, torch, numpy, scipy
- 可选: flash-attn, wandb
- 环境变量: `CUDA_DEVICE_ORDER=PCI_BUS_ID`

## GPU Auto-Detection

- `src/gpu_auto.py` 自动检测 GPU 数量和显存，智能分配 draft/target 模型
- 0 GPU → CPU-only (测试用)
- 1 GPU → 两个模型共用 cuda:0
- 2 GPU → draft 放小卡, target 放大卡
- 4+ GPU → target 用 device_map="auto" 跨卡, draft 占一张
