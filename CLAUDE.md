# Project: nips-specquant

## Project goal

**AcceptSpec: Acceptance-Preserving KV Cache Management for Speculative Decoding of Reasoning Models** — 发现投机解码中验证器的接受率对 KV cache 中的 token 具有稀疏敏感性（acceptance-critical tokens），利用这一发现实现面向接受率而非 perplexity 优化的 KV 压缩，在推理模型长链思考场景下实现超越独立 SD + KV 压缩的联合增益。

## Key models

- Qwen/Qwen3.5-{0.8B, 4B, 9B, 14B} — draft/target LLM 组合
- Meta-Llama/Llama-3.1-{8B, 70B} — 跨架构验证（可选）

## Key datasets

- GSM8K — 数学推理评测
- HumanEval — 代码生成评测
- MT-Bench — 多轮对话评测
- MMLU — 通用能力评测

## Repo map

- `src/` — 核心模块
  - `acceptspec.py` — **AcceptSpec 核心**: AcceptSensitivityOracle, AcceptPredictor, MixedPrecisionKV
  - `speculative_decode.py` — 投机解码引擎（基于 nips-specscale）
  - `turboquant_kv.py` — TurboQuant KV cache 量化原语（Hadamard rotation + scalar quant）
  - `quantized_verifier.py` — 量化验证器（monkey-patched attention forward）
  - `thinkcompress.py` — ThinkCompress（旧方向，ImportanceScorer/AdaptiveBitAllocator 可复用）
  - `utils.py` — 工具函数
- `scripts/` — 实验脚本
  - `oracle_sensitivity.py` — **Oracle 接受率敏感性研究**（M0/M1 gate）
  - `benchmark_specquant.py` — SpecQuant benchmark
  - `run_all_experiments.sh` — 全阶段编排
- `configs/` — 配置文件
- `results/` — 实验输出
- `refine-logs/` — 方法精炼日志
  - `FINAL_PROPOSAL.md` — AcceptSpec 最终 proposal (8.1/10 READY)
  - `EXPERIMENT_PLAN.md` — 实验计划 (20 runs, 147 GPU-hours)
  - `EXPERIMENT_TRACKER.md` — 实验跟踪
- `IDEA_REPORT.md` — Idea discovery 报告
- `LITERATURE_LANDSCAPE.md` — 文献综述

## Common commands

```bash
bash setup.sh
source .venv/bin/activate

bash run.sh
nohup bash run.sh > run.log 2>&1 &
FORCE_RERUN=1 bash run.sh
```

## Environment

- Python 3.10, PyTorch 2.10 (CUDA 12.8)
- 关键依赖: transformers, accelerate, datasets, torch, numpy
- 可选: flash-attn, wandb
- 环境变量: `CUDA_DEVICE_ORDER=PCI_BUS_ID`

## GPU Auto-Detection

- `src/gpu_auto.py` 自动检测 GPU 数量和显存，智能分配 draft/target 模型
- 0 GPU → CPU-only (测试用)
- 1 GPU → 两个模型共用 cuda:0
- 2 GPU → draft 放小卡, target 放大卡
- 4+ GPU → target 用 device_map="auto" 跨卡, draft 占一张
