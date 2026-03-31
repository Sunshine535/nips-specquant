# Project: nips-specquant

## Project goal

SpecQuant: TurboQuant-Accelerated Verification for Speculative Decoding — 将 TurboQuant 的近最优 KV cache 量化应用于投机解码验证阶段，减少验证时的显存带宽瓶颈，在保持接受率的同时显著加速端到端推理。

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
  - `speculative_decode.py` — 投机解码引擎（基于 nips-specscale 改进）
  - `turboquant_kv.py` — TurboQuant KV cache 量化实现
  - `quantized_verifier.py` — 量化验证器
  - `utils.py` — 工具函数
- `scripts/` — 实验脚本
  - `run_all_experiments.sh` — 全阶段编排
  - `benchmark_specquant.py` — SpecQuant benchmark
  - `eval_acceptance_rate.py` — 接受率评估
  - `run_ablations.py` — 消融实验
  - `gpu_utils.sh` — GPU 分配工具
- `configs/` — 配置文件
- `results/` — 实验输出
- `logs/` — 训练日志
- `paper/` — 论文素材

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

## Remote server

- SSH 存储: `ssh szs_cpu` (118.145.32.132:10072, key-based auth)
- SSH GPU: `ssh szs_gpu1` (118.145.32.133:11072)
- 项目目录: `/data/szs/250010072/nwh/nips-specquant`
- 数据目录: `/data/szs/share/specquant/`
- 共享目录: `/data/szs/share/`
- Conda: `source /data/szs/250010072/szs/anaconda3/bin/activate`
- ACP 启动: `bash /data/szs/250010072/nwh/nips-specquant/run_acp.sh`
