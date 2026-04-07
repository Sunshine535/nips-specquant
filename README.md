# SpecQuant — 投机解码与 KV Cache 量化

## 项目简介

将 KV cache 量化与 speculative decoding 的 verifier 深度结合。通过 monkey-patch attention forward（替代 post-hook），确保 native SDPA 不加载 FP prefix KV，实现真正的低精度 KV cache 验证。包含 ThinkCompress 模块，用于 thinking token 的高效压缩。

**Review 状态**: Round 3, Score 6.0/10, 42 tests passing

## 环境安装

```bash
cd /workspace/nips-specquant
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate datasets numpy scipy scikit-learn \
    tqdm pyyaml wandb matplotlib seaborn
```

## 快速开始

```bash
source .venv/bin/activate

# Microbenchmark verifier
python3 scripts/microbenchmark_verifier.py

# TV distance 评估
python3 scripts/eval_tv_distance.py --config configs/default.yaml
```

## 完整实验流程（10 阶段）

```bash
# 一键全流程（需要 2×GPU）
bash run.sh

# 分步：
# 1. Benchmark SpecQuant
python3 scripts/benchmark_specquant.py --config configs/default.yaml

# 2. 下游任务评估 (GSM8K/HumanEval/MMLU/MT-Bench)
python3 scripts/eval_downstream.py --config configs/default.yaml

# 3. Layer sensitivity 分析
python3 scripts/analyze_layer_sensitivity.py --config configs/default.yaml

# 4. 消融实验 (5 种)
python3 scripts/run_ablations.py --config configs/default.yaml

# 5. 生成论文图表
python3 scripts/generate_figures.py --results_dir results/
```

### 多卡配置
- Pipeline enforces dual-GPU
- 配置文件 `configs/default.yaml` 中 `model_pairs` 定义 draft/target 模型对

## 断点续训

- 结果按实验名存储在 `results/` 下
- 重跑会检查已有结果文件并跳过

## 已有结果

- **ThinkCompress**: 0% accuracy loss at up to 3.9x compression
- FP prefix KV evicted after compression
- 42 个单元测试全部通过
- Baselines: RTN, KIVI, Absmax

## 项目结构

```
src/
  baselines.py              # RTN/KIVI/Absmax 基线
  linear_attn_quantizer.py  # 线性注意力量化
  quantized_verifier.py     # 量化验证器
  speculative_decode.py     # 投机解码
  thinkcompress.py          # ThinkCompress
  turboquant_kv.py          # TurboQuant KV
  utils.py                  # 工具函数
scripts/
  benchmark_specquant.py    # Benchmark
  eval_downstream.py        # 下游任务
  eval_tv_distance.py       # TV distance
  analyze_layer_sensitivity.py  # Layer 分析
  run_ablations.py          # 消融实验
  generate_figures.py       # 图表生成
  microbenchmark_verifier.py    # Verifier benchmark
configs/
  default.yaml              # 全部配置
results/                    # 实验结果
tests/                      # 42 个测试
```

## 下一步

1. 完成全部 10 阶段 pipeline
2. 补充更多 model pairs
3. 论文撰写
