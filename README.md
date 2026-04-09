# AcceptSpec — Acceptance-Preserving KV Cache Management for Speculative Decoding

NeurIPS 2026 submission. Core finding: speculative verification has **sparse KV sensitivity** — a small fraction of tokens disproportionately influence acceptance. Optimizing KV compression for acceptance rate (not perplexity) yields super-additive gains over independent SD + KV compression.

**Review status**: 8.1/10 (GPT-5.4 nightmare, 3 rounds). READY for experiments.

## Quick Start (Resume from Here)

All idea discovery, literature survey, method refinement, and core implementation are complete. The next step is running experiments.

### 1. Setup

```bash
cd /workspace/nips-specquant
bash setup.sh              # venv + PyTorch (CUDA 12.8) + all deps
source .venv/bin/activate
```

GPU auto-detection handles any configuration:
- 0 GPU → CPU (testing only)
- 1 GPU → both models on cuda:0
- 2 GPU → draft on smaller GPU, target on larger GPU
- 4+ GPU → target uses device_map="auto" for tensor parallelism

### 2. Run M0 Gate (Oracle Sanity Check)

First experiment — validates the core hypothesis on 10 GSM8K problems:

```bash
python3 scripts/oracle_sensitivity.py \
    --draft_model Qwen/Qwen3-0.6B \
    --target_model Qwen/Qwen3-8B \
    --num_problems 10 \
    --output results/oracle_m0.json
```

**Decision gate**: Gini > 0.5 → continue. Else abort.

### 3. Run M1 Gate (Full Oracle Study)

If M0 passes, scale to 100 problems:

```bash
python3 scripts/oracle_sensitivity.py \
    --draft_model Qwen/Qwen3-0.6B \
    --target_model Qwen/Qwen3-8B \
    --num_problems 100 \
    --output results/oracle_m1.json
```

**Decision gate**: Top-20% tokens capture >80% sensitivity → continue. Else abort.

### 4. Run M2 (Acceptance vs Perplexity Divergence)

Core conceptual claim — acceptance-ranked ≠ perplexity-ranked:

```bash
python3 scripts/oracle_sensitivity.py \
    --draft_model Qwen/Qwen3-0.6B \
    --target_model Qwen/Qwen3-8B \
    --num_problems 100 \
    --output results/oracle_m2_divergence.json
```

The output includes Spearman ρ between acceptance sensitivity and attention importance. Need ρ < 0.7.

### 5. Full Experiment Pipeline

See `refine-logs/EXPERIMENT_PLAN.md` for the complete 20-run plan (147 GPU-hours). Run order:

| Milestone | Runs | GPU-hours | Gate |
|-----------|------|-----------|------|
| M0: Oracle sanity | R001 | 2 | Gini > 0.5 |
| M1: Full oracle | R002 | 10 | top-20% > 80% |
| M2: Divergence | R003-R005 | 15 | ρ < 0.7 |
| M3: Core comparison | R006-R009 | 40 | ≥3pp gap |
| M4: System benchmark | R010-R013 | 35 | ≥10% speedup |
| M5: Robustness + generalization | R014-R020 | 45 | Consistent |

## Project Structure

```
src/
  acceptspec.py              # AcceptSpec core: Oracle, Predictor, MixedPrecisionKV
  gpu_auto.py                # Auto-adaptive GPU detection and model placement
  speculative_decode.py      # Speculative decoding engine
  turboquant_kv.py           # TurboQuant (Hadamard rotation + scalar quant)
  quantized_verifier.py      # Monkey-patched attention forward
  baselines.py               # RTN/KIVI/Absmax baselines
  thinkcompress.py           # ThinkCompress (legacy, ImportanceScorer reusable)
  utils.py                   # Stats, KV cache compat, I/O
scripts/
  oracle_sensitivity.py      # Oracle acceptance sensitivity study (M0/M1/M2)
  benchmark_specquant.py     # SpecQuant benchmark
  run_all_experiments.sh     # Full pipeline
configs/
  default.yaml               # Model pairs, quant settings, hardware
results/                     # Experiment outputs (JSON)
refine-logs/
  FINAL_PROPOSAL.md          # AcceptSpec proposal (8.1/10 READY)
  EXPERIMENT_PLAN.md         # 6 blocks, 20 runs, 147 GPU-hours
  EXPERIMENT_TRACKER.md      # R001-R020 status tracker
  PIPELINE_SUMMARY.md        # Pipeline summary
  score-history.md           # Review score evolution
IDEA_REPORT.md               # Idea discovery report (10 ideas ranked)
LITERATURE_LANDSCAPE.md      # 80+ paper survey
tests/                       # Unit tests
```

## Key Claims to Prove

| Claim | Evidence | Status |
|-------|----------|--------|
| C1: Top-20% tokens → >80% sensitivity | Oracle masking sweep | TODO (R001-R002) |
| C2: Accept-ranked ≠ perplexity-ranked | Spearman ρ < 0.7 | TODO (R003) |
| C3: Accept-targeted > perplexity-targeted | ≥3pp accuracy gap | TODO (R006-R007) |
| C4: Beats naive EAGLE-3+ThinKV | ≥10% latency win | TODO (R010-R012) |
| C5: Predictor F1 > 0.75 vs oracle | Precision/recall | TODO (R004-R005) |

## GPU Auto-Detection

```python
from src.gpu_auto import plan_devices, load_models, print_gpu_summary

print_gpu_summary()  # shows detected GPUs and strategy

draft, target, tokenizer, plan = load_models(
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-8B",
)
# Models are placed optimally based on available hardware
```
