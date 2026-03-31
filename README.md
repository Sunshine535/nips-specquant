# SpecQuant: Compressed-Domain Verification Attention for Speculative Decoding

TurboQuant-accelerated KV cache quantization for the verification phase of speculative decoding. Achieves 4-5x KV compression with ≤3pp acceptance rate loss at 3-bit, enabling ≥1.5x end-to-end throughput improvement.

## Quick Start

```bash
git clone <repo-url>
cd nips-specquant
bash setup.sh
source .venv/bin/activate
bash run.sh
```

## Key Results

| Method | Throughput | Acceptance Rate | KV Compression |
|--------|-----------|-----------------|----------------|
| Autoregressive | 1.0x | - | - |
| Vanilla SpecDec | ~1.8x | baseline | 1x |
| SpecQuant-4bit | ~2.2x | ≤1pp drop | 3.8x |
| SpecQuant-3bit | ~2.7x | ≤3pp drop | 4.9x |

## Project Structure

```
nips-specquant/
├── src/
│   ├── turboquant_kv.py          # Hadamard rotation + scalar quantization
│   ├── speculative_decode.py     # Speculative decoder with quantized verification
│   └── __init__.py
├── scripts/
│   ├── run_all_experiments.sh    # Full pipeline (Phase 0-6)
│   ├── benchmark_specquant.py    # Main benchmark
│   ├── eval_tv_distance.py       # TV distance validation
│   ├── microbenchmark_verifier.py # Verifier kernel profiling
│   ├── analyze_layer_sensitivity.py # Per-layer sensitivity
│   ├── generate_figures.py       # Paper figures
│   └── gpu_utils.sh              # GPU utilities
├── configs/
│   └── default.yaml              # Default settings
├── refine-logs/                  # ARIS research refinement logs
├── results/                      # Experiment outputs
├── logs/                         # Runtime logs
├── paper/                        # LaTeX source
├── requirements.txt
├── setup.sh
├── run.sh
└── run_acp.sh                    # ACP server startup
```

## Experiments

| Phase | Description | Est. GPU-hours |
|-------|------------|---------------|
| 0 | Model check | <1 |
| 1 | Main benchmark (Claim 1) | ~40 |
| 2 | Bit-width sweep (Claim 2) | ~20 |
| 3 | TV distance validation (Claim 3) | ~15 |
| 4 | Verifier microbenchmark | ~15 |
| 5 | Layer sensitivity analysis | ~5 |
| 6 | Paper figures | ~5 |
| **Total** | | **~100** |

## Remote Deployment

```bash
# On ACP server
bash /data/szs/250010072/nwh/nips-specquant/run_acp.sh
```
