#!/bin/bash
# ============================================================================
# SpecQuant: Full experiment pipeline (NeurIPS submission)
# Phase 0: GPU validation (dual-GPU mandatory)
# Phase 1: Main benchmark (dual-GPU, 3 model pairs + baselines)
# Phase 2: Cross-architecture (Llama 3.1)
# Phase 3: Bit-width sweep
# Phase 4: TV distance validation (real empirical TV)
# Phase 5: Verifier microbenchmark
# Phase 6: Layer sensitivity (real activations)
# Phase 7: Ablation studies (5 types)
# Phase 8: Downstream tasks (GSM8K, HumanEval, MMLU, MT-Bench)
# Phase 9: Paper figures
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

RESULTS_DIR="${SPECQUANT_DATA_DIR:-results}"
MARKER_DIR="${RESULTS_DIR}/.phase_markers"
LOG_DIR="${RESULTS_DIR}/../logs"
mkdir -p "$RESULTS_DIR" "$MARKER_DIR" "$LOG_DIR"

QUICK="${QUICK:-0}"
FROM_PHASE="${FROM_PHASE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

phase_done() { [ -f "${MARKER_DIR}/phase_${1}_done" ] && [ "$FORCE_RERUN" = "0" ]; }
mark_done() { touch "${MARKER_DIR}/phase_${1}_done"; }

echo "============================================"
echo " SpecQuant Experiment Pipeline (NeurIPS)"
echo " $(date)"
echo " Results: $RESULTS_DIR"
echo " Quick: $QUICK | From Phase: $FROM_PHASE"
echo "============================================"

# ------------------------------------------------------------------
# Phase 0: GPU validation (dual-GPU mandatory)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 0 ] && ! phase_done 0; then
    echo ""
    echo "=== Phase 0: GPU validation & model check ==="

    GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
    echo "  Detected GPUs: $GPU_COUNT"
    if [ "$GPU_COUNT" -lt 2 ]; then
        echo "  [FATAL] SpecQuant requires at least 2 GPUs (draft→cuda:0, target→cuda:1)."
        echo "  Found $GPU_COUNT GPU(s). Aborting."
        exit 1
    fi

    for i in $(seq 0 $((GPU_COUNT - 1))); do
        python -c "import torch; print(f'  GPU {$i}: {torch.cuda.get_device_name($i)} ({torch.cuda.get_device_properties($i).total_mem / 1e9:.1f} GB)')"
    done

    python -c "
from transformers import AutoTokenizer
models = [
    'Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-4B',
    'Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-14B',
]
for m in models:
    try:
        t = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        print(f'  [ok] {m}')
    except Exception as e:
        print(f'  [WARN] {m}: {e}')
"
    mark_done 0
fi

# ------------------------------------------------------------------
# Phase 1: Main benchmark (dual-GPU, 3 model pairs + baselines)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 1 ] && ! phase_done 1; then
    echo ""
    echo "=== Phase 1: Main benchmark (dual-GPU) ==="

    DRAFT1="${QWEN35_0_8B:-Qwen/Qwen3.5-0.8B}"
    TARGET1="${QWEN35_9B:-Qwen/Qwen3.5-9B}"
    DRAFT2="${QWEN35_4B:-Qwen/Qwen3.5-4B}"
    TARGET2="${QWEN35_14B:-Qwen/Qwen3.5-14B}"
    DRAFT3="${QWEN35_0_8B:-Qwen/Qwen3.5-0.8B}"
    TARGET3="${QWEN35_4B:-Qwen/Qwen3.5-4B}"
    PAIRS=("${DRAFT1}:${TARGET1}" "${DRAFT2}:${TARGET2}" "${DRAFT3}:${TARGET3}")
    MAX_TOKENS=128
    [ "$QUICK" = "1" ] && MAX_TOKENS=32

    for pair in "${PAIRS[@]}"; do
        IFS=':' read -r draft target <<< "$pair"
        tag=$(basename "$target")
        echo "  Running: $draft -> $target"

        python scripts/benchmark_specquant.py \
            --draft-model "$draft" \
            --target-model "$target" \
            --draft-device cuda:0 \
            --target-device cuda:1 \
            --max-new-tokens $MAX_TOKENS \
            --gamma 5 \
            --quant-bits 0 3 4 \
            --baselines rtn kivi absmax \
            --num-trials 5 \
            --num-warmup 2 \
            --prompt-type reasoning code long_context \
            --output-dir "${RESULTS_DIR}/benchmark" \
            2>&1 | tee "${LOG_DIR}/phase1_${tag}.log"
    done
    mark_done 1
fi

# ------------------------------------------------------------------
# Phase 2: Cross-architecture (Llama 3.1)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 2 ] && ! phase_done 2; then
    echo ""
    echo "=== Phase 2: Cross-architecture (Llama 3.1-8B -> 70B) ==="

    LLAMA_DRAFT="${LLAMA31_8B:-meta-llama/Llama-3.1-8B}"
    LLAMA_TARGET="${LLAMA31_70B:-meta-llama/Llama-3.1-70B}"

    # Check if Llama models are accessible before running
    LLAMA_OK=$(python -c "
from transformers import AutoTokenizer
try:
    AutoTokenizer.from_pretrained('${LLAMA_DRAFT}', trust_remote_code=True)
    print('ok')
except:
    print('skip')
" 2>/dev/null || echo "skip")

    if [ "$LLAMA_OK" = "ok" ]; then
        python scripts/benchmark_specquant.py \
            --draft-model "$LLAMA_DRAFT" \
            --target-model "$LLAMA_TARGET" \
            --draft-device cuda:0 \
            --target-device cuda:1 \
            --max-new-tokens 128 \
            --gamma 5 \
            --quant-bits 0 3 4 \
            --baselines rtn kivi absmax \
            --num-trials 5 \
            --num-warmup 2 \
            --prompt-type reasoning code long_context \
            --output-dir "${RESULTS_DIR}/cross_arch" \
            2>&1 | tee "${LOG_DIR}/phase2_cross_arch.log"
    else
        echo "  [SKIP] Llama-3.1 models not available, skipping cross-architecture phase."
    fi
    mark_done 2
fi

# ------------------------------------------------------------------
# Phase 3: Bit-width sweep
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 3 ] && ! phase_done 3; then
    echo ""
    echo "=== Phase 3: Bit-width sweep ==="

    python scripts/benchmark_specquant.py \
        --draft-model "${QWEN35_4B:-Qwen/Qwen3.5-4B}" \
        --target-model "${QWEN35_14B:-Qwen/Qwen3.5-14B}" \
        --draft-device cuda:0 \
        --target-device cuda:1 \
        --max-new-tokens 128 \
        --gamma 5 \
        --quant-bits 0 2 3 4 \
        --num-trials 5 \
        --num-warmup 2 \
        --prompt-type all \
        --output-dir "${RESULTS_DIR}/bitwidth_sweep" \
        2>&1 | tee "${LOG_DIR}/phase3_bitwidth.log"
    mark_done 3
fi

# ------------------------------------------------------------------
# Phase 4: TV distance validation (real empirical TV)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 4 ] && ! phase_done 4; then
    echo ""
    echo "=== Phase 4: TV distance validation (empirical) ==="

    python scripts/eval_tv_distance.py \
        --draft-model "${QWEN35_4B:-Qwen/Qwen3.5-4B}" \
        --target-model "${QWEN35_14B:-Qwen/Qwen3.5-14B}" \
        --draft-device cuda:0 \
        --target-device cuda:1 \
        --bits 2 3 4 \
        --num-samples 100 \
        --output-dir "${RESULTS_DIR}/tv_validation" \
        2>&1 | tee "${LOG_DIR}/phase4_tv.log"
    mark_done 4
fi

# ------------------------------------------------------------------
# Phase 5: Verifier microbenchmark
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 5 ] && ! phase_done 5; then
    echo ""
    echo "=== Phase 5: Verifier microbenchmark ==="

    python scripts/microbenchmark_verifier.py \
        --target-model "${QWEN35_14B:-Qwen/Qwen3.5-14B}" \
        --draft-device cuda:0 \
        --target-device cuda:1 \
        --seq-lengths 1024 2048 4096 8192 16384 \
        --bits 0 3 4 \
        --num-trials 10 \
        --output-dir "${RESULTS_DIR}/microbenchmark" \
        2>&1 | tee "${LOG_DIR}/phase5_micro.log"
    mark_done 5
fi

# ------------------------------------------------------------------
# Phase 6: Layer sensitivity (real activations)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 6 ] && ! phase_done 6; then
    echo ""
    echo "=== Phase 6: Layer sensitivity analysis ==="

    python scripts/analyze_layer_sensitivity.py \
        --target-model "${QWEN35_14B:-Qwen/Qwen3.5-14B}" \
        --draft-device cuda:0 \
        --target-device cuda:1 \
        --bits 3 \
        --output-dir "${RESULTS_DIR}/robustness" \
        2>&1 | tee "${LOG_DIR}/phase6_layer_sens.log"
    mark_done 6
fi

# ------------------------------------------------------------------
# Phase 7: Ablation studies (5 types)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 7 ] && ! phase_done 7; then
    echo ""
    echo "=== Phase 7: Ablation studies ==="

    python scripts/run_ablations.py \
        --draft-model "${QWEN35_4B:-Qwen/Qwen3.5-4B}" \
        --target-model "${QWEN35_14B:-Qwen/Qwen3.5-14B}" \
        --draft-device cuda:0 \
        --target-device cuda:1 \
        --output-dir "${RESULTS_DIR}/ablations" \
        2>&1 | tee "${LOG_DIR}/phase7_ablations.log"
    mark_done 7
fi

# ------------------------------------------------------------------
# Phase 8: Downstream tasks (GSM8K, HumanEval, MMLU, MT-Bench)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 8 ] && ! phase_done 8; then
    echo ""
    echo "=== Phase 8: Downstream evaluation ==="

    python scripts/eval_downstream.py \
        --draft-model "${QWEN35_4B:-Qwen/Qwen3.5-4B}" \
        --target-model "${QWEN35_14B:-Qwen/Qwen3.5-14B}" \
        --draft-device cuda:0 \
        --target-device cuda:1 \
        --benchmarks gsm8k humaneval mmlu mt_bench \
        --num-samples 200 \
        --max-new-tokens 512 \
        --num-trials 3 \
        --output-dir "${RESULTS_DIR}/downstream" \
        2>&1 | tee "${LOG_DIR}/phase8_downstream.log"
    mark_done 8
fi

# ------------------------------------------------------------------
# Phase 9: Paper figures
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 9 ] && ! phase_done 9; then
    echo ""
    echo "=== Phase 9: Paper figures ==="

    python scripts/generate_figures.py \
        --results-dir "${RESULTS_DIR}" \
        --output-dir "${RESULTS_DIR}/figures" \
        2>&1 | tee "${LOG_DIR}/phase9_figures.log"
    mark_done 9
fi

echo ""
echo "============================================"
echo " Pipeline complete: $(date)"
echo " Results: $RESULTS_DIR"
echo "============================================"
touch "${RESULTS_DIR}/.pipeline_done"
