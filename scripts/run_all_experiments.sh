#!/bin/bash
# ============================================================================
# SpecQuant: Full experiment pipeline
# Phase 0: Model download
# Phase 1: Main benchmark (Claim 1)
# Phase 2: Bit-width sweep (Claim 2)
# Phase 3: TV distance validation (Claim 3)
# Phase 4: Verifier microbenchmark
# Phase 5: Robustness analysis
# Phase 6: Paper figures
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
echo " SpecQuant Experiment Pipeline"
echo " $(date)"
echo " Results: $RESULTS_DIR"
echo " Quick: $QUICK | From Phase: $FROM_PHASE"
echo "============================================"

# ------------------------------------------------------------------
# Phase 0: Model availability check
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 0 ] && ! phase_done 0; then
    echo ""
    echo "=== Phase 0: Model check ==="
    python -c "
from transformers import AutoTokenizer
for m in ['Qwen/Qwen3.5-0.8B', 'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-14B']:
    try:
        t = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
        print(f'  [ok] {m}')
    except Exception as e:
        print(f'  [WARN] {m}: {e}')
"
    mark_done 0
fi

# ------------------------------------------------------------------
# Phase 1: Main benchmark (Claim 1)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 1 ] && ! phase_done 1; then
    echo ""
    echo "=== Phase 1: Main benchmark ==="

    DRAFT1="${QWEN35_0_8B:-Qwen/Qwen3.5-0.8B}"
    TARGET1="${QWEN35_9B:-Qwen/Qwen3.5-9B}"
    DRAFT2="${QWEN35_4B:-Qwen/Qwen3.5-4B}"
    TARGET2="${QWEN35_27B:-Qwen/Qwen3.5-27B}"
    PAIRS=("${DRAFT1}:${TARGET1}" "${DRAFT2}:${TARGET2}")
    MAX_TOKENS=128
    [ "$QUICK" = "1" ] && MAX_TOKENS=32

    for pair in "${PAIRS[@]}"; do
        IFS=':' read -r draft target <<< "$pair"
        tag=$(basename "$target")
        echo "  Running: $draft -> $target"

        python scripts/benchmark_specquant.py \
            --draft-model "$draft" \
            --target-model "$target" \
            --max-new-tokens $MAX_TOKENS \
            --gamma 5 \
            --quant-bits 0 3 4 \
            --num-trials 3 \
            --prompt-type reasoning \
            --output-dir "${RESULTS_DIR}/benchmark" \
            2>&1 | tee "${LOG_DIR}/phase1_${tag}.log"
    done
    mark_done 1
fi

# ------------------------------------------------------------------
# Phase 2: Bit-width sweep (Claim 2)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 2 ] && ! phase_done 2; then
    echo ""
    echo "=== Phase 2: Bit-width sweep ==="

    python scripts/benchmark_specquant.py \
        --draft-model "${QWEN35_4B:-Qwen/Qwen3.5-4B}" \
        --target-model "${QWEN35_27B:-Qwen/Qwen3.5-27B}" \
        --max-new-tokens 128 \
        --gamma 5 \
        --quant-bits 0 2 3 4 \
        --num-trials 5 \
        --prompt-type all \
        --output-dir "${RESULTS_DIR}/bitwidth_sweep" \
        2>&1 | tee "${LOG_DIR}/phase2_bitwidth.log"
    mark_done 2
fi

# ------------------------------------------------------------------
# Phase 3: TV distance validation (Claim 3)
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 3 ] && ! phase_done 3; then
    echo ""
    echo "=== Phase 3: TV distance validation ==="

    python scripts/eval_tv_distance.py \
        --target-model "${QWEN35_27B:-Qwen/Qwen3.5-27B}" \
        --bits 2 3 4 \
        --num-samples 1000 \
        --output-dir "${RESULTS_DIR}/tv_validation" \
        2>&1 | tee "${LOG_DIR}/phase3_tv.log"
    mark_done 3
fi

# ------------------------------------------------------------------
# Phase 4: Verifier microbenchmark
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 4 ] && ! phase_done 4; then
    echo ""
    echo "=== Phase 4: Verifier microbenchmark ==="

    python scripts/microbenchmark_verifier.py \
        --target-model "${QWEN35_27B:-Qwen/Qwen3.5-27B}" \
        --seq-lengths 1024 2048 4096 8192 16384 \
        --bits 0 3 4 \
        --num-trials 10 \
        --output-dir "${RESULTS_DIR}/microbenchmark" \
        2>&1 | tee "${LOG_DIR}/phase4_micro.log"
    mark_done 4
fi

# ------------------------------------------------------------------
# Phase 5: Robustness analysis
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 5 ] && ! phase_done 5; then
    echo ""
    echo "=== Phase 5: Robustness analysis ==="

    python scripts/analyze_layer_sensitivity.py \
        --target-model "${QWEN35_27B:-Qwen/Qwen3.5-27B}" \
        --bits 3 \
        --output-dir "${RESULTS_DIR}/robustness" \
        2>&1 | tee "${LOG_DIR}/phase5_robust.log"
    mark_done 5
fi

# ------------------------------------------------------------------
# Phase 6: Paper figures
# ------------------------------------------------------------------
if [ "$FROM_PHASE" -le 6 ] && ! phase_done 6; then
    echo ""
    echo "=== Phase 6: Paper figures ==="

    python scripts/generate_figures.py \
        --results-dir "${RESULTS_DIR}" \
        --output-dir "${RESULTS_DIR}/figures" \
        2>&1 | tee "${LOG_DIR}/phase6_figures.log"
    mark_done 6
fi

echo ""
echo "============================================"
echo " Pipeline complete: $(date)"
echo " Results: $RESULTS_DIR"
echo "============================================"
touch "${RESULTS_DIR}/.pipeline_done"
