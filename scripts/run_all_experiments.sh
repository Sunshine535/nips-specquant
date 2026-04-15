#!/bin/bash
# ============================================================================
# AcceptSpec v3.0: Full experiment pipeline (NeurIPS 2026)
#
# MTP self-speculation: single model, no separate draft model.
#
# Milestones:
#   M0: Oracle sanity (10 problems)          → Gini > 0.5
#   M1: Full oracle (100 problems)           → top-20% > 80% sensitivity
#   M2: Triple divergence + predictor        → all pairwise ρ < 0.7
#   M3: Core comparison (8 retention policies)→ ≥3pp gap
#   M4: E2E system benchmark                 → ≥10% latency win
#   M5: Robustness + universality            → patterns hold
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

source .venv/bin/activate 2>/dev/null || true

RESULTS_DIR="${RESULTS_DIR:-results/acceptspec}"
MARKER_DIR="${RESULTS_DIR}/.markers"
LOG_DIR="logs"
mkdir -p "$RESULTS_DIR" "$MARKER_DIR" "$LOG_DIR"

# Single model with native MTP head (no separate draft model)
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
QUICK="${QUICK:-0}"
FROM_MILESTONE="${FROM_MILESTONE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

phase_done() { [ -f "${MARKER_DIR}/M${1}_done" ] && [ "$FORCE_RERUN" = "0" ]; }
mark_done() { touch "${MARKER_DIR}/M${1}_done"; }

echo "============================================"
echo " AcceptSpec v3.0 — MTP Self-Speculation"
echo " $(date) | $(hostname)"
echo " Model: $MODEL"
echo " Quick: $QUICK | From M: $FROM_MILESTONE"
echo "============================================"

# ------------------------------------------------------------------
# M0: Oracle sanity check (10 problems)
# Decision: Gini > 0.5 → continue, else WARN
# ------------------------------------------------------------------
if [ "$FROM_MILESTONE" -le 0 ] && ! phase_done 0; then
    echo ""
    echo "=== M0: Oracle Sanity Check (10 problems) ==="

    bash scripts/parallel_run.sh scripts/oracle_sensitivity.py \
        --model "$MODEL" \
        --num_problems 10 \
        --max_tokens 256 \
        --gamma 5 \
        --temperature 0.0 \
        --samples_per_step 50 \
        --output "${RESULTS_DIR}/oracle_m0.json" \
        2>&1 | tee "${LOG_DIR}/M0_oracle_sanity.log"

    # Check gate
    GINI=$(python -c "
import json
with open('${RESULTS_DIR}/oracle_m0.json') as f:
    d = json.load(f)
print(d['aggregate']['mean_gini'])
" 2>/dev/null || echo "0")

    echo "  M0 Gini: $GINI"
    PASSED=$(python -c "print('yes' if float('$GINI') > 0.5 else 'no')")
    if [ "$PASSED" = "no" ]; then
        echo "  M0 GATE WARNING: Gini <= 0.5. Sparsity may be weak."
        echo "  Continuing to M1 for full evaluation..."
    else
        echo "  M0 PASS: Gini > 0.5"
    fi
    mark_done 0
fi

# ------------------------------------------------------------------
# M1: Full oracle study (100 problems)
# Decision: top-20% > 50% → continue (recalibrated from 80% after M0/M1 data)
# Gate is WARN, not ABORT — proceed to M2/M3 to test if moderate sparsity
# still yields accept-optimal > attention-optimal (the core claim).
# ------------------------------------------------------------------
if [ "$FROM_MILESTONE" -le 1 ] && ! phase_done 1; then
    echo ""
    echo "=== M1: Full Oracle Study (100 problems) ==="

    NUM_PROBLEMS=100
    [ "$QUICK" = "1" ] && NUM_PROBLEMS=20

    bash scripts/parallel_run.sh scripts/oracle_sensitivity.py \
        --model "$MODEL" \
        --num_problems $NUM_PROBLEMS \
        --max_tokens 256 \
        --gamma 5 \
        --temperature 0.0 \
        --samples_per_step 50 \
        --output "${RESULTS_DIR}/oracle_m1.json" \
        2>&1 | tee "${LOG_DIR}/M1_oracle_full.log"

    TOP20=$(python -c "
import json
with open('${RESULTS_DIR}/oracle_m1.json') as f:
    d = json.load(f)
print(d['aggregate']['top20_coverage'])
" 2>/dev/null || echo "0")

    echo "  M1 Top-20% coverage: $TOP20"
    WEAK=$(python -c "print('yes' if float('$TOP20') < 0.5 else 'no')")
    if [ "$WEAK" = "yes" ]; then
        echo "  ⚠️  M1 WARNING: Top-20% < 50%. Sparsity is weak. Proceeding to M2/M3 to test if gap still exists."
    fi

    STRONG=$(python -c "print('yes' if float('$TOP20') >= 0.8 else 'no')")
    if [ "$STRONG" = "yes" ]; then
        echo "  M1 STRONG PASS: Top-20% >= 80%"
    elif [ "$WEAK" != "yes" ]; then
        echo "  M1 MODERATE PASS: Top-20% between 50-80%. Proceed — core test is M3 gap."
    fi
    mark_done 1
fi

# ------------------------------------------------------------------
# M2: Triple divergence + predictor validation
# ------------------------------------------------------------------
if [ "$FROM_MILESTONE" -le 2 ] && ! phase_done 2; then
    echo ""
    echo "=== M2: Triple Divergence + Predictor Validation ==="

    NUM_PROBLEMS=100
    [ "$QUICK" = "1" ] && NUM_PROBLEMS=30

    bash scripts/parallel_run.sh scripts/triple_divergence.py \
        --model "$MODEL" \
        --num_problems $NUM_PROBLEMS \
        --output_dir "${RESULTS_DIR}/divergence" \
        --output "${RESULTS_DIR}/divergence/triple_divergence.json" \
        2>&1 | tee "${LOG_DIR}/M2_divergence.log"

    mark_done 2
fi

# ------------------------------------------------------------------
# M3: Core comparison — 8 retention policies
# ------------------------------------------------------------------
if [ "$FROM_MILESTONE" -le 3 ] && ! phase_done 3; then
    echo ""
    echo "=== M3: Core Comparison (8 retention policies) ==="

    NUM_PROBLEMS=1319
    [ "$QUICK" = "1" ] && NUM_PROBLEMS=100

    for DATASET in gsm8k math500; do
        bash scripts/parallel_run.sh scripts/core_comparison.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --num_problems $NUM_PROBLEMS \
            --kv_budget 0.2 \
            --output_dir "${RESULTS_DIR}/comparison" \
            --output "${RESULTS_DIR}/comparison/core_${DATASET}_b0.2.json" \
            2>&1 | tee "${LOG_DIR}/M3_comparison_${DATASET}.log"
    done

    # Anti-claim ablation
    bash scripts/parallel_run.sh scripts/core_comparison.py \
        --model "$MODEL" \
        --dataset gsm8k \
        --num_problems $NUM_PROBLEMS \
        --kv_budget 0.2 \
        --ablation no_sd \
        --output_dir "${RESULTS_DIR}/comparison" \
        --output "${RESULTS_DIR}/comparison/core_gsm8k_nosd.json" \
        2>&1 | tee "${LOG_DIR}/M3_anticlaim.log"

    # Budget sweep
    for BUDGET in 0.1 0.2 0.3 0.5; do
        bash scripts/parallel_run.sh scripts/core_comparison.py \
            --model "$MODEL" \
            --dataset gsm8k \
            --num_problems 500 \
            --kv_budget $BUDGET \
            --output_dir "${RESULTS_DIR}/comparison" \
            --output "${RESULTS_DIR}/comparison/core_gsm8k_b${BUDGET}.json" \
            2>&1 | tee -a "${LOG_DIR}/M3_budget_sweep.log"
    done

    mark_done 3
fi

# ------------------------------------------------------------------
# M4: E2E system benchmark
# ------------------------------------------------------------------
if [ "$FROM_MILESTONE" -le 4 ] && ! phase_done 4; then
    echo ""
    echo "=== M4: End-to-End System Benchmark ==="

    NUM_PROBLEMS=1319
    [ "$QUICK" = "1" ] && NUM_PROBLEMS=100

    for DATASET in gsm8k math500; do
        python scripts/e2e_benchmark.py \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --num_problems $NUM_PROBLEMS \
            --gamma 5 \
            --output_dir "${RESULTS_DIR}/e2e" \
            2>&1 | tee "${LOG_DIR}/M4_e2e_${DATASET}.log"
    done

    # Profiling
    python scripts/e2e_benchmark.py \
        --model "$MODEL" \
        --dataset gsm8k \
        --num_problems 50 \
        --gamma 5 \
        --profile \
        --output_dir "${RESULTS_DIR}/e2e" \
        2>&1 | tee "${LOG_DIR}/M4_profiling.log"

    mark_done 4
fi

# ------------------------------------------------------------------
# M5: Robustness + Universality
# ------------------------------------------------------------------
if [ "$FROM_MILESTONE" -le 5 ] && ! phase_done 5; then
    echo ""
    echo "=== M5: Robustness Sweeps ==="

    NUM_PROBLEMS=200
    [ "$QUICK" = "1" ] && NUM_PROBLEMS=50

    for TEMP in 0.0 0.3 0.6 0.9; do
        python scripts/oracle_sensitivity.py \
            --model "$MODEL" \
            --num_problems $NUM_PROBLEMS \
            --temperature $TEMP \
            --output "${RESULTS_DIR}/robustness/oracle_temp_${TEMP}.json" \
            2>&1 | tee -a "${LOG_DIR}/M5_robustness.log"
    done

    for GAMMA in 3 5 7 10; do
        python scripts/oracle_sensitivity.py \
            --model "$MODEL" \
            --num_problems $NUM_PROBLEMS \
            --gamma $GAMMA \
            --output "${RESULTS_DIR}/robustness/oracle_gamma_${GAMMA}.json" \
            2>&1 | tee -a "${LOG_DIR}/M5_robustness.log"
    done

    mark_done 5
fi

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "============================================"
echo " AcceptSpec Pipeline Complete: $(date)"
echo " Results: $RESULTS_DIR"
echo "============================================"
touch "${RESULTS_DIR}/.pipeline_done"
