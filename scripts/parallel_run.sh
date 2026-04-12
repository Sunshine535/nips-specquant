#!/bin/bash
# ============================================================================
# Parallel launcher: run a Python script across N GPUs (1 model per GPU)
#
# Usage:
#   bash scripts/parallel_run.sh <script.py> [args...] --num_shards N
#
# Example:
#   bash scripts/parallel_run.sh scripts/oracle_sensitivity.py \
#       --model Qwen/Qwen3.5-9B --num_problems 100 --output results/oracle.json
#
# This will launch N parallel processes (one per GPU), each with:
#   CUDA_VISIBLE_DEVICES=X python <script.py> [args...] --shard I --num_shards N
#   --output results/oracle_shard_I.json
#
# Then merges shard results into the final output.
# ============================================================================
set -euo pipefail

# Detect GPUs
NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "[parallel] $NUM_GPUS GPUs detected"

# Parse: find --num_shards or default to NUM_GPUS
NUM_SHARDS=$NUM_GPUS
SCRIPT=""
ARGS=()
OUTPUT=""
for arg in "$@"; do
    if [ -z "$SCRIPT" ] && [[ "$arg" == *.py ]]; then
        SCRIPT="$arg"
    else
        ARGS+=("$arg")
    fi
done

# Extract --output from args
for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--output" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then
        OUTPUT="${ARGS[$((i+1))]}"
    fi
    if [ "${ARGS[$i]}" = "--output_dir" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then
        OUTPUT="${ARGS[$((i+1))]}"
    fi
done

if [ -z "$SCRIPT" ]; then
    echo "Usage: bash scripts/parallel_run.sh <script.py> [args...]"
    exit 1
fi

echo "[parallel] Script: $SCRIPT"
echo "[parallel] Shards: $NUM_SHARDS"
echo "[parallel] Output: $OUTPUT"

# Launch shards
PIDS=()
SHARD_OUTPUTS=()
for ((i=0; i<NUM_SHARDS; i++)); do
    GPU=$i

    # Build shard-specific output path
    if [ -n "$OUTPUT" ]; then
        BASE="${OUTPUT%.*}"
        EXT="${OUTPUT##*.}"
        SHARD_OUT="${BASE}_shard${i}.${EXT}"
    else
        SHARD_OUT="results/shard_${i}.json"
    fi
    SHARD_OUTPUTS+=("$SHARD_OUT")

    # Build args with shard info and shard-specific output
    SHARD_ARGS=()
    SKIP_NEXT=0
    for j in "${!ARGS[@]}"; do
        if [ "$SKIP_NEXT" -eq 1 ]; then
            SKIP_NEXT=0
            continue
        fi
        if [ "${ARGS[$j]}" = "--output" ]; then
            SHARD_ARGS+=("--output" "$SHARD_OUT")
            SKIP_NEXT=1
        elif [ "${ARGS[$j]}" = "--output_dir" ]; then
            SHARD_ARGS+=("--output_dir" "$SHARD_OUT")
            SKIP_NEXT=1
        else
            SHARD_ARGS+=("${ARGS[$j]}")
        fi
    done

    echo "[parallel] Launching shard $i on GPU $GPU → $SHARD_OUT"
    CUDA_VISIBLE_DEVICES=$GPU python "$SCRIPT" "${SHARD_ARGS[@]}" \
        --shard "$i" --num_shards "$NUM_SHARDS" &
    PIDS+=($!)
done

# Wait for all shards
echo "[parallel] Waiting for $NUM_SHARDS shards..."
FAILED=0
for ((i=0; i<NUM_SHARDS; i++)); do
    if ! wait "${PIDS[$i]}"; then
        echo "[parallel] Shard $i FAILED (PID ${PIDS[$i]})"
        FAILED=$((FAILED+1))
    else
        echo "[parallel] Shard $i done"
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "[parallel] $FAILED/$NUM_SHARDS shards failed"
fi

# Merge shard results
if [ -n "$OUTPUT" ]; then
    echo "[parallel] Merging ${#SHARD_OUTPUTS[@]} shards → $OUTPUT"
    python3 -c "
import json, sys, glob

shards = []
for path in sys.argv[1:]:
    try:
        with open(path) as f:
            shards.append(json.load(f))
    except Exception as e:
        print(f'  Skip {path}: {e}')

if not shards:
    print('ERROR: No shard results to merge')
    sys.exit(1)

# Merge: combine per_problem lists, recompute aggregates
merged = shards[0].copy()
if 'per_problem' in merged:
    all_problems = []
    for s in shards:
        all_problems.extend(s.get('per_problem', []))
    merged['per_problem'] = all_problems
    merged['num_problems'] = len(all_problems)

    # Recompute aggregates
    ginis = [p.get('mean_gini', 0) for p in all_problems if 'mean_gini' in p]
    top20s = [p.get('top20_coverage', 0) for p in all_problems if 'top20_coverage' in p]
    if ginis:
        merged.setdefault('aggregate', {})['mean_gini'] = sum(ginis)/len(ginis)
    if top20s:
        merged.setdefault('aggregate', {})['top20_coverage'] = sum(top20s)/len(top20s)

with open('$OUTPUT', 'w') as f:
    json.dump(merged, f, indent=2)
print(f'  Merged {len(shards)} shards → $OUTPUT')
" "${SHARD_OUTPUTS[@]}"
fi

echo "[parallel] Done"
