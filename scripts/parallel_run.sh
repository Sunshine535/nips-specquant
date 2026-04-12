#!/bin/bash
# ============================================================================
# Parallel launcher: run a Python script across GPUs
#
# Strategy: 2 GPUs per model instance (9B needs >80GB with KV cache)
#   8 GPUs → 4 parallel instances
#   4 GPUs → 2 parallel instances
#   2 GPUs → 1 instance
#
# Usage:
#   bash scripts/parallel_run.sh scripts/oracle_sensitivity.py \
#       --model Qwen/Qwen3.5-9B --num_problems 100 --output results/oracle.json
# ============================================================================
set -euo pipefail

NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
GPUS_PER_INSTANCE=2
NUM_INSTANCES=$((NUM_GPUS / GPUS_PER_INSTANCE))
[ "$NUM_INSTANCES" -lt 1 ] && NUM_INSTANCES=1

echo "[parallel] $NUM_GPUS GPUs, $GPUS_PER_INSTANCE per instance → $NUM_INSTANCES parallel"

# Parse script and args
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

# Extract --output
for i in "${!ARGS[@]}"; do
    if [ "${ARGS[$i]}" = "--output" ] && [ $((i+1)) -lt ${#ARGS[@]} ]; then
        OUTPUT="${ARGS[$((i+1))]}"
    fi
done

if [ -z "$SCRIPT" ]; then
    echo "Usage: bash scripts/parallel_run.sh <script.py> [args...]"
    exit 1
fi

echo "[parallel] Script: $SCRIPT | Instances: $NUM_INSTANCES"

# Launch instances
PIDS=()
SHARD_OUTPUTS=()
for ((i=0; i<NUM_INSTANCES; i++)); do
    GPU_START=$((i * GPUS_PER_INSTANCE))
    GPU_END=$((GPU_START + GPUS_PER_INSTANCE - 1))
    GPU_LIST=$(seq -s, $GPU_START $GPU_END)

    # Shard-specific output
    if [ -n "$OUTPUT" ]; then
        BASE="${OUTPUT%.*}"
        EXT="${OUTPUT##*.}"
        SHARD_OUT="${BASE}_shard${i}.${EXT}"
    else
        SHARD_OUT="results/shard_${i}.json"
    fi
    SHARD_OUTPUTS+=("$SHARD_OUT")

    # Build args with shard info
    SHARD_ARGS=()
    SKIP_NEXT=0
    for j in "${!ARGS[@]}"; do
        if [ "$SKIP_NEXT" -eq 1 ]; then SKIP_NEXT=0; continue; fi
        if [ "${ARGS[$j]}" = "--output" ]; then
            SHARD_ARGS+=("--output" "$SHARD_OUT")
            SKIP_NEXT=1
        else
            SHARD_ARGS+=("${ARGS[$j]}")
        fi
    done

    echo "[parallel] Instance $i: GPUs $GPU_LIST → $SHARD_OUT"
    CUDA_VISIBLE_DEVICES=$GPU_LIST python "$SCRIPT" "${SHARD_ARGS[@]}" \
        --shard "$i" --num_shards "$NUM_INSTANCES" &
    PIDS+=($!)
done

# Wait
echo "[parallel] Waiting for $NUM_INSTANCES instances..."
FAILED=0
for ((i=0; i<NUM_INSTANCES; i++)); do
    if ! wait "${PIDS[$i]}"; then
        echo "[parallel] Instance $i FAILED"
        FAILED=$((FAILED+1))
    else
        echo "[parallel] Instance $i done"
    fi
done

[ "$FAILED" -gt 0 ] && echo "[parallel] $FAILED/$NUM_INSTANCES failed"

# Merge
if [ -n "$OUTPUT" ]; then
    echo "[parallel] Merging → $OUTPUT"
    python3 -c "
import json, sys

shards = []
for path in sys.argv[1:]:
    try:
        with open(path) as f:
            shards.append(json.load(f))
    except Exception as e:
        print(f'  Skip {path}: {e}')

if not shards:
    print('ERROR: No shard results')
    sys.exit(1)

merged = shards[0].copy()
if 'per_problem' in merged:
    all_problems = []
    for s in shards:
        all_problems.extend(s.get('per_problem', []))
    merged['per_problem'] = all_problems
    merged['num_problems'] = len(all_problems)

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
