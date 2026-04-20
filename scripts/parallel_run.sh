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
NUM_INSTANCES=$NUM_GPUS

echo "[parallel] $NUM_GPUS GPUs → $NUM_INSTANCES parallel instances (1 GPU each)"

# Pre-download datasets so shards don't race on HF filelock (6/8 shards died
# with FileNotFoundError on fchmod when loading gsm8k/math500 simultaneously)
echo "[parallel] Pre-downloading datasets to avoid filelock race..."
python3 -c "
from datasets import load_dataset
try:
    load_dataset('openai/gsm8k', 'main', split='test')
    print('  gsm8k cached')
except Exception as e:
    print(f'  gsm8k fetch failed: {e}')
try:
    load_dataset('HuggingFaceH4/MATH-500', split='test')
    print('  MATH-500 cached')
except Exception:
    pass
" 2>&1 | grep -v "HTTP\|hf-mirror\|Fetching\|Downloading" | tail -5

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
    GPU_LIST=$i

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
    CUDA_VISIBLE_DEVICES=$GPU_LIST python scripts/torch_patch.py "$SCRIPT" "${SHARD_ARGS[@]}" \
        --shard "$i" --num_shards "$NUM_INSTANCES" &
    PIDS+=($!)
    # Stagger launches by 10s to avoid HF dataset cache filelock race
    # (6/8 shards died with fchmod FileNotFoundError on simultaneous load).
    # Combined with pre-download above, this should eliminate the race.
    sleep 10
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

# Merge — Round 1 reviewer fix: recompute aggregate from all shards'
# aggregate fields (not just shards[0]); flag missing/divergent keys.
if [ -n "$OUTPUT" ]; then
    echo "[parallel] Merging → $OUTPUT"
    python3 -c "
import json, sys

shards = []
shard_paths = []
for path in sys.argv[1:]:
    try:
        with open(path) as f:
            shards.append(json.load(f))
        shard_paths.append(path)
    except Exception as e:
        print(f'  Skip {path}: {e}')

if not shards:
    print('ERROR: No shard results')
    sys.exit(1)

# Start with a clean dict — do NOT inherit shard[0]'s stale aggregate
merged = {'num_shards': len(shards), 'shard_paths': shard_paths}
# Preserve any config from shard 0 (assumed identical across shards)
if 'config' in shards[0]:
    merged['config'] = shards[0]['config']

# Merge per_problem (top-level for M2 oracle/divergence)
all_problems = []
for s in shards:
    all_problems.extend(s.get('per_problem', []))
merged['per_problem'] = all_problems
merged['num_problems'] = len(all_problems)

# Round 2 reviewer fix: M3 core_comparison stores data under
# per_policy.<policy_name>.per_problem + per_policy.<policy_name>.accuracy
# Need to merge policy-by-policy and recompute accuracy.
if any('per_policy' in s for s in shards):
    merged_per_policy = {}
    # Collect the set of all policies across shards
    policy_names = set()
    for s in shards:
        for pname in s.get('per_policy', {}).keys():
            policy_names.add(pname)
    for pname in policy_names:
        policy_problems = []
        total_correct = 0
        total_probs = 0
        for s in shards:
            pdata = s.get('per_policy', {}).get(pname, {})
            policy_problems.extend(pdata.get('per_problem', []))
        # Recompute accuracy from merged per-problem entries
        correct_entries = [p for p in policy_problems if 'correct' in p]
        if correct_entries:
            total_correct = sum(1 for p in correct_entries if p.get('correct'))
            total_probs = len(correct_entries)
            accuracy = total_correct / max(total_probs, 1)
        else:
            accuracy = None
        # Recompute mean acceptance rate if present
        ar_vals = [p.get('acceptance_rate') for p in policy_problems
                   if p.get('acceptance_rate') is not None]
        mean_ar = sum(ar_vals) / len(ar_vals) if ar_vals else None
        merged_per_policy[pname] = {
            'per_problem': policy_problems,
            'num_problems': len(policy_problems),
            'accuracy': accuracy,
            'num_correct': total_correct,
            'mean_acceptance_rate': mean_ar,
        }
    merged['per_policy'] = merged_per_policy
    # Recompute claim_c3 from merged accuracies if present
    ac_map = {k: v.get('accuracy') for k, v in merged_per_policy.items()
              if v.get('accuracy') is not None}
    if 'oracle_accept' in ac_map and 'perplexity' in ac_map and 'attention_h2o' in ac_map:
        gap_ppl = (ac_map['oracle_accept'] - ac_map['perplexity']) * 100
        gap_attn = (ac_map['oracle_accept'] - ac_map['attention_h2o']) * 100
        merged['claim_c3'] = {
            'gap_vs_perplexity_pp': gap_ppl,
            'gap_vs_attention_pp': gap_attn,
            'passed': gap_ppl >= 3.0 and gap_attn >= 3.0,
        }
        print(f'  [M3] oracle_accept vs perplexity: {gap_ppl:+.2f}pp')
        print(f'  [M3] oracle_accept vs attention_h2o: {gap_attn:+.2f}pp')

# Recompute mean_gini + top-k coverage from per-problem data when present
agg = {}
for k in ('mean_gini', 'top10_coverage', 'top20_coverage', 'top50_coverage',
         'mean_alpha'):
    vals = [p.get(k) for p in all_problems if k in p and p.get(k) is not None]
    if vals:
        agg[k] = float(sum(vals) / len(vals))
        agg[f'{k}_n'] = len(vals)

# Also weighted-aggregate shards' own 'aggregate' block (for legacy fields)
shard_aggs = [s.get('aggregate', {}) for s in shards if 'aggregate' in s]
# Cross-shard sanity: if shards disagree on keys present, flag loudly
all_keys = set()
for sa in shard_aggs:
    all_keys.update(sa.keys())
key_divergence = {}
for k in sorted(all_keys):
    vals = [sa[k] for sa in shard_aggs if k in sa and isinstance(sa[k], (int, float))]
    if len(vals) >= 2:
        key_divergence[k] = {
            'min': float(min(vals)), 'max': float(max(vals)),
            'mean': float(sum(vals) / len(vals)), 'n_shards': len(vals),
        }
        if k not in agg:
            agg[k] = key_divergence[k]['mean']
merged['aggregate'] = agg
merged['per_shard_aggregate_diagnostics'] = key_divergence

# Merge Spearman correlations — do NOT average correlations across shards
# (that's wrong statistically). Report per-shard values and flag.
corr_by_shard = []
for s in shards:
    if 'spearman_correlations' in s:
        corr_by_shard.append(s['spearman_correlations'])
    if 'matched_spearman_correlations' in s:
        pass  # handled separately below
if corr_by_shard:
    merged['per_shard_spearman_correlations'] = corr_by_shard

matched_by_shard = [s.get('matched_spearman_correlations', {}) for s in shards
                    if s.get('matched_spearman_correlations')]
if matched_by_shard:
    merged['per_shard_matched_spearman_correlations'] = matched_by_shard

with open('$OUTPUT', 'w') as f:
    json.dump(merged, f, indent=2, default=str)
print(f'  Merged {len(shards)} shards → $OUTPUT')
print(f'  Aggregate keys: {sorted(agg.keys())}')
if key_divergence:
    for k, v in key_divergence.items():
        rng = v['max'] - v['min']
        if rng > 0.1 * abs(v['mean']):
            print(f'  WARN: shards disagree on {k}: min={v[\"min\"]:.3f} max={v[\"max\"]:.3f} (range {rng:.3f})')
" "${SHARD_OUTPUTS[@]}"
fi

echo "[parallel] Done"
