#!/bin/bash
# ============================================================================
# AcceptSpec: Run experiments
#
# Everything stays inside the project directory:
#   .venv/          — Python virtual environment
#   .cache/hf/      — HuggingFace model cache
#   results/        — experiment outputs
#   logs/           — experiment logs
#
# Usage:
#   bash run.sh                          # full pipeline M0→M5
#   QUICK=1 bash run.sh                  # quick test (fewer problems)
#   FROM_MILESTONE=2 bash run.sh         # resume from M2
#   FORCE_RERUN=1 bash run.sh            # re-run all
#
# Models: set DRAFT_MODEL / TARGET_MODEL to local paths or HF names.
#   DRAFT_MODEL=/path/to/Qwen3.5-0.8B TARGET_MODEL=/path/to/Qwen3.5-9B bash run.sh
# ============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ_DIR"

# --- HF cache: default to parent dir of project ---
export HF_HOME="${HF_HOME:-$(dirname "$PROJ_DIR")/.cache/hf}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJ_DIR}:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" logs results

# --- Activate venv ---
if [ ! -f ".venv/bin/activate" ]; then
    echo "ERROR: .venv not found. Run 'bash setup.sh' first."
    exit 1
fi
source .venv/bin/activate

# --- Log ---
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "============================================"
echo " AcceptSpec Experiment Pipeline"
echo " $(date) | $(hostname)"
echo " Draft : ${DRAFT_MODEL:-Qwen/Qwen3.5-0.8B}"
echo " Target: ${TARGET_MODEL:-Qwen/Qwen3.5-9B}"
echo " HF_HOME: $HF_HOME"
echo "============================================"

# --- Pre-flight ---
python -c "
import torch, sys
print(f'Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda or \"N/A\"}')
n = torch.cuda.device_count()
if n == 0:
    print('  No GPU detected (CPU mode)')
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    mem = getattr(p, 'total_memory', 0) or getattr(p, 'total_mem', 0)
    print(f'  GPU {i}: {p.name} ({mem / 1e9:.0f}GB)')
"

# --- Run ---
bash scripts/run_all_experiments.sh

echo ""
echo "Done: $(date)"
echo "Log: $LOG"
