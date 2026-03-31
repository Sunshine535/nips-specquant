#!/bin/bash
# ============================================================================
# SpecQuant ACP Training — Startup Script
#
# Startup command:
#   bash /data/szs/250010072/nwh/nips-specquant/run_acp.sh
# ============================================================================
set -euo pipefail

PROJECT_DIR=/data/szs/250010072/nwh/nips-specquant
DATA_DIR=/data/szs/share/specquant
SHARE_DIR=/data/szs/share

LOG_DIR=${DATA_DIR}/logs
mkdir -p ${DATA_DIR}/{results,logs,hf_cache,models}
LOG="${LOG_DIR}/acp_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "============================================"
echo " SpecQuant ACP Experiments"
echo " $(date) | $(hostname)"
echo " GPUs: ${SENSECORE_ACCELERATE_DEVICE_COUNT:-unknown}"
echo "============================================"

export HF_HOME="${DATA_DIR}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH=${CUDA_HOME}/bin:${PATH}
export SPECQUANT_DATA_DIR="${DATA_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# Use local model paths from shared storage
export QWEN35_0_8B="${SHARE_DIR}/Qwen3.5-0.8B"
export QWEN35_4B="${SHARE_DIR}/Qwen3.5-4B"
export QWEN35_9B="${SHARE_DIR}/Qwen3.5-9B"
export QWEN35_27B="${SHARE_DIR}/Qwen3.5-27B"

cd ${PROJECT_DIR}

# ========== PRE-FLIGHT ==========
python -c "
import torch, sys
print(f'Python {sys.version.split()[0]}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
n = torch.cuda.device_count()
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name} ({p.total_memory / 1e9:.0f}GB)')
import transformers, accelerate
print(f'Transformers {transformers.__version__}, Accelerate {accelerate.__version__}')
"

# ========== INSTALL DEPS ==========
pip install --quiet torch transformers accelerate datasets numpy scipy scikit-learn tqdm pyyaml wandb matplotlib seaborn 2>&1 | tail -5

# ========== RUN ALL EXPERIMENTS ==========
bash scripts/run_all_experiments.sh 2>&1

echo ""
echo "Done: $(date)"
echo "Log: ${LOG}"
