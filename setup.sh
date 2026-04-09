#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME="specquant"

echo "============================================"
echo " ${ENV_NAME}: Environment Setup (conda + CUDA 12.8)"
echo "============================================"

# --- Locate conda ---
CONDA_BIN="${CONDA_EXE:-$(which conda 2>/dev/null || echo "")}"
if [ -z "$CONDA_BIN" ] || [ ! -f "$CONDA_BIN" ]; then
    for p in /opt/conda/bin/conda "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /root/miniconda3/bin/conda; do
        if [ -f "$p" ]; then CONDA_BIN="$p"; break; fi
    done
fi
if [ -z "$CONDA_BIN" ] || [ ! -f "$CONDA_BIN" ]; then
    echo "ERROR: conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "[1/4] Using conda: $CONDA_BIN"
eval "$("$CONDA_BIN" shell.bash hook 2>/dev/null)" || {
    CONDA_DIR="$(dirname "$(dirname "$CONDA_BIN")")"
    source "$CONDA_DIR/etc/profile.d/conda.sh"
}

# --- Create / activate env ---
if ! conda env list 2>/dev/null | grep -qw "$ENV_NAME"; then
    echo "[2/4] Creating conda env '$ENV_NAME' (Python 3.11)..."
    conda create -y -n "$ENV_NAME" python=3.11 2>&1 | tail -3
else
    echo "[2/4] Conda env '$ENV_NAME' already exists"
fi
conda activate "$ENV_NAME"
echo "  Python: $(python --version) @ $(which python)"

# --- Install PyTorch + CUDA 12.8 ---
echo "[3/4] Installing PyTorch (CUDA 12.8)..."
pip install -U pip setuptools wheel
pip install "torch>=2.4.0" "torchvision" "torchaudio" \
    --index-url https://download.pytorch.org/whl/cu128

# --- Install project deps ---
echo "[4/4] Installing project dependencies..."
pip install -r "$PROJ_DIR/requirements.txt"

# Optional: flash-attn
pip install flash-attn --no-build-isolation 2>/dev/null || echo "  flash-attn skipped (optional)"

# --- Verify ---
echo ""
echo "============================================"
python -c "
import torch, transformers, accelerate
print(f'  PyTorch       : {torch.__version__}')
print(f'  Transformers  : {transformers.__version__}')
print(f'  Accelerate    : {accelerate.__version__}')
print(f'  CUDA          : {torch.version.cuda}')
print(f'  GPUs          : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
" 2>/dev/null || echo "  (import check skipped)"
echo "============================================"
echo ""
echo "Setup complete!"
echo "  Activate:  conda activate $ENV_NAME"
echo "  Run:       bash run.sh"
