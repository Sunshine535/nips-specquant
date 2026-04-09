#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " specquant: Environment Setup (venv + CUDA 12.8)"
echo "============================================"

# --- Find Python >= 3.10 ---
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver="$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")"
        major="${ver%%.*}"; minor="${ver##*.}"
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"; break
        fi
    fi
done
if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python >= 3.10 not found."; exit 1
fi
echo "[1/4] Using: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"

# --- Create venv ---
VENV_DIR="$PROJ_DIR/.venv"
if [ -d "$VENV_DIR" ] && { [ ! -f "$VENV_DIR/bin/activate" ] || [ ! -x "$VENV_DIR/bin/python" ]; }; then
    echo "[2/4] Removing incomplete .venv..."; rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating venv..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "[2/4] Venv already exists"
fi
source "$VENV_DIR/bin/activate"

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
echo "  Activate:  source .venv/bin/activate"
echo "  Run:       bash run.sh"
