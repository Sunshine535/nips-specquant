#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " specquant: Environment Setup"
echo " $(date)"
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
echo "[1/3] Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
echo ""

# --- Create venv ---
VENV_DIR="$PROJ_DIR/.venv"
if [ -d "$VENV_DIR" ] && { [ ! -f "$VENV_DIR/bin/activate" ] || [ ! -x "$VENV_DIR/bin/python" ]; }; then
    echo "[2/3] Removing incomplete .venv..."
    rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/3] Creating venv at $VENV_DIR ..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
    echo "  Done."
else
    echo "[2/3] Venv already exists: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "  Activated: $(which python)"
echo ""

# --- Detect CUDA version and pick PyTorch index ---
echo "[3/3] Detecting CUDA version..."
TORCH_INDEX="https://download.pytorch.org/whl/cu121"
if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "")
    if [ -n "$CUDA_VER" ]; then
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        echo "  System CUDA: $CUDA_VER"
        if [ "$CUDA_MAJOR" -ge 13 ] || { [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]; }; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu128"
            echo "  -> Using PyTorch cu128"
        elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            echo "  -> Using PyTorch cu124"
        else
            echo "  -> Using PyTorch cu121"
        fi
    else
        echo "  CUDA version not detected, defaulting to cu121"
    fi
else
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    echo "  No NVIDIA GPU detected, using CPU PyTorch"
fi
echo ""

pip install wheel

echo ">>> $(date) - Installing torch, torchvision, torchaudio from $TORCH_INDEX ..."
pip install "torch>=2.4.0" "torchvision" "torchaudio" \
    --index-url "$TORCH_INDEX"
echo ""

# Pin torch version so requirements.txt doesn't overwrite it from corporate mirror
TORCH_PIN="torch==$(pip show torch | grep '^Version:' | awk '{print $2}')"
TV_PIN="torchvision==$(pip show torchvision | grep '^Version:' | awk '{print $2}')"
TA_PIN="torchaudio==$(pip show torchaudio | grep '^Version:' | awk '{print $2}')"
CONSTRAINT_FILE=$(mktemp)
echo "$TORCH_PIN" > "$CONSTRAINT_FILE"
echo "$TV_PIN" >> "$CONSTRAINT_FILE"
echo "$TA_PIN" >> "$CONSTRAINT_FILE"
echo "  Pinned: $TORCH_PIN, $TV_PIN, $TA_PIN"
echo ""

echo ">>> $(date) - Installing requirements.txt (torch version pinned)..."
pip install -r "$PROJ_DIR/requirements.txt" -c "$CONSTRAINT_FILE"
rm -f "$CONSTRAINT_FILE"
echo ""

echo ">>> $(date) - Installing flash-attn (optional, may take a few minutes)..."
pip install flash-attn --no-build-isolation || echo "  flash-attn install failed (optional, skipping)"
echo ""

echo ">>> $(date) - Installing flash-linear-attention + causal-conv1d (optional, for Qwen3.5 GatedDeltaNet efficiency)..."
pip install causal-conv1d --no-build-isolation 2>/dev/null || echo "  causal-conv1d skipped (needs matching CUDA)"
pip install flash-linear-attention --no-build-isolation 2>/dev/null || echo "  flash-linear-attention skipped"
echo ""

# --- Verify ---
echo "============================================"
echo " Verification  $(date)"
echo "============================================"
python -c "
import torch, transformers, accelerate
print(f'  PyTorch       : {torch.__version__}')
print(f'  CUDA (torch)  : {torch.version.cuda}')
print(f'  GPUs          : {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'  transformers  : {transformers.__version__}')
print(f'  accelerate    : {accelerate.__version__}')
"
echo "============================================"
echo ""
echo "Setup complete!  $(date)"
echo "  Activate:  source .venv/bin/activate"
echo "  Run:       bash run.sh"
