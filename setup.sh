#!/bin/bash
set -e
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " AcceptSpec v3.0: Environment Setup"
echo " $(date)"
echo "============================================"

# --- Find Python >= 3.10 ---
PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver="$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")"
        major="${ver%%.*}"; minor="${ver##*.}"
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON_CMD="$cmd"; break
        fi
    fi
done
if [ -z "$PYTHON_CMD" ]; then echo "ERROR: Python >= 3.10 not found."; exit 1; fi
echo "[1/4] Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"

# --- Create venv ---
VENV_DIR="$PROJ_DIR/.venv"
if [ -d "$VENV_DIR" ] && { [ ! -f "$VENV_DIR/bin/activate" ] || [ ! -x "$VENV_DIR/bin/python" ]; }; then
    rm -rf "$VENV_DIR"
fi
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/4] Creating venv..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
else
    echo "[2/4] Venv exists"
fi
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel

# --- PyTorch + CUDA 12.8 ---
echo ""
echo "[3/4] Installing PyTorch (cu128)..."
pip install "torch>=2.7" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Pin torch for requirements.txt
TORCH_VER=$(python -c "import torch; print(torch.__version__)")
CONSTRAINT=$(mktemp)
echo "torch==$TORCH_VER" > "$CONSTRAINT"

# --- Dependencies ---
echo ""
echo "[4/4] Installing dependencies..."
pip install -r "$PROJ_DIR/requirements.txt" -c "$CONSTRAINT" 2>/dev/null || \
    pip install transformers accelerate datasets numpy scipy scikit-learn tqdm pyyaml safetensors huggingface_hub matplotlib seaborn -c "$CONSTRAINT"
rm -f "$CONSTRAINT"

# Flash attention + flash-linear-attention (Qwen3.5 GatedDeltaNet)
echo ""
echo ">>> Installing flash-attn..."
pip install flash-attn --no-build-isolation || echo "  flash-attn skipped"

echo ">>> Installing causal-conv1d + flash-linear-attention..."
pip install causal-conv1d --no-build-isolation || echo "  causal-conv1d skipped"
pip install flash-linear-attention --no-build-isolation || echo "  fla skipped"

# --- Verify ---
echo ""
echo "============================================"
python -c "
import torch, transformers
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
n = torch.cuda.device_count()
for i in range(n):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'transformers {transformers.__version__}')
try:
    from fla.modules import FusedRMSNormGated
    print('flash-linear-attention: OK')
except: print('flash-linear-attention: not available')
"
echo "============================================"
echo "Setup complete. Run: source .venv/bin/activate && bash run.sh"
