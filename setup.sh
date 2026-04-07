#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up nips-specquant (AcceptSpec) ==="

if [ -d ".venv" ]; then
    echo "[skip] .venv already exists"
else
    python3 -m venv .venv
    echo "[ok] Created .venv"
fi

source .venv/bin/activate
pip install --upgrade pip

# Detect CUDA driver version and install matching PyTorch
DRIVER_CUDA=""
if command -v nvidia-smi &>/dev/null; then
    DRIVER_CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')
    echo "NVIDIA driver: $DRIVER_CUDA"
fi

# Install PyTorch with best available CUDA support
# Try cu126 first (compatible with driver 12.6+), fall back to cu124, then CPU
TORCH_INSTALLED=0
for CUDA_TAG in cu126 cu124 cu121 cpu; do
    if [ "$CUDA_TAG" = "cpu" ]; then
        echo "Installing PyTorch (CPU only)..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu && TORCH_INSTALLED=1 && break
    else
        echo "Trying PyTorch with $CUDA_TAG..."
        if pip install "torch>=2.5" --index-url "https://download.pytorch.org/whl/$CUDA_TAG" 2>/dev/null; then
            TORCH_INSTALLED=1
            break
        fi
    fi
done

if [ "$TORCH_INSTALLED" -eq 0 ]; then
    echo "[WARN] Could not install PyTorch automatically. Install manually."
fi

# Install remaining dependencies
pip install transformers accelerate datasets numpy scipy scikit-learn \
    tqdm pyyaml matplotlib seaborn

echo ""
python3 -c "
import sys, torch
print(f'Python {sys.version.split()[0]}, PyTorch {torch.__version__}')
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f'CUDA available, {n} GPU(s):')
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        mem = getattr(p, 'total_memory', 0) or getattr(p, 'total_mem', 0)
        print(f'  [{i}] {p.name} ({mem / 1e9:.0f}GB)')
else:
    print('CUDA not available (CPU mode)')
print()

# Test gpu_auto
sys.path.insert(0, '.')
from src.gpu_auto import plan_devices
plan = plan_devices()
print(f'Device strategy: {plan.description}')
"
echo ""
echo "=== Setup complete. Run: source .venv/bin/activate ==="
