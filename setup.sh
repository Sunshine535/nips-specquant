#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Setting up nips-specquant ==="

if [ -d ".venv" ]; then
    echo "[skip] .venv already exists"
else
    python3 -m venv .venv
    echo "[ok] Created .venv"
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
python -c "
import torch, sys
print(f'Python {sys.version.split()[0]}, PyTorch {torch.__version__}')
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f'CUDA {torch.version.cuda}, {n} GPU(s)')
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name} ({p.total_memory / 1e9:.0f}GB)')
else:
    print('CUDA not available')
"
echo ""
echo "=== Setup complete. Run: source .venv/bin/activate ==="
