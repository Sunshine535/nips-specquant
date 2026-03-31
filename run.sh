#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate 2>/dev/null || true

echo "============================================"
echo " SpecQuant: TurboQuant-Accelerated Verification"
echo " $(date) | $(hostname)"
echo "============================================"

bash scripts/run_all_experiments.sh "$@"
