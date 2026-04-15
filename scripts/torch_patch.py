"""Monkey-patch torch.compile to no-op before fla is imported.

fla (flash-linear-attention) uses @torch.compile at module level, which
triggers a PyTorch inductor bug ('duplicate template name' in
flex_attention.py).  This patch must be applied BEFORE any transformers
or fla import.

Usage: python scripts/torch_patch.py scripts/triple_divergence.py [args...]
"""
import torch
torch.compile = lambda f=None, *a, **kw: f if f is not None else (lambda fn: fn)

import sys
import runpy

if len(sys.argv) < 2:
    print("Usage: python scripts/torch_patch.py <script.py> [args...]")
    sys.exit(1)

target = sys.argv[1]
sys.argv = sys.argv[1:]  # shift so target script sees correct argv
runpy.run_path(target, run_name="__main__")
