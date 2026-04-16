"""Monkey-patch torch.compile to no-op before fla is imported.

fla (flash-linear-attention) uses @torch.compile at module level, which
triggers a PyTorch inductor bug ('duplicate template name' in
flex_attention.py).  This patch must be applied BEFORE any transformers
or fla import.

Usage: python scripts/torch_patch.py scripts/triple_divergence.py [args...]
"""
import os
# Disable torch.compile / dynamo entirely before torch is imported.
# This prevents fla's module-level @torch.compile from triggering the
# inductor 'duplicate template name' AssertionError.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
torch.compile = lambda f=None, *a, **kw: f if f is not None else (lambda fn: fn)
# Also patch the internal optimize path used by older torch.compile implementations
try:
    import torch._dynamo
    torch._dynamo.config.disable = True
except Exception:
    pass

import sys
import runpy

if len(sys.argv) < 2:
    print("Usage: python scripts/torch_patch.py <script.py> [args...]")
    sys.exit(1)

target = sys.argv[1]
sys.argv = sys.argv[1:]  # shift so target script sees correct argv
runpy.run_path(target, run_name="__main__")
