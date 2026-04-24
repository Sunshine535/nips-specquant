"""Reproducibility utilities: deterministic seeds, split manifests, run metadata."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


def set_global_seed(seed: int) -> torch.Generator:
    """Set all random seeds and return a torch Generator for coupled randomness."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def make_coupled_uniforms(
    n_steps: int, gamma: int, gen: torch.Generator
) -> torch.Tensor:
    """Pre-sample uniform random variables for coupled acceptance measurement.

    Returns tensor of shape [n_steps, gamma] with values in [0, 1).
    Using the same uniforms for full-KV and perturbed-KV ensures paired measurement.
    """
    return torch.rand(n_steps, gamma, generator=gen)


@dataclass
class SplitManifest:
    """Records which problem IDs are used for calibration vs evaluation."""

    dataset: str
    total_problems: int
    calib_indices: List[int]
    eval_indices: List[int]
    seed: int
    split_hash: str = ""

    def __post_init__(self):
        overlap = set(self.calib_indices) & set(self.eval_indices)
        if overlap:
            raise ValueError(f"Calibration/eval overlap: {overlap}")
        self.split_hash = hashlib.md5(
            json.dumps(
                {"calib": sorted(self.calib_indices), "eval": sorted(self.eval_indices)},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SplitManifest":
        with open(path) as f:
            return cls(**json.load(f))


def make_calib_eval_split(
    total: int, calib_fraction: float = 0.3, seed: int = 42
) -> SplitManifest:
    """Create a deterministic calibration/evaluation split."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(total).tolist()
    n_calib = max(1, int(total * calib_fraction))
    return SplitManifest(
        dataset="",
        total_problems=total,
        calib_indices=sorted(indices[:n_calib]),
        eval_indices=sorted(indices[n_calib:]),
        seed=seed,
    )


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class RunMetadata:
    """Metadata saved with every experiment run."""

    model: str = ""
    dataset: str = ""
    seed: int = 42
    gamma: int = 5
    temperature: float = 0.0
    kv_budget: float = 0.2
    max_tokens: int = 512
    num_problems: int = 0
    git_hash: str = ""
    timestamp: str = ""
    command: str = ""
    hostname: str = ""
    gpu_count: int = 0
    extra: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.git_hash:
            self.git_hash = _get_git_hash()
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.hostname:
            import socket
            self.hostname = socket.gethostname()
        if not self.gpu_count and torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
