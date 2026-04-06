"""Statistical utilities for NeurIPS-level ML experiment reporting.

Provides confidence intervals, significance tests, effect-size computation,
GPU helpers, atomic result I/O, and KV cache compatibility helpers used
across all SpecQuant experiments.
"""

import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

NumericArray = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------------
# 0. KV cache API compatibility (transformers 4.x / 5.x / 5.5+)
# ---------------------------------------------------------------------------


def get_kv_tensors(kv_cache: Any, layer_idx: int) -> Tuple[Any, Any]:
    """Get K, V tensors from a KV cache, supporting both old and new transformers API.

    Handles three layouts (checked in this order for backward compatibility):
      - transformers 4.x-5.4: DynamicCache with .key_cache / .value_cache lists
      - transformers >= 5.5.0: DynamicCache with .layers[i].keys / .values
      - Legacy tuple format: tuple of (K, V) pairs per layer

    We check key_cache before layers because the older API is more common
    in existing code/tests, and the new API (5.5+) removes key_cache entirely
    so there is no ambiguity on real DynamicCache objects.
    """
    if hasattr(kv_cache, "key_cache"):
        # transformers 4.x-5.4: DynamicCache with list attributes
        return kv_cache.key_cache[layer_idx], kv_cache.value_cache[layer_idx]
    elif hasattr(kv_cache, "layers"):
        # transformers >= 5.5.0: DynamicCache with DynamicLayer objects
        layer = kv_cache.layers[layer_idx]
        return layer.keys, layer.values
    elif isinstance(kv_cache, tuple):
        # Legacy tuple format
        return kv_cache[layer_idx][0], kv_cache[layer_idx][1]
    else:
        raise TypeError(f"Unsupported KV cache type: {type(kv_cache)}")


def set_kv_tensors(kv_cache: Any, layer_idx: int, key: Any, value: Any) -> None:
    """Set K, V tensors in a KV cache, supporting both old and new transformers API.

    Note: tuple-based caches cannot be modified in-place.
    """
    if hasattr(kv_cache, "key_cache"):
        # transformers 4.x-5.4
        kv_cache.key_cache[layer_idx] = key
        kv_cache.value_cache[layer_idx] = value
    elif hasattr(kv_cache, "layers"):
        # transformers >= 5.5.0
        kv_cache.layers[layer_idx].keys = key
        kv_cache.layers[layer_idx].values = value
    else:
        raise TypeError(
            f"Cannot set KV tensors on {type(kv_cache).__name__}; "
            "tuple-based caches are not modifiable in-place."
        )


def get_num_kv_layers(kv_cache: Any) -> int:
    """Get number of layers in a KV cache object."""
    if hasattr(kv_cache, "key_cache"):
        return len(kv_cache.key_cache)
    elif hasattr(kv_cache, "layers"):
        return len(kv_cache.layers)
    elif isinstance(kv_cache, tuple):
        return len(kv_cache)
    return 0

# ---------------------------------------------------------------------------
# 1. Confidence intervals
# ---------------------------------------------------------------------------


def mean_confidence_interval(
    data: NumericArray,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Compute mean and *t*-distribution confidence interval.

    Parameters
    ----------
    data : array-like
        Sample observations (must contain >= 2 values).
    confidence : float
        Confidence level in (0, 1).

    Returns
    -------
    (mean, ci_lower, ci_upper)
    """
    a = np.asarray(data, dtype=np.float64)
    n = len(a)
    if n < 2:
        raise ValueError(f"Need >= 2 observations for a CI, got {n}")

    mean = float(np.mean(a))
    se = float(stats.sem(a))
    # Two-tailed critical value from the t-distribution
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    half_width = t_crit * se
    return mean, mean - half_width, mean + half_width


def bootstrap_ci(
    data: NumericArray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    statistic: str = "mean",
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Non-parametric bootstrap confidence interval (percentile method).

    Parameters
    ----------
    data : array-like
        Sample observations.
    n_bootstrap : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level in (0, 1).
    statistic : {"mean", "median"}
        Statistic to bootstrap.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
    """
    a = np.asarray(data, dtype=np.float64)
    n = len(a)
    if n < 1:
        raise ValueError("Need >= 1 observation for bootstrap")

    stat_fn = {
        "mean": np.mean,
        "median": np.median,
    }.get(statistic)
    if stat_fn is None:
        raise ValueError(f"Unknown statistic '{statistic}'; use 'mean' or 'median'")

    rng = np.random.default_rng(seed)
    # Vectorised: draw all resamples at once
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resamples = a[indices]
    boot_stats = np.array([stat_fn(row) for row in resamples])

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    point = float(stat_fn(a))
    return point, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# 2. Significance tests
# ---------------------------------------------------------------------------


def paired_ttest(
    a: NumericArray,
    b: NumericArray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Paired-sample *t*-test with Cohen's *d* for matched observations.

    Parameters
    ----------
    a, b : array-like
        Paired measurements (same length).
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    dict with keys ``t_stat``, ``p_value``, ``cohen_d``.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) != len(b_arr):
        raise ValueError(
            f"Arrays must have equal length; got {len(a_arr)} vs {len(b_arr)}"
        )

    result = stats.ttest_rel(a_arr, b_arr, alternative=alternative)
    diff = a_arr - b_arr
    cohen_d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else 0.0
    return {
        "t_stat": float(result.statistic),
        "p_value": float(result.pvalue),
        "cohen_d": cohen_d,
    }


def wilcoxon_test(
    a: NumericArray,
    b: NumericArray,
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to the paired *t*-test; appropriate when
    normality cannot be assumed.

    Parameters
    ----------
    a, b : array-like
        Paired measurements (same length).
    alternative : {"two-sided", "less", "greater"}

    Returns
    -------
    dict with keys ``statistic``, ``p_value``.
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if len(a_arr) != len(b_arr):
        raise ValueError(
            f"Arrays must have equal length; got {len(a_arr)} vs {len(b_arr)}"
        )

    # scipy raises if all differences are zero; handle gracefully
    diff = a_arr - b_arr
    if np.all(diff == 0):
        return {"statistic": 0.0, "p_value": 1.0}

    result = stats.wilcoxon(a_arr, b_arr, alternative=alternative)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def compute_effect_size(
    treatment: NumericArray,
    control: NumericArray,
) -> Dict[str, float]:
    """Cohen's *d* and relative improvement of *treatment* over *control*.

    Uses the pooled standard deviation for Cohen's *d* (independent-sample
    variant), which is the convention for ML ablation tables.

    Returns
    -------
    dict with keys ``cohen_d``, ``relative_improvement``,
    ``treatment_mean``, ``control_mean``.
    """
    t_arr = np.asarray(treatment, dtype=np.float64)
    c_arr = np.asarray(control, dtype=np.float64)

    t_mean = float(np.mean(t_arr))
    c_mean = float(np.mean(c_arr))

    # Pooled standard deviation (Hedges-style denominator)
    nt, nc = len(t_arr), len(c_arr)
    var_t = float(np.var(t_arr, ddof=1)) if nt > 1 else 0.0
    var_c = float(np.var(c_arr, ddof=1)) if nc > 1 else 0.0
    pooled_std = math.sqrt(
        ((nt - 1) * var_t + (nc - 1) * var_c) / max(nt + nc - 2, 1)
    )

    cohen_d = (t_mean - c_mean) / pooled_std if pooled_std > 0 else 0.0
    rel_improvement = (
        (t_mean - c_mean) / abs(c_mean) if abs(c_mean) > 1e-12 else 0.0
    )

    return {
        "cohen_d": cohen_d,
        "relative_improvement": rel_improvement,
        "treatment_mean": t_mean,
        "control_mean": c_mean,
    }


# ---------------------------------------------------------------------------
# 3. Result aggregation
# ---------------------------------------------------------------------------


def aggregate_trials(
    trial_values: NumericArray,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Summarise repeated-trial results with descriptive stats and CI.

    Parameters
    ----------
    trial_values : array-like
        One scalar per trial.
    confidence : float
        Confidence level for the CI.

    Returns
    -------
    dict with keys ``mean``, ``std``, ``median``, ``min``, ``max``,
    ``ci_lower``, ``ci_upper``, ``ci_confidence``, ``n_trials``.
    """
    a = np.asarray(trial_values, dtype=np.float64)
    n = len(a)
    if n == 0:
        raise ValueError("Cannot aggregate zero trials")

    result: Dict[str, Any] = {
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "n_trials": n,
        "ci_confidence": confidence,
    }

    if n >= 2:
        _, ci_lo, ci_hi = mean_confidence_interval(a, confidence=confidence)
        result["ci_lower"] = ci_lo
        result["ci_upper"] = ci_hi
    else:
        # Single trial: CI is degenerate
        result["ci_lower"] = result["mean"]
        result["ci_upper"] = result["mean"]

    return result


def format_with_ci(
    mean: float,
    ci_lo: float,
    ci_hi: float,
    fmt: str = ".2f",
) -> str:
    """Pretty-print ``mean (ci_lo, ci_hi)`` for paper tables.

    >>> format_with_ci(0.954, 0.931, 0.977)
    '0.95 (0.93, 0.98)'
    """
    return f"{mean:{fmt}} ({ci_lo:{fmt}}, {ci_hi:{fmt}})"


# ---------------------------------------------------------------------------
# 4. GPU utilities
# ---------------------------------------------------------------------------


def get_gpu_memory_info() -> List[Dict[str, Any]]:
    """Query per-GPU memory via PyTorch CUDA runtime.

    Returns a list of dicts with keys ``gpu_id``, ``name``, ``total_gb``,
    ``allocated_gb``, ``free_gb``.  Returns an empty list when no CUDA
    device is available.
    """
    try:
        import torch  # local import to keep module usable without torch
    except ImportError:
        logger.warning("PyTorch not available; cannot query GPU memory")
        return []

    if not torch.cuda.is_available():
        return []

    infos: List[Dict[str, Any]] = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        infos.append({
            "gpu_id": i,
            "name": props.name,
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "free_gb": round(total - allocated, 2),
        })
    return infos


def validate_dual_gpu() -> bool:
    """Assert that at least two CUDA GPUs are visible.

    Returns ``True`` on success; raises ``RuntimeError`` otherwise.
    Used as a pre-flight check for speculative-decoding experiments that
    place draft and target models on separate devices.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for GPU validation") from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Speculative decoding requires >= 2 GPUs."
        )
    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        raise RuntimeError(
            f"Need >= 2 GPUs for speculative decoding, but only {n_gpus} visible. "
            "Set CUDA_VISIBLE_DEVICES to expose more devices."
        )
    logger.info("Dual-GPU check passed: %d GPUs available", n_gpus)
    return True


# ---------------------------------------------------------------------------
# 5. Result I/O
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(
    results: Any,
    output_dir: Union[str, Path],
    filename: str,
) -> Path:
    """Atomically save *results* as pretty-printed JSON.

    Writes to a temporary file in the same directory first, then renames,
    so a crash mid-write never leaves a corrupt file.

    Returns the final file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dest = out / filename

    # Write to temp file in same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=str(out), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(results, f, indent=2, cls=_NumpyEncoder)
            f.write("\n")
        shutil.move(tmp_path, str(dest))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Results saved to %s", dest)
    return dest


def load_results(
    results_dir: Union[str, Path],
    pattern: str = "*.json",
) -> Dict[str, Any]:
    """Load all JSON result files matching *pattern* from *results_dir*.

    Returns a dict mapping filename (stem) to its parsed content.
    Skips files that fail to parse and logs a warning.
    """
    rd = Path(results_dir)
    if not rd.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {rd}")

    loaded: Dict[str, Any] = {}
    for path in sorted(rd.glob(pattern)):
        try:
            with open(path, "r") as f:
                loaded[path.stem] = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", path.name, exc)

    logger.info("Loaded %d result files from %s", len(loaded), rd)
    return loaded
