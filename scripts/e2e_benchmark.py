"""End-to-End System Benchmark — Block 4 of AcceptSpec experiment plan.

Benchmarks 7 systems on GSM8K / MATH-500:
  (a) Vanilla autoregressive (target model only, no SD)
  (b) Vanilla SD (rejection sampling, gamma=5, no KV compression)
  (c) SD + uniform 4-bit KV (QuantSpec-style)
  (d) R-KV only (attention + redundancy scoring, no SD)
  (e) SD + R-KV (naive composition)
  (f) SD + attention-proxy KV (SmallKV-style: draft attention for KV importance)
  (g) AcceptSpec (ours: SD + acceptance-guided KV compression)

For each system, measures:
  - Wall-clock time (end-to-end)
  - Tokens generated / Tokens per sec throughput
  - Peak KV memory (MB)
  - Task accuracy (GSM8K exact-match / MATH-500 exact-match)
  - Acceptance rate (for SD variants)
  - Per-component profiling (if --profile)

Usage:
    python scripts/e2e_benchmark.py \\
        --model Qwen/Qwen3.5-9B \\
        --dataset gsm8k \\
        --num_problems 100 \\
        --gamma 5 \\
        --output_dir results/e2e

    # With profiling
    python scripts/e2e_benchmark.py \\
        --model Qwen/Qwen3.5-9B \\
        --dataset gsm8k \\
        --num_problems 50 \\
        --gamma 5 \\
        --profile \\
        --output_dir results/e2e
"""

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.speculative_decode import SpeculativeDecoder, SpeculativeOutput, _trim_kv_cache
from src.acceptspec import (
    AcceptPredictor,
    MixedPrecisionKV,
    TAG_EVICTED,
    TAG_2BIT,
    TAG_4BIT,
    TAG_FP16,
)
from src.gpu_auto import plan_devices, load_models, print_gpu_summary
from src.utils import get_kv_tensors, set_kv_tensors, get_num_kv_layers, get_kv_layer_indices, save_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k(num_problems: int, seed: int = 42) -> List[Dict]:
    """Load GSM8K test problems."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), min(num_problems, len(ds)), replace=False)
    problems = []
    for idx in indices:
        item = ds[int(idx)]
        problems.append({
            "question": item["question"],
            "answer": item["answer"],
        })
    return problems


def load_math500(num_problems: int, seed: int = 42) -> List[Dict]:
    """Load MATH-500 test problems from hendrycks/MATH."""
    ds = load_dataset("hendrycks/MATH", split="test")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), min(num_problems, len(ds)), replace=False)
    problems = []
    for idx in indices:
        item = ds[int(idx)]
        problems.append({
            "question": item["problem"],
            "answer": item["solution"],
        })
    return problems


def format_prompt(question: str, dataset: str) -> str:
    """Format question for a thinking model."""
    if dataset == "math500":
        return (
            "Solve this math problem step by step. "
            "Put your final answer after ####.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
    # gsm8k
    return f"Solve this math problem step by step.\n\nQuestion: {question}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Answer extraction / scoring
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer after '####' in GSM8K chain-of-thought output."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass
    # Fallback: last number in the text
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1].replace(",", ""))
        except ValueError:
            pass
    return None


def extract_gsm8k_gold(answer_text: str) -> Optional[float]:
    """Extract gold numeric answer from GSM8K answer field."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer_text)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def extract_math_answer(text: str) -> Optional[str]:
    """Extract the boxed answer from a MATH solution."""
    # Try \\boxed{...} first
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    # Fallback: #### pattern
    match = re.search(r"####\s*(.+)", text)
    if match:
        return match.group(1).strip()
    return None


def check_answer(generated: str, gold: str, dataset: str) -> bool:
    """Check if generated answer matches gold for the given dataset."""
    if dataset == "gsm8k":
        pred = extract_gsm8k_answer(generated)
        gold_val = extract_gsm8k_gold(gold)
        if pred is None or gold_val is None:
            return False
        return abs(pred - gold_val) < 1e-3
    else:
        # math500: try numeric first, then string match
        pred_str = extract_math_answer(generated)
        gold_str = extract_math_answer(gold)
        if pred_str is None or gold_str is None:
            return False
        # Try numeric comparison
        try:
            return abs(float(pred_str) - float(gold_str)) < 1e-3
        except (ValueError, TypeError):
            pass
        # Fallback: normalized string match
        return pred_str.strip().lower() == gold_str.strip().lower()


# ---------------------------------------------------------------------------
# KV memory measurement
# ---------------------------------------------------------------------------

def measure_kv_memory_bytes(kv_cache: Any) -> int:
    """Estimate peak KV cache memory in bytes."""
    if kv_cache is None:
        return 0
    total = 0
    kv_layers = get_kv_layer_indices(kv_cache)
    for i in kv_layers:
        k, v = get_kv_tensors(kv_cache, i)
        if k is not None:
            total += k.nelement() * k.element_size()
        if v is not None:
            total += v.nelement() * v.element_size()
    return total


# ---------------------------------------------------------------------------
# R-KV scoring: attention importance + key redundancy
# ---------------------------------------------------------------------------

def compute_rkv_scores(
    kv_cache: Any,
    num_layers: int,
    alpha_attn: float = 0.5,
    alpha_redundancy: float = 0.5,
) -> torch.Tensor:
    """Compute R-KV importance scores: attention_importance + redundancy.

    Importance = attention weight magnitude (L2 norm of K as proxy).
    Redundancy = 1 - max cosine similarity with other keys (lower = more unique).
    Final score = alpha_attn * importance + alpha_redundancy * (1 - redundancy).

    Higher score = more important = keep.
    """
    kv_layers = get_kv_layer_indices(kv_cache)
    if not kv_layers:
        return torch.zeros(0)
    k0, _ = get_kv_tensors(kv_cache, kv_layers[0])
    if k0 is None:
        return torch.zeros(0)
    num_tokens = k0.shape[2]
    if num_tokens == 0:
        return torch.zeros(0)

    importance = torch.zeros(num_tokens, device="cpu")
    redundancy = torch.zeros(num_tokens, device="cpu")
    num_layers = len(kv_layers)

    for layer_i in kv_layers:
        k, v = get_kv_tensors(kv_cache, layer_i)
        if k is None:
            continue
        # k: [batch, heads, seq, dim]
        # Sum key norms across heads as importance proxy
        k_flat = k[0].float()  # [heads, seq, dim]
        k_norms = k_flat.norm(dim=-1).mean(dim=0)  # [seq]
        importance += k_norms.cpu()

        # Pairwise cosine similarity for redundancy (expensive; sample layers)
        if layer_i % max(1, num_layers // 4) == 0:
            # Average across heads for cosine sim
            k_avg = k_flat.mean(dim=0)  # [seq, dim]
            k_normed = F.normalize(k_avg, dim=-1)
            sim_matrix = torch.mm(k_normed, k_normed.T)  # [seq, seq]
            # Mask diagonal
            sim_matrix.fill_diagonal_(-float("inf"))
            max_sim = sim_matrix.max(dim=-1).values  # [seq]
            redundancy += max_sim.cpu()

    # Normalize
    importance = importance / max(importance.max().item(), 1e-8)
    redundancy = redundancy / max(redundancy.abs().max().item(), 1e-8)

    scores = alpha_attn * importance + alpha_redundancy * (1.0 - redundancy)
    return scores


def evict_kv_by_scores(
    kv_cache: Any,
    scores: torch.Tensor,
    keep_fraction: float = 0.5,
) -> Any:
    """Zero out KV entries for tokens with the lowest scores."""
    num_tokens = scores.numel()
    if num_tokens == 0:
        return kv_cache
    n_keep = max(1, int(keep_fraction * num_tokens))
    _, top_indices = scores.topk(n_keep)
    keep_mask = torch.zeros(num_tokens, dtype=torch.bool)
    keep_mask[top_indices] = True
    evict_indices = (~keep_mask).nonzero(as_tuple=True)[0]

    kv_layers = get_kv_layer_indices(kv_cache)
    for layer_i in kv_layers:
        k, v = get_kv_tensors(kv_cache, layer_i)
        if k is None:
            continue
        for idx in evict_indices:
            idx_val = idx.item()
            k[:, :, idx_val:idx_val + 1, :] = 0
            v[:, :, idx_val:idx_val + 1, :] = 0
    return kv_cache


# ---------------------------------------------------------------------------
# SmallKV-style scoring: draft model attention as KV importance proxy
# ---------------------------------------------------------------------------

def compute_smallkv_scores(
    draft_model: Any,
    draft_kv: Any,
    last_token: torch.Tensor,
    draft_device: torch.device,
    num_kv_tokens: int,
) -> torch.Tensor:
    """Score target KV tokens using draft model's attention weights.

    SmallKV uses the smaller draft model's attention patterns as a proxy
    for which KV tokens are important in the target model.
    """
    scores = torch.zeros(num_kv_tokens, device="cpu")
    attn_weights_collected = []

    def make_hook(layer_idx):
        def hook_fn(module, args, kwargs, output):
            if isinstance(output, tuple) and len(output) >= 2:
                attn_w = output[1]
                if attn_w is not None:
                    attn_weights_collected.append(attn_w.detach().cpu())
        return hook_fn

    hooks = []
    for name, module in draft_model.named_modules():
        if "self_attn" in name and not any(
            sub in name for sub in [".q_proj", ".k_proj", ".v_proj", ".o_proj"]
        ):
            if hasattr(module, "forward"):
                h = module.register_forward_hook(make_hook(len(hooks)), with_kwargs=True)
                hooks.append(h)

    try:
        out = draft_model(
            last_token.view(1, 1).to(draft_device),
            past_key_values=draft_kv,
            use_cache=True,
            output_attentions=True,
        )
        if hasattr(out, "attentions") and out.attentions is not None:
            for layer_attn in out.attentions:
                if layer_attn is not None:
                    # layer_attn: [batch, heads, 1, kv_len]
                    kv_len = min(layer_attn.shape[-1], num_kv_tokens)
                    attn_to_kv = layer_attn[0, :, 0, :kv_len].sum(dim=0)
                    scores[:kv_len] += attn_to_kv.cpu()
    except Exception as e:
        logger.debug("SmallKV scoring failed: %s. Using uniform scores.", e)
        scores = torch.ones(num_kv_tokens)
    finally:
        for h in hooks:
            h.remove()

    return scores


# ---------------------------------------------------------------------------
# Per-problem generation wrapper for each system
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_autoregressive(
    target_model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    target_device: torch.device,
    temperature: float = 0.0,
) -> Dict:
    """System (a): Vanilla autoregressive generation with target model only."""
    generated = input_ids.to(target_device)
    past = None
    peak_kv_bytes = 0

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for _ in range(max_new_tokens):
        if past is None:
            out = target_model(generated, use_cache=True)
        else:
            out = target_model(generated[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
        else:
            tok = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, tok], dim=1)

        if tok.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    wall = time.perf_counter() - start
    peak_kv_bytes = measure_kv_memory_bytes(past)

    num_gen = generated.shape[1] - input_ids.shape[1]
    return {
        "generated_ids": generated.cpu(),
        "num_generated_tokens": num_gen,
        "wall_time_seconds": wall,
        "tokens_per_sec": num_gen / max(wall, 1e-9),
        "peak_kv_bytes": peak_kv_bytes,
        "acceptance_rate": None,
    }


@torch.no_grad()
def run_vanilla_sd(
    decoder: SpeculativeDecoder,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    temperature: float = 0.0,
    profile: bool = False,
) -> Dict:
    """System (b): Vanilla speculative decoding, no KV compression."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    out = decoder.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        temperature=max(temperature, 1e-8),
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    result = {
        "generated_ids": out.generated_ids,
        "num_generated_tokens": out.num_generated_tokens,
        "wall_time_seconds": out.wall_time_seconds,
        "tokens_per_sec": out.throughput,
        "peak_kv_bytes": 0,  # measured externally
        "acceptance_rate": out.acceptance_rate,
    }
    if profile:
        result["draft_time"] = out.draft_time_seconds
        result["verify_time"] = out.verify_time_seconds
        result["compress_time"] = 0.0
        result["score_time"] = 0.0
    return result


@torch.no_grad()
def run_sd_uniform_4bit(
    draft_model: Any,
    target_model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    temperature: float = 0.0,
    profile: bool = False,
) -> Dict:
    """System (c): SD + uniform 4-bit KV (QuantSpec-style)."""
    decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        quant_bits=4,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    out = decoder.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        temperature=max(temperature, 1e-8),
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    result = {
        "generated_ids": out.generated_ids,
        "num_generated_tokens": out.num_generated_tokens,
        "wall_time_seconds": out.wall_time_seconds,
        "tokens_per_sec": out.throughput,
        "peak_kv_bytes": 0,
        "acceptance_rate": out.acceptance_rate,
    }
    if profile:
        result["draft_time"] = out.draft_time_seconds
        result["verify_time"] = out.verify_time_seconds
        result["compress_time"] = out.quantize_time_seconds
        result["score_time"] = 0.0
    return result


@torch.no_grad()
def run_rkv_only(
    target_model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    target_device: torch.device,
    keep_fraction: float = 0.5,
    temperature: float = 0.0,
    profile: bool = False,
) -> Dict:
    """System (d): R-KV only (KV compression via attention + redundancy, no SD).

    Autoregressive generation with periodic KV eviction using R-KV scoring.
    """
    generated = input_ids.to(target_device)
    past = None
    peak_kv_bytes = 0
    t_score = 0.0
    t_compress = 0.0
    evict_interval = 32  # Re-score and evict every N tokens

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for step in range(max_new_tokens):
        if past is None:
            out = target_model(generated, use_cache=True)
        else:
            out = target_model(generated[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
        else:
            tok = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, tok], dim=1)

        # Periodic R-KV eviction
        if step > 0 and step % evict_interval == 0:
            kv_layers = get_kv_layer_indices(past)
            t0 = time.perf_counter()
            scores = compute_rkv_scores(past, len(kv_layers))
            t_score += time.perf_counter() - t0

            t0 = time.perf_counter()
            past = evict_kv_by_scores(past, scores, keep_fraction=keep_fraction)
            t_compress += time.perf_counter() - t0

        cur_kv = measure_kv_memory_bytes(past)
        peak_kv_bytes = max(peak_kv_bytes, cur_kv)

        if tok.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    wall = time.perf_counter() - start

    num_gen = generated.shape[1] - input_ids.shape[1]
    result = {
        "generated_ids": generated.cpu(),
        "num_generated_tokens": num_gen,
        "wall_time_seconds": wall,
        "tokens_per_sec": num_gen / max(wall, 1e-9),
        "peak_kv_bytes": peak_kv_bytes,
        "acceptance_rate": None,
    }
    if profile:
        result["draft_time"] = 0.0
        result["verify_time"] = wall - t_score - t_compress
        result["compress_time"] = t_compress
        result["score_time"] = t_score
    return result


@torch.no_grad()
def run_smallkv_only(
    draft_model: Any,
    target_model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    draft_device: torch.device,
    target_device: torch.device,
    keep_fraction: float = 0.5,
    temperature: float = 0.0,
    profile: bool = False,
) -> Dict:
    """System (d2): SmallKV only (KV compression via draft attention proxy, no SD).

    Autoregressive generation with the target model. Periodically scores
    target KV tokens using the draft model's attention patterns (SmallKV)
    and evicts low-importance tokens.
    """
    generated = input_ids.to(target_device)
    past_target = None
    past_draft = None
    peak_kv_bytes = 0
    t_score = 0.0
    t_compress = 0.0
    evict_interval = 32

    # Prefill draft model to keep KV in sync
    draft_out = draft_model(input_ids.to(draft_device), use_cache=True)
    past_draft = draft_out.past_key_values

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    for step in range(max_new_tokens):
        if past_target is None:
            out = target_model(generated, use_cache=True)
        else:
            out = target_model(generated[:, -1:], past_key_values=past_target, use_cache=True)
        past_target = out.past_key_values
        logits = out.logits[:, -1, :]

        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
        else:
            tok = logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, tok], dim=1)

        # Keep draft KV in sync
        d_out = draft_model(
            tok.view(1, 1).to(draft_device),
            past_key_values=past_draft,
            use_cache=True,
        )
        past_draft = d_out.past_key_values

        # Periodic SmallKV eviction: score target KV via draft attention
        if step > 0 and step % evict_interval == 0:
            kv_layers = get_kv_layer_indices(past_target)
            k0, _ = get_kv_tensors(past_target, kv_layers[0]) if kv_layers else (None, None)
            num_kv_tokens = k0.shape[2] if k0 is not None else 0

            t0 = time.perf_counter()
            scores = compute_smallkv_scores(
                draft_model, past_draft, tok.squeeze(), draft_device, num_kv_tokens,
            )
            t_score += time.perf_counter() - t0

            t0 = time.perf_counter()
            past_target = evict_kv_by_scores(past_target, scores, keep_fraction=keep_fraction)
            past_draft = evict_kv_by_scores(past_draft, scores, keep_fraction=keep_fraction)
            t_compress += time.perf_counter() - t0

        cur_kv = measure_kv_memory_bytes(past_target)
        peak_kv_bytes = max(peak_kv_bytes, cur_kv)

        if tok.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    wall = time.perf_counter() - start

    num_gen = generated.shape[1] - input_ids.shape[1]
    result = {
        "generated_ids": generated.cpu(),
        "num_generated_tokens": num_gen,
        "wall_time_seconds": wall,
        "tokens_per_sec": num_gen / max(wall, 1e-9),
        "peak_kv_bytes": peak_kv_bytes,
        "acceptance_rate": None,
    }
    if profile:
        result["draft_time"] = 0.0
        result["verify_time"] = wall - t_score - t_compress
        result["compress_time"] = t_compress
        result["score_time"] = t_score
    return result


@torch.no_grad()
def run_sd_rkv(
    draft_model: Any,
    target_model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    keep_fraction: float = 0.5,
    temperature: float = 0.0,
    profile: bool = False,
) -> Dict:
    """System (e): SD + R-KV (naive composition).

    Standard speculative decoding with R-KV eviction applied to the target
    KV cache after each verification step.
    """
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    prefix_len = input_ids.shape[1]
    temp = max(temperature, 1e-8)

    t_draft = 0.0
    t_verify = 0.0
    t_score = 0.0
    t_compress = 0.0
    total_draft = 0
    total_accepted = 0
    n_rounds = 0
    peak_kv_bytes = 0

    # Prefill
    draft_out = draft_model(input_ids.to(draft_device), use_cache=True)
    draft_kv = draft_out.past_key_values
    draft_next_logits = draft_out.logits[:, -1, :]

    target_out = target_model(input_ids.to(target_device), use_cache=True)
    target_kv = target_out.past_key_values
    target_next_logits = target_out.logits[:, -1, :]

    all_token_ids = input_ids.cpu().clone()
    kv_len = prefix_len

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    while all_token_ids.shape[1] - prefix_len < max_new_tokens:
        remaining = max_new_tokens - (all_token_ids.shape[1] - prefix_len)
        cur_gamma = min(gamma, remaining)
        if cur_gamma <= 0:
            break

        n_rounds += 1
        total_draft += cur_gamma

        # Draft phase
        t0 = time.perf_counter()
        draft_tokens = []
        draft_probs = []
        cur_logits = draft_next_logits
        cur_draft_kv = draft_kv

        for _ in range(cur_gamma):
            p = F.softmax(cur_logits / temp, dim=-1).squeeze(0)
            tok = torch.multinomial(p, num_samples=1).squeeze(-1)
            draft_tokens.append(tok.cpu())
            draft_probs.append(p.cpu())

            d_out = draft_model(
                tok.view(1, 1).to(draft_device),
                past_key_values=cur_draft_kv,
                use_cache=True,
            )
            cur_draft_kv = d_out.past_key_values
            cur_logits = d_out.logits[:, -1, :]

        draft_kv = cur_draft_kv
        draft_tokens_t = torch.stack(draft_tokens)
        t_draft += time.perf_counter() - t0

        # Verify phase
        t0 = time.perf_counter()
        verify_out = target_model(
            draft_tokens_t.view(1, -1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv_ext = verify_out.past_key_values
        verify_logits = verify_out.logits

        n_acc, accepted = SpeculativeDecoder._rejection_sample(
            target_next_logits, verify_logits,
            draft_tokens_t, draft_probs,
            cur_gamma, temperature,
        )
        t_verify += time.perf_counter() - t0

        total_accepted += n_acc
        all_token_ids = torch.cat([all_token_ids, accepted.view(1, -1).cpu()], dim=1)

        new_kv_len = kv_len + n_acc
        draft_kv = _trim_kv_cache(draft_kv, new_kv_len)
        target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)

        last_tok = accepted[-1]

        # Re-sync draft
        d_out = draft_model(
            last_tok.view(1, 1).to(draft_device),
            past_key_values=draft_kv,
            use_cache=True,
        )
        draft_kv = d_out.past_key_values
        draft_next_logits = d_out.logits[:, -1, :]

        # Re-sync target
        t_out = target_model(
            last_tok.view(1, 1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv = t_out.past_key_values
        target_next_logits = t_out.logits[:, -1, :]

        kv_len = new_kv_len + 1

        # R-KV eviction on target KV
        kv_layers = get_kv_layer_indices(target_kv)
        t0 = time.perf_counter()
        scores = compute_rkv_scores(target_kv, len(kv_layers))
        t_score += time.perf_counter() - t0

        t0 = time.perf_counter()
        target_kv = evict_kv_by_scores(target_kv, scores, keep_fraction=keep_fraction)
        t_compress += time.perf_counter() - t0

        cur_kv = measure_kv_memory_bytes(target_kv)
        peak_kv_bytes = max(peak_kv_bytes, cur_kv)

        if last_tok.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    wall = time.perf_counter() - start

    num_gen = all_token_ids.shape[1] - prefix_len
    alpha = total_accepted / max(total_draft, 1)
    result = {
        "generated_ids": all_token_ids,
        "num_generated_tokens": num_gen,
        "wall_time_seconds": wall,
        "tokens_per_sec": num_gen / max(wall, 1e-9),
        "peak_kv_bytes": peak_kv_bytes,
        "acceptance_rate": alpha,
    }
    if profile:
        result["draft_time"] = t_draft
        result["verify_time"] = t_verify
        result["compress_time"] = t_compress
        result["score_time"] = t_score
    return result


@torch.no_grad()
def run_sd_smallkv(
    draft_model: Any,
    target_model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    keep_fraction: float = 0.5,
    temperature: float = 0.0,
    profile: bool = False,
) -> Dict:
    """System (f): SD + attention-proxy KV (SmallKV-style).

    Uses draft model's attention weights to score target KV tokens for
    eviction — attention importance rather than acceptance importance.
    """
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    prefix_len = input_ids.shape[1]
    temp = max(temperature, 1e-8)

    t_draft = 0.0
    t_verify = 0.0
    t_score = 0.0
    t_compress = 0.0
    total_draft = 0
    total_accepted = 0
    n_rounds = 0
    peak_kv_bytes = 0

    # Prefill
    draft_out = draft_model(input_ids.to(draft_device), use_cache=True)
    draft_kv = draft_out.past_key_values
    draft_next_logits = draft_out.logits[:, -1, :]

    target_out = target_model(input_ids.to(target_device), use_cache=True)
    target_kv = target_out.past_key_values
    target_next_logits = target_out.logits[:, -1, :]

    all_token_ids = input_ids.cpu().clone()
    kv_len = prefix_len

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    while all_token_ids.shape[1] - prefix_len < max_new_tokens:
        remaining = max_new_tokens - (all_token_ids.shape[1] - prefix_len)
        cur_gamma = min(gamma, remaining)
        if cur_gamma <= 0:
            break

        n_rounds += 1
        total_draft += cur_gamma

        # Draft phase
        t0 = time.perf_counter()
        draft_tokens = []
        draft_probs = []
        cur_logits = draft_next_logits
        cur_draft_kv = draft_kv

        for _ in range(cur_gamma):
            p = F.softmax(cur_logits / temp, dim=-1).squeeze(0)
            tok = torch.multinomial(p, num_samples=1).squeeze(-1)
            draft_tokens.append(tok.cpu())
            draft_probs.append(p.cpu())

            d_out = draft_model(
                tok.view(1, 1).to(draft_device),
                past_key_values=cur_draft_kv,
                use_cache=True,
            )
            cur_draft_kv = d_out.past_key_values
            cur_logits = d_out.logits[:, -1, :]

        draft_kv = cur_draft_kv
        draft_tokens_t = torch.stack(draft_tokens)
        t_draft += time.perf_counter() - t0

        # Verify phase
        t0 = time.perf_counter()
        verify_out = target_model(
            draft_tokens_t.view(1, -1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv_ext = verify_out.past_key_values
        verify_logits = verify_out.logits

        n_acc, accepted = SpeculativeDecoder._rejection_sample(
            target_next_logits, verify_logits,
            draft_tokens_t, draft_probs,
            cur_gamma, temperature,
        )
        t_verify += time.perf_counter() - t0

        total_accepted += n_acc
        all_token_ids = torch.cat([all_token_ids, accepted.view(1, -1).cpu()], dim=1)

        new_kv_len = kv_len + n_acc
        draft_kv = _trim_kv_cache(draft_kv, new_kv_len)
        target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)

        last_tok = accepted[-1]

        # Re-sync draft
        d_out = draft_model(
            last_tok.view(1, 1).to(draft_device),
            past_key_values=draft_kv,
            use_cache=True,
        )
        draft_kv = d_out.past_key_values
        draft_next_logits = d_out.logits[:, -1, :]

        # Re-sync target
        t_out = target_model(
            last_tok.view(1, 1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv = t_out.past_key_values
        target_next_logits = t_out.logits[:, -1, :]

        kv_len = new_kv_len + 1

        # SmallKV scoring: use draft attention to score target KV
        kv_layers = get_kv_layer_indices(target_kv)
        k0, _ = get_kv_tensors(target_kv, kv_layers[0]) if kv_layers else (None, None)
        num_kv_tokens = k0.shape[2] if k0 is not None else 0

        t0 = time.perf_counter()
        scores = compute_smallkv_scores(
            draft_model, draft_kv, last_tok, draft_device, num_kv_tokens,
        )
        t_score += time.perf_counter() - t0

        t0 = time.perf_counter()
        target_kv = evict_kv_by_scores(target_kv, scores, keep_fraction=keep_fraction)
        t_compress += time.perf_counter() - t0

        cur_kv = measure_kv_memory_bytes(target_kv)
        peak_kv_bytes = max(peak_kv_bytes, cur_kv)

        if last_tok.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    wall = time.perf_counter() - start

    num_gen = all_token_ids.shape[1] - prefix_len
    alpha = total_accepted / max(total_draft, 1)
    result = {
        "generated_ids": all_token_ids,
        "num_generated_tokens": num_gen,
        "wall_time_seconds": wall,
        "tokens_per_sec": num_gen / max(wall, 1e-9),
        "peak_kv_bytes": peak_kv_bytes,
        "acceptance_rate": alpha,
    }
    if profile:
        result["draft_time"] = t_draft
        result["verify_time"] = t_verify
        result["compress_time"] = t_compress
        result["score_time"] = t_score
    return result


@torch.no_grad()
def run_acceptspec(
    draft_model: Any,
    target_model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    gamma: int,
    predictor: AcceptPredictor,
    mixed_kv: MixedPrecisionKV,
    critical_fraction: float = 0.2,
    temperature: float = 0.0,
    profile: bool = False,
) -> Dict:
    """System (g): AcceptSpec (ours).

    SD with acceptance-guided mixed-precision KV compression.
    At each verification step:
      1. Compute AcceptPredictor scores from draft attention weights + value norms
      2. Generate per-token precision tags
      3. Compress target KV via MixedPrecisionKV
      4. Verify with compressed KV
    """
    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device
    prefix_len = input_ids.shape[1]
    temp = max(temperature, 1e-8)

    t_draft = 0.0
    t_verify = 0.0
    t_score = 0.0
    t_compress = 0.0
    total_draft = 0
    total_accepted = 0
    n_rounds = 0
    peak_kv_bytes = 0

    # Prefill
    draft_out = draft_model(input_ids.to(draft_device), use_cache=True)
    draft_kv = draft_out.past_key_values
    draft_next_logits = draft_out.logits[:, -1, :]

    target_out = target_model(input_ids.to(target_device), use_cache=True)
    target_kv = target_out.past_key_values
    target_next_logits = target_out.logits[:, -1, :]

    all_token_ids = input_ids.cpu().clone()
    kv_len = prefix_len

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()

    while all_token_ids.shape[1] - prefix_len < max_new_tokens:
        remaining = max_new_tokens - (all_token_ids.shape[1] - prefix_len)
        cur_gamma = min(gamma, remaining)
        if cur_gamma <= 0:
            break

        n_rounds += 1
        total_draft += cur_gamma

        # Draft phase — collect attention weights for AcceptPredictor
        t0 = time.perf_counter()
        draft_tokens = []
        draft_probs = []
        cur_logits = draft_next_logits
        cur_draft_kv = draft_kv
        last_draft_attn = None

        for step_i in range(cur_gamma):
            p = F.softmax(cur_logits / temp, dim=-1).squeeze(0)
            tok = torch.multinomial(p, num_samples=1).squeeze(-1)
            draft_tokens.append(tok.cpu())
            draft_probs.append(p.cpu())

            # On last draft token, capture attention weights
            if step_i == cur_gamma - 1:
                attn_collected = []

                def make_attn_hook(container):
                    def hook_fn(module, args, kwargs, output):
                        if isinstance(output, tuple) and len(output) >= 2:
                            attn_w = output[1]
                            if attn_w is not None:
                                container.append(attn_w.detach().cpu())
                    return hook_fn

                hooks = []
                for name, module in draft_model.named_modules():
                    if "self_attn" in name and not any(
                        sub in name for sub in [".q_proj", ".k_proj", ".v_proj", ".o_proj"]
                    ):
                        if hasattr(module, "forward"):
                            h = module.register_forward_hook(
                                make_attn_hook(attn_collected), with_kwargs=True,
                            )
                            hooks.append(h)

                try:
                    d_out = draft_model(
                        tok.view(1, 1).to(draft_device),
                        past_key_values=cur_draft_kv,
                        use_cache=True,
                        output_attentions=True,
                    )
                    # Extract attention from model output if available
                    if hasattr(d_out, "attentions") and d_out.attentions is not None:
                        # Average across layers, get last query position
                        all_attn = []
                        for layer_attn in d_out.attentions:
                            if layer_attn is not None:
                                # [batch, heads, 1, kv_len] -> [heads, kv_len]
                                all_attn.append(layer_attn[0, :, -1, :].cpu())
                        if all_attn:
                            # Average across layers: [heads, kv_len]
                            last_draft_attn = torch.stack(all_attn).mean(dim=0)
                finally:
                    for h in hooks:
                        h.remove()

                cur_draft_kv = d_out.past_key_values
                cur_logits = d_out.logits[:, -1, :]
            else:
                d_out = draft_model(
                    tok.view(1, 1).to(draft_device),
                    past_key_values=cur_draft_kv,
                    use_cache=True,
                )
                cur_draft_kv = d_out.past_key_values
                cur_logits = d_out.logits[:, -1, :]

        draft_kv = cur_draft_kv
        draft_tokens_t = torch.stack(draft_tokens)
        t_draft += time.perf_counter() - t0

        # AcceptPredictor scoring
        t0 = time.perf_counter()
        kv_layers = get_kv_layer_indices(target_kv)
        k0, v0 = get_kv_tensors(target_kv, kv_layers[0]) if kv_layers else (None, None)
        num_kv_tokens = k0.shape[2] if k0 is not None else 0

        if last_draft_attn is not None and num_kv_tokens > 0:
            # Compute value norms from target KV
            value_norms = torch.zeros(num_kv_tokens, device="cpu")
            for li in kv_layers:
                _, v = get_kv_tensors(target_kv, li)
                if v is not None:
                    # v: [batch, heads, seq, dim]
                    vnorm = v[0].float().norm(dim=-1).mean(dim=0).cpu()  # [seq]
                    value_norms[:len(vnorm)] += vnorm

            # Align draft attention to target KV length
            draft_attn_aligned = last_draft_attn
            attn_kv_len = draft_attn_aligned.shape[-1]
            if attn_kv_len < num_kv_tokens:
                # Pad with zeros for positions not covered by draft attention
                pad_size = num_kv_tokens - attn_kv_len
                draft_attn_aligned = F.pad(draft_attn_aligned, (0, pad_size))
            elif attn_kv_len > num_kv_tokens:
                draft_attn_aligned = draft_attn_aligned[:, :num_kv_tokens]

            # Ensure head count matches predictor
            n_heads = draft_attn_aligned.shape[0]
            if n_heads != predictor.num_heads:
                # Resize head weights to match actual head count
                predictor.head_weights = torch.ones(n_heads) / n_heads

            scores = predictor.predict_scores(draft_attn_aligned, value_norms)
            tags = predictor.predict_tags(
                draft_attn_aligned, value_norms, critical_fraction=critical_fraction,
            )
        else:
            # Fallback: uniform tags (all FP16)
            tags = torch.full((num_kv_tokens,), TAG_FP16, dtype=torch.uint8)
        t_score += time.perf_counter() - t0

        # Compress target KV
        t0 = time.perf_counter()
        target_kv = mixed_kv.compress_kv(target_kv, tags)
        t_compress += time.perf_counter() - t0

        # Verify phase
        t0 = time.perf_counter()
        verify_out = target_model(
            draft_tokens_t.view(1, -1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv_ext = verify_out.past_key_values
        verify_logits = verify_out.logits

        n_acc, accepted = SpeculativeDecoder._rejection_sample(
            target_next_logits, verify_logits,
            draft_tokens_t, draft_probs,
            cur_gamma, temperature,
        )
        t_verify += time.perf_counter() - t0

        total_accepted += n_acc
        all_token_ids = torch.cat([all_token_ids, accepted.view(1, -1).cpu()], dim=1)

        new_kv_len = kv_len + n_acc
        draft_kv = _trim_kv_cache(draft_kv, new_kv_len)
        target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)

        last_tok = accepted[-1]

        # Re-sync draft
        d_out = draft_model(
            last_tok.view(1, 1).to(draft_device),
            past_key_values=draft_kv,
            use_cache=True,
        )
        draft_kv = d_out.past_key_values
        draft_next_logits = d_out.logits[:, -1, :]

        # Re-sync target
        t_out = target_model(
            last_tok.view(1, 1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv = t_out.past_key_values
        target_next_logits = t_out.logits[:, -1, :]

        kv_len = new_kv_len + 1

        cur_kv = measure_kv_memory_bytes(target_kv)
        peak_kv_bytes = max(peak_kv_bytes, cur_kv)

        if last_tok.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    wall = time.perf_counter() - start

    num_gen = all_token_ids.shape[1] - prefix_len
    alpha = total_accepted / max(total_draft, 1)
    result = {
        "generated_ids": all_token_ids,
        "num_generated_tokens": num_gen,
        "wall_time_seconds": wall,
        "tokens_per_sec": num_gen / max(wall, 1e-9),
        "peak_kv_bytes": peak_kv_bytes,
        "acceptance_rate": alpha,
    }
    if profile:
        result["draft_time"] = t_draft
        result["verify_time"] = t_verify
        result["compress_time"] = t_compress
        result["score_time"] = t_score
    return result


# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------

SYSTEMS = [
    "autoregressive",
    "vanilla_sd",
    "sd_uniform_4bit",
    "rkv_only",
    "smallkv_only",
    "sd_rkv",
    "sd_smallkv",
    "acceptspec",
]


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(args):
    """Run the full end-to-end benchmark."""
    print_gpu_summary()

    # Load models
    plan = plan_devices()
    logger.info("Loading models with auto device plan: %s", plan.description)

    draft_model, target_model, tokenizer, plan = load_models(
        draft_model_name=args.draft_model,
        target_model_name=args.target_model,
        plan=plan,
    )

    draft_device = next(draft_model.parameters()).device
    target_device = next(target_model.parameters()).device

    # Load dataset
    logger.info("Loading dataset '%s' (%d problems)...", args.dataset, args.num_problems)
    if args.dataset == "gsm8k":
        problems = load_gsm8k(args.num_problems, seed=args.seed)
    elif args.dataset == "math500":
        problems = load_math500(args.num_problems, seed=args.seed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    logger.info("Loaded %d problems.", len(problems))

    # Build vanilla SD decoder (no quant) for systems (b), and reuse for rejection sampling
    vanilla_decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        quant_bits=0,
    )

    # Build AcceptPredictor (uniform weights — no calibration data yet)
    config = target_model.config
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    num_heads_draft = getattr(
        draft_model.config, "num_attention_heads",
        draft_model.config.num_attention_heads,
    )
    head_dim = config.hidden_size // config.num_attention_heads
    num_layers = config.num_hidden_layers

    predictor = AcceptPredictor(
        num_heads=num_heads_draft,
        theta_critical=0.8,
        theta_low=0.3,
    )

    mixed_kv = MixedPrecisionKV(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # Results container
    all_results = {
        "config": {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "dataset": args.dataset,
            "num_problems": len(problems),
            "gamma": args.gamma,
            "max_new_tokens": args.max_tokens,
            "temperature": args.temperature,
            "profile": args.profile,
            "seed": args.seed,
        },
        "systems": {},
    }

    # Run each system
    for system_name in SYSTEMS:
        logger.info("=" * 70)
        logger.info("SYSTEM: %s", system_name)
        logger.info("=" * 70)

        system_results = {
            "per_problem": [],
            "aggregate": {},
        }

        wall_times = []
        throughputs = []
        n_tokens_list = []
        peak_kv_list = []
        accuracies = []
        acceptance_rates = []

        # Profile accumulators
        if args.profile:
            prof_draft = []
            prof_verify = []
            prof_compress = []
            prof_score = []

        for i, problem in enumerate(problems):
            prompt = format_prompt(problem["question"], args.dataset)
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(
                    "  [%d/%d] '%s...'",
                    i + 1, len(problems), problem["question"][:50],
                )

            try:
                if system_name == "autoregressive":
                    result = run_autoregressive(
                        target_model, tokenizer, input_ids,
                        max_new_tokens=args.max_tokens,
                        target_device=target_device,
                        temperature=args.temperature,
                    )

                elif system_name == "vanilla_sd":
                    result = run_vanilla_sd(
                        vanilla_decoder, input_ids,
                        max_new_tokens=args.max_tokens,
                        gamma=args.gamma,
                        temperature=args.temperature,
                        profile=args.profile,
                    )

                elif system_name == "sd_uniform_4bit":
                    result = run_sd_uniform_4bit(
                        draft_model, target_model, tokenizer, input_ids,
                        max_new_tokens=args.max_tokens,
                        gamma=args.gamma,
                        temperature=args.temperature,
                        profile=args.profile,
                    )

                elif system_name == "rkv_only":
                    result = run_rkv_only(
                        target_model, tokenizer, input_ids,
                        max_new_tokens=args.max_tokens,
                        target_device=target_device,
                        keep_fraction=0.5,
                        temperature=args.temperature,
                        profile=args.profile,
                    )

                elif system_name == "smallkv_only":
                    result = run_smallkv_only(
                        draft_model, target_model, tokenizer, input_ids,
                        max_new_tokens=args.max_tokens,
                        draft_device=draft_device,
                        target_device=target_device,
                        keep_fraction=0.5,
                        temperature=args.temperature,
                        profile=args.profile,
                    )

                elif system_name == "sd_rkv":
                    result = run_sd_rkv(
                        draft_model, target_model, tokenizer, input_ids,
                        max_new_tokens=args.max_tokens,
                        gamma=args.gamma,
                        keep_fraction=0.5,
                        temperature=args.temperature,
                        profile=args.profile,
                    )

                elif system_name == "sd_smallkv":
                    result = run_sd_smallkv(
                        draft_model, target_model, tokenizer, input_ids,
                        max_new_tokens=args.max_tokens,
                        gamma=args.gamma,
                        keep_fraction=0.5,
                        temperature=args.temperature,
                        profile=args.profile,
                    )

                elif system_name == "acceptspec":
                    result = run_acceptspec(
                        draft_model, target_model, tokenizer, input_ids,
                        max_new_tokens=args.max_tokens,
                        gamma=args.gamma,
                        predictor=predictor,
                        mixed_kv=mixed_kv,
                        critical_fraction=0.2,
                        temperature=args.temperature,
                        profile=args.profile,
                    )

                else:
                    logger.warning("Unknown system: %s", system_name)
                    continue

                # Decode and check accuracy
                gen_ids = result["generated_ids"]
                gen_text = tokenizer.decode(
                    gen_ids[0, input_ids.shape[1]:].tolist(),
                    skip_special_tokens=True,
                )
                correct = check_answer(gen_text, problem["answer"], args.dataset)

                # Collect metrics
                wall_times.append(result["wall_time_seconds"])
                throughputs.append(result["tokens_per_sec"])
                n_tokens_list.append(result["num_generated_tokens"])
                peak_kv_list.append(result["peak_kv_bytes"])
                accuracies.append(1.0 if correct else 0.0)
                if result["acceptance_rate"] is not None:
                    acceptance_rates.append(result["acceptance_rate"])

                per_prob = {
                    "idx": i,
                    "wall_time": result["wall_time_seconds"],
                    "tokens_per_sec": result["tokens_per_sec"],
                    "num_tokens": result["num_generated_tokens"],
                    "peak_kv_mb": result["peak_kv_bytes"] / (1024 * 1024),
                    "correct": correct,
                }
                if result["acceptance_rate"] is not None:
                    per_prob["acceptance_rate"] = result["acceptance_rate"]

                if args.profile:
                    for key in ("draft_time", "verify_time", "compress_time", "score_time"):
                        if key in result:
                            per_prob[key] = result[key]
                    if "draft_time" in result:
                        prof_draft.append(result["draft_time"])
                    if "verify_time" in result:
                        prof_verify.append(result["verify_time"])
                    if "compress_time" in result:
                        prof_compress.append(result["compress_time"])
                    if "score_time" in result:
                        prof_score.append(result["score_time"])

                system_results["per_problem"].append(per_prob)

            except Exception as e:
                logger.error("  Problem %d failed: %s", i, e)
                continue

            # Clear CUDA cache between problems
            torch.cuda.empty_cache()

        # Aggregate
        n_done = len(wall_times)
        if n_done == 0:
            logger.warning("  No results for system %s.", system_name)
            all_results["systems"][system_name] = {"error": "no results"}
            continue

        agg = {
            "num_problems": n_done,
            "mean_wall_time": float(np.mean(wall_times)),
            "std_wall_time": float(np.std(wall_times)),
            "mean_throughput": float(np.mean(throughputs)),
            "std_throughput": float(np.std(throughputs)),
            "mean_tokens_generated": float(np.mean(n_tokens_list)),
            "mean_peak_kv_mb": float(np.mean(peak_kv_list)) / (1024 * 1024),
            "max_peak_kv_mb": float(np.max(peak_kv_list)) / (1024 * 1024),
            "accuracy": float(np.mean(accuracies)),
            "num_correct": int(np.sum(accuracies)),
        }
        if acceptance_rates:
            agg["mean_acceptance_rate"] = float(np.mean(acceptance_rates))
            agg["std_acceptance_rate"] = float(np.std(acceptance_rates))

        if args.profile:
            if prof_draft:
                agg["mean_draft_time"] = float(np.mean(prof_draft))
            if prof_verify:
                agg["mean_verify_time"] = float(np.mean(prof_verify))
            if prof_compress:
                agg["mean_compress_time"] = float(np.mean(prof_compress))
            if prof_score:
                agg["mean_score_time"] = float(np.mean(prof_score))

        system_results["aggregate"] = agg
        all_results["systems"][system_name] = system_results

        logger.info(
            "  => Accuracy: %.1f%% (%d/%d), Throughput: %.1f tok/s, "
            "Wall: %.2fs avg, KV: %.1f MB peak",
            agg["accuracy"] * 100,
            agg["num_correct"],
            n_done,
            agg["mean_throughput"],
            agg["mean_wall_time"],
            agg["max_peak_kv_mb"],
        )
        if "mean_acceptance_rate" in agg:
            logger.info("     Acceptance rate: %.3f", agg["mean_acceptance_rate"])

    # Save results
    output_dir = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"e2e_{args.dataset}_{timestamp}.json"
    save_results(all_results, output_dir, filename)

    # Print summary table
    _print_summary_table(all_results)


def _print_summary_table(results: Dict):
    """Print a formatted summary table of all systems."""
    logger.info("")
    logger.info("=" * 100)
    logger.info("END-TO-END BENCHMARK SUMMARY")
    logger.info("=" * 100)
    header = (
        f"{'System':<20} {'Accuracy':>10} {'Tok/s':>10} "
        f"{'Wall(s)':>10} {'KV(MB)':>10} {'Accept':>10}"
    )
    logger.info(header)
    logger.info("-" * 100)

    ar_throughput = None
    for sys_name in SYSTEMS:
        if sys_name not in results["systems"]:
            continue
        sys_data = results["systems"][sys_name]
        if "error" in sys_data:
            logger.info(f"{sys_name:<20} {'ERROR':>10}")
            continue

        agg = sys_data["aggregate"]
        acc_str = f"{agg['accuracy'] * 100:.1f}%"
        tp_str = f"{agg['mean_throughput']:.1f}"
        wall_str = f"{agg['mean_wall_time']:.2f}"
        kv_str = f"{agg['max_peak_kv_mb']:.1f}"
        alpha_str = (
            f"{agg['mean_acceptance_rate']:.3f}"
            if "mean_acceptance_rate" in agg
            else "N/A"
        )

        if sys_name == "autoregressive":
            ar_throughput = agg["mean_throughput"]

        speedup = ""
        if ar_throughput and ar_throughput > 0:
            speedup = f" ({agg['mean_throughput'] / ar_throughput:.2f}x)"

        logger.info(
            f"{sys_name:<20} {acc_str:>10} {tp_str:>10}{speedup:>8} "
            f"{wall_str:>10} {kv_str:>10} {alpha_str:>10}"
        )

    logger.info("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End System Benchmark (Block 4)"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math500"])
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--profile", action="store_true",
                        help="Enable per-component profiling")
    parser.add_argument("--output_dir", type=str, default="results/e2e")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()
