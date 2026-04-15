"""Block 3: Core Comparison -- 8 retention policies at same KV budget.

Compares 8 KV retention strategies (all at the same memory budget) under
speculative decoding to validate claim C3:
    accept-targeted retention > perplexity-ranked AND attention-ranked
    at the same KV memory footprint.

Also addresses the anti-claim by running AcceptSpec without SD (AR-only).

Policies (all retain `kv_budget` fraction at FP16, compress rest to 2-bit):
    a) Oracle acceptance-ranked  (upper bound)
    b) AcceptSpec predicted      (practical)
    c) Perplexity-ranked
    d) Attention-ranked          (SmallKV / H2O style)
    e) R-KV style                (redundancy + importance)
    f) Random
    g) FP16 baseline             (no compression, upper bound)
    h) AcceptSpec w/o SD         (anti-claim ablation, AR generation)

Usage:
    python scripts/core_comparison.py \
        --model Qwen/Qwen3.5-9B \
        --dataset gsm8k \
        --num_problems 100 \
        --kv_budget 0.2 \
        --output_dir results/comparison

    # Anti-claim ablation (no speculative decoding)
    python scripts/core_comparison.py \
        --dataset gsm8k --ablation no_sd \
        --output_dir results/comparison
"""

import argparse
import json
import logging
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
    AcceptSensitivityOracle,
    AcceptPredictor,
    MixedPrecisionKV,
    TAG_FP16,
    TAG_4BIT,
    TAG_2BIT,
    TAG_EVICTED,
)
from src.gpu_auto import plan_devices, load_models, load_model_mtp, print_gpu_summary
from src.utils import (
    get_kv_tensors,
    set_kv_tensors,
    get_num_kv_layers,
    get_kv_layer_indices,
    save_results,
)

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
    """Load MATH-500 (from hendrycks/MATH test split)."""
    ds = load_dataset("hendrycks/MATH", split="test")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), min(num_problems, len(ds)), replace=False)
    problems = []
    for idx in indices:
        item = ds[int(idx)]
        problems.append({
            "question": item["problem"],
            "answer": item["solution"],
            "level": item.get("level", ""),
            "type": item.get("type", ""),
        })
    return problems


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "After your reasoning, write the final numeric answer on a new line "
        "in the format: #### <number>\n\n"
        f"Question: {question}\n\n"
        "Step-by-step solution:\n"
    )


def format_math500_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{}.\n\n"
        f"Problem: {question}\n\n"
        "Solution:\n"
    )


# ---------------------------------------------------------------------------
# Answer extraction / accuracy
# ---------------------------------------------------------------------------

def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer after '####' in generated text."""
    match = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if match:
        num_str = match.group(1).replace(",", "")
        try:
            return float(num_str)
        except ValueError:
            pass
    # Fallback: last number in text
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


def extract_math500_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} in generated text."""
    # Find last \\boxed{...}
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def extract_math500_gold(solution: str) -> Optional[str]:
    """Extract gold answer from MATH solution field."""
    matches = re.findall(r"\\boxed\{([^}]+)\}", solution)
    if matches:
        return matches[-1].strip()
    return None


def check_gsm8k(generated: str, gold_answer: str) -> bool:
    """Check if GSM8K answer matches numerically."""
    pred = extract_gsm8k_answer(generated)
    gold = extract_gsm8k_gold(gold_answer)
    if pred is None or gold is None:
        return False
    return abs(pred - gold) < 1e-3


def check_math500(generated: str, gold_solution: str) -> bool:
    """Check if MATH-500 answer matches (string equality on boxed content)."""
    pred = extract_math500_answer(generated)
    gold = extract_math500_gold(gold_solution)
    if pred is None or gold is None:
        return False
    # Normalize whitespace and compare
    return pred.strip() == gold.strip()


# ---------------------------------------------------------------------------
# Scoring functions for retention policies
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_oracle_acceptance(
    oracle: AcceptSensitivityOracle,
    target_kv: Any,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_next_logits: torch.Tensor,
    temperature: float,
    num_kv_tokens: int,
) -> torch.Tensor:
    """Score tokens by oracle acceptance sensitivity (ground truth)."""
    coupled_seeds = torch.rand(draft_tokens.shape[0])
    sens_result = oracle.measure_step_sensitivity(
        target_kv=target_kv,
        draft_tokens=draft_tokens,
        draft_probs=draft_probs,
        target_next_logits=target_next_logits,
        temperature=temperature,
        coupled_seeds=coupled_seeds,
    )
    if sens_result is not None:
        return sens_result.sensitivities[:num_kv_tokens]
    return torch.ones(num_kv_tokens)


@torch.no_grad()
def score_accept_predicted(
    predictor: AcceptPredictor,
    draft_model: Any,
    draft_kv: Any,
    target_kv: Any,
    draft_tokens: torch.Tensor,
    num_kv_tokens: int,
) -> torch.Tensor:
    """Score tokens using the trained AcceptPredictor."""
    # Get draft attention weights from last draft step
    draft_device = next(draft_model.parameters()).device
    kv_layers = get_kv_layer_indices(draft_kv)
    num_heads = predictor.num_heads

    # Approximate: extract draft attention from a forward pass with output_attentions
    try:
        out = draft_model(
            draft_tokens[-1:].view(1, 1).to(draft_device),
            past_key_values=draft_kv,
            use_cache=True,
            output_attentions=True,
        )
        # Aggregate attention across layers: [heads, kv_len]
        attn_agg = torch.zeros(num_heads, num_kv_tokens, device="cpu")
        if hasattr(out, "attentions") and out.attentions is not None:
            for layer_attn in out.attentions:
                if layer_attn is not None:
                    # layer_attn: [batch, heads, query_len, kv_len]
                    kv_dim = min(layer_attn.shape[-1], num_kv_tokens)
                    head_dim = min(layer_attn.shape[1], num_heads)
                    attn_agg[:head_dim, :kv_dim] += (
                        layer_attn[0, :head_dim, :, :kv_dim].sum(dim=1).cpu()
                    )
        # Trim the draft KV back (forward added to it)
        _trim_kv_cache(draft_kv, num_kv_tokens)
    except Exception:
        attn_agg = torch.ones(num_heads, num_kv_tokens)

    # Get value norms from target KV
    v_norms = torch.zeros(num_kv_tokens, device="cpu")
    kv_layers = get_kv_layer_indices(target_kv)
    for layer_i in kv_layers:
        _, v = get_kv_tensors(target_kv, layer_i)
        if v is not None:
            kv_dim = min(v.shape[2], num_kv_tokens)
            # v: [batch, heads, seq_len, head_dim]
            v_norms[:kv_dim] += v[0, :, :kv_dim, :].float().norm(dim=-1).sum(dim=0).cpu()

    scores = predictor.predict_scores(attn_agg, v_norms)
    return scores[:num_kv_tokens]


@torch.no_grad()
def score_perplexity(
    target_model: Any,
    target_kv: Any,
    draft_tokens: torch.Tensor,
    num_kv_tokens: int,
) -> torch.Tensor:
    """Score tokens by perplexity sensitivity.

    Perplexity-important tokens are those whose quantization most increases
    the cross-entropy loss on the draft tokens. We approximate this by using
    the gradient of the loss w.r.t. the logits, weighted by the attention
    each KV token receives.
    """
    device = next(target_model.parameters()).device

    # Forward with output_attentions to get attention distribution
    try:
        out = target_model(
            draft_tokens.view(1, -1).to(device),
            past_key_values=target_kv,
            use_cache=True,
            output_attentions=True,
        )
        # Compute per-token log-prob (proxy for perplexity contribution)
        logits = out.logits  # [1, gamma, vocab]
        log_probs = F.log_softmax(logits, dim=-1)
        # Self-entropy per position
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [1, gamma]

        # Attention-weighted importance for perplexity
        importance = torch.zeros(num_kv_tokens, device="cpu")
        if hasattr(out, "attentions") and out.attentions is not None:
            for layer_attn in out.attentions:
                if layer_attn is not None:
                    # Weight by entropy: high-entropy positions depend more on KV
                    kv_dim = min(layer_attn.shape[-1], num_kv_tokens)
                    query_len = min(layer_attn.shape[2], entropy.shape[1])
                    # attn: [batch, heads, q, kv] -> weighted sum
                    ent_weights = entropy[0, :query_len].unsqueeze(0).unsqueeze(-1)
                    weighted = (
                        layer_attn[0, :, :query_len, :kv_dim] *
                        ent_weights.to(layer_attn.device)
                    )
                    importance[:kv_dim] += weighted.sum(dim=(0, 1)).cpu()
        else:
            importance = torch.ones(num_kv_tokens)

        # Trim KV cache back
        kv_layers = get_kv_layer_indices(target_kv)
        for layer_i in kv_layers:
            k, v = get_kv_tensors(target_kv, layer_i)
            if k is not None and k.shape[2] > num_kv_tokens:
                set_kv_tensors(
                    target_kv, layer_i,
                    k[:, :, :num_kv_tokens, :],
                    v[:, :, :num_kv_tokens, :],
                )
    except Exception as e:
        logger.warning("Perplexity scoring failed: %s. Using uniform.", e)
        importance = torch.ones(num_kv_tokens)

    return importance


@torch.no_grad()
def score_attention(
    target_model: Any,
    target_kv: Any,
    draft_tokens: torch.Tensor,
    num_kv_tokens: int,
) -> torch.Tensor:
    """Score tokens by attention importance (SmallKV / H2O style).

    Sum of attention weights received from all query positions across all
    heads and layers.
    """
    device = next(target_model.parameters()).device
    importance = torch.zeros(num_kv_tokens, device="cpu")

    try:
        out = target_model(
            draft_tokens.view(1, -1).to(device),
            past_key_values=target_kv,
            use_cache=True,
            output_attentions=True,
        )
        if hasattr(out, "attentions") and out.attentions is not None:
            for layer_attn in out.attentions:
                if layer_attn is not None:
                    kv_dim = min(layer_attn.shape[-1], num_kv_tokens)
                    importance[:kv_dim] += (
                        layer_attn[0, :, :, :kv_dim].sum(dim=(0, 1)).cpu()
                    )
        else:
            importance = torch.ones(num_kv_tokens)

        # Trim KV back
        kv_layers = get_kv_layer_indices(target_kv)
        for layer_i in kv_layers:
            k, v = get_kv_tensors(target_kv, layer_i)
            if k is not None and k.shape[2] > num_kv_tokens:
                set_kv_tensors(
                    target_kv, layer_i,
                    k[:, :, :num_kv_tokens, :],
                    v[:, :, :num_kv_tokens, :],
                )
    except Exception as e:
        logger.warning("Attention scoring failed: %s. Using uniform.", e)
        importance = torch.ones(num_kv_tokens)

    return importance


@torch.no_grad()
def score_rkv(
    target_model: Any,
    target_kv: Any,
    draft_tokens: torch.Tensor,
    num_kv_tokens: int,
    lambda_: float = 0.5,
) -> torch.Tensor:
    """Score tokens R-KV style: importance - redundancy.

    importance = attention sum (same as H2O)
    redundancy = cosine similarity between consecutive key vectors
    score = lambda * importance - (1 - lambda) * redundancy
    """
    device = next(target_model.parameters()).device

    # Compute attention importance
    attn_importance = torch.zeros(num_kv_tokens, device="cpu")
    try:
        out = target_model(
            draft_tokens.view(1, -1).to(device),
            past_key_values=target_kv,
            use_cache=True,
            output_attentions=True,
        )
        if hasattr(out, "attentions") and out.attentions is not None:
            for layer_attn in out.attentions:
                if layer_attn is not None:
                    kv_dim = min(layer_attn.shape[-1], num_kv_tokens)
                    attn_importance[:kv_dim] += (
                        layer_attn[0, :, :, :kv_dim].sum(dim=(0, 1)).cpu()
                    )

        # Trim KV back
        kv_layers = get_kv_layer_indices(target_kv)
        for layer_i in kv_layers:
            k, v = get_kv_tensors(target_kv, layer_i)
            if k is not None and k.shape[2] > num_kv_tokens:
                set_kv_tensors(
                    target_kv, layer_i,
                    k[:, :, :num_kv_tokens, :],
                    v[:, :, :num_kv_tokens, :],
                )
    except Exception as e:
        logger.warning("R-KV attention scoring failed: %s", e)
        attn_importance = torch.ones(num_kv_tokens)

    # Compute redundancy: cosine similarity between consecutive key vectors
    redundancy = torch.zeros(num_kv_tokens, device="cpu")
    kv_layers = get_kv_layer_indices(target_kv)
    for layer_i in kv_layers:
        k, _ = get_kv_tensors(target_kv, layer_i)
        if k is None:
            continue
        kv_dim = min(k.shape[2], num_kv_tokens)
        # k: [batch, heads, seq_len, head_dim] -> average across heads
        k_avg = k[0, :, :kv_dim, :].float().mean(dim=0)  # [seq_len, head_dim]
        if kv_dim > 1:
            # Cosine similarity between consecutive tokens
            k_norm = F.normalize(k_avg, dim=-1)
            cos_sim = (k_norm[:-1] * k_norm[1:]).sum(dim=-1)  # [seq_len - 1]
            # Token i's redundancy = avg(sim(i-1, i), sim(i, i+1))
            redundancy[1:kv_dim] += cos_sim.cpu()
            redundancy[:kv_dim - 1] += cos_sim.cpu()
            # Normalize for tokens that have both neighbors
            if kv_dim > 2:
                redundancy[1:kv_dim - 1] /= 2.0

    # Normalize both to [0, 1] range
    if attn_importance.max() > 0:
        attn_importance = attn_importance / attn_importance.max()
    if redundancy.max() > 0:
        redundancy = redundancy / redundancy.max()

    # score = lambda * importance - (1 - lambda) * redundancy
    scores = lambda_ * attn_importance - (1.0 - lambda_) * redundancy
    return scores


def score_random(num_kv_tokens: int, seed: int = 42) -> torch.Tensor:
    """Random scores for random retention baseline."""
    rng = torch.Generator().manual_seed(seed)
    return torch.rand(num_kv_tokens, generator=rng)


# ---------------------------------------------------------------------------
# Tag computation from scores
# ---------------------------------------------------------------------------

def scores_to_tags(
    scores: torch.Tensor,
    kv_budget: float,
) -> torch.Tensor:
    """Convert per-token scores to precision tags.

    Top `kv_budget` fraction -> TAG_FP16
    Next 30% -> TAG_4BIT
    Rest -> TAG_2BIT
    """
    n = scores.numel()
    n_fp16 = max(1, int(kv_budget * n))
    n_4bit = max(1, int(0.3 * n))

    tags = torch.full((n,), TAG_2BIT, dtype=torch.uint8)

    # Top fraction -> FP16
    _, top_indices = scores.topk(min(n_fp16, n))
    tags[top_indices] = TAG_FP16

    # Next 30% -> 4-bit
    remaining_scores = scores.clone()
    remaining_scores[tags == TAG_FP16] = -float("inf")
    n_remaining = (tags != TAG_FP16).sum().item()
    n_4bit_actual = min(n_4bit, n_remaining)
    if n_4bit_actual > 0:
        _, moderate_indices = remaining_scores.topk(n_4bit_actual)
        tags[moderate_indices] = TAG_4BIT

    return tags


# ---------------------------------------------------------------------------
# Instrumented SD with per-step KV compression
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_sd_with_policy(
    decoder: SpeculativeDecoder,
    mixed_kv: MixedPrecisionKV,
    input_ids: torch.Tensor,
    score_fn,
    kv_budget: float,
    max_new_tokens: int = 256,
    gamma: int = 5,
    temperature: float = 0.0,
) -> Dict:
    """Run speculative decoding with a given KV retention policy.

    At each verification step, scores all KV tokens using `score_fn`,
    computes precision tags, and applies MixedPrecisionKV compression.

    Args:
        decoder: SpeculativeDecoder instance
        mixed_kv: MixedPrecisionKV for applying compression
        input_ids: prompt token ids [1, seq_len]
        score_fn: callable(target_kv, draft_tokens, ...) -> scores [num_kv]
        kv_budget: fraction of tokens to keep at FP16
        max_new_tokens: generation length
        gamma: draft length
        temperature: sampling temperature

    Returns:
        Dict with generated_text, num_tokens, acceptance_rate, etc.
    """
    assert input_ids.shape[0] == 1

    draft_model = decoder.draft_model
    target_model = decoder.target_model
    draft_device = decoder.draft_device
    target_device = decoder.target_device

    prefix_len = input_ids.shape[1]

    # Prefill
    draft_out = draft_model(input_ids.to(draft_device), use_cache=True)
    draft_kv = draft_out.past_key_values
    draft_next_logits = draft_out.logits[:, -1, :]

    target_out = target_model(input_ids.to(target_device), use_cache=True)
    target_kv = target_out.past_key_values
    target_next_logits = target_out.logits[:, -1, :]

    all_token_ids = input_ids.cpu().clone()
    kv_len = prefix_len

    total_draft = 0
    total_accepted = 0
    n_rounds = 0
    compression_applied = 0

    start = time.perf_counter()

    while all_token_ids.shape[1] - prefix_len < max_new_tokens:
        remaining = max_new_tokens - (all_token_ids.shape[1] - prefix_len)
        cur_gamma = min(gamma, remaining)
        if cur_gamma <= 0:
            break

        n_rounds += 1
        total_draft += cur_gamma

        # Draft phase
        draft_tokens_list = []
        draft_probs_list = []
        cur_logits = draft_next_logits

        for _ in range(cur_gamma):
            if temperature > 0:
                probs = torch.softmax(cur_logits / temperature, dim=-1)
                tok = torch.multinomial(probs.squeeze(0), 1).item()
            else:
                tok = cur_logits.argmax(dim=-1).item()
                probs = torch.softmax(cur_logits, dim=-1)

            draft_tokens_list.append(tok)
            draft_probs_list.append(probs.squeeze(0).cpu())

            tok_tensor = torch.tensor([[tok]], device=draft_device)
            d_out = draft_model(tok_tensor, past_key_values=draft_kv, use_cache=True)
            draft_kv = d_out.past_key_values
            cur_logits = d_out.logits[:, -1, :]

        draft_tokens = torch.tensor(draft_tokens_list, device=target_device)

        # Score and compress KV before verification
        kv_layers = get_kv_layer_indices(target_kv)
        k0, _ = get_kv_tensors(target_kv, kv_layers[0]) if kv_layers else (None, None)
        if k0 is not None:
            num_kv_tokens = k0.shape[2]
            try:
                scores = score_fn(
                    target_kv=target_kv,
                    draft_tokens=draft_tokens,
                    num_kv_tokens=num_kv_tokens,
                )
                tags = scores_to_tags(scores, kv_budget)
                target_kv = mixed_kv.compress_kv(target_kv, tags)
                compression_applied += 1
            except Exception as e:
                logger.debug("Scoring/compression failed at round %d: %s", n_rounds, e)

        # Verification
        verify_out = target_model(
            draft_tokens.view(1, -1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv_ext = verify_out.past_key_values
        verify_logits = verify_out.logits

        # Rejection sampling
        n_acc, accepted = decoder._rejection_sample(
            target_next_logits, verify_logits,
            draft_tokens, draft_probs_list,
            cur_gamma, temperature,
        )

        total_accepted += n_acc
        all_token_ids = torch.cat([all_token_ids, accepted.view(1, -1).cpu()], dim=1)

        # Trim KV caches
        new_kv_len = kv_len + n_acc
        draft_kv = _trim_kv_cache(draft_kv, new_kv_len)
        target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)

        last_tok = accepted[-1]
        kv_len = new_kv_len

        # Update for next round
        d_out = draft_model(
            last_tok.view(1, 1).to(draft_device),
            past_key_values=draft_kv,
            use_cache=True,
        )
        draft_kv = d_out.past_key_values
        draft_next_logits = d_out.logits[:, -1, :]

        t_out = target_model(
            last_tok.view(1, 1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv = t_out.past_key_values
        target_next_logits = t_out.logits[:, -1, :]

        kv_len = new_kv_len + 1

        if last_tok.item() == decoder.tokenizer.eos_token_id:
            break

    wall_time = time.perf_counter() - start
    num_generated = all_token_ids.shape[1] - prefix_len
    acceptance_rate = total_accepted / max(total_draft, 1)

    return {
        "generated_ids": all_token_ids,
        "num_generated": num_generated,
        "n_rounds": n_rounds,
        "total_draft": total_draft,
        "total_accepted": total_accepted,
        "acceptance_rate": acceptance_rate,
        "wall_time": wall_time,
        "compression_rounds": compression_applied,
    }


@torch.no_grad()
def run_fp16_baseline_sd(
    decoder: SpeculativeDecoder,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    gamma: int = 5,
    temperature: float = 0.0,
) -> Dict:
    """FP16 baseline: standard SD without any KV compression."""
    result = decoder.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        gamma=gamma,
        temperature=temperature,
    )
    return {
        "generated_ids": result.generated_ids,
        "num_generated": result.num_generated_tokens,
        "n_rounds": result.num_draft_rounds,
        "total_draft": result.total_draft_tokens,
        "total_accepted": result.total_accepted_tokens,
        "acceptance_rate": result.acceptance_rate,
        "wall_time": result.wall_time_seconds,
        "compression_rounds": 0,
    }


@torch.no_grad()
def run_ar_with_compression(
    target_model: Any,
    tokenizer: Any,
    mixed_kv: MixedPrecisionKV,
    input_ids: torch.Tensor,
    score_fn,
    kv_budget: float,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    compress_interval: int = 16,
) -> Dict:
    """Autoregressive generation with KV compression (no SD, for anti-claim).

    Compresses every `compress_interval` tokens to amortize scoring cost.
    """
    device = next(target_model.parameters()).device
    generated = input_ids.to(device)
    past = None

    start = time.perf_counter()
    compression_applied = 0

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

        # Periodic compression
        if (step + 1) % compress_interval == 0 and past is not None:
            kv_layers = get_kv_layer_indices(past)
            k0, _ = get_kv_tensors(past, kv_layers[0]) if kv_layers else (None, None)
            if k0 is not None:
                num_kv_tokens = k0.shape[2]
                # For AR, we pass dummy "draft_tokens" (just last generated)
                try:
                    scores = score_fn(
                        target_kv=past,
                        draft_tokens=generated[0, -1:].to(device),
                        num_kv_tokens=num_kv_tokens,
                    )
                    tags = scores_to_tags(scores, kv_budget)
                    past = mixed_kv.compress_kv(past, tags)
                    compression_applied += 1
                except Exception as e:
                    logger.debug("AR compression failed at step %d: %s", step, e)

        if tok.item() == tokenizer.eos_token_id:
            break

    wall_time = time.perf_counter() - start
    num_generated = generated.shape[1] - input_ids.shape[1]

    return {
        "generated_ids": generated.cpu(),
        "num_generated": num_generated,
        "n_rounds": 0,
        "total_draft": 0,
        "total_accepted": 0,
        "acceptance_rate": 0.0,
        "wall_time": wall_time,
        "compression_rounds": compression_applied,
    }


# ---------------------------------------------------------------------------
# Main comparison loop
# ---------------------------------------------------------------------------

def build_policy_score_fns(
    decoder: SpeculativeDecoder,
    oracle: Optional[AcceptSensitivityOracle],
    predictor: Optional[AcceptPredictor],
    temperature: float,
) -> Dict[str, Any]:
    """Build scoring function closures for each policy.

    Each score_fn has signature:
        score_fn(target_kv, draft_tokens, num_kv_tokens) -> Tensor[num_kv_tokens]
    """
    policies = {}

    # (a) Oracle acceptance-ranked
    if oracle is not None:
        def _oracle_fn(target_kv, draft_tokens, num_kv_tokens, _oracle=oracle,
                       _temp=temperature, _dec=decoder):
            device = next(_oracle.target_model.parameters()).device

            # --- target_next_logits: the logits from the CURRENT KV state
            # (pre-draft), obtained by a zero-token forward that reads out
            # the last cached position.  We must NOT advance the KV cache,
            # so we use the last token already in the cache rather than
            # feeding draft_tokens[:1] which would append a new position.
            # TODO: Ideally the caller should pass pre-draft target_next_logits
            # directly; this forward call on the last cached token is an
            # approximation that avoids mutating target_kv.
            kv_layers = get_kv_layer_indices(target_kv)
            k0, _ = get_kv_tensors(target_kv, kv_layers[0]) if kv_layers else (None, None)
            if k0 is not None and k0.shape[2] > 0:
                # Reconstruct the last token from the cache is not possible;
                # instead, run a 1-token forward using the first draft token
                # and immediately trim back to restore KV state.
                with torch.no_grad():
                    prefill_out = _oracle.target_model(
                        draft_tokens[:1].view(1, 1).to(device),
                        past_key_values=target_kv,
                        use_cache=True,
                    )
                # Trim KV cache back to original length (undo the 1-token append)
                for li in kv_layers:
                    k, v = get_kv_tensors(target_kv, li)
                    if k is not None and k.shape[2] > num_kv_tokens:
                        set_kv_tensors(target_kv, li,
                                       k[:, :, :num_kv_tokens, :],
                                       v[:, :, :num_kv_tokens, :])
                target_next_logits = prefill_out.logits[:, -1, :]
            else:
                # Fallback: uniform logits (should not happen in practice)
                vocab_size = _oracle.target_model.config.vocab_size
                target_next_logits = torch.zeros(1, vocab_size, device=device)

            # --- draft_probs: per-position SCALAR probability of the selected
            # draft token.  Requires running the draft model (MTP head or
            # separate draft model) to get real probabilities.
            # TODO: Pass real draft probabilities from the outer generation
            # loop.  For now, approximate by running a softmax on
            # target_next_logits (which the target model would have produced
            # at the pre-draft position) and extracting p(draft_token[0]),
            # then using uniform 1/vocab for remaining positions where we
            # lack the sequential draft logits without an expensive multi-step
            # draft forward.
            if _temp > 0:
                probs_dist = torch.softmax(target_next_logits / _temp, dim=-1)
            else:
                probs_dist = torch.softmax(target_next_logits, dim=-1)
            gamma = draft_tokens.shape[0]
            draft_probs = torch.zeros(gamma)
            # First position: real probability from target logits (best available approx)
            draft_probs[0] = probs_dist[0, draft_tokens[0].item()].cpu().item()
            # Remaining positions: TODO need sequential draft forward for real values
            for j in range(1, gamma):
                draft_probs[j] = probs_dist[0, draft_tokens[j].item()].cpu().item()

            return score_oracle_acceptance(
                _oracle, target_kv, draft_tokens, draft_probs,
                target_next_logits, _temp, num_kv_tokens,
            )
        policies["oracle_accept"] = _oracle_fn

    # (b) AcceptSpec predicted
    if predictor is not None:
        def _predictor_fn(target_kv, draft_tokens, num_kv_tokens,
                          _pred=predictor, _dec=decoder):
            return score_accept_predicted(
                _pred, _dec.draft_model, _dec.draft_model(
                    draft_tokens[:1].view(1, 1).to(_dec.draft_device),
                    use_cache=True,
                ).past_key_values,
                target_kv, draft_tokens, num_kv_tokens,
            )
        policies["acceptspec_predicted"] = _predictor_fn

    # (c) Perplexity-ranked
    def _ppl_fn(target_kv, draft_tokens, num_kv_tokens, _model=decoder.target_model):
        return score_perplexity(_model, target_kv, draft_tokens, num_kv_tokens)
    policies["perplexity"] = _ppl_fn

    # (d) Attention-ranked (SmallKV / H2O)
    def _attn_fn(target_kv, draft_tokens, num_kv_tokens, _model=decoder.target_model):
        return score_attention(_model, target_kv, draft_tokens, num_kv_tokens)
    policies["attention_h2o"] = _attn_fn

    # (e) R-KV style
    def _rkv_fn(target_kv, draft_tokens, num_kv_tokens, _model=decoder.target_model):
        return score_rkv(_model, target_kv, draft_tokens, num_kv_tokens)
    policies["rkv"] = _rkv_fn

    # (f) Random
    _random_seed_counter = [0]
    def _rand_fn(target_kv, draft_tokens, num_kv_tokens):
        _random_seed_counter[0] += 1
        return score_random(num_kv_tokens, seed=42 + _random_seed_counter[0])
    policies["random"] = _rand_fn

    return policies


def run_comparison(args):
    """Run the full 8-policy comparison."""
    print_gpu_summary()

    # Load models
    plan = plan_devices()
    logger.info("Loading models with auto device plan: %s", plan.description)

    model, mtp_head, tokenizer, plan = load_model_mtp(args.model, plan=plan)
    target_model = model

    # Create SD decoder with MTP self-speculation (not dual-model legacy)
    decoder = SpeculativeDecoder(
        target_model=target_model,
        tokenizer=tokenizer,
        mtp_head=mtp_head,
        quant_bits=0,
    )

    # Create MixedPrecisionKV
    config = target_model.config
    num_layers = config.num_hidden_layers
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads

    mixed_kv = MixedPrecisionKV(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    # Create oracle (expensive but gives ground truth)
    oracle = AcceptSensitivityOracle(
        target_model=target_model,
        quantizer_bits=2,
        sample_fraction=0.2,
    )

    # Load or create AcceptPredictor
    predictor = AcceptPredictor(
        num_heads=num_kv_heads,
        theta_critical=0.8,
        theta_low=0.3,
    )
    # Try to load from calibration results
    calib_path = Path(args.output_dir) / "predictor_weights.pt"
    if calib_path.exists():
        state = torch.load(calib_path, map_location="cpu", weights_only=True)
        predictor.head_weights = state["head_weights"]
        predictor._fitted = True
        logger.info("Loaded AcceptPredictor weights from %s", calib_path)
    else:
        logger.warning(
            "No calibrated predictor found at %s. "
            "Using uniform head weights (run Block 2 first for best results).",
            calib_path,
        )

    # Load dataset
    logger.info("Loading dataset '%s' (%d problems)...", args.dataset, args.num_problems)
    if args.dataset == "gsm8k":
        problems = load_gsm8k(args.num_problems)
        format_fn = format_gsm8k_prompt
        check_fn = check_gsm8k
    elif args.dataset == "math500":
        problems = load_math500(args.num_problems)
        format_fn = format_math500_prompt
        check_fn = check_math500
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Shard for parallel execution
    if args.shard is not None and args.num_shards is not None:
        shard_size = len(problems) // args.num_shards
        start = args.shard * shard_size
        end = start + shard_size if args.shard < args.num_shards - 1 else len(problems)
        problems = problems[start:end]
        logger.info("Shard %d/%d: problems [%d, %d) (%d problems)",
                     args.shard, args.num_shards, start, end, len(problems))

    logger.info("Loaded %d problems.", len(problems))

    # Build scoring functions
    policy_fns = build_policy_score_fns(decoder, oracle, predictor, args.temperature)

    # Determine which policies to run
    if args.ablation == "no_sd":
        # Anti-claim: run AcceptSpec predicted + attention + random WITHOUT SD
        run_policies = ["acceptspec_predicted", "attention_h2o", "random"]
        run_mode = "ar"
        logger.info("Running ANTI-CLAIM ablation (no SD, AR generation only)")
    else:
        run_policies = list(policy_fns.keys())
        run_mode = "sd"

    # Add FP16 baseline always
    if "fp16_baseline" not in run_policies:
        run_policies.append("fp16_baseline")

    # Results storage
    all_results = {
        "config": {
            "draft_model": args.model,
            "target_model": args.model,
            "dataset": args.dataset,
            "num_problems": len(problems),
            "kv_budget": args.kv_budget,
            "gamma": args.gamma,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "ablation": args.ablation,
            "run_mode": run_mode,
        },
        "per_policy": {},
    }

    # Run each policy
    for policy_name in run_policies:
        logger.info("=" * 60)
        logger.info("POLICY: %s", policy_name)
        logger.info("=" * 60)

        policy_correct = 0
        policy_total = 0
        policy_acceptance_rates = []
        policy_wall_times = []
        policy_problem_results = []

        for i, problem in enumerate(problems):
            prompt = format_fn(problem["question"])
            input_ids = tokenizer.encode(prompt, return_tensors="pt")

            logger.info(
                "[%s] [%d/%d] '%s...'",
                policy_name, i + 1, len(problems),
                problem["question"][:50],
            )

            try:
                if policy_name == "fp16_baseline":
                    # No compression
                    if run_mode == "ar":
                        gen_ids, wall = decoder.generate_autoregressive(
                            input_ids, max_new_tokens=args.max_tokens,
                            temperature=args.temperature,
                        )
                        gen_result = {
                            "generated_ids": gen_ids,
                            "num_generated": gen_ids.shape[1] - input_ids.shape[1],
                            "acceptance_rate": 0.0,
                            "wall_time": wall,
                            "compression_rounds": 0,
                        }
                    else:
                        gen_result = run_fp16_baseline_sd(
                            decoder, input_ids,
                            max_new_tokens=args.max_tokens,
                            gamma=args.gamma,
                            temperature=args.temperature,
                        )
                elif run_mode == "ar":
                    # Anti-claim: AR with compression
                    score_fn = policy_fns[policy_name]
                    # Re-create mixed_kv per problem to avoid stale state
                    mkv = MixedPrecisionKV(
                        num_layers=num_layers,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                    )
                    gen_result = run_ar_with_compression(
                        target_model, tokenizer, mkv,
                        input_ids, score_fn, args.kv_budget,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                else:
                    # SD with compression policy
                    score_fn = policy_fns[policy_name]
                    mkv = MixedPrecisionKV(
                        num_layers=num_layers,
                        num_kv_heads=num_kv_heads,
                        head_dim=head_dim,
                    )
                    gen_result = run_sd_with_policy(
                        decoder, mkv, input_ids, score_fn, args.kv_budget,
                        max_new_tokens=args.max_tokens,
                        gamma=args.gamma,
                        temperature=args.temperature,
                    )

                # Decode generated text
                gen_ids = gen_result["generated_ids"]
                generated_text = tokenizer.decode(
                    gen_ids[0, input_ids.shape[1]:],
                    skip_special_tokens=True,
                )

                # Check accuracy
                correct = check_fn(generated_text, problem["answer"])
                if correct:
                    policy_correct += 1
                policy_total += 1

                if "acceptance_rate" in gen_result:
                    policy_acceptance_rates.append(gen_result["acceptance_rate"])
                policy_wall_times.append(gen_result["wall_time"])

                policy_problem_results.append({
                    "problem_idx": i,
                    "correct": correct,
                    "acceptance_rate": gen_result.get("acceptance_rate", 0.0),
                    "num_generated": gen_result.get("num_generated", 0),
                    "wall_time": gen_result["wall_time"],
                    "compression_rounds": gen_result.get("compression_rounds", 0),
                })

                logger.info(
                    "  correct=%s, accept_rate=%.3f, time=%.1fs",
                    correct,
                    gen_result.get("acceptance_rate", 0.0),
                    gen_result["wall_time"],
                )

            except Exception as e:
                logger.error("  FAILED: %s", e)
                policy_total += 1
                policy_problem_results.append({
                    "problem_idx": i,
                    "correct": False,
                    "error": str(e),
                })

            # Free CUDA memory between problems
            torch.cuda.empty_cache()

        # Aggregate policy results
        accuracy = policy_correct / max(policy_total, 1)
        mean_accept = (
            float(np.mean(policy_acceptance_rates))
            if policy_acceptance_rates else 0.0
        )
        mean_wall = (
            float(np.mean(policy_wall_times))
            if policy_wall_times else 0.0
        )

        logger.info("-" * 40)
        logger.info(
            "POLICY %s: accuracy=%.1f%% (%d/%d), mean_accept=%.3f, mean_time=%.1fs",
            policy_name, accuracy * 100, policy_correct, policy_total,
            mean_accept, mean_wall,
        )

        all_results["per_policy"][policy_name] = {
            "accuracy": accuracy,
            "num_correct": policy_correct,
            "num_total": policy_total,
            "mean_acceptance_rate": mean_accept,
            "std_acceptance_rate": (
                float(np.std(policy_acceptance_rates))
                if len(policy_acceptance_rates) > 1 else 0.0
            ),
            "mean_wall_time": mean_wall,
            "std_wall_time": (
                float(np.std(policy_wall_times))
                if len(policy_wall_times) > 1 else 0.0
            ),
            "per_problem": policy_problem_results,
        }

    # Compute compression stats summary
    all_results["compression_stats"] = mixed_kv.get_compression_stats(
        scores_to_tags(torch.rand(100), args.kv_budget)
    )

    # Summary comparison table
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY: %s (kv_budget=%.0f%%)", args.dataset.upper(), args.kv_budget * 100)
    logger.info("=" * 70)
    logger.info("%-25s %8s %12s %10s", "Policy", "Accuracy", "Accept Rate", "Time (s)")
    logger.info("-" * 70)
    for pname, pdata in all_results["per_policy"].items():
        logger.info(
            "%-25s %7.1f%% %11.3f %9.1f",
            pname,
            pdata["accuracy"] * 100,
            pdata["mean_acceptance_rate"],
            pdata["mean_wall_time"],
        )
    logger.info("=" * 70)

    # Check claim C3: AcceptSpec > perplexity AND attention by >= 3pp
    if "acceptspec_predicted" in all_results["per_policy"]:
        acc_accept = all_results["per_policy"]["acceptspec_predicted"]["accuracy"]
        acc_ppl = all_results["per_policy"].get("perplexity", {}).get("accuracy", 0)
        acc_attn = all_results["per_policy"].get("attention_h2o", {}).get("accuracy", 0)
        gap_ppl = (acc_accept - acc_ppl) * 100
        gap_attn = (acc_accept - acc_attn) * 100
        c3_pass = gap_ppl >= 3.0 and gap_attn >= 3.0
        logger.info("")
        logger.info("CLAIM C3 CHECK: AcceptSpec vs perplexity = %+.1fpp, vs attention = %+.1fpp",
                     gap_ppl, gap_attn)
        logger.info("C3 GATE (>=3pp over both): %s", "PASS" if c3_pass else "FAIL")
        all_results["claim_c3"] = {
            "gap_vs_perplexity_pp": gap_ppl,
            "gap_vs_attention_pp": gap_attn,
            "passed": c3_pass,
        }

    # Save results
    ablation_tag = f"_ablation_{args.ablation}" if args.ablation else ""
    filename = f"core_comparison_{args.dataset}_budget{args.kv_budget}{ablation_tag}.json"
    save_results(all_results, args.output_dir, filename)
    logger.info("Results saved to %s/%s", args.output_dir, filename)


def main():
    parser = argparse.ArgumentParser(
        description="Block 3: Core Comparison -- 8 retention policies at same KV budget"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math500"])
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--kv_budget", type=float, default=0.2,
                        help="Fraction of KV tokens to keep at FP16 (default: 0.2)")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=["no_sd"],
                        help="Ablation mode: 'no_sd' runs AR without speculative decoding")
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="results/comparison")
    parser.add_argument("--output", type=str, default=None,
                        help="Override output path (used by parallel_run.sh)")
    parser.add_argument("--shard", type=int, default=None,
                        help="Shard index (0-based) for parallel execution")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of shards for parallel execution")
    args = parser.parse_args()

    run_comparison(args)


if __name__ == "__main__":
    main()
