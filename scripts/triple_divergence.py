"""Triple Divergence + Predictor Validation (Block 2).

Validates that acceptance-critical tokens != attention-important tokens !=
perplexity-sensitive tokens.  Trains AcceptPredictor and an AttentionProxy
predictor on a 50/50 split and compares F1/precision/recall.

Usage:
    python scripts/triple_divergence.py \
        --model Qwen/Qwen3.5-9B \
        --num_problems 100 \
        --output_dir results/triple_divergence

Decision gate:
    All pairwise Spearman rho < 0.7 -> PASS
    AcceptPredictor F1 > 0.75 -> PASS
    AcceptPredictor F1 > AttentionProxy F1 -> PASS (AcceptSpec adds value)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.speculative_decode import SpeculativeDecoder, _trim_kv_cache
from src.acceptspec import AcceptSensitivityOracle, AcceptPredictor
from src.gpu_auto import plan_devices, load_models, load_model_mtp, print_gpu_summary
from src.turboquant_kv import HadamardRotation, ScalarQuantizer
from src.utils import get_kv_tensors, set_kv_tensors, get_num_kv_layers, get_kv_layer_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (same as oracle_sensitivity.py)
# ---------------------------------------------------------------------------

def load_gsm8k(num_problems: int, seed: int = 42) -> list:
    """Load GSM8K test problems."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), min(num_problems, len(ds)), replace=False)
    problems = []
    for idx in indices:
        item = ds[int(idx)]
        problems.append({
            'question': item['question'],
            'answer': item['answer'],
        })
    return problems


def format_prompt(question: str) -> str:
    """Format GSM8K question for thinking model."""
    return f"Solve this math problem step by step.\n\nQuestion: {question}\n\nAnswer:"


# ---------------------------------------------------------------------------
# Perplexity sensitivity measurement
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_perplexity_sensitivity(
    target_model: Any,
    target_kv: Any,
    draft_tokens: torch.Tensor,
    quantizer: ScalarQuantizer,
    rotation: HadamardRotation,
    sample_fraction: float = 0.2,
    num_lookahead: int = 3,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Measure per-token perplexity sensitivity via KV perturbation.

    For each sampled token i, quantize its KV to 2-bit, then measure the
    change in cross-entropy loss on the next few draft tokens.

    Args:
        target_model: the verifier model
        target_kv: current KV cache (will be restored after each perturbation)
        draft_tokens: [gamma] draft token ids
        quantizer: 2-bit scalar quantizer
        rotation: Hadamard rotation for quantization
        sample_fraction: fraction of KV tokens to perturb
        num_lookahead: how many tokens ahead to measure CE loss
        rng: random generator for reproducible sampling

    Returns:
        sensitivities: [num_kv_tokens] perplexity sensitivity per token
    """
    device = next(target_model.parameters()).device
    kv_layers = get_kv_layer_indices(target_kv)
    if not kv_layers:
        return torch.zeros(1)
    k0, v0 = get_kv_tensors(target_kv, kv_layers[0])
    if k0 is None:
        return torch.zeros(1)
    num_kv_tokens = k0.shape[2]
    gamma = draft_tokens.shape[0]
    lookahead = min(num_lookahead, gamma)

    # Step 1: Compute baseline loss with full-precision KV
    baseline_out = target_model(
        draft_tokens.view(1, -1).to(device),
        past_key_values=target_kv,
        use_cache=True,
    )
    baseline_logits = baseline_out.logits  # [1, gamma, vocab]
    # Trim KV back (the forward appended to cache)
    for layer_i in kv_layers:
        k, v = get_kv_tensors(target_kv, layer_i)
        if k is not None and k.shape[2] > num_kv_tokens:
            set_kv_tensors(
                target_kv, layer_i,
                k[:, :, :num_kv_tokens, :],
                v[:, :, :num_kv_tokens, :],
            )

    # Cross-entropy on next tokens: use positions [0..lookahead-1] predicting
    # tokens [1..lookahead] (shifted by 1)
    if lookahead < 2:
        # Need at least 2 tokens for a meaningful CE measurement
        return torch.zeros(num_kv_tokens)

    target_ids = draft_tokens[1:lookahead].to(device)
    baseline_ce = F.cross_entropy(
        baseline_logits[0, :lookahead - 1, :],
        target_ids,
        reduction='mean',
    ).item()

    # Step 2: Sample tokens to perturb
    n_sample = max(1, int(sample_fraction * num_kv_tokens))
    n_sample = min(n_sample, num_kv_tokens)
    if rng is not None:
        sample_indices = torch.randperm(num_kv_tokens, generator=rng)[:n_sample]
    else:
        sample_indices = torch.randperm(num_kv_tokens)[:n_sample]

    sensitivities = torch.zeros(num_kv_tokens, device='cpu')

    # Step 3: For each sampled token, perturb KV to 2-bit and re-measure CE
    for idx in sample_indices:
        idx_val = idx.item()

        # Save and perturb
        orig_kvs = {}
        for layer_i in kv_layers:
            k, v = get_kv_tensors(target_kv, layer_i)
            if k is None:
                orig_kvs[layer_i] = (None, None)
                continue
            orig_kvs[layer_i] = (
                k[:, :, idx_val:idx_val + 1, :].clone(),
                v[:, :, idx_val:idx_val + 1, :].clone(),
            )
            # Quantize token KV to 2-bit
            k_tok = k[:, :, idx_val:idx_val + 1, :].float()
            v_tok = v[:, :, idx_val:idx_val + 1, :].float()
            k_rot = rotation.rotate(k_tok)
            v_rot = rotation.rotate(v_tok)
            k_codes, k_scales, k_zeros = quantizer.quantize(k_rot)
            v_codes, v_scales, v_zeros = quantizer.quantize(v_rot)
            k_deq = rotation.inverse_rotate(
                quantizer.dequantize(k_codes, k_scales, k_zeros)
            ).to(k.dtype)
            v_deq = rotation.inverse_rotate(
                quantizer.dequantize(v_codes, v_scales, v_zeros)
            ).to(v.dtype)
            k[:, :, idx_val:idx_val + 1, :] = k_deq
            v[:, :, idx_val:idx_val + 1, :] = v_deq

        # Re-measure CE
        perturbed_out = target_model(
            draft_tokens.view(1, -1).to(device),
            past_key_values=target_kv,
            use_cache=True,
        )
        perturbed_logits = perturbed_out.logits
        perturbed_ce = F.cross_entropy(
            perturbed_logits[0, :lookahead - 1, :],
            target_ids,
            reduction='mean',
        ).item()

        sensitivities[idx_val] = abs(perturbed_ce - baseline_ce)

        # Restore original KV
        for layer_i in kv_layers:
            k, v = get_kv_tensors(target_kv, layer_i)
            if k is None:
                continue
            orig_k, orig_v = orig_kvs[layer_i]
            if orig_k is not None:
                k[:, :, idx_val:idx_val + 1, :] = orig_k
                v[:, :, idx_val:idx_val + 1, :] = orig_v

        # Trim extended KV
        for layer_i in kv_layers:
            k, v = get_kv_tensors(target_kv, layer_i)
            if k is not None and k.shape[2] > num_kv_tokens:
                set_kv_tensors(
                    target_kv, layer_i,
                    k[:, :, :num_kv_tokens, :],
                    v[:, :, :num_kv_tokens, :],
                )

    return sensitivities


# ---------------------------------------------------------------------------
# Instrumented speculative decoding (collects all 3 rankings per step)
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_instrumented_sd(
    decoder: SpeculativeDecoder,
    oracle: AcceptSensitivityOracle,
    ppl_quantizer: ScalarQuantizer,
    ppl_rotation: HadamardRotation,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    gamma: int = 5,
    temperature: float = 0.0,
    num_samples_per_step: int = 50,
    sample_fraction: float = 0.2,
) -> Optional[dict]:
    """Run SD and collect acceptance / perplexity / attention rankings at each step.

    Returns dict with per-step rankings and draft-model features for predictor
    training, or None if generation fails completely.
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

    # Collected per-step data
    step_accept_sens = []   # acceptance sensitivity rankings
    step_ppl_sens = []      # perplexity sensitivity rankings
    step_attn_imp = []      # attention importance rankings
    # Features for predictor training
    step_draft_attns = []   # draft model attention weights per head
    step_value_norms = []   # value vector L2 norms
    n_steps = 0

    ppl_rng = torch.Generator().manual_seed(123)

    while all_token_ids.shape[1] - prefix_len < max_new_tokens:
        remaining = max_new_tokens - (all_token_ids.shape[1] - prefix_len)
        cur_gamma = min(gamma, remaining)
        if cur_gamma <= 0:
            break

        n_steps += 1

        # --- Draft phase ---
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

            prob_val = probs.squeeze(0)[tok].item()
            draft_tokens_list.append(tok)
            draft_probs_list.append(prob_val)

            tok_tensor = torch.tensor([[tok]], device=draft_device)
            d_out = draft_model(tok_tensor, past_key_values=draft_kv, use_cache=True)
            draft_kv = d_out.past_key_values
            cur_logits = d_out.logits[:, -1, :]

        draft_tokens = torch.tensor(draft_tokens_list, device=target_device)
        draft_probs = torch.tensor(draft_probs_list)
        coupled_seeds = torch.rand(cur_gamma)

        # --- Measure all 3 rankings (every few steps to save compute) ---
        if n_steps % max(1, gamma) == 1 or n_steps <= 3:
            try:
                # (a) Acceptance sensitivity (from oracle)
                sens_result = oracle.measure_step_sensitivity(
                    target_kv=target_kv,
                    draft_tokens=draft_tokens,
                    draft_probs=draft_probs,
                    target_next_logits=target_next_logits,
                    temperature=temperature,
                    num_samples=num_samples_per_step,
                    coupled_seeds=coupled_seeds,
                )

                if sens_result is not None:
                    accept_sens = sens_result.sensitivities
                    attn_imp = sens_result.attention_importance

                    # (b) Perplexity sensitivity
                    ppl_sens = measure_perplexity_sensitivity(
                        target_model=target_model,
                        target_kv=target_kv,
                        draft_tokens=draft_tokens,
                        quantizer=ppl_quantizer,
                        rotation=ppl_rotation,
                        sample_fraction=sample_fraction,
                        num_lookahead=min(3, cur_gamma),
                        rng=ppl_rng,
                    )

                    step_accept_sens.append(accept_sens)
                    step_ppl_sens.append(ppl_sens)
                    step_attn_imp.append(attn_imp)

                    # Collect draft-model features for predictor training
                    draft_attn, v_norms = _extract_draft_features(
                        draft_model, draft_kv, draft_tokens, draft_device,
                    )
                    step_draft_attns.append(draft_attn)
                    step_value_norms.append(v_norms)

            except Exception as e:
                logger.debug("Measurement failed at step %d: %s", n_steps, e)

        # --- Standard verification ---
        verify_out = target_model(
            draft_tokens.view(1, -1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
        )
        target_kv_ext = verify_out.past_key_values
        verify_logits = verify_out.logits

        n_acc, accepted = decoder._rejection_sample(
            target_next_logits, verify_logits,
            draft_tokens, draft_probs,
            cur_gamma, temperature,
        )

        all_token_ids = torch.cat([all_token_ids, accepted.view(1, -1).cpu()], dim=1)

        # Trim KV caches
        new_kv_len = kv_len + n_acc
        draft_kv = _trim_kv_cache(draft_kv, new_kv_len)
        target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)

        last_tok = accepted[-1]
        kv_len = new_kv_len

        # Update draft and target next logits
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

        # Check for EOS
        if last_tok.item() == decoder.tokenizer.eos_token_id:
            break

    if not step_accept_sens:
        return None

    return {
        'num_steps': n_steps,
        'num_tokens': all_token_ids.shape[1] - prefix_len,
        'step_accept_sens': step_accept_sens,
        'step_ppl_sens': step_ppl_sens,
        'step_attn_imp': step_attn_imp,
        'step_draft_attns': step_draft_attns,
        'step_value_norms': step_value_norms,
    }


def _extract_draft_features(
    draft_model: Any,
    draft_kv: Any,
    draft_tokens: torch.Tensor,
    draft_device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract draft-model attention weights and value norms for predictor features.

    Returns:
        draft_attn: [num_heads, num_kv_tokens] — attention from last draft query
        value_norms: [num_kv_tokens] — L2 norm of value vectors
    """
    kv_layers = get_kv_layer_indices(draft_kv)
    if not kv_layers:
        return torch.zeros(1, 1), torch.zeros(1)
    k0, v0 = get_kv_tensors(draft_kv, kv_layers[0])
    if k0 is None:
        return torch.zeros(1, 1), torch.zeros(1)
    num_kv_tokens = k0.shape[2]
    num_heads = k0.shape[1]

    # Value norms: average across layers, sum across head dim
    v_norms = torch.zeros(num_kv_tokens, device='cpu')
    for layer_i in kv_layers:
        _, v = get_kv_tensors(draft_kv, layer_i)
        if v is not None and v.shape[2] >= num_kv_tokens:
            # v: [batch, heads, seq, head_dim]
            layer_norms = v[0, :, :num_kv_tokens, :].float().norm(dim=-1).mean(dim=0)
            v_norms += layer_norms.cpu()
    v_norms /= max(len(kv_layers), 1)

    # Draft attention: run a forward pass with output_attentions to get attention weights
    # Use the last draft token as the query
    attn_weights = torch.zeros(num_heads, num_kv_tokens, device='cpu')
    try:
        last_tok = draft_tokens[-1:]
        out = draft_model(
            last_tok.view(1, 1).to(draft_device),
            past_key_values=draft_kv,
            use_cache=True,
            output_attentions=True,
        )
        if hasattr(out, 'attentions') and out.attentions is not None:
            # Average across layers; each layer_attn: [batch, heads, 1, kv_len]
            for layer_attn in out.attentions:
                if layer_attn is not None:
                    a = layer_attn[0, :, 0, :num_kv_tokens].cpu()
                    attn_weights[:a.shape[0], :a.shape[1]] += a
            attn_weights /= max(len(out.attentions), 1)

        # Trim KV back (forward appended one position)
        for layer_i in kv_layers:
            k, v = get_kv_tensors(draft_kv, layer_i)
            if k is not None and k.shape[2] > num_kv_tokens:
                set_kv_tensors(
                    draft_kv, layer_i,
                    k[:, :, :num_kv_tokens, :],
                    v[:, :, :num_kv_tokens, :],
                )
    except Exception as e:
        logger.debug("Draft feature extraction failed: %s", e)
        attn_weights = torch.ones(num_heads, num_kv_tokens) / num_kv_tokens

    return attn_weights, v_norms


# ---------------------------------------------------------------------------
# Spearman correlation helpers
# ---------------------------------------------------------------------------

def pairwise_spearman(
    accept: torch.Tensor,
    ppl: torch.Tensor,
    attn: torch.Tensor,
    min_nonzero: int = 10,
) -> Dict[str, Tuple[float, float]]:
    """Compute pairwise Spearman rho between 3 ranking vectors.

    Only considers tokens where at least one of the pair is nonzero.

    Returns dict mapping pair name -> (rho, p-value).
    """
    pairs = {
        'accept_vs_ppl': (accept, ppl),
        'accept_vs_attn': (accept, attn),
        'ppl_vs_attn': (ppl, attn),
    }
    results = {}
    for name, (a, b) in pairs.items():
        mask = (a > 0) | (b > 0)
        if mask.sum() >= min_nonzero:
            rho, pval = spearmanr(a[mask].numpy(), b[mask].numpy())
            results[name] = (float(rho), float(pval))
        else:
            results[name] = (0.0, 1.0)
    return results


# ---------------------------------------------------------------------------
# Predictor training and evaluation
# ---------------------------------------------------------------------------

def _make_oracle_labels(
    sensitivities: torch.Tensor,
    critical_fraction: float = 0.2,
) -> torch.Tensor:
    """Binarize sensitivity scores: top critical_fraction -> 1, rest -> 0."""
    n = sensitivities.numel()
    n_critical = max(1, int(critical_fraction * n))
    _, top_idx = sensitivities.topk(n_critical)
    labels = torch.zeros(n)
    labels[top_idx] = 1.0
    return labels


def train_and_evaluate_predictor(
    train_draft_attns: List[torch.Tensor],
    train_value_norms: List[torch.Tensor],
    train_labels: List[torch.Tensor],
    test_draft_attns: List[torch.Tensor],
    test_value_norms: List[torch.Tensor],
    test_labels: List[torch.Tensor],
    num_heads: int,
) -> Dict[str, float]:
    """Train AcceptPredictor on train set, evaluate on test set.

    Returns dict with F1, precision, recall.
    """
    predictor = AcceptPredictor(num_heads=num_heads)
    predictor.fit(train_draft_attns, train_value_norms, train_labels)

    # Evaluate on test set
    all_preds = []
    all_true = []
    for attn, vnorm, labels in zip(test_draft_attns, test_value_norms, test_labels):
        scores = predictor.predict_scores(attn, vnorm)
        # Use same threshold: top 20% predicted as critical
        n = scores.numel()
        n_critical = max(1, int(0.2 * n))
        pred_labels = torch.zeros(n)
        _, top_idx = scores.topk(min(n_critical, n))
        pred_labels[top_idx] = 1.0
        all_preds.append(pred_labels)
        all_true.append(labels)

    preds = torch.cat(all_preds)
    true = torch.cat(all_true)
    return _compute_classification_metrics(preds, true)


def _compute_classification_metrics(
    preds: torch.Tensor,
    true: torch.Tensor,
) -> Dict[str, float]:
    """Compute precision, recall, F1 for binary predictions."""
    tp = ((preds == 1) & (true == 1)).sum().float()
    fp = ((preds == 1) & (true == 0)).sum().float()
    fn = ((preds == 0) & (true == 1)).sum().float()

    precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp.item()),
        'fp': int(fp.item()),
        'fn': int(fn.item()),
    }


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def run_triple_divergence(args):
    """Run the full Block 2 experiment."""
    print_gpu_summary()

    # Auto-detect GPUs and plan device placement
    plan = plan_devices()
    logger.info("Loading models with auto device plan: %s", plan.description)

    model, mtp_head, tokenizer, plan = load_model_mtp(args.model, plan=plan)
    target_model = model
    draft_model = model  # MTP mode: same model for both

    # Create SD decoder (no quantization -- baseline acceptance)
    decoder = SpeculativeDecoder(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        quant_bits=0,
    )

    # Create oracle
    oracle = AcceptSensitivityOracle(
        target_model=target_model,
        quantizer_bits=2,
        sample_fraction=args.sample_fraction,
    )

    # Create perplexity-sensitivity quantizer (separate instance, same 2-bit config)
    head_dim = target_model.config.hidden_size // target_model.config.num_attention_heads
    ppl_quantizer = ScalarQuantizer(bits=2, block_size=128)
    ppl_rotation = HadamardRotation(dim=head_dim, seed=43)

    # Load problems
    logger.info("Loading GSM8K (%d problems)...", args.num_problems)
    problems = load_gsm8k(args.num_problems)

    # Split 50/50 for predictor train/test
    split_idx = len(problems) // 2
    train_problems = problems[:split_idx]
    test_problems = problems[split_idx:]
    logger.info("Split: %d train, %d test problems", len(train_problems), len(test_problems))

    # Collect data from all problems
    all_accept_sens = []
    all_ppl_sens = []
    all_attn_imp = []

    # Predictor training features: per-split
    train_draft_attns = []
    train_value_norms = []
    train_accept_labels = []
    train_attn_labels = []

    test_draft_attns = []
    test_value_norms = []
    test_accept_labels = []
    test_attn_labels = []

    problem_results = []

    for i, problem in enumerate(problems):
        is_train = i < split_idx
        split_name = "TRAIN" if is_train else "TEST"
        prompt = format_prompt(problem['question'])
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        logger.info(
            "[%d/%d %s] Running SD for '%s...'",
            i + 1, len(problems), split_name, problem['question'][:50],
        )

        try:
            result = run_instrumented_sd(
                decoder, oracle, ppl_quantizer, ppl_rotation,
                input_ids,
                max_new_tokens=args.max_tokens,
                gamma=args.gamma,
                temperature=args.temperature,
                num_samples_per_step=args.samples_per_step,
                sample_fraction=args.sample_fraction,
            )

            if result is None:
                logger.warning("  No measurements collected, skipping.")
                continue

            # Aggregate per-step rankings
            for step_i in range(len(result['step_accept_sens'])):
                a_sens = result['step_accept_sens'][step_i]
                p_sens = result['step_ppl_sens'][step_i]
                attn = result['step_attn_imp'][step_i]

                # Align lengths (in case of sampling differences)
                min_len = min(a_sens.numel(), p_sens.numel(), attn.numel())
                if min_len < 10:
                    continue
                a_sens = a_sens[:min_len]
                p_sens = p_sens[:min_len]
                attn = attn[:min_len]

                all_accept_sens.append(a_sens)
                all_ppl_sens.append(p_sens)
                all_attn_imp.append(attn)

                # Make binary labels for predictor training
                accept_labels = _make_oracle_labels(a_sens, critical_fraction=0.2)
                attn_labels = _make_oracle_labels(attn, critical_fraction=0.2)

                d_attn = result['step_draft_attns'][step_i]
                v_norms = result['step_value_norms'][step_i]

                # Align draft features to min_len
                d_attn = d_attn[:, :min_len]
                v_norms = v_norms[:min_len]

                if is_train:
                    train_draft_attns.append(d_attn)
                    train_value_norms.append(v_norms)
                    train_accept_labels.append(accept_labels)
                    train_attn_labels.append(attn_labels)
                else:
                    test_draft_attns.append(d_attn)
                    test_value_norms.append(v_norms)
                    test_accept_labels.append(accept_labels)
                    test_attn_labels.append(attn_labels)

            # Per-problem Spearman (average across steps)
            if result['step_accept_sens']:
                step_rhos = []
                for s_i in range(len(result['step_accept_sens'])):
                    a = result['step_accept_sens'][s_i]
                    p = result['step_ppl_sens'][s_i]
                    t = result['step_attn_imp'][s_i]
                    min_l = min(a.numel(), p.numel(), t.numel())
                    if min_l >= 10:
                        rhos = pairwise_spearman(a[:min_l], p[:min_l], t[:min_l])
                        step_rhos.append(rhos)
                if step_rhos:
                    avg_rhos = {}
                    for key in step_rhos[0]:
                        rho_vals = [r[key][0] for r in step_rhos]
                        avg_rhos[key] = float(np.mean(rho_vals))
                else:
                    avg_rhos = {}
            else:
                avg_rhos = {}

            problem_results.append({
                'question': problem['question'][:100],
                'split': split_name,
                'num_steps': result['num_steps'],
                'num_tokens': result['num_tokens'],
                'num_measured_steps': len(result['step_accept_sens']),
                'avg_spearman': avg_rhos,
            })
            logger.info(
                "  Steps: %d, Measured: %d, Rhos: %s",
                result['num_steps'],
                len(result['step_accept_sens']),
                {k: f"{v:.3f}" for k, v in avg_rhos.items()} if avg_rhos else "N/A",
            )

        except Exception as e:
            logger.error("  Failed: %s", e)
            continue

        # Clear CUDA cache between problems
        torch.cuda.empty_cache()

    # --- Aggregate Spearman correlations ---
    if not all_accept_sens:
        logger.error("No results collected. Aborting.")
        return

    cat_accept = torch.cat(all_accept_sens)
    cat_ppl = torch.cat(all_ppl_sens)
    cat_attn = torch.cat(all_attn_imp)

    global_rhos = pairwise_spearman(cat_accept, cat_ppl, cat_attn)

    # --- Train and evaluate predictors ---
    num_heads = draft_model.config.num_attention_heads
    accept_predictor_metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
    attn_proxy_metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

    if train_draft_attns and test_draft_attns:
        logger.info("Training AcceptPredictor (%d train steps, %d test steps)...",
                     len(train_draft_attns), len(test_draft_attns))
        accept_predictor_metrics = train_and_evaluate_predictor(
            train_draft_attns, train_value_norms, train_accept_labels,
            test_draft_attns, test_value_norms, test_accept_labels,
            num_heads=num_heads,
        )

        logger.info("Training AttentionProxy predictor...")
        attn_proxy_metrics = train_and_evaluate_predictor(
            train_draft_attns, train_value_norms, train_attn_labels,
            test_draft_attns, test_value_norms, test_accept_labels,
            num_heads=num_heads,
        )
    else:
        logger.warning("Insufficient data for predictor training/testing.")

    # --- Decision gates ---
    logger.info("=" * 60)
    logger.info("TRIPLE DIVERGENCE RESULTS (%d problems)", len(problem_results))
    logger.info("=" * 60)

    logger.info("Pairwise Spearman correlations:")
    all_rho_pass = True
    for name, (rho, pval) in global_rhos.items():
        status = "< 0.7" if abs(rho) < 0.7 else ">= 0.7"
        logger.info("  %s: rho=%.3f (p=%.4f) [%s]", name, rho, pval, status)
        if abs(rho) >= 0.7:
            all_rho_pass = False

    divergence_pass = all_rho_pass
    logger.info("Gate: All pairwise rho < 0.7? %s", "PASS" if divergence_pass else "FAIL")

    logger.info("")
    logger.info("AcceptPredictor: F1=%.3f, Prec=%.3f, Rec=%.3f",
                accept_predictor_metrics['f1'],
                accept_predictor_metrics['precision'],
                accept_predictor_metrics['recall'])
    logger.info("AttentionProxy:  F1=%.3f, Prec=%.3f, Rec=%.3f",
                attn_proxy_metrics['f1'],
                attn_proxy_metrics['precision'],
                attn_proxy_metrics['recall'])

    predictor_f1_pass = accept_predictor_metrics['f1'] > 0.75
    logger.info("Gate: AcceptPredictor F1 > 0.75? %s (%.3f)",
                "PASS" if predictor_f1_pass else "FAIL",
                accept_predictor_metrics['f1'])

    accept_beats_attn = accept_predictor_metrics['f1'] > attn_proxy_metrics['f1']
    logger.info("Gate: AcceptPredictor F1 > AttentionProxy F1? %s (%.3f vs %.3f)",
                "PASS" if accept_beats_attn else "FAIL",
                accept_predictor_metrics['f1'],
                attn_proxy_metrics['f1'])

    overall_pass = divergence_pass and predictor_f1_pass and accept_beats_attn
    logger.info("")
    logger.info("OVERALL: %s", "PASS -- proceed to Block 3" if overall_pass else "FAIL -- investigate before proceeding")

    # --- Save results ---
    output = {
        'config': {
            'model': args.model,
            'num_problems': args.num_problems,
            'num_train': len(train_problems),
            'num_test': len(test_problems),
            'gamma': args.gamma,
            'temperature': args.temperature,
            'samples_per_step': args.samples_per_step,
            'sample_fraction': args.sample_fraction,
            'max_tokens': args.max_tokens,
        },
        'spearman_correlations': {
            name: {'rho': rho, 'pval': pval}
            for name, (rho, pval) in global_rhos.items()
        },
        'accept_predictor': accept_predictor_metrics,
        'attention_proxy': attn_proxy_metrics,
        'gates': {
            'divergence': {
                'criterion': 'all pairwise rho < 0.7',
                'passed': divergence_pass,
                'details': {
                    name: {'rho': rho, 'passed': abs(rho) < 0.7}
                    for name, (rho, _) in global_rhos.items()
                },
            },
            'predictor_f1': {
                'criterion': 'AcceptPredictor F1 > 0.75',
                'passed': predictor_f1_pass,
                'value': accept_predictor_metrics['f1'],
            },
            'accept_beats_attn': {
                'criterion': 'AcceptPredictor F1 > AttentionProxy F1',
                'passed': accept_beats_attn,
                'accept_f1': accept_predictor_metrics['f1'],
                'attn_f1': attn_proxy_metrics['f1'],
            },
            'overall_passed': overall_pass,
        },
        'aggregate': {
            'total_tokens_measured': int(cat_accept.numel()),
            'num_problems_completed': len(problem_results),
            'num_train_steps': len(train_draft_attns),
            'num_test_steps': len(test_draft_attns),
        },
        'per_problem': problem_results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'triple_divergence.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Block 2: Triple Divergence + Predictor Validation",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--num_problems", type=int, default=100)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samples_per_step", type=int, default=50)
    parser.add_argument("--sample_fraction", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="results/triple_divergence")
    args = parser.parse_args()
    run_triple_divergence(args)


if __name__ == "__main__":
    main()
