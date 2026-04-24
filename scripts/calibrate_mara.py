"""MARA Calibration: Collect oracle risk labels and train risk predictor.

Usage:
    python scripts/calibrate_mara.py \
        --model Qwen/Qwen3.5-9B \
        --num_calib 8 \
        --num_eval 8 \
        --sample_fraction 0.3 \
        --output_dir results/mara/calib
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.accept_risk import (
    AcceptanceRiskOracle,
    AcceptanceRiskPredictor,
    RiskBudgetAllocator,
    MarginUncertaintyGate,
    RiskLabel,
    RiskLabelSet,
)
from src.gpu_auto import load_model_mtp, print_gpu_summary
from src.mtp_loop import mtp_draft_step, verify_and_accept, resync_after_accept
from src.repro import (
    RunMetadata,
    SplitManifest,
    make_calib_eval_split,
    make_coupled_uniforms,
    set_global_seed,
)
from src.speculative_decode import SpeculativeDecoder, _trim_kv_cache
from src.utils import get_kv_tensors, set_kv_tensors, get_kv_layer_indices

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_gsm8k_split(split: str, num_problems: int, seed: int = 42):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(ds), min(num_problems, len(ds)), replace=False)
    return [{"question": ds[int(i)]["question"], "answer": ds[int(i)]["answer"]} for i in indices]


def format_prompt(question: str) -> str:
    return f"Solve this math problem step by step.\n\nQuestion: {question}\n\nSolution:"


@torch.no_grad()
def collect_risk_labels(
    target_model,
    mtp_head,
    tokenizer,
    decoder,
    problems,
    gamma: int = 5,
    temperature: float = 0.0,
    max_tokens: int = 128,
    sample_fraction: float = 0.3,
    margin_threshold: float = 2.0,
    seed: int = 42,
) -> RiskLabelSet:
    """Run MTP SD on problems, perturb KV tokens, collect continuous risk labels."""
    gen = set_global_seed(seed)
    device = next(target_model.parameters()).device
    all_labels = []

    for pidx, problem in enumerate(problems):
        prompt = format_prompt(problem["question"])
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        logger.info("[%d/%d] Collecting risk labels for '%s...'",
                    pidx + 1, len(problems), problem["question"][:50])

        try:
            # Prefill
            t_out = target_model(
                input_ids.to(device), use_cache=True, output_hidden_states=True,
            )
            target_kv = t_out.past_key_values
            target_next_logits = t_out.logits[:, -1, :]
            kv_len = input_ids.shape[1]

            # MTP initial logits
            resync_pos = torch.tensor([[kv_len]], device=device)
            last_tok_id = input_ids[0, -1]
            mtp_logits, _, _ = mtp_head(
                last_tok_id.view(1, 1).to(device),
                t_out.hidden_states[-1][:, -1:, :],
                resync_pos,
            )
            draft_next_logits = mtp_logits.squeeze(1)

            step_idx = 0
            tokens_generated = 0

            while tokens_generated < max_tokens:
                remaining = max_tokens - tokens_generated
                cur_gamma = min(gamma, remaining)
                if cur_gamma <= 0:
                    break

                # Draft using shared MTP helper
                draft_result, draft_kv = mtp_draft_step(
                    target_model, mtp_head,
                    draft_next_logits, target_kv,
                    kv_len, cur_gamma, temperature, device,
                )

                # Pre-sample coupled uniforms for paired measurement
                coupled_u = make_coupled_uniforms(1, cur_gamma, gen)[0]

                # Verify
                verify_out = target_model(
                    draft_result.tokens.view(1, -1).to(device),
                    past_key_values=target_kv,
                    use_cache=True,
                )
                target_kv_ext = verify_out.past_key_values
                verify_logits = verify_out.logits[0]  # [gamma, vocab]

                # Compute full-KV acceptance and margin
                target_probs_full = F.softmax(verify_logits, dim=-1)
                full_accept = _compute_acceptance_count(
                    target_probs_full, draft_result.probs,
                    draft_result.tokens, coupled_u,
                )
                full_margin = _compute_margin(verify_logits)

                # Sample KV tokens to perturb
                kv_layers = target_kv
                if isinstance(kv_layers, (list, tuple)) and len(kv_layers) > 0:
                    actual_kv_len = kv_layers[0][0].shape[2]
                else:
                    actual_kv_len = kv_len

                n_sample = max(1, int(actual_kv_len * sample_fraction))
                sample_indices = torch.randperm(actual_kv_len, generator=gen)[:n_sample].sort().values

                # Perturb each sampled token under each action
                # Use get_kv_layer_indices to only touch MHA layers (skip LinearAttention)
                mha_layers = get_kv_layer_indices(target_kv)

                for tok_idx in sample_indices.tolist():
                    for action in ["4bit", "2bit"]:
                        bits = 2 if action == "2bit" else 4
                        n_levels = 2**bits - 1

                        # Quantize single token in MHA layers only
                        import copy
                        kv_perturbed = copy.deepcopy(target_kv)
                        for li in mha_layers:
                            k, v = get_kv_tensors(kv_perturbed, li)
                            if k is None or tok_idx >= k.shape[2]:
                                continue
                            for tensor in [k, v]:
                                val = tensor[:, :, tok_idx, :]
                                vmin, vmax = val.min(), val.max()
                                if vmax > vmin:
                                    norm = (val - vmin) / (vmax - vmin)
                                    q = torch.round(norm * n_levels) / n_levels
                                    tensor[:, :, tok_idx, :] = q * (vmax - vmin) + vmin
                            set_kv_tensors(kv_perturbed, li, k, v)

                        # Re-verify with perturbed KV
                        perturbed_out = target_model(
                            draft_result.tokens.view(1, -1).to(device),
                            past_key_values=kv_perturbed,
                            use_cache=True,
                        )
                        perturbed_logits = perturbed_out.logits[0]
                        perturbed_probs = F.softmax(perturbed_logits, dim=-1)

                        pert_accept = _compute_acceptance_count(
                            perturbed_probs, draft_result.probs,
                            draft_result.tokens, coupled_u,
                        )
                        pert_margin = _compute_margin(perturbed_logits)
                        tv = 0.5 * (target_probs_full - perturbed_probs).abs().sum(-1).mean().item()

                        acceptance_risk = max(0.0, full_accept - pert_accept)
                        margin_risk = 0.0
                        if full_margin < margin_threshold:
                            margin_risk = max(0.0, full_margin - pert_margin)

                        all_labels.append(RiskLabel(
                            step_idx=step_idx,
                            token_idx=tok_idx,
                            action=action,
                            alpha_full=full_accept,
                            alpha_perturbed=pert_accept,
                            acceptance_risk=acceptance_risk,
                            tv_distance=tv,
                            margin_full=full_margin,
                            margin_perturbed=pert_margin,
                            margin_risk=margin_risk,
                        ))

                        del kv_perturbed

                # Accept and advance
                n_acc, accepted = decoder._rejection_sample(
                    target_next_logits, verify_out.logits,
                    draft_result.tokens, draft_result.probs,
                    cur_gamma, temperature,
                )

                new_kv_len = kv_len + n_acc
                target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)
                last_tok = accepted[-1]
                tokens_generated += n_acc
                step_idx += 1

                # Resync
                target_next_logits, draft_next_logits, target_kv, kv_len = \
                    resync_after_accept(target_model, mtp_head, last_tok, target_kv, new_kv_len, device)

                if last_tok.item() == tokenizer.eos_token_id:
                    break

            logger.info("  Collected %d risk labels from %d steps",
                        len(all_labels) - sum(1 for _ in []), step_idx)

        except Exception as e:
            import traceback
            logger.error("  Failed: %s\n%s", e, traceback.format_exc())
            continue

        torch.cuda.empty_cache()

    return RiskLabelSet(
        labels=all_labels,
        metadata={"num_problems": len(problems), "seed": seed, "gamma": gamma},
    )


def _compute_acceptance_count(
    target_probs, draft_probs_list, draft_tokens, uniforms
) -> float:
    gamma = len(draft_tokens)
    accepted = 0
    for j in range(gamma):
        t_idx = draft_tokens[j].item()
        p_t = target_probs[j, t_idx].item()
        p_d = draft_probs_list[j][t_idx].item()
        ratio = min(1.0, p_t / p_d) if p_d > 0 else (1.0 if p_t > 0 else 0.0)
        if uniforms[j].item() < ratio:
            accepted += 1
        else:
            break
    return accepted / max(1, gamma)


def _compute_margin(logits: torch.Tensor) -> float:
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    top2 = logits.topk(2, dim=-1).values
    return (top2[:, 0] - top2[:, 1]).abs().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B")
    parser.add_argument("--num_calib", type=int, default=8)
    parser.add_argument("--num_eval", type=int, default=8)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--sample_fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results/mara/calib")
    parser.add_argument("--shard", type=int, default=None, help="Shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=None, help="Total number of shards")
    parser.add_argument("--output", type=str, default=None, help="Output file path (for shard mode)")
    args = parser.parse_args()

    print_gpu_summary()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save run metadata
    meta = RunMetadata(
        model=args.model, dataset="gsm8k", seed=args.seed,
        gamma=args.gamma, temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_problems=args.num_calib + args.num_eval,
    )
    meta.save(os.path.join(args.output_dir, "run_metadata.json"))

    # Load model with MTP
    logger.info("Loading model: %s", args.model)
    target_model, mtp_head, tokenizer, _plan = load_model_mtp(args.model)
    device = next(target_model.parameters()).device

    decoder = SpeculativeDecoder(
        target_model=target_model,
        draft_model=None,
        tokenizer=tokenizer,
        mtp_head=mtp_head,
    )

    # Load calibration problems (from TRAIN split, not test)
    logger.info("Loading %d calibration problems from train split", args.num_calib)
    all_calib_problems = load_gsm8k_split("train", args.num_calib, seed=args.seed)

    # Shard support: split problems across GPUs
    if args.shard is not None and args.num_shards is not None:
        n = len(all_calib_problems)
        shard_size = (n + args.num_shards - 1) // args.num_shards
        start = args.shard * shard_size
        end = min(start + shard_size, n)
        calib_problems = all_calib_problems[start:end]
        logger.info("Shard %d/%d: problems [%d, %d) (%d problems)",
                     args.shard, args.num_shards, start, end, len(calib_problems))
    else:
        calib_problems = all_calib_problems

    # Save split manifest
    split = SplitManifest(
        dataset="gsm8k",
        total_problems=args.num_calib + args.num_eval,
        calib_indices=list(range(args.num_calib)),
        eval_indices=list(range(args.num_calib, args.num_calib + args.num_eval)),
        seed=args.seed,
    )
    split.save(os.path.join(args.output_dir, "split_manifest.json"))

    # Collect risk labels
    logger.info("Collecting risk labels...")
    risk_labels = collect_risk_labels(
        target_model, mtp_head, tokenizer, decoder,
        calib_problems,
        gamma=args.gamma,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )

    logger.info("Collected %d total risk labels", len(risk_labels.labels))
    risk_labels.save(os.path.join(args.output_dir, "risk_labels.json"))

    if len(risk_labels.labels) == 0:
        logger.error("No risk labels collected. Cannot train predictor.")
        return

    # Train risk predictor
    X, y = risk_labels.to_tensors()
    logger.info("Training predictor on %d samples (features=%d)", len(X), X.shape[1])

    # Split into train/val for calibration metrics
    n_train = max(1, int(len(X) * 0.8))
    perm = torch.randperm(len(X))
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_val, y_val = X[perm[n_train:]], y[perm[n_train:]]

    predictor = AcceptanceRiskPredictor(input_dim=X.shape[1], hidden_dim=32)
    losses = predictor.fit(X_train, y_train, lr=0.01, epochs=200)
    logger.info("Training losses: %s", losses)

    # Calibration metrics on held-out
    if len(X_val) > 2:
        cal_metrics = predictor.compute_calibration_metrics(X_val, y_val)
        logger.info("Calibration metrics: %s", cal_metrics)
    else:
        cal_metrics = {"spearman_rho": float("nan"), "ece": float("nan"), "n_samples": 0}

    # Save predictor
    predictor.save(os.path.join(args.output_dir, "predictor_weights.pt"))

    # Save calibration report
    report = {
        "num_labels": len(risk_labels.labels),
        "training_losses": losses,
        "calibration_metrics": cal_metrics,
        "risk_stats": {
            "mean_risk": float(y.mean()),
            "std_risk": float(y.std()),
            "max_risk": float(y.max()),
            "frac_nonzero": float((y > 0).float().mean()),
        },
        "metadata": meta.to_dict(),
    }
    with open(os.path.join(args.output_dir, "calibration_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Calibration complete. Output: %s", args.output_dir)
    logger.info("Risk stats: mean=%.4f, std=%.4f, nonzero=%.1f%%",
                report["risk_stats"]["mean_risk"],
                report["risk_stats"]["std_risk"],
                report["risk_stats"]["frac_nonzero"] * 100)


if __name__ == "__main__":
    main()
