"""Oracle Acceptance Sensitivity Study (M0/M1 gate).

Measures per-token acceptance sensitivity in speculative decoding to validate
the core AcceptSpec hypothesis: acceptance sensitivity is sparse.

Usage:
    python scripts/oracle_sensitivity.py \
        --model Qwen/Qwen3.5-9B \
        --num_problems 10 \
        --output results/oracle_sensitivity.json

Decision gate:
    M0 (10 problems): Gini > 0.5
    M1 (100 problems): top-20% tokens capture >80% sensitivity, else ABORT
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from src.speculative_decode import SpeculativeDecoder
from src.acceptspec import (
    AcceptSensitivityOracle,
    OracleStudyResult,
    SensitivityResult,
)
from src.gpu_auto import plan_devices, load_models, load_model_mtp, print_gpu_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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


def run_oracle_study(args):
    """Run the oracle acceptance sensitivity study."""
    print_gpu_summary()

    # Auto-detect GPUs and plan device placement
    plan = plan_devices()
    logger.info("Loading models with auto device plan: %s", plan.description)

    model, mtp_head, tokenizer, plan = load_model_mtp(args.model, plan=plan)
    target_model = model

    # Create SD decoder (no quantization — we want baseline acceptance)
    decoder = SpeculativeDecoder(
        target_model=target_model,
        tokenizer=tokenizer,
        mtp_head=mtp_head,
        quant_bits=0,
    )

    # Create oracle
    oracle = AcceptSensitivityOracle(
        target_model=target_model,
        quantizer_bits=2,
        sample_fraction=args.sample_fraction,
    )

    # Load problems
    logger.info("Loading GSM8K (%d problems)...", args.num_problems)
    problems = load_gsm8k(args.num_problems)

    # Run study
    all_sensitivities = []
    all_attention_importances = []
    all_ginis = []
    all_alphas = []
    problem_results = []

    for i, problem in enumerate(problems):
        prompt = format_prompt(problem['question'])
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        logger.info("[%d/%d] Running SD for '%s...'", i + 1, len(problems), problem['question'][:50])

        try:
            # Run speculative decoding and measure sensitivity at each step
            result = run_instrumented_sd(
                decoder, oracle, input_ids,
                max_new_tokens=args.max_tokens,
                gamma=args.gamma,
                temperature=args.temperature,
                num_samples_per_step=args.samples_per_step,
            )

            if result is not None:
                all_sensitivities.extend(result['step_sensitivities'])
                all_attention_importances.extend(result['step_attentions'])
                all_ginis.extend(result['step_ginis'])
                all_alphas.extend(result['step_alphas'])
                problem_results.append({
                    'question': problem['question'][:100],
                    'num_steps': result['num_steps'],
                    'mean_gini': float(np.mean(result['step_ginis'])) if result['step_ginis'] else 0,
                    'mean_alpha': float(np.mean(result['step_alphas'])) if result['step_alphas'] else 0,
                    'num_tokens_generated': result['num_tokens'],
                })
                logger.info("  Steps: %d, Mean Gini: %.3f, Mean α: %.3f",
                            result['num_steps'],
                            problem_results[-1]['mean_gini'],
                            problem_results[-1]['mean_alpha'])

        except Exception as e:
            logger.error("  Failed: %s", e)
            continue

        # Clear CUDA cache between problems
        torch.cuda.empty_cache()

    # Aggregate results
    if not all_sensitivities:
        logger.error("No results collected. Aborting.")
        return

    # Compute cumulative curve
    all_sens = torch.cat(all_sensitivities)
    all_attn = torch.cat(all_attention_importances)

    sorted_sens, sort_idx = all_sens.sort(descending=True)
    total_sens = sorted_sens.sum().item()

    # Cumulative sensitivity at different retention fractions
    retained_fracs = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    captured = []
    for f in retained_fracs:
        k = max(1, int(f * len(sorted_sens)))
        cap = sorted_sens[:k].sum().item() / max(total_sens, 1e-10)
        captured.append(cap)

    top10 = captured[retained_fracs.index(0.1)]
    top20 = captured[retained_fracs.index(0.2)]
    top50 = captured[retained_fracs.index(0.5)]
    mean_gini = float(np.mean(all_ginis)) if all_ginis else 0
    std_gini = float(np.std(all_ginis)) if all_ginis else 0

    # Spearman correlation between sensitivity and attention importance
    from scipy.stats import spearmanr
    # Only correlate for tokens that were actually sampled (non-zero sensitivity)
    nonzero_mask = all_sens > 0
    if nonzero_mask.sum() > 10:
        rho, pval = spearmanr(
            all_sens[nonzero_mask].numpy(),
            all_attn[nonzero_mask].numpy(),
        )
    else:
        rho, pval = 0.0, 1.0

    # Decision gates
    logger.info("=" * 60)
    logger.info("ORACLE STUDY RESULTS (%d problems)", len(problem_results))
    logger.info("=" * 60)
    logger.info("Top-10%% tokens capture %.1f%% of sensitivity", top10 * 100)
    logger.info("Top-20%% tokens capture %.1f%% of sensitivity", top20 * 100)
    logger.info("Top-50%% tokens capture %.1f%% of sensitivity", top50 * 100)
    logger.info("Mean Gini: %.3f ± %.3f", mean_gini, std_gini)
    logger.info("Spearman ρ (sensitivity vs attention): %.3f (p=%.4f)", rho, pval)

    if args.num_problems <= 10:
        gate = "M0"
        passed = mean_gini > 0.5
        logger.info("Gate %s: Gini > 0.5? %s (%.3f)", gate, "PASS ✓" if passed else "FAIL ✗", mean_gini)
    else:
        gate = "M1"
        passed = top20 > 0.8
        logger.info("Gate %s: Top-20%% > 80%%? %s (%.1f%%)", gate, "PASS ✓" if passed else "FAIL ✗", top20 * 100)

    if not passed:
        logger.warning("⚠️  GATE %s FAILED — consider aborting project", gate)

    # Save results
    output = {
        'config': {
            'model': args.model,
            'num_problems': args.num_problems,
            'gamma': args.gamma,
            'temperature': args.temperature,
            'samples_per_step': args.samples_per_step,
        },
        'aggregate': {
            'retained_fractions': retained_fracs,
            'sensitivity_captured': captured,
            'top10_coverage': top10,
            'top20_coverage': top20,
            'top50_coverage': top50,
            'mean_gini': mean_gini,
            'std_gini': std_gini,
            'spearman_rho': float(rho),
            'spearman_pval': float(pval),
            'total_tokens_measured': int(nonzero_mask.sum()),
            'total_tokens': len(all_sens),
        },
        'gate': {
            'name': gate,
            'passed': passed,
            'criterion': 'gini > 0.5' if gate == 'M0' else 'top20 > 0.8',
        },
        'per_problem': problem_results,
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", args.output)


def run_instrumented_sd(
    decoder: SpeculativeDecoder,
    oracle: AcceptSensitivityOracle,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    gamma: int = 5,
    temperature: float = 0.0,
    num_samples_per_step: int = 50,
) -> dict:
    """Run SD with oracle sensitivity measurement at each step.

    This is a modified version of SpeculativeDecoder.generate() that
    instruments each verification step with oracle sensitivity measurement.
    """
    assert input_ids.shape[0] == 1

    target_model = decoder.target_model
    mtp_head = decoder.mtp_head
    target_device = decoder.target_device

    prefix_len = input_ids.shape[1]

    # Prefill (single model — MTP head provides draft logits from target hidden states)
    target_out = target_model(input_ids.to(target_device), use_cache=True, output_hidden_states=True)
    target_kv = target_out.past_key_values
    target_next_logits = target_out.logits[:, -1, :]
    draft_next_logits = mtp_head(target_out.hidden_states[-1][:, -1:, :]).squeeze(1)

    all_token_ids = input_ids.cpu().clone()
    kv_len = prefix_len

    step_sensitivities = []
    step_attentions = []
    step_ginis = []
    step_alphas = []
    n_steps = 0

    while all_token_ids.shape[1] - prefix_len < max_new_tokens:
        remaining = max_new_tokens - (all_token_ids.shape[1] - prefix_len)
        cur_gamma = min(gamma, remaining)
        if cur_gamma <= 0:
            break

        n_steps += 1

        # Draft phase (MTP head — single forward per token using target KV)
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

            # Feed drafted token through target model to get hidden state for next MTP prediction
            tok_tensor = torch.tensor([[tok]], device=target_device)
            d_out = target_model(tok_tensor, past_key_values=target_kv, use_cache=True, output_hidden_states=True)
            target_kv = d_out.past_key_values
            cur_logits = mtp_head(d_out.hidden_states[-1][:, -1:, :]).squeeze(1)

        draft_tokens = torch.tensor(draft_tokens_list, device=target_device)
        draft_probs = torch.tensor(draft_probs_list)

        # Generate coupled random seeds for paired comparison
        coupled_seeds = torch.rand(cur_gamma)

        # Measure oracle sensitivity BEFORE standard verification
        # Only do this every few steps to save compute
        if n_steps % max(1, gamma) == 1 or n_steps <= 3:
            try:
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
                    step_sensitivities.append(sens_result.sensitivities)
                    step_attentions.append(sens_result.attention_importance)
                    step_ginis.append(sens_result.gini)
                    step_alphas.append(sens_result.alpha_full)
            except Exception as e:
                logger.debug("Oracle measurement failed at step %d: %s", n_steps, e)

        # Standard verification (rewind target KV to pre-draft state, then verify)
        from src.speculative_decode import _trim_kv_cache as _tkv
        target_kv = _tkv(target_kv, kv_len)
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
            draft_tokens, draft_probs,
            cur_gamma, temperature,
        )

        all_token_ids = torch.cat([all_token_ids, accepted.view(1, -1).cpu()], dim=1)

        # Trim KV cache to accepted length
        new_kv_len = kv_len + n_acc
        target_kv = _tkv(target_kv_ext, new_kv_len)

        last_tok = accepted[-1]
        kv_len = new_kv_len

        # Update target and draft (MTP) next logits from single model
        t_out = target_model(
            last_tok.view(1, 1).to(target_device),
            past_key_values=target_kv,
            use_cache=True,
            output_hidden_states=True,
        )
        target_kv = t_out.past_key_values
        target_next_logits = t_out.logits[:, -1, :]
        draft_next_logits = mtp_head(t_out.hidden_states[-1][:, -1:, :]).squeeze(1)

        # Check for EOS
        if last_tok.item() == decoder.tokenizer.eos_token_id:
            break

    return {
        'num_steps': n_steps,
        'num_tokens': all_token_ids.shape[1] - prefix_len,
        'step_sensitivities': step_sensitivities,
        'step_attentions': step_attentions,
        'step_ginis': step_ginis,
        'step_alphas': step_alphas,
    }


def main():
    parser = argparse.ArgumentParser(description="Oracle Acceptance Sensitivity Study")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-9B",
                        help="Model with native MTP head for self-speculation")
    parser.add_argument("--num_problems", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--gamma", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--samples_per_step", type=int, default=50)
    parser.add_argument("--sample_fraction", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="results/oracle_sensitivity.json")
    args = parser.parse_args()
    run_oracle_study(args)


if __name__ == "__main__":
    main()
