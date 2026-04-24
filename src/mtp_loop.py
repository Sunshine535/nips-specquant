"""Shared MTP draft/verify step — single source of truth for all scripts.

All experiment scripts (oracle_sensitivity, core_comparison, calibrate_mara)
must use this helper for MTP speculative decoding. This prevents the P0 bug
where core_comparison.py used decoder.draft_model (aliased to target_model
in MTP mode) instead of the MTP head.

Usage:
    from src.mtp_loop import mtp_draft_step, verify_and_accept
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DraftResult:
    """Output of a single MTP draft step."""
    tokens: torch.Tensor       # [gamma] draft token ids
    probs: List[torch.Tensor]  # list of [vocab_size] probability distributions
    gamma: int


@dataclass
class VerifyResult:
    """Output of verification + rejection sampling."""
    accepted_tokens: torch.Tensor   # [n_accepted] accepted token ids
    n_accepted: int
    target_logits: torch.Tensor     # [gamma, vocab_size] target verification logits
    target_next_logits: torch.Tensor  # [1, vocab_size] next-token logits after accepted
    acceptance_rate: float
    target_kv: object               # updated KV cache after trim


@torch.no_grad()
def mtp_draft_step(
    target_model,
    mtp_head,
    target_next_logits: torch.Tensor,
    target_kv,
    kv_len: int,
    gamma: int,
    temperature: float = 0.0,
    device: Optional[torch.device] = None,
) -> Tuple[DraftResult, object]:
    """Draft gamma tokens using MTP head on target model's hidden states.

    This is the CORRECT MTP drafting path. It must NOT use decoder.draft_model,
    which in MTP mode is aliased to the target model itself.

    Args:
        target_model: The target/verifier model
        mtp_head: The MTP head for drafting
        target_next_logits: [1, vocab_size] or [vocab_size] logits from last target forward
        target_kv: Current target KV cache
        kv_len: Current KV cache length
        gamma: Number of tokens to draft
        temperature: Sampling temperature (0 = greedy)
        device: Target device

    Returns:
        (DraftResult, updated_target_kv)
    """
    if device is None:
        device = target_next_logits.device

    draft_tokens = []
    draft_probs = []

    cur_logits = target_next_logits.squeeze(0) if target_next_logits.dim() > 1 else target_next_logits

    for step in range(gamma):
        # Sample or argmax from current logits
        if temperature > 0:
            probs = F.softmax(cur_logits / temperature, dim=-1)
            tok = torch.multinomial(probs, 1).squeeze(-1)
        else:
            tok = cur_logits.argmax(dim=-1)
            probs = F.softmax(cur_logits, dim=-1)

        draft_tokens.append(tok.item() if tok.dim() == 0 else tok.item())
        draft_probs.append(probs.cpu())

        # Feed token through target model to get hidden state for MTP
        tok_tensor = torch.tensor([[draft_tokens[-1]]], device=device)
        t_out = target_model(
            tok_tensor,
            past_key_values=target_kv,
            use_cache=True,
            output_hidden_states=True,
        )
        target_kv = t_out.past_key_values

        # MTP head predicts next token from hidden state
        mtp_pos = torch.tensor([[kv_len + len(draft_tokens)]], device=device)
        mtp_logits, _, _ = mtp_head(
            tok_tensor,
            t_out.hidden_states[-1][:, -1:, :],
            mtp_pos,
        )
        cur_logits = mtp_logits.squeeze(0).squeeze(0)

    tokens_tensor = torch.tensor(draft_tokens, device=device)

    return DraftResult(
        tokens=tokens_tensor,
        probs=draft_probs,
        gamma=gamma,
    ), target_kv


@torch.no_grad()
def verify_and_accept(
    decoder,
    target_model,
    target_kv,
    draft_result: DraftResult,
    target_next_logits: torch.Tensor,
    kv_len: int,
    temperature: float = 0.0,
    device: Optional[torch.device] = None,
) -> VerifyResult:
    """Verify draft tokens and do rejection sampling.

    Uses decoder._rejection_sample for consistency with the main SD engine.
    """
    if device is None:
        device = draft_result.tokens.device

    draft_tokens = draft_result.tokens
    gamma = draft_result.gamma

    # Target model verifies all draft tokens at once
    verify_out = target_model(
        draft_tokens.view(1, -1).to(device),
        past_key_values=target_kv,
        use_cache=True,
    )
    target_kv_ext = verify_out.past_key_values
    verify_logits = verify_out.logits

    # Rejection sampling
    n_acc, accepted = decoder._rejection_sample(
        target_next_logits, verify_logits,
        draft_tokens, draft_result.probs,
        gamma, temperature,
    )

    # Trim KV cache to accepted length
    from src.speculative_decode import _trim_kv_cache
    new_kv_len = kv_len + n_acc
    target_kv_trimmed = _trim_kv_cache(target_kv_ext, new_kv_len)

    acceptance_rate = n_acc / max(gamma, 1)

    # Get next-token logits from the last accepted position
    # (This will be done by the caller after resync)
    return VerifyResult(
        accepted_tokens=accepted,
        n_accepted=n_acc,
        target_logits=verify_logits,
        target_next_logits=target_next_logits,
        acceptance_rate=acceptance_rate,
        target_kv=target_kv_trimmed,
    )


@torch.no_grad()
def resync_after_accept(
    target_model,
    mtp_head,
    last_tok: torch.Tensor,
    target_kv,
    new_kv_len: int,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, object, int]:
    """Resync target model and MTP head after acceptance.

    Returns:
        (target_next_logits, draft_next_logits, updated_kv, new_kv_len)

    CRITICAL: kv_len is advanced to new_kv_len + 1 after the target forward.
    This matches src/speculative_decode.py:498.
    """
    if device is None:
        device = last_tok.device

    t_out = target_model(
        last_tok.view(1, 1).to(device),
        past_key_values=target_kv,
        use_cache=True,
        output_hidden_states=True,
    )
    target_kv = t_out.past_key_values
    target_next_logits = t_out.logits[:, -1, :]

    # Advance kv_len AFTER target forward
    kv_len = new_kv_len + 1
    assert target_kv[0][0].shape[2] == kv_len, \
        f"KV length mismatch after resync: expected {kv_len}, got {target_kv[0][0].shape[2]}"

    # MTP head for next draft
    resync_pos = torch.tensor([[kv_len]], device=device)
    mtp_logits, _, _ = mtp_head(
        last_tok.view(1, 1).to(device),
        t_out.hidden_states[-1][:, -1:, :],
        resync_pos,
    )
    draft_next_logits = mtp_logits.squeeze(1)

    return target_next_logits, draft_next_logits, target_kv, kv_len
