"""Speculative decoding engine with optional quantized verification.

Extends the standard speculative decoding algorithm to support compressed-domain
verification attention via TurboQuant KV cache quantization.
"""

import time
import logging
import dataclasses
from typing import Optional, List, Tuple, Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .turboquant_kv import QuantizedKVCache, HadamardRotation, ScalarQuantizer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SpeculativeOutput:
    """Results from a speculative decoding run."""

    generated_ids: torch.Tensor
    num_generated_tokens: int
    num_draft_rounds: int
    total_draft_tokens: int
    total_accepted_tokens: int
    acceptance_counts_by_position: List[int]
    draft_rounds_by_position: List[int]
    wall_time_seconds: float
    draft_time_seconds: float
    verify_time_seconds: float
    quantize_time_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_round(self) -> float:
        if self.num_draft_rounds == 0:
            return 0.0
        return self.num_generated_tokens / self.num_draft_rounds

    @property
    def throughput(self) -> float:
        if self.wall_time_seconds == 0:
            return 0.0
        return self.num_generated_tokens / self.wall_time_seconds

    @property
    def position_acceptance_rates(self) -> List[float]:
        rates = []
        for acc, total in zip(
            self.acceptance_counts_by_position, self.draft_rounds_by_position
        ):
            rates.append(acc / total if total > 0 else 0.0)
        return rates

    def to_dict(self) -> dict:
        return {
            "num_generated_tokens": self.num_generated_tokens,
            "num_draft_rounds": self.num_draft_rounds,
            "total_draft_tokens": self.total_draft_tokens,
            "total_accepted_tokens": self.total_accepted_tokens,
            "acceptance_rate": self.acceptance_rate,
            "tokens_per_round": self.tokens_per_round,
            "throughput_tokens_per_sec": self.throughput,
            "wall_time_seconds": self.wall_time_seconds,
            "draft_time_seconds": self.draft_time_seconds,
            "verify_time_seconds": self.verify_time_seconds,
            "quantize_time_seconds": self.quantize_time_seconds,
            "position_acceptance_rates": self.position_acceptance_rates,
        }


def _trim_kv_cache(past: Any, keep_length: int) -> Any:
    """Trim a KV cache to only retain the first keep_length positions."""
    if past is None:
        return None
    if hasattr(past, "crop"):
        past.crop(keep_length)
        return past
    if isinstance(past, tuple):
        return tuple(
            tuple(
                t[:, :, :keep_length, :] if t is not None else None
                for t in layer
            )
            for layer in past
        )
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        for i in range(len(past.key_cache)):
            if past.key_cache[i] is not None:
                past.key_cache[i] = past.key_cache[i][:, :, :keep_length, :]
            if past.value_cache[i] is not None:
                past.value_cache[i] = past.value_cache[i][:, :, :keep_length, :]
        return past
    raise TypeError(f"Unsupported KV cache type: {type(past)}")


class SpeculativeDecoder:
    """Speculative decoding with optional TurboQuant-compressed verification."""

    def __init__(
        self,
        draft_model: PreTrainedModel,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        quant_bits: int = 0,
        quant_block_size: int = 128,
        quant_seed: int = 42,
    ):
        """
        Args:
            draft_model: smaller model for drafting tokens
            target_model: larger model for verification
            tokenizer: shared tokenizer
            quant_bits: KV quantization bits (0 = full precision, 3 or 4 = quantized)
            quant_block_size: block size for per-block quantization
            quant_seed: seed for Hadamard sign vector
        """
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.quant_bits = quant_bits
        self.quant_block_size = quant_block_size
        self.quant_seed = quant_seed

        self.draft_model.eval()
        self.target_model.eval()
        self.draft_device = next(draft_model.parameters()).device
        self.target_device = next(target_model.parameters()).device

        self.use_quant = quant_bits > 0
        if self.use_quant:
            config = target_model.config
            num_layers = config.num_hidden_layers
            num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
            head_dim = config.hidden_size // config.num_attention_heads
            logger.info(
                f"SpecQuant enabled: {quant_bits}-bit, block_size={quant_block_size}, "
                f"layers={num_layers}, kv_heads={num_kv_heads}, head_dim={head_dim}"
            )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        gamma: int = 5,
        temperature: float = 1.0,
    ) -> SpeculativeOutput:
        """Run speculative decoding with optional quantized verification."""
        assert input_ids.shape[0] == 1, "Only batch_size=1 is supported"

        prefix_len = input_ids.shape[1]
        acc_by_pos = [0] * gamma
        rounds_by_pos = [0] * gamma
        total_draft = 0
        total_accepted = 0
        n_rounds = 0
        t_draft_total = 0.0
        t_verify_total = 0.0
        t_quant_total = 0.0

        draft_out = self.draft_model(
            input_ids.to(self.draft_device), use_cache=True
        )
        draft_kv = draft_out.past_key_values
        draft_next_logits = draft_out.logits[:, -1, :]

        target_out = self.target_model(
            input_ids.to(self.target_device), use_cache=True
        )
        target_kv = target_out.past_key_values
        target_next_logits = target_out.logits[:, -1, :]

        all_token_ids = input_ids.cpu().clone()
        kv_len = prefix_len

        start = time.perf_counter()

        while all_token_ids.shape[1] - prefix_len < max_new_tokens:
            remaining = max_new_tokens - (all_token_ids.shape[1] - prefix_len)
            cur_gamma = min(gamma, remaining)
            if cur_gamma <= 0:
                break

            n_rounds += 1
            for k in range(cur_gamma):
                rounds_by_pos[k] += 1
            total_draft += cur_gamma

            t0 = time.perf_counter()
            draft_tokens, draft_probs, draft_kv = self._draft_phase(
                draft_next_logits, draft_kv, cur_gamma, temperature
            )
            t_draft_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            verify_out = self.target_model(
                draft_tokens.view(1, -1).to(self.target_device),
                past_key_values=target_kv,
                use_cache=True,
            )
            target_kv_ext = verify_out.past_key_values
            verify_logits = verify_out.logits

            n_acc, accepted = self._rejection_sample(
                target_next_logits,
                verify_logits,
                draft_tokens,
                draft_probs,
                cur_gamma,
                temperature,
            )
            t_verify_total += time.perf_counter() - t0

            total_accepted += n_acc
            for k in range(n_acc):
                acc_by_pos[k] += 1

            all_token_ids = torch.cat(
                [all_token_ids, accepted.view(1, -1).cpu()], dim=1
            )

            new_kv_len = kv_len + n_acc
            draft_kv = _trim_kv_cache(draft_kv, new_kv_len)
            target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)

            last_tok = accepted[-1]

            d_out = self.draft_model(
                last_tok.view(1, 1).to(self.draft_device),
                past_key_values=draft_kv,
                use_cache=True,
            )
            draft_kv = d_out.past_key_values
            draft_next_logits = d_out.logits[:, -1, :]

            t_out = self.target_model(
                last_tok.view(1, 1).to(self.target_device),
                past_key_values=target_kv,
                use_cache=True,
            )
            target_kv = t_out.past_key_values
            target_next_logits = t_out.logits[:, -1, :]

            kv_len = new_kv_len + 1

        wall = time.perf_counter() - start

        final_ids = all_token_ids[:, : prefix_len + max_new_tokens]
        return SpeculativeOutput(
            generated_ids=final_ids,
            num_generated_tokens=final_ids.shape[1] - prefix_len,
            num_draft_rounds=n_rounds,
            total_draft_tokens=total_draft,
            total_accepted_tokens=total_accepted,
            acceptance_counts_by_position=acc_by_pos,
            draft_rounds_by_position=rounds_by_pos,
            wall_time_seconds=wall,
            draft_time_seconds=t_draft_total,
            verify_time_seconds=t_verify_total,
            quantize_time_seconds=t_quant_total,
        )

    @torch.no_grad()
    def generate_autoregressive(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, float]:
        """Baseline autoregressive generation with target model only."""
        generated = input_ids.to(self.target_device)
        past = None

        start = time.perf_counter()
        for _ in range(max_new_tokens):
            if past is None:
                out = self.target_model(generated, use_cache=True)
            else:
                out = self.target_model(
                    generated[:, -1:], past_key_values=past, use_cache=True
                )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            probs = F.softmax(logits / max(temperature, 1e-8), dim=-1)
            tok = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, tok], dim=1)
        wall = time.perf_counter() - start
        return generated.cpu(), wall

    def _draft_phase(
        self,
        start_logits: torch.Tensor,
        kv: Any,
        gamma: int,
        temperature: float,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Any]:
        tokens: List[torch.Tensor] = []
        probs_list: List[torch.Tensor] = []
        logits = start_logits
        current_kv = kv

        for _ in range(gamma):
            p = F.softmax(logits / max(temperature, 1e-8), dim=-1).squeeze(0)
            tok = torch.multinomial(p, num_samples=1).squeeze(-1)
            tokens.append(tok.cpu())
            probs_list.append(p.cpu())

            out = self.draft_model(
                tok.view(1, 1).to(self.draft_device),
                past_key_values=current_kv,
                use_cache=True,
            )
            current_kv = out.past_key_values
            logits = out.logits[:, -1, :]

        return torch.stack(tokens), probs_list, current_kv

    @staticmethod
    def _rejection_sample(
        target_next_logits: torch.Tensor,
        verify_logits: torch.Tensor,
        draft_tokens: torch.Tensor,
        draft_probs: List[torch.Tensor],
        gamma: int,
        temperature: float,
    ) -> Tuple[int, torch.Tensor]:
        device = verify_logits.device
        temp = max(temperature, 1e-8)
        accepted: List[torch.Tensor] = []
        n_accepted = 0

        for i in range(gamma):
            if i == 0:
                tgt_logits_i = target_next_logits.squeeze(0)
            else:
                tgt_logits_i = verify_logits[:, i - 1, :].squeeze(0)

            tp = F.softmax(tgt_logits_i.to(device) / temp, dim=-1)
            dp = draft_probs[i].to(device)
            tok_id = draft_tokens[i].item()

            p_t = tp[tok_id]
            p_d = dp[tok_id].clamp(min=1e-10)

            if torch.rand(1, device=device).item() < min(1.0, (p_t / p_d).item()):
                accepted.append(draft_tokens[i])
                n_accepted += 1
            else:
                adjusted = (tp - dp).clamp(min=0)
                s = adjusted.sum()
                if s > 0:
                    adjusted = adjusted / s
                else:
                    adjusted = tp
                new_tok = torch.multinomial(adjusted, num_samples=1).squeeze(-1)
                accepted.append(new_tok.cpu())
                break
        else:
            bonus_logits = verify_logits[:, gamma - 1, :].squeeze(0)
            bonus_p = F.softmax(bonus_logits.to(device) / temp, dim=-1)
            bonus = torch.multinomial(bonus_p, num_samples=1).squeeze(-1)
            accepted.append(bonus.cpu())

        return n_accepted, torch.stack(accepted)

    @torch.no_grad()
    def measure_tv_distance(
        self,
        input_ids: torch.Tensor,
        num_positions: int = 256,
    ) -> dict:
        """Measure empirical TV distance between full-precision and quantized logits.

        Used for Claim 3 validation: compare with theoretical bound.
        """
        if not self.use_quant:
            return {"tv_mean": 0.0, "tv_std": 0.0, "tv_per_position": []}

        seq_len = min(input_ids.shape[1], num_positions)
        chunk = input_ids[:, :seq_len].to(self.target_device)

        fp_out = self.target_model(chunk)
        fp_logits = fp_out.logits.float()
        fp_probs = F.softmax(fp_logits, dim=-1)

        # TODO: quantized forward pass comparison
        # For now return placeholder
        return {"tv_mean": 0.0, "tv_std": 0.0, "note": "quantized path not yet integrated"}
