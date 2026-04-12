"""Speculative decoding engine with MTP self-speculation and optional KV compression.

Supports two drafting modes:
  - MTP self-speculation (preferred): target model's own MTP head as draft
  - Dual-model (legacy): separate draft model

And two verification architectures:
  - Standard MHA (e.g., Llama, Qwen3.5 MHA layers): QuantizedKVCache
  - GatedDeltaNet linear attention (Qwen3.5 linear layers): LinearAttnVerifier
"""

import time
import logging
import dataclasses
from typing import Optional, List, Tuple, Any

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .turboquant_kv import QuantizedKVCache, HadamardRotation, ScalarQuantizer
from .utils import get_kv_tensors, set_kv_tensors, get_num_kv_layers
from .linear_attn_quantizer import (
    LinearAttnVerifier,
    QuantizedStateCache,
    is_linear_attention_model,
)

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


def _evict_prefix_kv(past: Any, keep_last_n: int = 0) -> None:
    """Release FP prefix KV tensors from the HF cache to free GPU memory.

    For standard MHA caches, replaces K/V with empty tensors.
    For linear attention caches (GatedDeltaNet), this is a no-op since
    recurrent states are fixed-size and managed by the LinearAttnVerifier.
    """
    if past is None:
        return
    # Linear attention (GatedDeltaNet): fixed-size state, nothing to evict.
    # Check for actual LinearAttentionLayer, not just any object with the attr.
    if hasattr(past, "layers") and len(past.layers) > 0:
        layer0_type = type(past.layers[0]).__name__
        if "Linear" in layer0_type or "Recurrent" in layer0_type:
            return
    num_layers = get_num_kv_layers(past)
    for i in range(num_layers):
        k, v = get_kv_tensors(past, i)
        if k is None:
            continue
        device, dtype, shape = k.device, k.dtype, k.shape
        if keep_last_n <= 0:
            new_k = torch.empty(shape[0], shape[1], 0, shape[3], device=device, dtype=dtype)
            new_v = torch.empty(shape[0], shape[1], 0, shape[3], device=device, dtype=dtype)
        else:
            new_k = k[:, :, -keep_last_n:, :]
            new_v = v[:, :, -keep_last_n:, :]
        set_kv_tensors(past, i, new_k, new_v)


def _trim_kv_cache(past: Any, keep_length: int) -> Any:
    """Trim a KV cache to only retain the first keep_length positions.

    For linear-attention caches (GatedDeltaNet), the recurrent state is
    fixed-size and does not need trimming -- we return it unchanged.
    """
    if past is None:
        return None

    # GatedDeltaNet object cache: recurrent states are fixed-size, no trimming.
    if hasattr(past, "recurrent_states"):
        return past

    if hasattr(past, "crop"):
        past.crop(keep_length)
        return past
    if isinstance(past, tuple):
        # Check if this looks like a linear-attn tuple cache.
        # Linear-attn layers store (recurrent_state, conv_state) where
        # recurrent_state is [b, h, d, d] (square last two dims).
        first_layer = past[0]
        if isinstance(first_layer, tuple) and len(first_layer) >= 2:
            rs = first_layer[0]
            if rs is not None and rs.dim() == 4 and rs.shape[-1] == rs.shape[-2]:
                # Fixed-size recurrent state -- nothing to trim.
                return past

        # Standard MHA KV cache as tuple of tuples.
        return tuple(
            tuple(
                t[:, :, :keep_length, :] if t is not None else None
                for t in layer
            )
            for layer in past
        )
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        # Check if key_cache entries look like recurrent states (square last dims).
        if (
            len(past.key_cache) > 0
            and past.key_cache[0] is not None
            and past.key_cache[0].dim() == 4
            and past.key_cache[0].shape[-1] == past.key_cache[0].shape[-2]
        ):
            return past  # Linear-attn in DynamicCache clothing -- no trim.

        for i in range(len(past.key_cache)):
            if past.key_cache[i] is not None:
                past.key_cache[i] = past.key_cache[i][:, :, :keep_length, :]
            if past.value_cache[i] is not None:
                past.value_cache[i] = past.value_cache[i][:, :, :keep_length, :]
        return past
    raise TypeError(f"Unsupported KV cache type: {type(past)}")


class SpeculativeDecoder:
    """Speculative decoding with MTP self-speculation and optional KV compression.

    Preferred mode: single model + MTP head (self-speculation).
    Legacy mode:   separate draft model + target model.

    Automatically detects GatedDeltaNet (linear attention) layers and uses
    ``LinearAttnVerifier`` for state-matrix compression on those layers.
    """

    def __init__(
        self,
        target_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        mtp_head: Optional[Any] = None,
        draft_model: Optional[PreTrainedModel] = None,
        quant_bits: int = 0,
        quant_block_size: int = 128,
        quant_seed: int = 42,
    ):
        """
        Args:
            target_model: the main model for generation and verification
            tokenizer: tokenizer
            mtp_head: Qwen35MTPHead for self-speculation (preferred)
            draft_model: separate draft model (legacy, used if mtp_head is None)
            quant_bits: KV quantization bits (0 = full precision)
            quant_block_size: block size for per-block quantization
            quant_seed: seed for Hadamard sign vector
        """
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.mtp_head = mtp_head
        self.draft_model = draft_model
        self.quant_bits = quant_bits
        self.quant_block_size = quant_block_size
        self.quant_seed = quant_seed

        self.use_mtp = mtp_head is not None
        if not self.use_mtp and draft_model is None:
            raise ValueError("Either mtp_head or draft_model must be provided")

        self.target_model.eval()
        self.target_device = next(target_model.parameters()).device

        if self.use_mtp:
            logger.info("Speculative decoding: MTP self-speculation mode")
            self.draft_device = self.target_device
        else:
            self.draft_model.eval()
            self.draft_device = next(draft_model.parameters()).device
            logger.info("Speculative decoding: dual-model mode (legacy)")

        self.use_quant = quant_bits > 0

        # Detect architecture: GatedDeltaNet (linear attention) vs standard MHA.
        self._is_linear_attn = is_linear_attention_model(target_model)
        self._linear_verifier: Optional[LinearAttnVerifier] = None

        if self.use_quant:
            if self._is_linear_attn:
                self._linear_verifier = LinearAttnVerifier(
                    target_model,
                    bits=quant_bits,
                    block_size=quant_block_size,
                    seed=quant_seed,
                )
                logger.info(
                    "SpecQuant (linear-attn): %d-bit state compression, "
                    "block_size=%d",
                    quant_bits, quant_block_size,
                )
            else:
                config = target_model.config
                num_layers = config.num_hidden_layers
                num_kv_heads = getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                )
                head_dim = config.hidden_size // config.num_attention_heads
                logger.info(
                    "SpecQuant (MHA): %d-bit, block_size=%d, "
                    "layers=%d, kv_heads=%d, head_dim=%d",
                    quant_bits, quant_block_size,
                    num_layers, num_kv_heads, head_dim,
                )

    def _build_qkv_cache(self) -> QuantizedKVCache:
        """Create a QuantizedKVCache matched to the target model architecture."""
        config = self.target_model.config
        num_layers = config.num_hidden_layers
        num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        head_dim = config.hidden_size // config.num_attention_heads
        return QuantizedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            bits=self.quant_bits,
            block_size=self.quant_block_size,
            seed=self.quant_seed,
        )

    def _compress_kv(self, kv: Any, qkv_cache: QuantizedKVCache) -> Any:
        """Round-trip compress target KV cache: quantize then dequantize."""
        if isinstance(kv, tuple):
            new_layers = []
            for i, layer in enumerate(kv):
                k, v = layer[0], layer[1]
                qkv_cache.compress_and_store(i, k.float(), v.float())
                k_rot, v_rot = qkv_cache.get_rotated_kv(i)
                k_deq = qkv_cache.rotation.inverse_rotate(k_rot).to(k.dtype)
                v_deq = qkv_cache.rotation.inverse_rotate(v_rot).to(v.dtype)
                new_layers.append((k_deq, v_deq))
            return tuple(new_layers)

        if hasattr(kv, "key_cache") and hasattr(kv, "value_cache"):
            for i in range(min(len(kv.key_cache), qkv_cache.num_layers)):
                k = kv.key_cache[i]
                v = kv.value_cache[i]
                if k is None:
                    continue
                qkv_cache.compress_and_store(i, k.float(), v.float())
                k_rot, v_rot = qkv_cache.get_rotated_kv(i)
                kv.key_cache[i] = qkv_cache.rotation.inverse_rotate(
                    k_rot
                ).to(k.dtype)
                kv.value_cache[i] = qkv_cache.rotation.inverse_rotate(
                    v_rot
                ).to(v.dtype)
            return kv

        return kv

    def _compress_cache(self, cache: Any, qkv_cache: Optional[QuantizedKVCache]) -> Any:
        """Unified compression entry point for both MHA and linear-attn caches."""
        if self._is_linear_attn and self._linear_verifier is not None:
            return self._linear_verifier.compress_cache(cache)
        if qkv_cache is not None:
            return self._compress_kv(cache, qkv_cache)
        return cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        gamma: int = 5,
        temperature: float = 1.0,
    ) -> SpeculativeOutput:
        """Run speculative decoding with optional quantized verification.

        In MTP mode: target model produces hidden states → MTP head drafts
        → target model verifies.  No separate draft model needed.
        """
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

        # Prefill target model
        target_out = self.target_model(
            input_ids.to(self.target_device),
            use_cache=True,
            output_hidden_states=self.use_mtp,
        )
        target_kv = target_out.past_key_values
        target_next_logits = target_out.logits[:, -1, :]

        # For MTP: extract last hidden state for the MTP head
        if self.use_mtp:
            last_hidden = target_out.hidden_states[-1][:, -1, :]  # [B, D]
        else:
            # Legacy: prefill draft model
            draft_out = self.draft_model(
                input_ids.to(self.draft_device), use_cache=True
            )
            draft_kv = draft_out.past_key_values
            draft_next_logits = draft_out.logits[:, -1, :]

        all_token_ids = input_ids.cpu().clone()
        kv_len = prefix_len

        qkv_cache = None
        if self.use_quant:
            if self._is_linear_attn:
                target_kv = self._compress_cache(target_kv, None)
            else:
                qkv_cache = self._build_qkv_cache()
                target_kv = self._compress_kv(target_kv, qkv_cache)

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

            # --- Draft phase ---
            t0 = time.perf_counter()
            if self.use_mtp:
                # Sample first token from target logits
                temp = max(temperature, 1e-8)
                p0 = F.softmax(target_next_logits.squeeze(0) / temp, dim=-1)
                tok0 = torch.multinomial(p0, num_samples=1).squeeze(-1)

                # MTP head drafts remaining tokens
                draft_tokens_mtp, draft_probs_mtp, draft_hiddens, draft_attns = (
                    self.mtp_head.draft(
                        tok0, last_hidden, kv_len, cur_gamma - 1, temperature,
                    )
                )
                # Combine: tok0 + MTP drafted tokens
                draft_tokens = torch.cat([tok0.cpu().unsqueeze(0), draft_tokens_mtp]) if cur_gamma > 1 else tok0.cpu().unsqueeze(0)
                draft_probs = [p0.cpu()] + draft_probs_mtp if cur_gamma > 1 else [p0.cpu()]
            else:
                draft_tokens, draft_probs, draft_kv = self._draft_phase(
                    draft_next_logits, draft_kv, cur_gamma, temperature
                )
            t_draft_total += time.perf_counter() - t0

            # --- Verify phase ---
            t0 = time.perf_counter()
            verify_out = self.target_model(
                draft_tokens.view(1, -1).to(self.target_device),
                past_key_values=target_kv,
                use_cache=True,
                output_hidden_states=self.use_mtp,
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
            target_kv = _trim_kv_cache(target_kv_ext, new_kv_len)

            last_tok = accepted[-1]

            if self.use_mtp:
                # Re-sync target model: advance by last accepted token
                t_out = self.target_model(
                    last_tok.view(1, 1).to(self.target_device),
                    past_key_values=target_kv,
                    use_cache=True,
                    output_hidden_states=True,
                )
                target_kv = t_out.past_key_values
                target_next_logits = t_out.logits[:, -1, :]
                last_hidden = t_out.hidden_states[-1][:, -1, :]
            else:
                # Legacy: re-sync both models
                draft_kv = _trim_kv_cache(draft_kv, new_kv_len)

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

            if self.use_quant:
                t0_q = time.perf_counter()
                target_kv = self._compress_cache(target_kv, qkv_cache)
                t_quant_total += time.perf_counter() - t0_q

            kv_len = new_kv_len + 1

            if last_tok.item() == self.tokenizer.eos_token_id:
                break

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
            if tp.shape[-1] != dp.shape[-1]:
                mv = max(tp.shape[-1], dp.shape[-1])
                tp = F.pad(tp, (0, mv - tp.shape[-1])) if tp.shape[-1] < mv else tp
                dp = F.pad(dp, (0, mv - dp.shape[-1])) if dp.shape[-1] < mv else dp
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

        Splits input into prefix + suffix.  Builds KV from the prefix, then
        runs suffix verification twice -- once with full-precision KV and once
        with the quantized round-trip KV -- and reports per-position TV.
        """
        if not self.use_quant:
            return {"tv_mean": 0.0, "tv_std": 0.0, "tv_per_position": []}

        device = self.target_device
        seq_len = min(input_ids.shape[1], num_positions)
        tokens = input_ids[:, :seq_len].to(device)

        split = max(1, seq_len // 2)
        prefix = tokens[:, :split]
        suffix = tokens[:, split:]

        if suffix.shape[1] == 0:
            return {"tv_mean": 0.0, "tv_std": 0.0, "tv_per_position": []}

        fp_out = self.target_model(prefix, use_cache=True)
        fp_kv = fp_out.past_key_values

        q_out = self.target_model(prefix, use_cache=True)
        q_kv = q_out.past_key_values

        # Compress the quantized cache using the appropriate backend.
        if self._is_linear_attn:
            q_kv = self._compress_cache(q_kv, None)
        else:
            qkv_cache = self._build_qkv_cache()
            q_kv = self._compress_kv(q_kv, qkv_cache)

        fp_verify = self.target_model(
            suffix, past_key_values=fp_kv, use_cache=False
        )
        fp_probs = F.softmax(fp_verify.logits.float(), dim=-1)

        q_verify = self.target_model(
            suffix, past_key_values=q_kv, use_cache=False
        )
        q_probs = F.softmax(q_verify.logits.float(), dim=-1)

        tv_per_pos = 0.5 * (fp_probs - q_probs).abs().sum(dim=-1).squeeze(0)
        tv_values = tv_per_pos.cpu().tolist()
        if isinstance(tv_values, float):
            tv_values = [tv_values]

        tv_t = torch.tensor(tv_values)
        return {
            "tv_mean": tv_t.mean().item(),
            "tv_std": tv_t.std().item() if len(tv_values) > 1 else 0.0,
            "tv_per_position": tv_values,
            "num_positions": len(tv_values),
            "prefix_len": split,
            "quant_bits": self.quant_bits,
            "architecture": "linear_attn" if self._is_linear_attn else "mha",
        }

    def get_verifier_stats(self) -> dict:
        """Return compression stats from the active verifier (if any)."""
        if self._linear_verifier is not None:
            return self._linear_verifier.get_memory_stats()
        return {"architecture": "mha", "quant_bits": self.quant_bits}
