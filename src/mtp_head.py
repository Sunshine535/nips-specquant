"""Qwen3.5 Multi-Token Prediction (MTP) head for self-speculative decoding.

Loads the MTP weights from a Qwen3.5 checkpoint (under the `mtp.` prefix)
that HuggingFace transformers ignores by default.  Shares embed_tokens and
lm_head with the main model.

Architecture (from Qwen3.5 config):
    input_ids  → embed_tokens (shared) → RMSNorm (pre_fc_norm_embedding)
    hidden_states                       → RMSNorm (pre_fc_norm_hidden)
    concat([emb, hidden], dim=-1)       → Linear(2d → d)
    → single full-attention decoder layer (own KV cache)
    → RMSNorm → lm_head (shared) → logits

Usage:
    mtp = Qwen35MTPHead.from_pretrained("Qwen/Qwen3.5-9B", main_model)
    logits, hidden, mtp_kv = mtp(token_ids, hidden_states, position_ids)
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight RMSNorm (avoids transformers version dependency)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


# ---------------------------------------------------------------------------
# Simplified attention layer for the MTP head
# ---------------------------------------------------------------------------

class MTPAttention(nn.Module):
    """Grouped-query attention for the MTP decoder layer."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int = 131072,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # RoPE
        self.rope_theta = rope_theta
        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._rope_max_len = 0

    def _build_rope(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._rope_cache is not None and seq_len <= self._rope_max_len:
            return
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(max(seq_len, 4096), device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        self._rope_cache = (freqs.cos().to(dtype), freqs.sin().to(dtype))
        self._rope_max_len = max(seq_len, 4096)

    def _apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self._rope_cache
        cos = cos[positions].unsqueeze(1)  # [B, 1, seq, dim//2]
        sin = sin[positions].unsqueeze(1)
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        self._build_rope(position_ids.max().item() + 1, device, q.dtype)
        q = self._apply_rope(q, position_ids)
        k = self._apply_rope(k, position_ids)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        new_kv = (k, v)

        # GQA: expand KV heads
        if self.num_kv_groups > 1:
            k = k.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)
            v = v.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1).reshape(bsz, self.num_heads, -1, self.head_dim)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Causal mask
        kv_len = k.shape[2]
        if seq_len > 1:
            causal = torch.triu(torch.full((seq_len, kv_len), float("-inf"), device=device), diagonal=kv_len - seq_len + 1)
            attn_weights = attn_weights + causal.unsqueeze(0).unsqueeze(0)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)

        return self.o_proj(attn_output), new_kv, attn_weights


class MTPDecoderLayer(nn.Module):
    """Single decoder layer for MTP head (full attention, not GatedDeltaNet)."""

    def __init__(self, config: Dict):
        super().__init__()
        hidden = config["hidden_size"]
        intermediate = config.get("intermediate_size", hidden * 4)
        num_heads = config["num_attention_heads"]
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim", hidden // num_heads)

        self.self_attn = MTPAttention(
            hidden_size=hidden,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rope_theta=config.get("rope_theta", 1000000.0),
        )
        self.input_layernorm = RMSNorm(hidden)
        self.post_attention_layernorm = RMSNorm(hidden)

        # SwiGLU MLP
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, new_kv, attn_weights = self.self_attn(hidden_states, position_ids, past_key_value)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        hidden_states = residual + hidden_states

        return hidden_states, new_kv, attn_weights


# ---------------------------------------------------------------------------
# MTP Head
# ---------------------------------------------------------------------------

class Qwen35MTPHead(nn.Module):
    """Qwen3.5 MTP head for self-speculative decoding.

    Loads ``mtp.*`` weights from checkpoint.  Shares ``embed_tokens`` and
    ``lm_head`` with the main model (no extra copies).
    """

    def __init__(self, config: Dict, embed_tokens: nn.Embedding, lm_head: nn.Linear):
        super().__init__()
        hidden = config["hidden_size"]

        self.embed_tokens = embed_tokens  # shared, not owned
        self.lm_head = lm_head            # shared, not owned

        self.pre_fc_norm_embedding = RMSNorm(hidden)
        self.pre_fc_norm_hidden = RMSNorm(hidden)
        self.fc = nn.Linear(hidden * 2, hidden, bias=False)
        self.layer = MTPDecoderLayer(config)
        self.norm = RMSNorm(hidden)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        main_model: nn.Module,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Qwen35MTPHead":
        """Load MTP head from a Qwen3.5 checkpoint.

        Extracts ``mtp.*`` weights from safetensors files and builds the head,
        sharing embed_tokens and lm_head with *main_model*.
        """
        from safetensors import safe_open
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        cfg = hf_config.to_dict() if hasattr(hf_config, "to_dict") else vars(hf_config)

        # Get shared modules from main model
        embed = _find_module(main_model, "embed_tokens")
        lm_head = _find_module(main_model, "lm_head")
        if embed is None or lm_head is None:
            raise RuntimeError("Cannot find embed_tokens / lm_head on main model")

        if device is None:
            device = next(main_model.parameters()).device
        if dtype is None:
            dtype = next(main_model.parameters()).dtype

        head = cls(cfg, embed, lm_head)

        # Load mtp.* weights from checkpoint
        mtp_weights = _load_mtp_weights(model_name_or_path)
        if not mtp_weights:
            raise FileNotFoundError(
                f"No mtp.* weights found in {model_name_or_path}. "
                "Is this a Qwen3.5 model with MTP?"
            )

        # Map checkpoint keys → module keys
        state = {}
        for ck, tensor in mtp_weights.items():
            # Strip "mtp." prefix
            key = ck[4:] if ck.startswith("mtp.") else ck
            # Map layers.0.* → layer.*
            key = key.replace("layers.0.", "layer.")
            state[key] = tensor.to(dtype=dtype)

        missing, unexpected = head.load_state_dict(state, strict=False)
        # embed_tokens and lm_head are shared, not in state_dict
        missing = [k for k in missing if "embed_tokens" not in k and "lm_head" not in k]
        if missing:
            logger.warning("MTP head missing keys: %s", missing)
        if unexpected:
            logger.warning("MTP head unexpected keys: %s", unexpected)

        head = head.to(device=device, dtype=dtype)
        head.eval()
        logger.info(
            "Loaded Qwen3.5 MTP head (%d params, device=%s, dtype=%s)",
            sum(p.numel() for p in head.parameters() if p not in list(embed.parameters()) + list(lm_head.parameters())),
            device, dtype,
        )
        return head

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[torch.Tensor]]:
        """Single MTP step.

        Args:
            input_ids:     [B, 1] last predicted token
            hidden_states: [B, 1, D] from main model or previous MTP step
            position_ids:  [B, 1]
            past_key_value: MTP layer KV cache (tuple of K, V)

        Returns:
            logits:        [B, 1, V]
            hidden_states: [B, 1, D] for next MTP step
            past_key_value: updated MTP KV cache
            attn_weights:  [B, H, 1, kv_len] from MTP attention
        """
        emb = self.embed_tokens(input_ids)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(1)

        emb = self.pre_fc_norm_embedding(emb)
        hidden_states = self.pre_fc_norm_hidden(hidden_states)

        fused = torch.cat([emb, hidden_states], dim=-1)
        fused = self.fc(fused)

        fused, new_kv, attn_weights = self.layer(fused, position_ids, past_key_value)
        normed = self.norm(fused)
        logits = self.lm_head(normed)

        return logits, fused, new_kv, attn_weights

    @torch.no_grad()
    def draft(
        self,
        start_token: torch.Tensor,
        hidden_states: torch.Tensor,
        start_position: int,
        gamma: int,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[Optional[torch.Tensor]]]:
        """Multi-step MTP drafting.

        Args:
            start_token:   [B] the token just generated by the main model
            hidden_states: [B, D] last hidden state from main model
            start_position: current sequence position
            gamma:         number of tokens to draft
            temperature:   sampling temperature

        Returns:
            draft_tokens: [gamma] drafted token ids
            draft_probs:  list of [V] probability vectors per step
            draft_hiddens: list of [B, D] hidden states per step
            draft_attns:  list of attention weights per step
        """
        device = hidden_states.device
        temp = max(temperature, 1e-8)

        tokens = []
        probs_list = []
        hiddens = []
        attns = []
        mtp_kv = None

        cur_token = start_token.view(1, 1)
        cur_hidden = hidden_states.unsqueeze(1) if hidden_states.dim() == 2 else hidden_states

        for step in range(gamma):
            pos = torch.tensor([[start_position + step + 1]], device=device)

            logits, cur_hidden, mtp_kv, attn_w = self.forward(
                cur_token, cur_hidden, pos, mtp_kv,
            )

            p = F.softmax(logits.squeeze(0).squeeze(0) / temp, dim=-1)
            tok = torch.multinomial(p, num_samples=1).squeeze(-1)

            tokens.append(tok.cpu())
            probs_list.append(p.cpu())
            hiddens.append(cur_hidden.squeeze(1).cpu())
            if attn_w is not None:
                attns.append(attn_w.cpu())

            cur_token = tok.view(1, 1)

        return torch.stack(tokens), probs_list, hiddens, attns


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------

def _find_module(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Find a named module anywhere in the model tree."""
    for n, m in model.named_modules():
        if n.endswith(name):
            return m
    return None


def _load_mtp_weights(model_name_or_path: str) -> Dict[str, torch.Tensor]:
    """Load all mtp.* weights from safetensors files in a model directory."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    # Resolve to local path
    if os.path.isdir(model_name_or_path):
        model_dir = model_name_or_path
    else:
        model_dir = snapshot_download(model_name_or_path)

    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}")

    mtp_weights = {}
    for path in st_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("mtp."):
                    mtp_weights[key] = f.get_tensor(key)

    logger.info("Loaded %d MTP weight tensors from %s", len(mtp_weights), model_dir)
    return mtp_weights
