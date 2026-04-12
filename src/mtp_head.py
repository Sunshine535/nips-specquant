"""Qwen3.5 Multi-Token Prediction (MTP) head for self-speculative decoding.

Loads the MTP weights from a Qwen3.5 checkpoint (under the ``mtp.`` prefix)
that HuggingFace transformers ignores by default.  Reuses the model's own
decoder layer class to guarantee architecture compatibility.

Usage:
    mtp = Qwen35MTPHead.from_pretrained("Qwen/Qwen3.5-9B", main_model)
    logits, hidden, mtp_kv = mtp(token_ids, hidden_states, position_ids)
"""

from __future__ import annotations

import copy
import glob
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(orig_dtype)


# ---------------------------------------------------------------------------
# MTP Head — uses the model's own decoder layer class
# ---------------------------------------------------------------------------

class Qwen35MTPHead(nn.Module):
    """Qwen3.5 MTP head for self-speculative decoding.

    Loads ``mtp.*`` weights from checkpoint.  Shares ``embed_tokens`` and
    ``lm_head`` with the main model.  Uses the model's own decoder layer
    class (copied from the first full-attention layer) to guarantee
    architecture compatibility.
    """

    def __init__(self, hidden_size: int, embed_tokens: nn.Embedding,
                 lm_head: nn.Linear, decoder_layer: nn.Module,
                 rotary_emb: nn.Module, rms_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed_tokens = embed_tokens  # shared
        self.lm_head = lm_head            # shared
        self.rotary_emb = rotary_emb      # shared — for computing position_embeddings

        self.pre_fc_norm_embedding = RMSNorm(hidden_size, eps=rms_eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, eps=rms_eps)
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.layer = decoder_layer  # actual model decoder layer
        self.norm = RMSNorm(hidden_size, eps=rms_eps)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        main_model: nn.Module,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Qwen35MTPHead":
        """Load MTP head from a Qwen3.5 checkpoint."""

        # Get config
        model_config = main_model.config
        if hasattr(model_config, "text_config"):
            model_config = model_config.text_config
        hidden_size = model_config.hidden_size
        rms_eps = getattr(model_config, "rms_norm_eps", 1e-6)

        logger.info("MTP config: hidden_size=%d, rms_eps=%s", hidden_size, rms_eps)

        # Get shared modules
        embed = _find_module(main_model, "embed_tokens")
        lm_head_mod = _find_module(main_model, "lm_head")
        if embed is None or lm_head_mod is None:
            raise RuntimeError("Cannot find embed_tokens / lm_head on main model")

        if device is None:
            device = next(main_model.parameters()).device
        if dtype is None:
            dtype = next(main_model.parameters()).dtype

        # Find the rotary embedding module (shared, not cloned)
        rotary_emb = _find_module(main_model, "rotary_emb")
        if rotary_emb is None:
            logger.warning("rotary_emb not found — decoder layer may handle RoPE internally")

        # Find a full-attention decoder layer to clone as MTP layer
        decoder_layer = _clone_full_attn_layer(main_model)
        if decoder_layer is None:
            raise RuntimeError(
                "Cannot find a full-attention decoder layer in the model. "
                "Is this a Qwen3.5 model?"
            )
        logger.info("Cloned decoder layer type: %s", type(decoder_layer).__name__)

        head = cls(hidden_size, embed, lm_head_mod, decoder_layer, rotary_emb, rms_eps)

        # Load mtp.* weights
        mtp_weights = _load_mtp_weights(model_name_or_path)
        if not mtp_weights:
            raise FileNotFoundError(f"No mtp.* weights in {model_name_or_path}")

        # Map checkpoint keys → module keys
        state = {}
        for ck, tensor in mtp_weights.items():
            key = ck[4:] if ck.startswith("mtp.") else ck
            key = key.replace("layers.0.", "layer.")
            state[key] = tensor.to(dtype=dtype)

        missing, unexpected = head.load_state_dict(state, strict=False)
        # Filter out shared modules
        missing = [k for k in missing if "embed_tokens" not in k and "lm_head" not in k]
        if missing:
            logger.warning("MTP head missing keys: %s", missing)
        if unexpected:
            logger.warning("MTP head unexpected keys: %s", unexpected)

        head = head.to(device=device, dtype=dtype)
        head.eval()

        own_params = sum(
            p.numel() for n, p in head.named_parameters()
            if "embed_tokens" not in n and "lm_head" not in n
        )
        logger.info("MTP head loaded: %.1fM own params, device=%s", own_params / 1e6, device)
        return head

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """Single MTP step.

        Args:
            input_ids:      [B, 1] last predicted token
            hidden_states:  [B, 1, D] from main model or previous MTP step
            position_ids:   [B, 1]
            past_key_values: cache object for the MTP layer

        Returns:
            logits:         [B, 1, V]
            hidden_states:  [B, 1, D] for next MTP step
            past_key_values: updated cache
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

        # Compute position embeddings (RoPE cos/sin) from position_ids
        position_embeddings = None
        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(fused, position_ids)

        # Call the decoder layer — use the model's native interface
        layer_kwargs = dict(use_cache=True)
        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings
        else:
            layer_kwargs["position_ids"] = position_ids
        if past_key_values is not None:
            layer_kwargs["past_key_value"] = past_key_values

        layer_out = self.layer(fused, **layer_kwargs)
        # Decoder layers return (hidden_states, ...) or (hidden_states, present_kv, ...)
        if isinstance(layer_out, tuple):
            fused = layer_out[0]
            new_kv = layer_out[1] if len(layer_out) > 1 else None
        else:
            fused = layer_out
            new_kv = None

        normed = self.norm(fused)
        logits = self.lm_head(normed)

        return logits, fused, new_kv

    @torch.no_grad()
    def draft(
        self,
        start_token: torch.Tensor,
        hidden_states: torch.Tensor,
        start_position: int,
        gamma: int,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Multi-step MTP drafting.

        Returns:
            draft_tokens: [gamma] drafted token ids
            draft_probs:  list of [V] probability vectors per step
            draft_hiddens: list of [B, D] hidden states per step
        """
        device = hidden_states.device
        temp = max(temperature, 1e-8)

        tokens = []
        probs_list = []
        hiddens = []
        mtp_kv = None

        cur_token = start_token.view(1, 1)
        cur_hidden = hidden_states.unsqueeze(1) if hidden_states.dim() == 2 else hidden_states

        for step in range(gamma):
            pos = torch.tensor([[start_position + step + 1]], device=device)

            logits, cur_hidden, mtp_kv = self.forward(
                cur_token, cur_hidden, pos, mtp_kv,
            )

            p = F.softmax(logits.squeeze(0).squeeze(0) / temp, dim=-1)
            tok = torch.multinomial(p, num_samples=1).squeeze(-1)

            tokens.append(tok.cpu())
            probs_list.append(p.cpu())
            hiddens.append(cur_hidden.squeeze(1).cpu())

            cur_token = tok.view(1, 1)

        return torch.stack(tokens), probs_list, hiddens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_module(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Find a named module anywhere in the model tree."""
    for n, m in model.named_modules():
        if n.endswith(name):
            return m
    return None


def _clone_full_attn_layer(model: nn.Module) -> Optional[nn.Module]:
    """Find and deep-copy a full-attention decoder layer from the model.

    For Qwen3.5 hybrid models, selects a layer with ``self_attn``
    (not ``linear_attn`` / GatedDeltaNet).
    """
    # Find decoder layers
    layers = None
    for attr_chain in (("model", "layers"), ("transformer", "h"), ("transformer", "layers")):
        obj = model
        for attr in attr_chain:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__len__"):
            layers = list(obj)
            break

    if not layers:
        return None

    # Find a full-attention layer (has self_attn, not linear_attn)
    for layer in layers:
        if hasattr(layer, "self_attn") and not hasattr(layer, "linear_attn"):
            return copy.deepcopy(layer)

    # Fallback: first layer
    return copy.deepcopy(layers[0])


def _load_mtp_weights(model_name_or_path: str) -> Dict[str, torch.Tensor]:
    """Load all mtp.* weights from safetensors files."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

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
