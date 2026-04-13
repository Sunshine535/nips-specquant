"""Auto-adaptive GPU detection and model placement.

Detects available GPUs, their memory, and assigns draft/target models
to optimal devices. Supports 1-GPU, 2-GPU, and multi-GPU configurations.

Strategies:
  0 GPUs → CPU-only (slow, for testing)
  1 GPU  → Both models on cuda:0, draft loaded first then offloaded if needed
  2 GPUs → Draft on smaller-VRAM GPU, target on larger-VRAM GPU
  4+ GPUs → Target with device_map="auto" across GPUs, draft on remaining GPU
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    index: int
    name: str
    total_gb: float
    free_gb: float


@dataclass
class DevicePlan:
    """Describes how to load draft and target models."""
    draft_device: str                      # "cpu", "cuda:0", etc.
    target_device: str                     # "cpu", "cuda:0", "auto", etc.
    target_device_map: Optional[str] = None  # "auto" for multi-GPU TP
    num_gpus: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    dtype: torch.dtype = torch.float16
    description: str = ""


def detect_gpus() -> List[GPUInfo]:
    """Detect all available CUDA GPUs and their memory."""
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_bytes = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        total = total_bytes / (1024 ** 3)
        free = (total_bytes - torch.cuda.memory_allocated(i)) / (1024 ** 3)
        gpus.append(GPUInfo(index=i, name=props.name, total_gb=total, free_gb=free))
    return gpus


def plan_devices(
    draft_model_size_gb: float = 2.0,
    target_model_size_gb: float = 16.0,
    prefer_dtype: str = "float16",
) -> DevicePlan:
    """Create an optimal device placement plan based on available hardware.

    Args:
        draft_model_size_gb: estimated draft model size in GB
        target_model_size_gb: estimated target model size in GB
        prefer_dtype: "float16" or "bfloat16"
    """
    dtype = torch.bfloat16 if prefer_dtype == "bfloat16" else torch.float16
    gpus = detect_gpus()
    n = len(gpus)

    if n == 0:
        return DevicePlan(
            draft_device="cpu",
            target_device="cpu",
            num_gpus=0,
            gpus=[],
            dtype=dtype,
            description="CPU-only mode (no CUDA GPUs detected)",
        )

    if n == 1:
        gpu = gpus[0]
        both_fit = gpu.free_gb > (draft_model_size_gb + target_model_size_gb) * 1.2
        if both_fit:
            desc = f"Single GPU ({gpu.name}, {gpu.total_gb:.0f}GB) — both models fit"
        else:
            desc = f"Single GPU ({gpu.name}, {gpu.total_gb:.0f}GB) — tight fit, may need offloading"
        return DevicePlan(
            draft_device="cuda:0",
            target_device="cuda:0",
            num_gpus=1,
            gpus=gpus,
            dtype=dtype,
            description=desc,
        )

    if n == 2:
        # Put target on the GPU with more memory
        sorted_gpus = sorted(gpus, key=lambda g: g.total_gb, reverse=True)
        target_idx = sorted_gpus[0].index
        draft_idx = sorted_gpus[1].index
        return DevicePlan(
            draft_device=f"cuda:{draft_idx}",
            target_device=f"cuda:{target_idx}",
            num_gpus=2,
            gpus=gpus,
            dtype=dtype,
            description=f"Dual GPU — draft→cuda:{draft_idx} ({sorted_gpus[1].name}), target→cuda:{target_idx} ({sorted_gpus[0].name})",
        )

    # 3+ GPUs: target uses device_map="auto" across all but one, draft gets remaining
    # Pick smallest GPU for draft, rest for target
    sorted_gpus = sorted(gpus, key=lambda g: g.total_gb)
    draft_idx = sorted_gpus[0].index
    target_gpus = [g.index for g in sorted_gpus[1:]]
    return DevicePlan(
        draft_device=f"cuda:{draft_idx}",
        target_device=f"cuda:{target_gpus[0]}",
        target_device_map="auto",
        num_gpus=n,
        gpus=gpus,
        dtype=dtype,
        description=f"Multi-GPU ({n}) — draft→cuda:{draft_idx}, target→device_map='auto' across {target_gpus}",
    )


def load_models(
    draft_model_name: str,
    target_model_name: str,
    plan: Optional[DevicePlan] = None,
    trust_remote_code: bool = True,
):
    """Load draft and target models according to a device plan.

    Returns: (draft_model, target_model, tokenizer, plan)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if plan is None:
        plan = plan_devices()

    logger.info("Device plan: %s", plan.description)
    for gpu in plan.gpus:
        logger.info("  GPU %d: %s (%.1f GB total, %.1f GB free)",
                     gpu.index, gpu.name, gpu.total_gb, gpu.free_gb)

    tokenizer = AutoTokenizer.from_pretrained(
        target_model_name, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load draft model
    draft_kwargs = dict(
        torch_dtype=plan.dtype,
        trust_remote_code=trust_remote_code,
    )
    if plan.draft_device == "cpu":
        draft_kwargs["device_map"] = "cpu"
    else:
        draft_kwargs["device_map"] = plan.draft_device

    logger.info("Loading draft model: %s → %s", draft_model_name, plan.draft_device)
    draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name, **draft_kwargs)
    draft_model.eval()

    # Load target model
    target_kwargs = dict(
        torch_dtype=plan.dtype,
        trust_remote_code=trust_remote_code,
    )
    if plan.target_device_map == "auto":
        target_kwargs["device_map"] = "auto"
        logger.info("Loading target model: %s → device_map='auto'", target_model_name)
    elif plan.target_device == "cpu":
        target_kwargs["device_map"] = "cpu"
        logger.info("Loading target model: %s → cpu", target_model_name)
    else:
        target_kwargs["device_map"] = plan.target_device
        logger.info("Loading target model: %s → %s", target_model_name, plan.target_device)

    target_model = AutoModelForCausalLM.from_pretrained(target_model_name, **target_kwargs)
    target_model.eval()

    return draft_model, target_model, tokenizer, plan


def load_model_mtp(
    model_name: str,
    plan: Optional[DevicePlan] = None,
    trust_remote_code: bool = True,
):
    """Load a single model with its MTP head for self-speculation.

    Returns: (model, mtp_head, tokenizer, plan)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if plan is None:
        plan = plan_devices()

    logger.info("Device plan: %s", plan.description)
    for gpu in plan.gpus:
        logger.info("  GPU %d: %s (%.1f GB total, %.1f GB free)",
                     gpu.index, gpu.name, gpu.total_gb, gpu.free_gb)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Single GPU per instance (with fla, 9B fits on one 80GB GPU)
    kwargs = dict(
        torch_dtype=plan.dtype,
        trust_remote_code=trust_remote_code,
    )
    if plan.num_gpus >= 1:
        kwargs["device_map"] = "cuda:0"
        logger.info("Loading model: %s → cuda:0", model_name)
    else:
        kwargs["device_map"] = "cpu"
        logger.info("Loading model: %s → cpu", model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    # Load MTP head on same device
    from .mtp_head import Qwen35MTPHead
    mtp_head = Qwen35MTPHead.from_pretrained(model_name, model)

    return model, mtp_head, tokenizer, plan


class _MultiGPUModelWrapper(torch.nn.Module):
    """Wraps a device_map='auto' model to handle input_ids device placement.

    accelerate hooks don't move integer tensors (input_ids) correctly.
    This wrapper computes inputs_embeds on the right device and passes
    that instead, which accelerate handles correctly.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        # Find embed_tokens and its device
        self._embed = None
        self._embed_device = None
        for name, mod in model.named_modules():
            if name.endswith("embed_tokens"):
                self._embed = mod
                self._embed_device = next(mod.parameters()).device
                break
        self.config = model.config

    def __call__(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None and inputs_embeds is None and self._embed is not None:
            # Compute embeddings on the correct device, pass as float tensor
            inputs_embeds = self._embed(input_ids.to(self._embed_device))
            input_ids = None
        return self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()

    def named_modules(self, *args, **kwargs):
        return self.model.named_modules(*args, **kwargs)

    @property
    def hf_device_map(self):
        return getattr(self.model, "hf_device_map", None)


def _get_output_device(model) -> torch.device:
    """Find the device where a model's output hidden states land.

    For device_map='auto' models, this is the device of the last decoder layer.
    """
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        # Find the highest-numbered layer in the device map
        last_device = None
        for key, device in model.hf_device_map.items():
            last_device = device
        if last_device is not None:
            return torch.device(last_device) if isinstance(last_device, (str, int)) else last_device
    # Fallback: device of the last parameter
    *_, last_param = model.parameters()
    return last_param.device


def print_gpu_summary():
    """Print a human-readable GPU summary."""
    gpus = detect_gpus()
    if not gpus:
        print("No CUDA GPUs detected. Running on CPU.")
        return
    print(f"\n{'='*50}")
    print(f"  {len(gpus)} GPU(s) detected")
    print(f"{'='*50}")
    for g in gpus:
        print(f"  [{g.index}] {g.name}: {g.total_gb:.1f} GB total, {g.free_gb:.1f} GB free")
    plan = plan_devices()
    print(f"\n  Strategy: {plan.description}")
    print(f"{'='*50}\n")
