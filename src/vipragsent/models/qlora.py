from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..orchestration.status import RuntimeBlocked
from ..runtime.device import resolve_selected_cuda_device


def build_qlora_backbone(
    repo_id: str,
    *,
    revision: str,
    local_path: str | None = None,
    selected_device: torch.device | str | int | None = None,
    transformers_module: Any | None = None,
    peft_module: Any | None = None,
) -> nn.Module:
    """Construct a 4-bit NF4 base transformer with trainable LoRA adapters."""
    try:
        transformers = transformers_module or __import__("transformers")
        peft = peft_module or __import__("peft")
        BitsAndBytesConfig = transformers.BitsAndBytesConfig
        AutoModel = transformers.AutoModel
        LoraConfig = peft.LoraConfig
        get_peft_model = peft.get_peft_model
        prepare_model_for_kbit_training = peft.prepare_model_for_kbit_training
    except (ImportError, AttributeError) as exc:
        raise RuntimeBlocked("QLoRA requires compatible transformers, PEFT, and bitsandbytes packages") from exc
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        device = resolve_selected_cuda_device(selected_device)
        if device.type == "cuda":
            device_map: dict[str, Any] = {"": int(device.index or 0)}
        else:
            device_map = {"": str(device)}
        model = AutoModel.from_pretrained(
            local_path or repo_id,
            revision=revision,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
            local_files_only=local_path is not None,
            device_map=device_map,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = get_peft_model(model, lora_config)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.gradient_checkpointing = True
    except Exception as exc:
        if isinstance(exc, RuntimeBlocked):
            raise
        raise RuntimeBlocked(f"Unable to construct pinned NF4 QLoRA backbone: {exc}") from exc
    for name, parameter in model.named_parameters():
        if "lora_" not in name:
            parameter.requires_grad = False
    model._vipragsent_qlora_contract = {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "double_quant": True,
        "compute_dtype": "bf16",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "gradient_checkpointing": True,
        "selected_device": str(device),
        "device_map": device_map,
    }
    model._vipragsent_quantized = True
    return model


def trainable_parameter_report(model: nn.Module) -> dict[str, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}
