from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ..atomic import atomic_write_json
from ..orchestration.status import RuntimeBlocked


def resolve_selected_cuda_device(
    selected_device: int | str | torch.device | None = None,
    *,
    torch_module: Any = torch,
    require_cuda: bool = False,
) -> torch.device:
    """Resolve the one device selected by preflight without silently sharding."""
    value = selected_device
    if value is None:
        value = __import__("os").environ.get("VIPRAGSENT_SELECTED_CUDA_DEVICE", "cpu")
    if isinstance(value, int):
        value = f"cuda:{value}"
    device = torch_module.device(value)
    if device.type == "cuda":
        if not bool(torch_module.cuda.is_available()):
            raise RuntimeBlocked("selected CUDA device is unavailable")
        count = int(torch_module.cuda.device_count())
        if device.index is None or device.index < 0 or device.index >= count:
            raise RuntimeBlocked(f"selected CUDA device index is invalid: {device}")
    elif require_cuda:
        raise RuntimeBlocked("a CUDA device is required by the selected runtime")
    return device


def _parameter_devices(module: nn.Module) -> set[torch.device]:
    return {parameter.device for parameter in module.parameters()} | {buffer.device for buffer in module.buffers()}


def _device_string(device: torch.device) -> str:
    return str(device)


def resolve_model_input_device(model: nn.Module, fallback: torch.device | str | None = None) -> torch.device:
    devices = _parameter_devices(model)
    if len(devices) > 1:
        raise RuntimeBlocked(f"model parameters span multiple devices: {sorted(map(str, devices))}")
    if devices:
        return next(iter(devices))
    if fallback is None:
        return torch.device("cpu")
    return torch.device(fallback)


def place_non_quantized_model(model: nn.Module, device: torch.device | str, *, model_family: str = "unknown") -> nn.Module:
    """Move a complete non-quantized model once and verify the result."""
    target = torch.device(device)
    if getattr(model, "_vipragsent_quantized", False):
        raise RuntimeBlocked("quantized models must be placed during quantized loading")
    model.to(target)
    devices = _parameter_devices(model)
    if devices != {target}:
        raise RuntimeBlocked(f"{model_family} model placement failed: {sorted(map(str, devices))}")
    return model


def place_task_modules(model: nn.Module, device: torch.device | str) -> nn.Module:
    """Place newly attached heads/decoders without moving a quantized backbone."""
    target = torch.device(device)
    for name, child in model.named_children():
        if name != "backbone":
            child.to(target)
    return model


def move_batch_to_device(value: Any, device: torch.device | str) -> Any:
    """Recursively transfer tensors while preserving all non-tensor metadata."""
    target = torch.device(device)
    if isinstance(value, Tensor):
        return value.to(target)
    if isinstance(value, Mapping):
        return type(value)((key, move_batch_to_device(item, target)) for key, item in value.items())
    if isinstance(value, list):
        return [move_batch_to_device(item, target) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, target) for item in value)
    return value


def _tensor_devices(value: Any) -> set[torch.device]:
    if isinstance(value, Tensor):
        return {value.device}
    if isinstance(value, Mapping):
        devices: set[torch.device] = set()
        for item in value.values():
            devices.update(_tensor_devices(item))
        return devices
    if isinstance(value, (list, tuple)):
        devices: set[torch.device] = set()
        for item in value:
            devices.update(_tensor_devices(item))
        return devices
    return set()


def tensor_devices(value: Any) -> set[torch.device]:
    return _tensor_devices(value)


def assert_runtime_device_contract(
    model: nn.Module,
    selected_device: torch.device | str,
    *,
    model_family: str = "unknown",
    quantized: bool = False,
    device_map: Mapping[str, Any] | None = None,
    batch: Any | None = None,
    loss: Tensor | None = None,
    uncertainty_module: nn.Module | None = None,
    require_lora: bool = False,
) -> dict[str, Any]:
    """Validate placement and QLoRA trainability before an optimizer step."""
    target = torch.device(selected_device)
    blockers: list[str] = []
    model_devices = _parameter_devices(model)
    if len(model_devices) > 1:
        blockers.append("mixed-device model parameters")
    if target not in model_devices and model_devices:
        blockers.append(f"model is not on selected device {target}")
    if target.type == "cuda" and len({device.index for device in model_devices if device.type == "cuda"}) > 1:
        blockers.append("multiple unapproved CUDA devices are used")
    if quantized:
        base_trainable = [name for name, parameter in model.named_parameters() if "backbone" in name and parameter.requires_grad and "lora_" not in name.lower()]
        if base_trainable:
            blockers.append("QLoRA base parameters are trainable")
        if require_lora and not any("lora_" in name.lower() and parameter.requires_grad for name, parameter in model.named_parameters()):
            blockers.append("required LoRA parameters are frozen or absent")
    if require_lora and not any(name.split(".")[0] != "backbone" and parameter.requires_grad for name, parameter in model.named_parameters()):
        blockers.append("required task-head parameters are frozen or absent")
    if uncertainty_module is not None:
        uncertainty_devices = _parameter_devices(uncertainty_module)
        if uncertainty_devices != {target}:
            blockers.append("uncertainty parameters are on an incompatible device")
    batch_devices = _tensor_devices(batch) if batch is not None else set()
    if batch_devices and batch_devices != {target}:
        blockers.append("required batch tensors are on incompatible devices")
    if loss is not None and loss.device != target:
        blockers.append("loss is not on the optimizer device")
    if blockers:
        raise RuntimeBlocked("; ".join(blockers))
    return {
        "selected_device": _device_string(target),
        "device_index": target.index,
        "device_name": (torch.cuda.get_device_name(target) if target.type == "cuda" else "cpu"),
        "compute_capability": (".".join(map(str, torch.cuda.get_device_capability(target))) if target.type == "cuda" else "n/a"),
        "model_family": model_family,
        "quantized": bool(quantized),
        "device_map": dict(device_map or {}),
        "backbone_devices": sorted({_device_string(parameter.device) for name, parameter in model.named_parameters() if name.startswith("backbone.")}),
        "trainable_parameter_devices": sorted({_device_string(parameter.device) for parameter in model.parameters() if parameter.requires_grad}),
        "frozen_parameter_devices": sorted({_device_string(parameter.device) for parameter in model.parameters() if not parameter.requires_grad}),
        "head_devices": sorted({_device_string(parameter.device) for name, parameter in model.named_parameters() if any(token in name for token in ("heads", "classifier"))}),
        "rationale_device": sorted({_device_string(parameter.device) for name, parameter in model.named_parameters() if "rationale" in name}),
        "uncertainty_device": sorted({_device_string(parameter.device) for parameter in uncertainty_module.parameters()}) if uncertainty_module is not None else [],
        "first_batch_tensor_devices": sorted({_device_string(device) for device in batch_devices}),
        "loss_device": _device_string(loss.device) if loss is not None else None,
        "status": "PASS",
        "blockers": [],
    }


def write_device_report(path: str | Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(report))
