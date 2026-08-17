from __future__ import annotations

import gc
import os
from collections.abc import Mapping, Sequence
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
        value = os.environ.get("VIPRAGSENT_SELECTED_CUDA_DEVICE", "cpu")
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


def _map_device(value: Any) -> torch.device:
    if isinstance(value, int):
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def _quantized_input_device(model: nn.Module) -> torch.device | None:
    contract = getattr(model, "_vipragsent_qlora_contract", None)
    if not isinstance(contract, Mapping):
        return None
    selected = contract.get("input_device") or contract.get("selected_device")
    if selected is not None:
        return _map_device(selected)
    device_map = contract.get("device_map")
    if isinstance(device_map, Mapping) and device_map.get("") is not None:
        return _map_device(device_map[""])
    return None


def resolve_model_input_device(model: nn.Module, fallback: torch.device | str | None = None) -> torch.device:
    quantized_device = _quantized_input_device(model)
    if quantized_device is not None:
        return quantized_device
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
    task_devices = {
        parameter.device
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    }
    if task_devices and task_devices != {target}:
        raise RuntimeBlocked(f"task modules are not on selected device {target}: {sorted(map(str, task_devices))}")
    return model


def move_batch_to_device(
    value: Any,
    device: torch.device | str,
    *,
    preserve_keys: Sequence[str] = (),
) -> Any:
    """Recursively transfer tensors while preserving all non-tensor metadata."""
    target = torch.device(device)
    if isinstance(value, Tensor):
        return value.to(target)
    if isinstance(value, Mapping):
        items = [
            (key, item if str(key) in preserve_keys else move_batch_to_device(item, target, preserve_keys=()))
            for key, item in value.items()
        ]
        try:
            return type(value)(items)
        except (TypeError, ValueError):
            return dict(items)
    if isinstance(value, list):
        return [move_batch_to_device(item, target) for item in value]
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, target) for item in value)
    return value


DEFAULT_METADATA_KEYS = ("sample_ids", "sample_id", "text", "raw_text", "metadata", "examples")


def move_batch_to_model_device(
    batch: Mapping[str, Any],
    model: nn.Module,
    *,
    device: torch.device | str | None = None,
    preserve_keys: Sequence[str] = DEFAULT_METADATA_KEYS,
) -> dict[str, Any]:
    """Move tensor inputs/targets while leaving IDs and text in host memory."""
    target = device or resolve_model_input_device(model)
    moved = move_batch_to_device(batch, target, preserve_keys=preserve_keys)
    if not isinstance(moved, Mapping):
        raise RuntimeBlocked("model batch must remain a mapping after device preparation")
    return dict(moved)


def _tensor_devices(value: Any) -> set[torch.device]:
    if isinstance(value, Tensor):
        return {value.device}
    if isinstance(value, Mapping):
        devices: set[torch.device] = set()
        for item in value.values():
            devices.update(_tensor_devices(item))
        return devices
    if isinstance(value, list | tuple):
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
    quantized_input = _quantized_input_device(model)
    if quantized_input is not None and quantized_input != target:
        blockers.append(f"quantized model input device is {quantized_input}, not selected device {target}")
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
        "input_device": _device_string(quantized_input or target),
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


class DeviceContractReporter:
    """Write exactly one first-batch device report for a custom executor."""

    def __init__(
        self,
        path: str | Path,
        model: nn.Module,
        selected_device: torch.device | str,
        *,
        model_family: str = "unknown",
        quantized: bool = False,
        device_map: Mapping[str, Any] | None = None,
        uncertainty_module: nn.Module | None = None,
        require_lora: bool = False,
    ) -> None:
        self.path = Path(path)
        self.model = model
        self.selected_device = torch.device(selected_device)
        self.kwargs = {
            "model_family": model_family,
            "quantized": quantized,
            "device_map": device_map,
            "uncertainty_module": uncertainty_module,
            "require_lora": require_lora,
        }
        self.written = False

    def observe(self, batch: Mapping[str, Any], loss: Tensor | None = None) -> dict[str, Any]:
        if self.written:
            return {}
        try:
            report = assert_runtime_device_contract(self.model, self.selected_device, batch=batch, loss=loss, **self.kwargs)
        except RuntimeBlocked as exc:
            report = {
                "selected_device": str(self.selected_device),
                "input_device": str(self.selected_device),
                "status": "BLOCKED",
                "blockers": [str(exc)],
            }
            write_device_report(self.path, report)
            self.written = True
            raise
        write_device_report(self.path, report)
        self.written = True
        return report


def release_model_resources(
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    loader: Any | None = None,
    *,
    clear_cuda_cache: bool = True,
) -> None:
    """Release one-at-a-time executor resources without moving quantized models."""
    if optimizer is not None:
        optimizer.state.clear()
    if loader is not None:
        close = getattr(loader, "close", None)
        if callable(close):
            close()
    if model is not None and not getattr(model, "_vipragsent_quantized", False):
        model.to("cpu")
    gc.collect()
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
