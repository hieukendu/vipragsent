from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from ..orchestration.status import RuntimeBlocked


def _trainable_groups(model: nn.Module, *, weight_decay: float, uncertainty_parameters: Iterable[nn.Parameter] = ()) -> tuple[list[dict[str, Any]], dict[str, int]]:
    uncertainty_parameters = list(uncertainty_parameters)
    uncertainty_ids = {id(parameter) for parameter in uncertainty_parameters}
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    uncertainty: list[nn.Parameter] = []
    names: dict[str, list[str]] = {"model_decay": [], "model_no_decay": [], "uncertainty_no_decay": []}
    trainable = 0
    frozen = 0
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        if not parameter.requires_grad:
            frozen += count
            continue
        trainable += count
        if id(parameter) in uncertainty_ids:
            uncertainty.append(parameter)
            names["uncertainty_no_decay"].append(name)
        elif name.endswith(".bias") or "norm" in name.casefold() or "layernorm" in name.casefold():
            no_decay.append(parameter)
            names["model_no_decay"].append(name)
        else:
            decay.append(parameter)
            names["model_decay"].append(name)
    uncertainty = [parameter for parameter in uncertainty_parameters if parameter.requires_grad]
    if uncertainty:
        names["uncertainty_no_decay"] = [f"loss_aggregator.{index}" for index in range(len(uncertainty))]
        trainable += sum(int(parameter.numel()) for parameter in uncertainty)
    groups: list[dict[str, Any]] = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "name": "model_decay", "parameter_names": names["model_decay"]})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "name": "model_no_decay", "parameter_names": names["model_no_decay"]})
    if uncertainty:
        groups.append({"params": uncertainty, "weight_decay": 0.0, "name": "uncertainty_no_decay", "parameter_names": names["uncertainty_no_decay"]})
    return groups, {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def build_optimizer(
    model: nn.Module,
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    uncertainty_parameters: Iterable[nn.Parameter] = (),
    optimizer_module: Any | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    groups, totals = _trainable_groups(model, weight_decay=weight_decay, uncertainty_parameters=uncertainty_parameters)
    if not groups:
        raise ValueError("No trainable parameters are available for the optimizer")
    if optimizer_name == "AdamW":
        optimizer_cls = torch.optim.AdamW
    elif optimizer_name == "paged_adamw_8bit":
        module = optimizer_module
        if module is None:
            try:
                import bitsandbytes as module  # type: ignore[no-redef]
            except ImportError as exc:
                raise RuntimeBlocked("paged AdamW 8-bit requires bitsandbytes") from exc
        optimizer_cls = getattr(getattr(module, "optim", module), "PagedAdamW8bit", None)
        if optimizer_cls is None:
            raise RuntimeBlocked("bitsandbytes does not expose PagedAdamW8bit")
    else:
        raise ValueError(f"Unsupported locked optimizer: {optimizer_name}")
    optimizer = optimizer_cls(groups, lr=float(learning_rate))
    summary_groups = []
    for group in groups:
        parameters = list(group["params"])
        summary_groups.append({
            "name": group["name"],
            "parameter_count": sum(int(parameter.numel()) for parameter in parameters),
            "parameter_names": list(group.get("parameter_names", [])),
            "weight_decay": float(group["weight_decay"]),
            "learning_rate": float(learning_rate),
        })
    summary = {"optimizer": optimizer_name, "learning_rate": float(learning_rate), "weight_decay": float(weight_decay), "groups": summary_groups, **totals}
    return optimizer, summary
