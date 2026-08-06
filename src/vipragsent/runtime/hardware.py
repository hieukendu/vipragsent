from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import yaml


def _device_records(torch_module: Any = torch) -> list[dict[str, Any]]:
    if not bool(torch_module.cuda.is_available()):
        return []
    records: list[dict[str, Any]] = []
    for index in range(int(torch_module.cuda.device_count())):
        properties = torch_module.cuda.get_device_properties(index)
        records.append({
            "index": index,
            "name": str(torch_module.cuda.get_device_name(index)),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_gb": float(properties.total_memory / (1024**3)),
        })
    return records


def validate_hardware(root: str | Path = ".", *, torch_module: Any = torch, selected_index: int | None = None) -> dict[str, Any]:
    root = Path(root)
    config_path = root / "configs/runtime/a100_20gb.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    records = _device_records(torch_module)
    index = int(selected_index if selected_index is not None else (torch_module.cuda.current_device() if records else 0))
    selected = next((record for record in records if record["index"] == index), None)
    required_pattern = str(config.get("gpu_name_pattern", "A100"))
    name_ok = bool(selected and (re.search(required_pattern, selected["name"], re.IGNORECASE) or "MIG" in selected["name"]))
    memory_ok = bool(selected and selected["total_memory_gb"] <= float(config.get("max_gpu_memory_gb", 20)) + 0.5 and selected["total_memory_gb"] > 0)
    bf16_ok = bool(getattr(torch_module.cuda, "is_bf16_supported", lambda: False)()) if records else False
    checks = {
        "cuda_available": bool(records),
        "selected_device": selected is not None,
        "a100_or_approved_mig": name_ok,
        "memory_profile": memory_ok,
        "bf16_supported": bf16_ok,
        "one_gpu_policy": len(records) == 1,
        "cuda_torch_runtime": bool(getattr(torch_module, "version", None) and getattr(torch_module.version, "cuda", None)) if records else False,
    }
    blockers = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS" if not blockers else "BLOCKED",
        "selected_device_index": index,
        "selected_device": selected,
        "devices": records,
        "accepted_runtime": {"gpu_name_pattern": required_pattern, "max_gpu_memory_gb": config.get("max_gpu_memory_gb"), "precision": config.get("precision")},
        "checks": checks,
        "blockers": blockers,
    }


def hardware_identity(report: dict[str, Any]) -> str:
    selected = report.get("selected_device") or {}
    if not selected:
        return "unavailable"
    return f"{selected.get('name')}|cc={selected.get('compute_capability')}|memory_gb={selected.get('total_memory_gb')}"
