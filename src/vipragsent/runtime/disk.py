from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml


def derive_minimum_free_disk_bytes(root: str | Path, model_family: str, *, snapshot_size_bytes: int | None = None) -> dict[str, Any]:
    root = Path(root)
    registry = yaml.safe_load((root / "configs/models/model_registry.yaml").read_text(encoding="utf-8")) or {}
    model = (registry.get("models") or {}).get(model_family, {})
    snapshot = int(snapshot_size_bytes or model.get("snapshot_size_bytes") or (8 * 1024**3 if model.get("quantization") == "nf4" else 3 * 1024**3))
    checkpoint = int(snapshot * (0.35 if model.get("quantization") == "nf4" else 0.20))
    optimizer = int(snapshot * (0.25 if model.get("quantization") == "nf4" else 0.10))
    artifacts = 512 * 1024**2
    safety_margin = 2 * 1024**3
    minimum = snapshot + checkpoint + optimizer + artifacts + safety_margin
    usage = shutil.disk_usage(root)
    return {
        "model_family": model_family,
        "snapshot_estimate_bytes": snapshot,
        "checkpoint_estimate_bytes": checkpoint,
        "optimizer_state_estimate_bytes": optimizer,
        "prediction_artifact_estimate_bytes": artifacts,
        "safety_margin_bytes": safety_margin,
        "minimum_free_bytes": minimum,
        "available_free_bytes": usage.free,
        "passed": usage.free >= minimum,
    }
