from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from ..atomic import atomic_write_json, atomic_write_text
from ..hashing import sha256_json
from ..orchestration.contracts import RunEntry
from ..orchestration.system_registry import SystemExecutionSpec


@dataclass(frozen=True)
class ResolvedTrainingConfig:
    system_id: str
    model_family: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    scheduler: str
    warmup_ratio: float
    physical_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    maximum_epochs: int
    precision: str
    gradient_clipping: float
    patience: int
    minimum_delta: float
    gradient_checkpointing: bool
    deterministic_algorithms: str
    uncertainty_weighting_enabled: bool
    active_uncertainty_tasks: tuple[str, ...]
    rationale_training: bool
    rationale_beta: float
    rationale_inference: bool
    qlora: dict[str, Any]
    selection_metric: str
    config_hash: str

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["active_uncertainty_tasks"] = list(self.active_uncertainty_tasks)
        return result


def _load_training_values(root: Path) -> dict[str, Any]:
    path = root / "configs/runtime/training.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("configs/runtime/training.yaml must contain a mapping")
    return payload


def _family_values(payload: dict[str, Any], model_family: str) -> dict[str, Any]:
    if model_family in {"phobert_base", "xlmr_large"}:
        values = payload.get("encoder")
    elif model_family in {"sailor_7b", "vistral_7b"}:
        values = payload.get("qlora_7b")
    else:
        raise ValueError(f"No locked training family exists for {model_family}")
    if not isinstance(values, dict):
        raise ValueError(f"Missing locked training values for {model_family}")
    return values


def _selection_metric(entry: RunEntry, spec: SystemExecutionSpec) -> str:
    # This is an exact protocol-key translation, not a system-name heuristic.
    values = {
        "macro_prag_f1_dev": "dev_macro_pragmatic_f1",
        "sarcasm_dev_f1": "dev_sarcasm_binary_macro_f1",
        "dev_macro_pragmatic_f1": "dev_macro_pragmatic_f1",
        "dev_sarcasm_macro_f1": "dev_sarcasm_macro_f1",
        "dev_sarcasm_binary_macro_f1": "dev_sarcasm_binary_macro_f1",
        "dev_polarity_macro_f1": "dev_polarity_macro_f1",
        "dev_emotion_macro_f1": "dev_emotion_macro_f1",
    }
    key = str(entry.raw.get("selection_metric") or spec.selection_metric)
    try:
        return values[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported locked selection metric for {entry.run_id}: {key}") from exc


def _physical_batch(values: dict[str, Any], model_family: str, runtime_status: dict[str, Any] | None) -> int:
    status = runtime_status or {}
    candidate = status.get("successful_batch")
    if candidate is None:
        raise ValueError(f"Phase 15 physical batch is unresolved for {model_family}")
    physical = int(candidate)
    probe = values.get("physical_batch_probe", {})
    candidates = probe.get(model_family) if isinstance(probe, dict) else None
    if not candidates or physical not in {int(item) for item in candidates}:
        raise ValueError(f"Physical batch {physical} is not in the locked probe order for {model_family}")
    return physical


def resolve_training_config(
    entry: RunEntry,
    execution_spec: SystemExecutionSpec,
    *,
    root: str | Path = ".",
    runtime_status: dict[str, Any] | None = None,
) -> ResolvedTrainingConfig:
    root = Path(root)
    if execution_spec.executor_kind not in {"single_model_trainable", "single_task_bundle", "independent_checkpoint_bundle", "generation_baseline"}:
        raise ValueError(f"Training configuration is not applicable to executor {execution_spec.executor_kind}")
    values = _family_values(_load_training_values(root), execution_spec.model_family)
    physical = _physical_batch(values, execution_spec.model_family, runtime_status)
    effective = int(values["effective_batch_size"])
    if effective % physical:
        raise ValueError(f"Effective batch {effective} is not exactly divisible by physical batch {physical}")
    accumulation = effective // physical
    rationale = _load_training_values(root).get("rationale", {})
    uncertainty_enabled = bool(execution_spec.uncertainty_tasks)
    qlora = {"quantization": dict(values.get("quantization", {})), "lora": dict(values.get("lora", {}))} if execution_spec.model_family in {"sailor_7b", "vistral_7b"} else {"quantization": {"type": "none"}, "lora": {}}
    raw = {
        "system_id": entry.system_id,
        "model_family": execution_spec.model_family,
        "optimizer": str(values["optimizer"]),
        "learning_rate": float(values["learning_rate"]),
        "weight_decay": float(values["weight_decay"]),
        "scheduler": str(values["scheduler"]),
        "warmup_ratio": float(values["warmup_ratio"]),
        "physical_batch_size": physical,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective,
        "maximum_epochs": int(values["maximum_epochs"]),
        "precision": str(values["precision"]),
        "gradient_clipping": float(values["gradient_clipping"]),
        "patience": int(values["patience"]),
        "minimum_delta": float(values["minimum_delta"]),
        "gradient_checkpointing": bool(values["gradient_checkpointing"]),
        "deterministic_algorithms": str(_load_training_values(root)["deterministic_algorithms"]),
        "uncertainty_weighting_enabled": uncertainty_enabled,
        "active_uncertainty_tasks": list(execution_spec.uncertainty_tasks),
        "rationale_training": bool(execution_spec.rationale_training),
        "rationale_beta": float(rationale["beta"]),
        "rationale_inference": bool(execution_spec.rationale_inference),
        "qlora": qlora,
        "selection_metric": _selection_metric(entry, execution_spec),
    }
    config_hash = sha256_json(raw)
    return ResolvedTrainingConfig(config_hash=config_hash, **raw)


def persist_resolved_training_config(root: str | Path, run_root: str | Path, config: ResolvedTrainingConfig) -> dict[str, str]:
    root = Path(root)
    run_root = Path(run_root)
    payload = config.as_dict()
    json_path = run_root / "training/resolved_training_config.json"
    snapshot_path = run_root / "config_snapshot.yaml"
    atomic_write_json(json_path, payload)
    snapshot = {
        "configuration_source": "configs/runtime/training.yaml and exact system execution registry entry",
        "resolved_training_config": payload,
    }
    atomic_write_text(snapshot_path, yaml.safe_dump(snapshot, sort_keys=True, allow_unicode=False))
    def display(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

    return {
        "resolved_training_config": display(json_path),
        "config_snapshot": display(snapshot_path),
        "config_hash": config.config_hash,
    }


def resolved_config_from_json(path: str | Path) -> ResolvedTrainingConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    payload["active_uncertainty_tasks"] = tuple(payload["active_uncertainty_tasks"])
    return ResolvedTrainingConfig(**payload)
