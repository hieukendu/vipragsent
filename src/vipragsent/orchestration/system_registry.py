from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..constants import PRAGMATIC_LABELS
from ..hashing import sha256_file, sha256_json

ALLOWED_EXECUTOR_KINDS = frozenset(
    {
        "single_model_trainable",
        "single_task_bundle",
        "independent_checkpoint_bundle",
        "checkpoint_reuse",
        "evaluation_only",
        "artifact_extraction",
        "generation_baseline",
        "generation_trainable",
        "rationale_checkpoint_reuse",
        "azure",
    }
)
REQUIRED_FIELDS = (
    "system_id",
    "model_family",
    "executor_kind",
    "variant_id",
    "active_heads",
    "active_losses",
    "uncertainty_tasks",
    "rationale_training",
    "rationale_inference",
    "checkpoint_semantics",
    "selection_metric",
    "evaluation_strategy",
    "external_evaluation_strategy",
    "output_source",
    "reusable_checkpoint_key_pattern",
    "approved_source_reference",
)


@dataclass(frozen=True)
class SystemExecutionSpec:
    system_id: str
    model_family: str
    executor_kind: str
    variant_id: str
    active_heads: tuple[str, ...]
    active_losses: tuple[str, ...]
    uncertainty_tasks: tuple[str, ...]
    rationale_training: bool
    rationale_inference: bool
    checkpoint_semantics: str
    selection_metric: str
    evaluation_strategy: str
    external_evaluation_strategy: str
    output_source: str
    reusable_checkpoint_key_pattern: str
    approved_source_reference: str
    source_system_id: str = ""
    source_checkpoint_key_pattern: str = ""
    additional_training: bool = True
    direct_classification_outputs_used: bool = True

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SystemExecutionSpec:
        missing = [field for field in REQUIRED_FIELDS if field not in value]
        if missing:
            raise ValueError(f"Execution registry entry is missing fields: {missing}")
        system_id = str(value["system_id"])
        executor_kind = str(value["executor_kind"])
        if not system_id:
            raise ValueError("Execution registry system_id cannot be empty")
        if executor_kind not in ALLOWED_EXECUTOR_KINDS:
            raise ValueError(f"Unsupported executor kind for {system_id}: {executor_kind}")
        active_heads = tuple(str(item) for item in value["active_heads"])
        uncertainty_tasks = tuple(str(item) for item in value["uncertainty_tasks"])
        if not set(uncertainty_tasks).issubset(set(PRAGMATIC_LABELS) | {"polarity", "emotion"}):
            raise ValueError(f"Unknown uncertainty task for {system_id}")
        if any(not str(value[field]) for field in ("model_family", "variant_id", "checkpoint_semantics", "selection_metric", "evaluation_strategy", "external_evaluation_strategy", "output_source", "reusable_checkpoint_key_pattern", "approved_source_reference")):
            raise ValueError(f"Execution registry entry has an empty semantic field: {system_id}")
        return cls(
            system_id=system_id,
            model_family=str(value["model_family"]),
            executor_kind=executor_kind,
            variant_id=str(value["variant_id"]),
            active_heads=active_heads,
            active_losses=tuple(str(item) for item in value["active_losses"]),
            uncertainty_tasks=uncertainty_tasks,
            rationale_training=bool(value["rationale_training"]),
            rationale_inference=bool(value["rationale_inference"]),
            checkpoint_semantics=str(value["checkpoint_semantics"]),
            selection_metric=str(value["selection_metric"]),
            evaluation_strategy=str(value["evaluation_strategy"]),
            external_evaluation_strategy=str(value["external_evaluation_strategy"]),
            output_source=str(value["output_source"]),
            reusable_checkpoint_key_pattern=str(value["reusable_checkpoint_key_pattern"]),
            approved_source_reference=str(value["approved_source_reference"]),
            source_system_id=str(value.get("source_system_id", "")),
            source_checkpoint_key_pattern=str(value.get("source_checkpoint_key_pattern", "")),
            additional_training=bool(value.get("additional_training", executor_kind not in {"evaluation_only", "checkpoint_reuse"})),
            direct_classification_outputs_used=bool(value.get("direct_classification_outputs_used", True)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "system_id": self.system_id,
            "model_family": self.model_family,
            "executor_kind": self.executor_kind,
            "variant_id": self.variant_id,
            "active_heads": list(self.active_heads),
            "active_losses": list(self.active_losses),
            "uncertainty_tasks": list(self.uncertainty_tasks),
            "rationale_training": self.rationale_training,
            "rationale_inference": self.rationale_inference,
            "checkpoint_semantics": self.checkpoint_semantics,
            "selection_metric": self.selection_metric,
            "evaluation_strategy": self.evaluation_strategy,
            "external_evaluation_strategy": self.external_evaluation_strategy,
            "output_source": self.output_source,
            "reusable_checkpoint_key_pattern": self.reusable_checkpoint_key_pattern,
            "approved_source_reference": self.approved_source_reference,
            "source_system_id": self.source_system_id,
            "source_checkpoint_key_pattern": self.source_checkpoint_key_pattern,
            "additional_training": self.additional_training,
            "direct_classification_outputs_used": self.direct_classification_outputs_used,
        }


def load_execution_registry(root: str | Path = ".") -> dict[str, SystemExecutionSpec]:
    path = Path(root) / "configs/experiments/system_execution_registry.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = payload.get("systems")
    if not isinstance(entries, list):
        raise ValueError("system_execution_registry.yaml must contain a systems list")
    specs: dict[str, SystemExecutionSpec] = {}
    duplicates: list[str] = []
    for raw in entries:
        if not isinstance(raw, Mapping):
            raise ValueError("Every system execution registry entry must be a mapping")
        spec = SystemExecutionSpec.from_mapping(raw)
        if spec.system_id in specs:
            duplicates.append(spec.system_id)
        specs[spec.system_id] = spec
    if duplicates:
        raise ValueError(f"Duplicate system execution registry IDs: {sorted(set(duplicates))}")
    return specs


def validate_execution_registry(root: str | Path = ".", inventory_rows: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    root = Path(root)
    from .inventory import build_expected_runs

    rows = inventory_rows if inventory_rows is not None else build_expected_runs(root)["rows"]
    specs = load_execution_registry(root)
    inventory_ids = {str(row.get("system_id") or row.get("system")) for row in rows}
    registry_ids = set(specs)
    missing = sorted(inventory_ids - registry_ids)
    extra = sorted(registry_ids - inventory_ids)
    invalid_model_rows = sorted(
        spec.system_id
        for spec in specs.values()
        if not spec.model_family or not spec.executor_kind
    )
    semantic_errors: list[str] = []
    for system_id, spec in sorted(specs.items()):
        if system_id == "cot_only_vistral":
            if spec.executor_kind != "generation_trainable":
                semantic_errors.append("cot_only_vistral.executor_kind must be generation_trainable")
            expected = {
                "active_heads": (),
                "active_losses": ("generation_cross_entropy",),
                "rationale_training": True,
                "rationale_inference": True,
                "checkpoint_semantics": "own_generation_checkpoint",
                "selection_metric": "full_split_macro_pragmatic_f1_all_zero_fallback_dev",
                "evaluation_strategy": "reasoning_generation_shared_judge",
                "output_source": "judged_generated_reasoning",
                "additional_training": True,
                "direct_classification_outputs_used": False,
            }
            for field, value in expected.items():
                if getattr(spec, field) != value:
                    semantic_errors.append(f"cot_only_vistral.{field} must equal {value!r}")
            if spec.source_system_id or spec.source_checkpoint_key_pattern:
                semantic_errors.append("cot_only_vistral cannot declare a source checkpoint")
        if system_id == "explanation_only_vistral":
            if spec.executor_kind != "rationale_checkpoint_reuse":
                semantic_errors.append("explanation_only_vistral.executor_kind must be rationale_checkpoint_reuse")
            expected = {
                "active_heads": (),
                "active_losses": (),
                "rationale_training": False,
                "rationale_inference": True,
                "checkpoint_semantics": "reuse_approved_full_vistral_same_seed",
                "selection_metric": "inherited_from_source_checkpoint",
                "evaluation_strategy": "rationale_only_shared_judge",
                "output_source": "judged_rationale_decoder_output",
                "source_system_id": "vipragsent_full_vistral",
                "source_checkpoint_key_pattern": "vipragsent_full_vistral:{seed}",
                "additional_training": False,
                "direct_classification_outputs_used": False,
            }
            for field, value in expected.items():
                if getattr(spec, field) != value:
                    semantic_errors.append(f"explanation_only_vistral.{field} must equal {value!r}")
    passed = not missing and not extra and not invalid_model_rows and not semantic_errors and len(specs) == len(registry_ids)
    return {
        "status": "PASS" if passed else "FAIL",
        "inventory_system_count": len(inventory_ids),
        "registry_system_count": len(registry_ids),
        "inventory_row_count": len(rows),
        "missing_system_ids": missing,
        "extra_system_ids": extra,
        "invalid_entries": invalid_model_rows,
        "semantic_errors": semantic_errors,
        "registry_hash": sha256_json({key: specs[key].as_dict() for key in sorted(specs)}),
        "registry_file_sha256": sha256_file(root / "configs/experiments/system_execution_registry.yaml"),
        "executor_kinds": {kind: sum(spec.executor_kind == kind for spec in specs.values()) for kind in sorted(ALLOWED_EXECUTOR_KINDS)},
    }


def resolve_execution_spec(root: str | Path, system_id: str) -> SystemExecutionSpec:
    specs = load_execution_registry(root)
    try:
        return specs[system_id]
    except KeyError as exc:
        raise ValueError(f"Unknown system_id is BLOCKED before model construction: {system_id}") from exc
