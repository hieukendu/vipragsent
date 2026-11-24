from __future__ import annotations

from pathlib import Path
from typing import Any

from .system_registry import resolve_execution_spec


def changed_components_against_full_phobert(root: str | Path, system_id: str) -> dict[str, Any]:
    spec = resolve_execution_spec(root, system_id)
    full = resolve_execution_spec(root, "full_phobert")
    changes: dict[str, Any] = {}
    if tuple(spec.active_heads) != tuple(full.active_heads):
        changes["active_heads"] = {"full": list(full.active_heads), "variant": list(spec.active_heads)}
    if tuple(spec.active_losses) != tuple(full.active_losses):
        changes["active_losses"] = {"full": list(full.active_losses), "variant": list(spec.active_losses)}
    if tuple(spec.uncertainty_tasks) != tuple(full.uncertainty_tasks):
        changes["uncertainty_tasks"] = {"full": list(full.uncertainty_tasks), "variant": list(spec.uncertainty_tasks)}
    if spec.rationale_training != full.rationale_training:
        changes["rationale_training"] = {"full": full.rationale_training, "variant": spec.rationale_training}
    if spec.executor_kind != full.executor_kind:
        changes["executor_kind"] = {"full": full.executor_kind, "variant": spec.executor_kind}
    if spec.checkpoint_semantics != full.checkpoint_semantics:
        changes["checkpoint_semantics"] = {"full": full.checkpoint_semantics, "variant": spec.checkpoint_semantics}
    if not changes:
        return {"baseline_system_id": "full_phobert", "variant_system_id": system_id, "changed_components": [], "status": "IDENTICAL_TO_FULL"}
    return {"baseline_system_id": "full_phobert", "variant_system_id": system_id, "changed_components": changes, "status": "RESOLVED"}
