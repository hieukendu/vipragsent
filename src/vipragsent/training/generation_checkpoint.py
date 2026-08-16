"""Integrity-checked resume checkpoints for rationale generation.

This module deliberately delegates tensor/state serialization to
``training.checkpoints``.  The sidecar manifest is small metadata only; it
never contains a second copy of model weights.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..atomic import atomic_write_json
from ..hashing import sha256_file, sha256_json
from .checkpoints import (
    CheckpointContractError,
    CheckpointLoadResult,
    build_checkpoint_payload,
    load_checkpoint,
    save_checkpoint,
)

GENERATION_CHECKPOINT_SCHEMA_VERSION = 1
PROVENANCE_FIELDS = (
    "model",
    "optimizer",
    "scheduler",
    "rng",
    "data_order",
    "config",
    "model_environment",
)


class GenerationCheckpointError(CheckpointContractError):
    """A generation checkpoint is corrupt or does not identify this run."""


@dataclass(frozen=True)
class GenerationCheckpointManifest:
    schema_version: int
    checkpoint_sha256: str
    provenance: dict[str, Any]
    provenance_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "provenance": self.provenance,
            "provenance_sha256": self.provenance_sha256,
        }


@dataclass(frozen=True)
class GenerationCheckpointLoadResult:
    checkpoint: CheckpointLoadResult
    manifest: GenerationCheckpointManifest | None

    @property
    def payload(self) -> dict[str, Any]:
        return self.checkpoint.payload

    @property
    def run_state(self) -> Mapping[str, Any]:
        return self.checkpoint.run_state


def _manifest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".manifest.json")


def _validated_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in PROVENANCE_FIELDS if field not in value]
    if missing:
        raise GenerationCheckpointError(f"generation provenance is missing: {missing}")
    # Round-trip through JSON to reject non-portable identities at the boundary.
    try:
        encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise GenerationCheckpointError(f"generation provenance is not JSON serializable: {exc}") from exc


def _read_manifest(path: Path, checkpoint_path: Path) -> GenerationCheckpointManifest:
    if not path.exists():
        raise GenerationCheckpointError(f"generation checkpoint sidecar manifest is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        provenance = _validated_provenance(raw["provenance"])
        manifest = GenerationCheckpointManifest(
            schema_version=int(raw["schema_version"]),
            checkpoint_sha256=str(raw["checkpoint_sha256"]),
            provenance=provenance,
            provenance_sha256=str(raw["provenance_sha256"]),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GenerationCheckpointError(f"invalid generation checkpoint manifest {path}: {exc}") from exc
    if manifest.schema_version != GENERATION_CHECKPOINT_SCHEMA_VERSION:
        raise GenerationCheckpointError(f"unsupported generation checkpoint manifest schema: {manifest.schema_version}")
    if manifest.provenance_sha256 != sha256_json(manifest.provenance):
        raise GenerationCheckpointError("generation checkpoint provenance hash mismatch")
    if manifest.checkpoint_sha256 != sha256_file(checkpoint_path):
        raise GenerationCheckpointError("generation checkpoint content hash mismatch")
    return manifest


def _legacy_payload_info(path: Path) -> tuple[bool, dict[str, Any] | None]:
    """Inspect a legacy fixture without making a canonical checkpoint optional."""
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping) or raw.get("schema_version") == 2 or "model" not in raw:
        return False, None
    metadata = raw.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("provenance"), Mapping):
        return True, None
    provenance = _validated_provenance(metadata["provenance"])
    if metadata.get("provenance_sha256") != sha256_json(provenance):
        raise GenerationCheckpointError("legacy generation provenance hash mismatch")
    return True, provenance


def save_generation_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    run_state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    loss_aggregator: nn.Module | None = None,
    rng_state: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> GenerationCheckpointManifest:
    """Atomically save one canonical checkpoint plus its integrity manifest."""
    checkpoint_path = Path(path)
    normalized = _validated_provenance(provenance)
    payload = build_checkpoint_payload(
        model, optimizer, scheduler, loss_aggregator, run_state,
        metadata={
            **dict(metadata or {}),
            "checkpoint_kind": "generation_resume",
            "provenance": normalized,
        },
        rng_state=rng_state,
    )
    save_checkpoint(checkpoint_path, payload)
    manifest = GenerationCheckpointManifest(
        GENERATION_CHECKPOINT_SCHEMA_VERSION,
        sha256_file(checkpoint_path),
        normalized,
        sha256_json(normalized),
    )
    atomic_write_json(_manifest_path(checkpoint_path), manifest.as_dict())
    return manifest


def load_generation_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    expected_provenance: Mapping[str, Any] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    loss_aggregator: nn.Module | None = None,
    allow_legacy_fixture: bool = False,
    restore_training_state: bool = True,
    report_path: str | Path | None = None,
    restore_rng: Callable[[Mapping[str, Any]], None] | None = None,
    restore_hooks: Mapping[str, Callable[[Any], None]] | None = None,
) -> GenerationCheckpointLoadResult:
    """Validate identity/integrity, then delegate loading to canonical primitives."""
    checkpoint_path = Path(path)
    manifest_path = _manifest_path(checkpoint_path)
    manifest: GenerationCheckpointManifest | None = None
    try:
        manifest = _read_manifest(manifest_path, checkpoint_path)
    except GenerationCheckpointError:
        # A canonical checkpoint without its sidecar is never loadable.  Only
        # an explicitly opted-in historical fixture may use this recovery path.
        if manifest_path.exists() or not allow_legacy_fixture:
            raise
        is_legacy, legacy_provenance = _legacy_payload_info(checkpoint_path)
        if not is_legacy:
            raise GenerationCheckpointError("canonical generation checkpoint cannot load without its sidecar manifest")
        if expected_provenance is not None:
            expected = _validated_provenance(expected_provenance)
            if legacy_provenance is None:
                raise GenerationCheckpointError("legacy checkpoint lacks validated provenance for identity checking")
            if legacy_provenance != expected:
                raise GenerationCheckpointError("legacy generation checkpoint provenance identity mismatch")
    if manifest is not None and expected_provenance is not None:
        expected = _validated_provenance(expected_provenance)
        if manifest.provenance != expected:
            raise GenerationCheckpointError("generation checkpoint provenance identity mismatch")
    kwargs: dict[str, Any] = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_aggregator": loss_aggregator,
        "restore_training_state": restore_training_state,
        "allow_legacy_fixture": allow_legacy_fixture,
        "report_path": report_path,
    }
    if restore_rng is not None:
        kwargs["restore_rng"] = restore_rng
    result = load_checkpoint(checkpoint_path, model, **kwargs)
    hooks = restore_hooks or {}
    if optimizer is not None and result.payload.get("optimizer_state_dict") is not None and "optimizer" in hooks:
        hooks["optimizer"](result.payload["optimizer_state_dict"])
    if scheduler is not None and result.payload.get("scheduler_state_dict") is not None and "scheduler" in hooks:
        hooks["scheduler"](result.payload["scheduler_state_dict"])
    if result.payload.get("rng_state") and "rng" in hooks:
        hooks["rng"](result.payload["rng_state"])
    return GenerationCheckpointLoadResult(result, manifest)
