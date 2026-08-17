"""Integrity-checked resume checkpoints for rationale generation.

This module deliberately delegates tensor/state serialization to
``training.checkpoints``.  The sidecar manifest is small metadata only; it
never contains a second copy of model weights.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
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

GENERATION_CHECKPOINT_SCHEMA_VERSION = 2
GENERATION_CHECKPOINT_POINTER_SCHEMA_VERSION = 1
GENERATION_SELECTION_METRIC_NAME = "full_split_macro_pragmatic_f1_all_zero_fallback_dev"
GENERATION_CHECKPOINT_POINTER_KINDS = ("latest", "best")
_CHECKPOINT_SHA_RE = re.compile(r"[0-9A-Fa-f]{64}")
_CANONICAL_EPOCH_PATH_RE = re.compile(r"checkpoints/epoch_(\d{4})/model\.pt")
_LEGACY_EPOCH_PATH_RE = re.compile(r"checkpoints/epoch_(\d+)/model\.pt")
_PLACEHOLDER_DATA_HASHES = {
    "",
    "NONE",
    "NULL",
    "NOT_PROVIDED",
    "NOT PROVIDED",
    "TO_BE_FILLED",
    "TO_BE_FILLED_AFTER_PROTOCOL_FILES_ARE_WRITTEN",
}
PROVENANCE_FIELDS = (
    "model",
    "model_artifact",
    "tokenizer_artifact",
    "dataset",
    "data_hash",
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
    epoch: int | None = None
    variant_fingerprint: str = "NOT_PROVIDED"
    selection_metric_name: str = GENERATION_SELECTION_METRIC_NAME
    selection_metric_value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "provenance": self.provenance,
            "provenance_sha256": self.provenance_sha256,
            "epoch": self.epoch,
            "variant_fingerprint": self.variant_fingerprint,
            "selection_metric_name": self.selection_metric_name,
            "selection_metric_value": self.selection_metric_value,
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


def _finite_metric(value: Any, *, allow_none: bool = True) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GenerationCheckpointError("generation checkpoint selection metric must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise GenerationCheckpointError("generation checkpoint selection metric must be finite")
    return result


def _positive_epoch(value: Any, *, field: str = "epoch") -> int:
    if isinstance(value, bool) or not isinstance(value, int | float) or not float(value).is_integer():
        raise GenerationCheckpointError(f"generation checkpoint {field} must be a positive integer")
    result = int(value)
    if result < 1:
        raise GenerationCheckpointError(f"generation checkpoint {field} must be a positive integer")
    return result


def _sha256(value: Any, *, field: str) -> str:
    result = str(value or "").strip().upper()
    if not _CHECKPOINT_SHA_RE.fullmatch(result):
        raise GenerationCheckpointError(f"generation checkpoint {field} must be a SHA-256 digest")
    return result


def _run_root_path(run_root: str | Path, path: str | Path, *, require_relative: bool = False) -> Path:
    """Resolve a checkpoint path while refusing paths outside the run root."""
    root = Path(run_root).resolve()
    candidate = Path(path)
    if require_relative and candidate.is_absolute():
        raise GenerationCheckpointError("generation checkpoint pointer path must be relative to the run root")
    resolved = (root / candidate if not candidate.is_absolute() else candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise GenerationCheckpointError("generation checkpoint path is outside the run root") from exc
    return resolved


def _relative_run_path(run_root: str | Path, path: str | Path) -> str:
    root = Path(run_root).resolve()
    resolved = _run_root_path(root, path)
    return resolved.relative_to(root).as_posix()


def _pointer_path(run_root: str | Path, kind: str) -> Path:
    if kind not in GENERATION_CHECKPOINT_POINTER_KINDS:
        raise GenerationCheckpointError(f"unsupported generation checkpoint pointer kind: {kind}")
    return Path(run_root) / "checkpoints" / f"{kind}_checkpoint.json"


def generation_checkpoint_pointer_path(run_root: str | Path, kind: str) -> Path:
    """Return the atomic pointer path for ``latest`` or ``best``."""
    return _pointer_path(run_root, kind)


def canonical_generation_epoch_path(epoch: int) -> str:
    """Return the immutable, zero-padded physical path for one epoch."""
    return f"checkpoints/epoch_{_positive_epoch(epoch):04d}/model.pt"


def _epoch_from_checkpoint_path(path: str, *, canonical_only: bool = False) -> int:
    normalized = Path(path).as_posix()
    match = _CANONICAL_EPOCH_PATH_RE.fullmatch(normalized)
    if match is None and not canonical_only:
        match = _LEGACY_EPOCH_PATH_RE.fullmatch(normalized)
    if match is None:
        raise GenerationCheckpointError(
            "generation checkpoint pointer path must identify checkpoints/epoch_NNNN/model.pt"
        )
    return _positive_epoch(int(match.group(1)))


def is_real_dataset_hash(value: Any) -> bool:
    """Return whether a data hash is usable as production provenance."""
    if value is None:
        return False
    normalized = str(value).strip()
    upper = normalized.upper()
    return bool(re.fullmatch(r"[0-9A-F]{64}", upper)) and upper not in _PLACEHOLDER_DATA_HASHES


def _model_is_cpu(model: nn.Module) -> bool:
    devices = {parameter.device for parameter in model.parameters()} | {buffer.device for buffer in model.buffers()}
    return all(device.type == "cpu" for device in devices)


def _validate_provenance_mode(
    provenance: Mapping[str, Any],
    model: nn.Module,
    *,
    production_provenance_required: bool,
    fixture_mode: bool,
) -> None:
    if fixture_mode and not _model_is_cpu(model):
        raise GenerationCheckpointError("fixture/legacy generation checkpoint mode requires a CPU model")
    if is_real_dataset_hash(provenance.get("data_hash")):
        return
    if production_provenance_required:
        raise GenerationCheckpointError(
            "production generation checkpoint requires a real dataset hash in canonical SHA-256 format"
        )
    if not fixture_mode:
        raise GenerationCheckpointError("dataset hash must be a canonical SHA-256 digest or explicit CPU fixture mode")


def _manifest_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(checkpoint_path.suffix + ".manifest.json")


def _validated_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in PROVENANCE_FIELDS if field not in value]
    if missing:
        raise GenerationCheckpointError(f"generation provenance is missing: {missing}")
    for field in ("model_artifact", "tokenizer_artifact"):
        artifact = value[field]
        if not isinstance(artifact, Mapping) or not str(artifact.get("identity", "")):
            raise GenerationCheckpointError(f"generation provenance {field} identity is missing")
    dataset = value["dataset"]
    if not isinstance(dataset, Mapping) or not str(dataset.get("identity", "")) or not str(dataset.get("hash", "")):
        raise GenerationCheckpointError("generation provenance dataset identity/hash is missing")
    if value["data_hash"] in (None, ""):
        raise GenerationCheckpointError("generation provenance data_hash is missing")
    if str(value["data_hash"]).strip().upper() != str(dataset["hash"]).strip().upper():
        raise GenerationCheckpointError("generation provenance data_hash does not match dataset hash")
    # Round-trip through JSON to reject non-portable identities at the boundary.
    try:
        encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise GenerationCheckpointError(f"generation provenance is not JSON serializable: {exc}") from exc


def _read_manifest(
    path: Path,
    checkpoint_path: Path,
    *,
    observed_checkpoint_sha256: str | None = None,
) -> GenerationCheckpointManifest:
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
            epoch=(None if raw.get("epoch") is None else _positive_epoch(raw["epoch"])),
            variant_fingerprint=str(raw.get("variant_fingerprint", "NOT_PROVIDED")),
            selection_metric_name=str(raw.get("selection_metric_name", GENERATION_SELECTION_METRIC_NAME)),
            selection_metric_value=_finite_metric(raw.get("selection_metric_value")),
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GenerationCheckpointError(f"invalid generation checkpoint manifest {path}: {exc}") from exc
    if manifest.schema_version != GENERATION_CHECKPOINT_SCHEMA_VERSION:
        raise GenerationCheckpointError(f"unsupported generation checkpoint manifest schema: {manifest.schema_version}")
    if manifest.provenance_sha256 != sha256_json(manifest.provenance):
        raise GenerationCheckpointError("generation checkpoint provenance hash mismatch")
    observed = sha256_file(checkpoint_path) if observed_checkpoint_sha256 is None else str(observed_checkpoint_sha256)
    if str(manifest.checkpoint_sha256).upper() != observed.upper():
        raise GenerationCheckpointError("generation checkpoint content hash mismatch")
    if not manifest.variant_fingerprint:
        raise GenerationCheckpointError("generation checkpoint variant fingerprint is missing")
    if not manifest.selection_metric_name:
        raise GenerationCheckpointError("generation checkpoint selection metric name is missing")
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
    production_provenance_required: bool = False,
    fixture_mode: bool = False,
    epoch: int | None = None,
    variant_fingerprint: str | None = None,
    selection_metric_name: str = GENERATION_SELECTION_METRIC_NAME,
    selection_metric_value: float | None = None,
) -> GenerationCheckpointManifest:
    """Atomically save one canonical checkpoint plus its integrity manifest."""
    checkpoint_path = Path(path)
    normalized = _validated_provenance(provenance)
    normalized_epoch = None if epoch is None else _positive_epoch(epoch)
    if not str(selection_metric_name).strip():
        raise GenerationCheckpointError("generation checkpoint selection metric name is missing")
    normalized_metric = _finite_metric(selection_metric_value)
    normalized_variant = str(variant_fingerprint or "NOT_PROVIDED").strip()
    if not normalized_variant:
        raise GenerationCheckpointError("generation checkpoint variant fingerprint is missing")
    _validate_provenance_mode(
        normalized,
        model,
        production_provenance_required=production_provenance_required,
        fixture_mode=fixture_mode,
    )
    payload = build_checkpoint_payload(
        model, optimizer, scheduler, loss_aggregator, run_state,
        metadata={
            **dict(metadata or {}),
            "checkpoint_kind": "generation_resume",
            "provenance": normalized,
            "provenance_sha256": sha256_json(normalized),
            "epoch": normalized_epoch,
            "variant_fingerprint": normalized_variant,
            "selection_metric_name": str(selection_metric_name),
            "selection_metric_value": normalized_metric,
        },
        rng_state=rng_state,
    )
    save_checkpoint(checkpoint_path, payload)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    manifest = GenerationCheckpointManifest(
        GENERATION_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_sha256,
        normalized,
        sha256_json(normalized),
        normalized_epoch,
        normalized_variant,
        str(selection_metric_name),
        normalized_metric,
    )
    atomic_write_json(_manifest_path(checkpoint_path), manifest.as_dict())
    # Verify both atomic outputs before making the checkpoint eligible for a
    # latest/best pointer.  A pointer must never reference a partially written
    # or stale canonical payload.
    verified = _read_manifest(
        _manifest_path(checkpoint_path),
        checkpoint_path,
        observed_checkpoint_sha256=checkpoint_sha256,
    )
    if verified.checkpoint_sha256 != manifest.checkpoint_sha256 or verified.provenance_sha256 != manifest.provenance_sha256:
        raise GenerationCheckpointError("generation checkpoint sidecar verification failed")
    return manifest


def _validate_pointer_target(
    run_root: str | Path,
    path: str | Path,
    *,
    expected_epoch: int | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_provenance_sha256: str | None = None,
    expected_variant_fingerprint: str | None = None,
    expected_selection_metric_name: str | None = None,
    expected_selection_metric_value: float | None = None,
) -> tuple[Path, GenerationCheckpointManifest, int]:
    relative = _relative_run_path(run_root, path)
    epoch = _epoch_from_checkpoint_path(relative, canonical_only=True)
    if expected_epoch is not None and epoch != _positive_epoch(expected_epoch, field="pointer epoch"):
        raise GenerationCheckpointError("generation checkpoint pointer epoch disagrees with its path")
    checkpoint_path = _run_root_path(run_root, relative)
    if not checkpoint_path.is_file():
        raise GenerationCheckpointError(f"generation checkpoint pointer target is missing: {relative}")
    observed_hash = _sha256(sha256_file(checkpoint_path), field="checkpoint_sha256")
    manifest = _read_manifest(
        _manifest_path(checkpoint_path),
        checkpoint_path,
        observed_checkpoint_sha256=observed_hash,
    )
    if manifest.epoch is None:
        raise GenerationCheckpointError("canonical generation checkpoint sidecar epoch is missing")
    if manifest.epoch != epoch:
        raise GenerationCheckpointError("generation checkpoint sidecar epoch disagrees with its path")
    sidecar_hash = _sha256(manifest.checkpoint_sha256, field="sidecar checkpoint_sha256")
    if observed_hash != sidecar_hash:
        raise GenerationCheckpointError("generation checkpoint content hash does not match its sidecar")
    if expected_checkpoint_sha256 is not None and observed_hash != _sha256(expected_checkpoint_sha256, field="pointer checkpoint_sha256"):
        raise GenerationCheckpointError("generation checkpoint pointer SHA does not match its target")
    sidecar_provenance = _sha256(manifest.provenance_sha256, field="sidecar provenance_sha256")
    if expected_provenance_sha256 is not None and sidecar_provenance != _sha256(expected_provenance_sha256, field="pointer provenance_sha256"):
        raise GenerationCheckpointError("generation checkpoint pointer provenance SHA does not match its sidecar")
    if expected_variant_fingerprint is not None and str(manifest.variant_fingerprint) != str(expected_variant_fingerprint):
        raise GenerationCheckpointError("generation checkpoint pointer variant fingerprint does not match its sidecar")
    if expected_selection_metric_name is not None and str(manifest.selection_metric_name) != str(expected_selection_metric_name):
        raise GenerationCheckpointError("generation checkpoint pointer selection metric name does not match its sidecar")
    if expected_selection_metric_value is not None and manifest.selection_metric_value is not None:
        observed_metric = _finite_metric(manifest.selection_metric_value)
        expected_metric = _finite_metric(expected_selection_metric_value, allow_none=False)
        if observed_metric is None or not math.isclose(observed_metric, expected_metric, rel_tol=0.0, abs_tol=1e-12):
            raise GenerationCheckpointError("generation checkpoint pointer selection metric value does not match its sidecar")
    return checkpoint_path, manifest, epoch


def write_generation_checkpoint_pointer(
    run_root: str | Path,
    kind: str,
    path: str | Path,
    *,
    selection_metric_name: str = GENERATION_SELECTION_METRIC_NAME,
    selection_metric_value: float | None = None,
    variant_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Atomically publish one tiny pointer to an immutable epoch payload."""
    if kind not in GENERATION_CHECKPOINT_POINTER_KINDS:
        raise GenerationCheckpointError(f"unsupported generation checkpoint pointer kind: {kind}")
    if not str(selection_metric_name).strip():
        raise GenerationCheckpointError("generation checkpoint pointer selection metric name is missing")
    metric = _finite_metric(selection_metric_value)
    checkpoint_path, manifest, epoch = _validate_pointer_target(
        run_root,
        path,
        expected_selection_metric_name=None,
    )
    normalized_variant = str(variant_fingerprint or manifest.variant_fingerprint).strip()
    if not normalized_variant or normalized_variant == "NOT_PROVIDED":
        raise GenerationCheckpointError("generation checkpoint pointer variant fingerprint is missing")
    if manifest.variant_fingerprint != normalized_variant:
        raise GenerationCheckpointError("generation checkpoint pointer variant fingerprint does not match its sidecar")
    sidecar_metric_name = str(manifest.selection_metric_name)
    if sidecar_metric_name != str(selection_metric_name):
        raise GenerationCheckpointError("generation checkpoint pointer selection metric name does not match its sidecar")
    sidecar_metric = _finite_metric(manifest.selection_metric_value)
    if metric is not None and sidecar_metric is not None:
        if not math.isclose(sidecar_metric, metric, rel_tol=0.0, abs_tol=1e-12):
            raise GenerationCheckpointError("generation checkpoint pointer selection metric value does not match its sidecar")
    elif sidecar_metric is not None:
        # A caller may omit the metric when the canonical sidecar already
        # contains it.  The pointer still records the verified value.
        metric = sidecar_metric
    pointer = {
        "schema_version": GENERATION_CHECKPOINT_POINTER_SCHEMA_VERSION,
        "path": checkpoint_path.relative_to(Path(run_root).resolve()).as_posix(),
        "epoch": epoch,
        "checkpoint_sha256": _sha256(manifest.checkpoint_sha256, field="checkpoint_sha256"),
        "provenance_sha256": _sha256(manifest.provenance_sha256, field="provenance_sha256"),
        "variant_fingerprint": normalized_variant,
        "selection_metric_name": str(selection_metric_name),
        "selection_metric_value": metric,
    }
    atomic_write_json(_pointer_path(run_root, kind), pointer)
    return pointer


def _legacy_pointer_record(run_root: str | Path, kind: str, path: Path) -> dict[str, Any]:
    """Describe an old physical best/latest file without rewriting it."""
    relative = _relative_run_path(run_root, path)
    if not path.is_file():
        raise GenerationCheckpointError(f"legacy generation checkpoint is missing: {relative}")
    epoch: int | None = None
    metric: float | None = None
    manifest_path = _manifest_path(path)
    checkpoint_sha256: str | None = None
    if manifest_path.exists():
        manifest = _read_manifest(manifest_path, path)
        checkpoint_sha256 = manifest.checkpoint_sha256
        epoch = manifest.epoch
        metric = manifest.selection_metric_value
        provenance_sha256 = manifest.provenance_sha256
        variant_fingerprint = manifest.variant_fingerprint
        metric_name = manifest.selection_metric_name
    else:
        try:
            try:
                raw = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                raw = torch.load(path, map_location="cpu")
        except Exception:
            raw = None
        state = raw.get("state", {}) if isinstance(raw, Mapping) else {}
        if isinstance(state, Mapping) and state.get("epoch") is not None:
            try:
                epoch = _positive_epoch(state.get("epoch"), field="legacy epoch")
            except GenerationCheckpointError:
                epoch = None
        provenance_sha256 = "NOT_PROVIDED"
        variant_fingerprint = "NOT_PROVIDED"
        metric_name = GENERATION_SELECTION_METRIC_NAME
        checkpoint_sha256 = sha256_file(path)
    if epoch is None:
        # Legacy physical files did not always persist an epoch.  Readers can
        # still load them, but pointer consumers must not claim a false epoch.
        epoch = 0
    return {
        "schema_version": GENERATION_CHECKPOINT_POINTER_SCHEMA_VERSION,
        "path": relative,
        "epoch": epoch,
        "checkpoint_sha256": _sha256(checkpoint_sha256, field="checkpoint_sha256"),
        "provenance_sha256": provenance_sha256,
        "variant_fingerprint": variant_fingerprint,
        "selection_metric_name": metric_name,
        "selection_metric_value": metric,
        "legacy": True,
    }


def read_generation_checkpoint_pointer(
    run_root: str | Path,
    kind: str,
    *,
    allow_legacy: bool = True,
) -> dict[str, Any]:
    """Read and fully validate a latest/best pointer, with legacy fallback."""
    pointer_path = _pointer_path(run_root, kind)
    if not pointer_path.exists():
        if allow_legacy:
            legacy_path = Path(run_root) / "checkpoints" / kind / "model.pt"
            if legacy_path.exists():
                return _legacy_pointer_record(run_root, kind, legacy_path)
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer is missing")
    try:
        raw = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer is unreadable: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer must be a JSON object")
    required = (
        "schema_version",
        "path",
        "epoch",
        "checkpoint_sha256",
        "provenance_sha256",
        "variant_fingerprint",
        "selection_metric_name",
        "selection_metric_value",
    )
    missing = [field for field in required if field not in raw]
    if missing:
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer is missing fields: {missing}")
    try:
        schema_version = int(raw["schema_version"])
    except (TypeError, ValueError) as exc:
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer schema is invalid") from exc
    if schema_version != GENERATION_CHECKPOINT_POINTER_SCHEMA_VERSION:
        raise GenerationCheckpointError(f"unsupported generation checkpoint pointer schema: {schema_version}")
    path_value = raw["path"]
    if not isinstance(path_value, str) or not path_value.strip():
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer path is missing")
    relative = Path(path_value).as_posix()
    if Path(path_value).is_absolute() or relative != path_value.replace("\\", "/"):
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer path must be a normalized relative path")
    epoch = _positive_epoch(raw["epoch"], field="pointer epoch")
    checkpoint_sha256 = _sha256(raw["checkpoint_sha256"], field="pointer checkpoint_sha256")
    provenance_sha256 = _sha256(raw["provenance_sha256"], field="pointer provenance_sha256")
    variant_fingerprint = str(raw["variant_fingerprint"]).strip()
    if not variant_fingerprint:
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer variant fingerprint is missing")
    metric_name = str(raw["selection_metric_name"]).strip()
    if not metric_name:
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer selection metric name is missing")
    metric = _finite_metric(raw["selection_metric_value"])
    checkpoint_path, manifest, target_epoch = _validate_pointer_target(
        run_root,
        relative,
        expected_epoch=epoch,
        expected_checkpoint_sha256=checkpoint_sha256,
        expected_provenance_sha256=provenance_sha256,
        expected_variant_fingerprint=variant_fingerprint,
        expected_selection_metric_name=metric_name,
        expected_selection_metric_value=metric,
    )
    if target_epoch != epoch:
        raise GenerationCheckpointError(f"generation checkpoint {kind} pointer epoch disagrees with its path")
    return {
        "schema_version": schema_version,
        "path": checkpoint_path.relative_to(Path(run_root).resolve()).as_posix(),
        "epoch": epoch,
        "checkpoint_sha256": checkpoint_sha256,
        "provenance_sha256": provenance_sha256,
        "variant_fingerprint": variant_fingerprint,
        "selection_metric_name": metric_name,
        "selection_metric_value": metric,
    }


def resolve_generation_checkpoint_pointer(
    run_root: str | Path,
    kind: str,
    *,
    allow_legacy: bool = True,
) -> Path:
    """Resolve a validated pointer to its physical payload path."""
    pointer = read_generation_checkpoint_pointer(run_root, kind, allow_legacy=allow_legacy)
    return _run_root_path(run_root, pointer["path"])


# Concise aliases are useful to stage/executor callers and keep the public
# surface discoverable for existing integrations.
read_checkpoint_pointer = read_generation_checkpoint_pointer
write_checkpoint_pointer = write_generation_checkpoint_pointer
resolve_checkpoint_pointer = resolve_generation_checkpoint_pointer


def _provenance_matches(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    fields: Sequence[str] | None,
) -> bool:
    if fields is None:
        return dict(observed) == dict(expected)
    unknown = [field for field in fields if field not in PROVENANCE_FIELDS]
    if unknown:
        raise GenerationCheckpointError(f"unknown provenance comparison fields: {unknown}")
    return all(observed.get(field) == expected.get(field) for field in fields)


def load_generation_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    expected_provenance: Mapping[str, Any] | None = None,
    compare_provenance_fields: Sequence[str] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    loss_aggregator: nn.Module | None = None,
    allow_legacy_fixture: bool = False,
    restore_training_state: bool = True,
    report_path: str | Path | None = None,
    restore_rng: Callable[[Mapping[str, Any]], None] | None = None,
    restore_hooks: Mapping[str, Callable[[Any], None]] | None = None,
    production_provenance_required: bool = False,
    fixture_mode: bool = False,
    observed_checkpoint_sha256: str | None = None,
) -> GenerationCheckpointLoadResult:
    """Validate identity/integrity, then delegate loading to canonical primitives."""
    checkpoint_path = Path(path)
    if allow_legacy_fixture and not fixture_mode:
        raise GenerationCheckpointError("legacy loading requires explicit fixture_mode=True")
    if production_provenance_required and expected_provenance is None:
        raise GenerationCheckpointError("production generation checkpoint load requires expected provenance")
    if fixture_mode and not _model_is_cpu(model):
        raise GenerationCheckpointError("fixture/legacy generation checkpoint mode requires a CPU model")
    manifest_path = _manifest_path(checkpoint_path)
    manifest: GenerationCheckpointManifest | None = None
    legacy_provenance: dict[str, Any] | None = None
    try:
        manifest = _read_manifest(
            manifest_path,
            checkpoint_path,
            observed_checkpoint_sha256=observed_checkpoint_sha256,
        )
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
            if not _provenance_matches(legacy_provenance, expected, compare_provenance_fields):
                raise GenerationCheckpointError("legacy generation checkpoint provenance identity mismatch")
    if manifest is not None and expected_provenance is not None:
        expected = _validated_provenance(expected_provenance)
        if not _provenance_matches(manifest.provenance, expected, compare_provenance_fields):
            raise GenerationCheckpointError("generation checkpoint provenance identity mismatch")
    if manifest is not None:
        _validate_provenance_mode(
            manifest.provenance,
            model,
            production_provenance_required=production_provenance_required,
            fixture_mode=fixture_mode,
        )
    elif production_provenance_required:
        if legacy_provenance is None:
            raise GenerationCheckpointError("production generation checkpoint requires validated legacy provenance")
        _validate_provenance_mode(
            legacy_provenance,
            model,
            production_provenance_required=production_provenance_required,
            fixture_mode=fixture_mode,
        )
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
