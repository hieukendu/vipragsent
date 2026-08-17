"""Future-only, inference-only runtime for the Vistral explanation system.

This module is intentionally an adapter around the Wave-2 generation
contracts.  It does not introduce a second checkpoint or chunk format:
``GenerationChunkStore`` remains the commit boundary and the generation
batch-policy/context helpers remain the source of truth for inference
execution.  The production explanation stage registry routes through this
runtime; legacy artifacts remain preserved under their separate artifact
root and are never reinterpreted as canonical runtime chunks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..atomic import atomic_write_json, atomic_write_text
from ..hashing import sha256_file, sha256_json
from ..runtime.device import (
    assert_runtime_device_contract,
    move_batch_to_model_device,
    resolve_model_input_device,
    write_device_report,
)
from ..training.generation_checkpoint import is_real_dataset_hash
from .approval import validate_approval_record
from .executors.explanation_reuse import (
    ApprovedFullVistralSource,
    resolve_approved_full_vistral_source,
    validate_source_checkpoint,
)
from .executors.generation import (
    SUPPORTED_GENERATION_BATCH_SIZES,
    reversible_inference_context,
    select_generation_batch_size,
)
from .generation_persistence import GenerationChunkStore
from .provenance import expected_inference_provenance, validate_inference_provenance
from .run_store import git_commit, git_source_fingerprint

EXPLANATION_SYSTEM_ID = "explanation_only_vistral"
SOURCE_SYSTEM_ID = "vipragsent_full_vistral"
EXPLANATION_ENGINE_ID = "luna_explanation_engine"
EXPLANATION_ENGINE_VERSION = "task-h-v1"
SHARED_GENERATION_ENGINE_ID = "wave2_generation_engine"
SHARED_GENERATION_ENGINE_VERSION = "wave2-v1"
SHARED_PROTOCOL_ID = "reasoning_generation_shared_judge_v1"
SHARED_PROTOCOL_VERSION = "v1"
SHARED_BATCH_POLICY_ID = "generation_inference_batch_policy"
SHARED_BATCH_POLICY_VERSION = "wave2-v1"
CONTRACT_VERSION = 1
EXPECTED_EXPLANATION_SEEDS = (20260521, 20260522, 20260523)


class ExplanationRuntimeError(RuntimeError):
    """The explanation-only contract or artifact state is unsafe."""


class ExplanationRecordError(ExplanationRuntimeError):
    """An explanation failure attributable to one input record or its decoding."""


ExplanationContractError = ExplanationRuntimeError


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _record_identity(value: Any) -> Any:
    """Canonicalize tensor-backed input records for the generation contract."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "values": tensor.tolist(),
        }
    if isinstance(value, Mapping):
        return {str(key): _record_identity(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_record_identity(item) for item in value]
    return _jsonable(value)


def _same_json(left: Any, right: Any) -> bool:
    return _jsonable(left) == _jsonable(right)


def _dataset_binding(payload: Mapping[str, Any], run_manifest_path: Path | None = None) -> dict[str, str]:
    """Read the canonical dataset identity/hash binding from source metadata."""
    candidates: list[Mapping[str, Any]] = [payload]
    nested = payload.get("dataset")
    if isinstance(nested, Mapping):
        candidates.insert(0, nested)
    if run_manifest_path is not None and run_manifest_path.exists():
        try:
            manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, Mapping):
            candidates.append(manifest)
            nested_manifest = manifest.get("dataset")
            if isinstance(nested_manifest, Mapping):
                candidates.insert(0, nested_manifest)
    identity_keys = ("identity", "dataset_identity", "dataset_id", "manifest_id", "dataset_manifest_id")
    hash_keys = ("hash", "dataset_hash", "data_hash", "dataset_fingerprint", "manifest_hash", "dataset_manifest_hash")
    identity = next((str(item[key]) for item in candidates for key in identity_keys if str(item.get(key, "")).strip()), "")
    digest = next((str(item[key]) for item in candidates for key in hash_keys if str(item.get(key, "")).strip()), "")
    return {"identity": identity, "hash": digest.upper()}


def _dataset_binding_from_run_root(root: Path) -> dict[str, str]:
    for name in ("review_summary.json", "run_manifest.json", "state.json"):
        path = root / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            binding = _dataset_binding(payload, root / "run_manifest.json")
            if binding["identity"] and binding["hash"]:
                return binding
    return {"identity": "", "hash": ""}


@dataclass(frozen=True)
class SharedInferenceIdentity:
    """Frozen engine/protocol/batch/environment identity shared by all seeds."""

    engine_id: str = EXPLANATION_ENGINE_ID
    engine_version: str = EXPLANATION_ENGINE_VERSION
    generation_engine_id: str = SHARED_GENERATION_ENGINE_ID
    generation_engine_version: str = SHARED_GENERATION_ENGINE_VERSION
    protocol_id: str = SHARED_PROTOCOL_ID
    protocol_version: str = SHARED_PROTOCOL_VERSION
    protocol_hash: str = "NOT_PROVIDED"
    batch_policy_id: str = SHARED_BATCH_POLICY_ID
    batch_policy_version: str = SHARED_BATCH_POLICY_VERSION
    environment_identity: str = "NOT_PROVIDED"
    environment_version: str = "NOT_PROVIDED"

    def as_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.as_dict())

    def validate(self) -> None:
        expected = {
            "engine_id": EXPLANATION_ENGINE_ID,
            "generation_engine_id": SHARED_GENERATION_ENGINE_ID,
            "protocol_id": SHARED_PROTOCOL_ID,
            "batch_policy_id": SHARED_BATCH_POLICY_ID,
        }
        for field_name, expected_value in expected.items():
            if str(getattr(self, field_name)) != expected_value:
                raise ExplanationRuntimeError(
                    f"unsupported explanation engine identity {field_name}={getattr(self, field_name)!r}"
                )
        for field_name, value in self.as_dict().items():
            if not value or value == "NOT_PROVIDED":
                raise ExplanationRuntimeError(f"explanation engine identity {field_name} is not frozen")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SharedInferenceIdentity:
        aliases = {
            "engine": "engine_id",
            "engine_identity": "engine_id",
            "engine_version_id": "engine_version",
            "generation_protocol_id": "protocol_id",
            "generation_protocol_version": "protocol_version",
            "batch_policy": "batch_policy_id",
            "batch_policy_version_id": "batch_policy_version",
            "environment": "environment_identity",
        }
        normalized = {aliases.get(str(key), str(key)): item for key, item in value.items()}
        known = {key: normalized[key] for key in asdict(cls()) if key in normalized}
        return cls(**known)


@dataclass(frozen=True)
class SourceCheckpointIdentity:
    """Exact approved same-seed full-Vistral checkpoint identity."""

    seed: int | str
    checkpoint_path: Path
    checkpoint_sha256: str
    source_system_id: str = SOURCE_SYSTEM_ID
    source_checkpoint_key: str = ""
    variant_fingerprint: str = ""
    model_revision: str = ""
    tokenizer_revision: str = ""
    review_summary_sha256: str = ""
    approval_sha256: str = ""
    checksum_file_sha256: str = ""
    config_sha256: str = ""
    source_run_root: Path | None = None
    dataset_identity: str = ""
    dataset_hash: str = ""

    def __post_init__(self) -> None:
        if not self.source_checkpoint_key:
            object.__setattr__(self, "source_checkpoint_key", f"{SOURCE_SYSTEM_ID}:{self.seed}")
        object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        object.__setattr__(self, "checkpoint_sha256", str(self.checkpoint_sha256).upper())
        if self.source_run_root is not None:
            object.__setattr__(self, "source_run_root", Path(self.source_run_root))
        object.__setattr__(self, "dataset_hash", str(self.dataset_hash).upper())

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def _validate_identity_fields(self, requested_seed: int | str | None = None) -> None:
        if self.source_system_id != SOURCE_SYSTEM_ID:
            raise ExplanationRuntimeError("explanation-only requires a full Vistral source checkpoint")
        if requested_seed is not None and str(self.seed) != str(requested_seed):
            raise ExplanationRuntimeError(
                f"source checkpoint seed mismatch: source={self.seed!r}, request={requested_seed!r}"
            )
        expected_key = f"{SOURCE_SYSTEM_ID}:{self.seed}"
        if self.source_checkpoint_key != expected_key:
            raise ExplanationRuntimeError(
                f"unauthorized explanation-only source key {self.source_checkpoint_key!r}; expected {expected_key!r}"
            )
        if not str(self.checkpoint_sha256):
            raise ExplanationRuntimeError("source checkpoint hash is missing")
        if not self.checkpoint_path.exists():
            raise ExplanationRuntimeError(f"source checkpoint is missing: {self.checkpoint_path}")

    def validate(
        self,
        requested_seed: int | str | None = None,
        *,
        require_approval_bindings: bool = False,
    ) -> ValidatedSourceCheckpointIdentity:
        self._validate_identity_fields(requested_seed)
        observed = sha256_file(self.checkpoint_path)
        if observed != str(self.checkpoint_sha256).upper():
            raise ExplanationRuntimeError("source checkpoint hash mismatch")
        if require_approval_bindings:
            self._validate_approval_bindings(requested_seed)
        return ValidatedSourceCheckpointIdentity(self)

    def _validate_approval_bindings(self, requested_seed: int | str | None) -> None:
        """Validate the on-disk approval chain, not just copied hash fields."""
        root = self.source_run_root
        if root is None:
            raise ExplanationRuntimeError(
                "production explanation inference requires a source resolved through resolve_explanation_source"
            )
        root = root.resolve()
        try:
            self.checkpoint_path.resolve().relative_to(root)
        except ValueError as exc:
            raise ExplanationRuntimeError("approved source checkpoint is outside its approved run root") from exc

        required = {
            "variant fingerprint": self.variant_fingerprint,
            "model revision": self.model_revision,
            "tokenizer revision": self.tokenizer_revision,
            "review-summary SHA": self.review_summary_sha256,
            "approval SHA": self.approval_sha256,
            "checksum-file SHA": self.checksum_file_sha256,
            "config SHA": self.config_sha256,
            "dataset identity": self.dataset_identity,
            "dataset SHA": self.dataset_hash,
        }
        for label, value in required.items():
            if not str(value).strip():
                raise ExplanationRuntimeError(f"approved source {label} is missing")
        for label, value in (
            ("review-summary SHA", self.review_summary_sha256),
            ("approval SHA", self.approval_sha256),
            ("checksum-file SHA", self.checksum_file_sha256),
            ("checkpoint SHA", self.checkpoint_sha256),
            ("config SHA", self.config_sha256),
            ("dataset SHA", self.dataset_hash),
        ):
            if not re.fullmatch(r"[0-9A-Fa-f]{64}", str(value)):
                raise ExplanationRuntimeError(f"approved source {label} is not a SHA-256 digest")

        summary_path = root / "review_summary.json"
        approval_path = root / "approval_status.json"
        checksums_path = root / "checksums.sha256"
        manifest_path = root / "checkpoints/checkpoint_manifest.json"
        config_path = root / "config_snapshot.yaml"
        if not all(path.exists() for path in (summary_path, approval_path, checksums_path, manifest_path, config_path)):
            raise ExplanationRuntimeError("approved source approval bindings are incomplete")
        try:
            summary = _read_json(summary_path)
            _read_json(approval_path)
            manifest = _read_json(manifest_path)
        except ExplanationRuntimeError as exc:
            raise ExplanationRuntimeError("approved source approval bindings are unreadable") from exc
        approval_errors = validate_approval_record(root, expected_run_id=root.name)
        if approval_errors:
            raise ExplanationRuntimeError("approved source approval record is incomplete: " + "; ".join(approval_errors))
        if str(summary.get("system_id")) != SOURCE_SYSTEM_ID or str(summary.get("seed")) != str(self.seed):
            raise ExplanationRuntimeError("approved source summary system or seed binding mismatch")
        expected_key = f"{SOURCE_SYSTEM_ID}:{self.seed}"
        if str(summary.get("reusable_checkpoint_key") or summary.get("source_checkpoint_id")) != expected_key:
            raise ExplanationRuntimeError("approved source summary checkpoint key binding mismatch")
        if sha256_file(summary_path) != self.review_summary_sha256.upper():
            raise ExplanationRuntimeError("approved source review-summary SHA mismatch")
        if sha256_file(approval_path) != self.approval_sha256.upper():
            raise ExplanationRuntimeError("approved source approval SHA mismatch")
        if sha256_file(checksums_path) != self.checksum_file_sha256.upper():
            raise ExplanationRuntimeError("approved source checksum-file SHA mismatch")
        if sha256_file(config_path) != self.config_sha256.upper():
            raise ExplanationRuntimeError("approved source config SHA mismatch")
        if str(manifest.get("checkpoint_sha256", "")).upper() != self.checkpoint_sha256.upper():
            raise ExplanationRuntimeError("approved source checkpoint manifest SHA mismatch")
        manifest_variant = str(manifest.get("variant_fingerprint") or summary.get("variant_fingerprint") or "")
        if manifest_variant != self.variant_fingerprint:
            raise ExplanationRuntimeError("approved source variant binding mismatch")
        for field_name, summary_key in (("model_revision", "model_revision"), ("tokenizer_revision", "tokenizer_revision")):
            observed = str(summary.get(summary_key) or manifest.get(summary_key) or "")
            if observed != str(getattr(self, field_name)):
                raise ExplanationRuntimeError(f"approved source {field_name} binding mismatch")
        observed_dataset = _dataset_binding(summary, root / "run_manifest.json")
        if observed_dataset["identity"] != self.dataset_identity or observed_dataset["hash"].upper() != self.dataset_hash.upper():
            raise ExplanationRuntimeError("approved source dataset identity binding mismatch")

    @classmethod
    def from_approved_source(cls, source: ApprovedFullVistralSource) -> SourceCheckpointIdentity:
        dataset = _dataset_binding_from_run_root(source.run_root)
        return cls(
            seed=source.seed,
            checkpoint_path=source.checkpoint_path,
            checkpoint_sha256=source.checkpoint_sha256,
            variant_fingerprint=source.variant_fingerprint,
            model_revision=source.model_revision,
            tokenizer_revision=source.tokenizer_revision,
            review_summary_sha256=source.review_summary_sha256,
            approval_sha256=source.approval_sha256,
            checksum_file_sha256=source.checksum_file_sha256,
            config_sha256=source.config_sha256,
            source_run_root=source.run_root,
            dataset_identity=dataset["identity"],
            dataset_hash=dataset["hash"],
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SourceCheckpointIdentity:
        nested = value.get("source_checkpoint")
        source = nested if isinstance(nested, Mapping) else value
        path = source.get("checkpoint_path", source.get("path"))
        if path is None:
            raise ExplanationRuntimeError("source checkpoint path is missing")
        return cls(
            seed=source.get("seed", ""),
            checkpoint_path=Path(str(path)),
            checkpoint_sha256=str(source.get("checkpoint_sha256", source.get("sha256", ""))),
            source_system_id=str(source.get("source_system_id", source.get("system_id", SOURCE_SYSTEM_ID))),
            source_checkpoint_key=str(source.get("source_checkpoint_key", source.get("checkpoint_key", ""))),
            variant_fingerprint=str(source.get("variant_fingerprint", "")),
            model_revision=str(source.get("model_revision", "")),
            tokenizer_revision=str(source.get("tokenizer_revision", "")),
            review_summary_sha256=str(source.get("review_summary_sha256", "")),
            approval_sha256=str(source.get("approval_sha256", "")),
            checksum_file_sha256=str(source.get("checksum_file_sha256", "")),
            config_sha256=str(source.get("config_sha256", "")),
            source_run_root=source.get("source_run_root"),
            dataset_identity=str(source.get("dataset_identity", "")),
            dataset_hash=str(source.get("dataset_hash", source.get("dataset_sha256", ""))),
        )


@dataclass(frozen=True)
class ValidatedSourceCheckpointIdentity:
    """Immutable source boundary after the exact checkpoint was verified once."""

    identity: SourceCheckpointIdentity

    def __getattr__(self, name: str) -> Any:
        return getattr(self.identity, name)

    def as_dict(self) -> dict[str, Any]:
        return self.identity.as_dict()

    def validate(
        self,
        requested_seed: int | str | None = None,
        *,
        require_approval_bindings: bool = False,
    ) -> ValidatedSourceCheckpointIdentity:
        # Re-check immutable metadata and the exact path, but deliberately do
        # not hash the large checkpoint again.  Approval files remain small
        # and are independently checked when production bindings are required.
        self.identity._validate_identity_fields(requested_seed)
        if require_approval_bindings:
            self.identity._validate_approval_bindings(requested_seed)
        return self


@dataclass(frozen=True)
class ExplanationOnlyConfig:
    """Explicit immutable inference contract; there is no training field."""

    identity: SharedInferenceIdentity = field(default_factory=SharedInferenceIdentity)
    system_id: str = EXPLANATION_SYSTEM_ID
    execution_kind: str = "checkpoint_reuse"
    source_system_id: str = SOURCE_SYSTEM_ID
    inference_output_source: str = "judge_of_rationale_decoder_output"
    decoder_max_tokens: int = 160
    generation_profile: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        provenance = expected_inference_provenance(self.system_id, execution_kind=self.execution_kind)
        return {
            "identity": self.identity.as_dict(),
            "system_id": self.system_id,
            "execution_kind": self.execution_kind,
            "source_system_id": self.source_system_id,
            "additional_training": False,
            "optimizer_created": False,
            "scheduler_created": False,
            "inference_output_source": self.inference_output_source,
            "provenance": provenance,
            "decoder_max_tokens": self.decoder_max_tokens,
            "generation_profile": _jsonable(self.generation_profile),
        }

    def validate(self) -> None:
        self.identity.validate()
        if self.system_id != EXPLANATION_SYSTEM_ID or self.execution_kind != "checkpoint_reuse":
            raise ExplanationRuntimeError("request is not an explanation-only checkpoint-reuse request")
        if self.source_system_id != SOURCE_SYSTEM_ID:
            raise ExplanationRuntimeError("explanation-only source system is not full Vistral")
        if self.inference_output_source != "judge_of_rationale_decoder_output":
            raise ExplanationRuntimeError("explanation-only cannot use a classification or causal-generation output")
        if int(self.decoder_max_tokens) < 2:
            raise ExplanationRuntimeError("decoder_max_tokens must allow BOS and EOS")
        provenance = {
            "system_id": self.system_id,
            "mode": "full",
            **expected_inference_provenance(self.system_id, execution_kind=self.execution_kind),
        }
        errors = validate_inference_provenance(provenance, source="explanation runtime")
        if errors:
            raise ExplanationRuntimeError("invalid explanation inference provenance: " + "; ".join(errors))


@dataclass(frozen=True)
class ExplanationOnlyRequest:
    """One seed's fully bound, inference-only request."""

    seed: int | str
    source_checkpoint: SourceCheckpointIdentity | ValidatedSourceCheckpointIdentity | ApprovedFullVistralSource | Mapping[str, Any]
    config: ExplanationOnlyConfig = field(default_factory=ExplanationOnlyConfig)
    data_hash: str = ""
    dataset_identity: str = ""
    batch_size: int | None = None
    artifact_root: Path | None = None
    legacy_artifact_root: Path | None = None
    fixture_mode: bool = False

    def __post_init__(self) -> None:
        source = self.source_checkpoint
        if isinstance(source, ApprovedFullVistralSource):
            source = SourceCheckpointIdentity.from_approved_source(source)
        elif isinstance(source, Mapping):
            source = SourceCheckpointIdentity.from_mapping(source)
        if not isinstance(source, (SourceCheckpointIdentity, ValidatedSourceCheckpointIdentity)):
            raise ExplanationRuntimeError("invalid explanation source checkpoint identity")
        # Verify the exact large checkpoint at construction.  Production
        # approval files are checked later by request/runtime validation, but
        # the verified checkpoint boundary is retained across that step.
        verified = source.validate(self.seed, require_approval_bindings=False)
        object.__setattr__(self, "source_checkpoint", verified)

    def normalized_source(self) -> ValidatedSourceCheckpointIdentity:
        source = self.source_checkpoint
        if isinstance(source, ApprovedFullVistralSource):
            source = SourceCheckpointIdentity.from_approved_source(source)
        elif isinstance(source, Mapping):
            source = SourceCheckpointIdentity.from_mapping(source)
        if isinstance(source, SourceCheckpointIdentity):
            source = source.validate(self.seed, require_approval_bindings=not self.fixture_mode)
        elif isinstance(source, ValidatedSourceCheckpointIdentity):
            source.validate(self.seed, require_approval_bindings=not self.fixture_mode)
        else:
            raise ExplanationRuntimeError("invalid explanation source checkpoint identity")
        return source

    def validate(self, run_root: str | Path | None = None) -> None:
        self.config.validate()
        source = self.normalized_source()
        if not is_real_dataset_hash(self.data_hash) and not self.fixture_mode:
            raise ExplanationRuntimeError("production explanation inference requires a canonical SHA-256 dataset hash")
        if not self.fixture_mode:
            if not self.dataset_identity.strip():
                raise ExplanationRuntimeError("production explanation inference requires a dataset identity")
            if source.dataset_hash.upper() != self.data_hash.strip().upper():
                raise ExplanationRuntimeError("explanation data_hash does not match the approved source dataset hash")
            if source.dataset_identity != self.dataset_identity:
                raise ExplanationRuntimeError("explanation dataset identity does not match the approved source dataset identity")
        if self.batch_size is not None and int(self.batch_size) not in SUPPORTED_GENERATION_BATCH_SIZES:
            raise ExplanationRuntimeError(f"batch size must be one of {SUPPORTED_GENERATION_BATCH_SIZES}")
        artifact = Path(self.artifact_root) if self.artifact_root is not None else Path(run_root or ".")
        legacy = Path(self.legacy_artifact_root) if self.legacy_artifact_root is not None else None
        if legacy is not None and artifact.resolve() == legacy.resolve():
            raise ExplanationRuntimeError("legacy artifacts must use a separate artifact root")
        if source.source_system_id != self.config.source_system_id:
            raise ExplanationRuntimeError("source checkpoint and explanation config identify different systems")

    def as_dict(self, run_root: str | Path | None = None) -> dict[str, Any]:
        self.validate(run_root)
        source = self.normalized_source()
        artifact = Path(self.artifact_root) if self.artifact_root is not None else Path(run_root or ".")
        return {
            "seed": self.seed,
            "source_checkpoint": source.as_dict(),
            "config": self.config.as_dict(),
            "data_hash": self.data_hash,
            "dataset_identity": self.dataset_identity,
            "batch_size": self.batch_size,
            "artifact_root": str(artifact),
            "legacy_artifact_root": str(self.legacy_artifact_root) if self.legacy_artifact_root else None,
            "fixture_mode": self.fixture_mode,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.as_dict())


@dataclass(frozen=True)
class ExplanationOnlyState:
    """Small auditable state sidecar; chunk rows remain in GenerationChunkStore."""

    contract_version: int
    request_fingerprint: str
    engine_fingerprint: str
    source_checkpoint_sha256: str
    split: str
    sample_ids: tuple[str, ...]
    committed_sample_ids: tuple[str, ...] = ()
    complete: bool = False
    finalized: bool = False

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExplanationRuntimeError(f"invalid explanation runtime contract: {path}") from exc
    if not isinstance(value, dict):
        raise ExplanationRuntimeError(f"explanation runtime contract must be an object: {path}")
    return value


def _write_idempotent_text(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ExplanationRuntimeError(f"attempted to rewrite finalized explanation artifact: {path}")
    if not path.exists():
        atomic_write_text(path, text)


class ExplanationOnlyRuntime:
    """Rationale-decoder-only execution with committed, resumable chunks."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        request: ExplanationOnlyRequest,
        *,
        run_root: str | Path,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.request = request
        self.run_root = Path(run_root)
        self.request.validate(self.run_root)
        self.source = request.normalized_source()
        if getattr(model, "rationale_decoder", None) is None or not callable(getattr(model.rationale_decoder, "greedy_decode", None)):
            raise ExplanationRuntimeError("explanation-only requires the source model rationale decoder")
        self.device = resolve_model_input_device(model)
        if request.fixture_mode and self.device.type != "cpu":
            raise ExplanationRuntimeError("fixture explanation mode requires a CPU model")
        self.artifact_root = Path(request.artifact_root) if request.artifact_root is not None else self.run_root
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.contract_path = self.artifact_root / "explanation_runtime_contract.json"
        self.state_path = self.artifact_root / "explanation_runtime_state.json"
        self.engine_identity = request.config.identity
        self.selected_batch_size = select_generation_batch_size(
            request.config.generation_profile,
            requested=request.batch_size,
        )
        self._device_report_written = False
        self._ensure_contract()

    @property
    def engine_fingerprint(self) -> str:
        return self.engine_identity.fingerprint

    def _ensure_contract(self) -> None:
        if self.contract_path.exists():
            observed = _read_json(self.contract_path)
            expected = self._contract()
            if not _same_json(observed.get("engine_identity"), expected["engine_identity"]):
                raise ExplanationRuntimeError("explanation engine identity mismatch")
            if not _same_json(observed.get("source_checkpoint"), expected["source_checkpoint"]):
                raise ExplanationRuntimeError("explanation source checkpoint identity mismatch")
            if str(observed.get("request_fingerprint")) != str(expected["request_fingerprint"]):
                raise ExplanationRuntimeError("explanation runtime request identity mismatch")
        elif (self.artifact_root / "reasoning").exists():
            # A chunk manifest without our contract may belong to a legacy
            # executor.  Do not reinterpret or overwrite it.
            manifests = list((self.artifact_root / "reasoning").glob("*_chunks_manifest.json"))
            if manifests:
                raise ExplanationRuntimeError("existing reasoning chunks lack an explanation-only runtime contract")
        if not self.contract_path.exists():
            atomic_write_json(self.contract_path, self._contract())

    def _contract(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "system_id": EXPLANATION_SYSTEM_ID,
            "execution_kind": "checkpoint_reuse",
            "request_fingerprint": self.request.fingerprint,
            "engine_identity": self.engine_identity.as_dict(),
            "engine_fingerprint": self.engine_fingerprint,
            "source_checkpoint": self.source.as_dict(),
            "inference_only": True,
            "additional_training": False,
            "optimizer_created": False,
            "scheduler_created": False,
            "legacy_artifact_root": str(self.request.legacy_artifact_root) if self.request.legacy_artifact_root else None,
            "legacy_artifacts_preserved": True,
        }

    def _write_state(self, state: ExplanationOnlyState) -> None:
        atomic_write_json(self.state_path, state.as_dict())

    def _record_inputs(self, record: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        if "input_ids" not in record:
            raise ExplanationRecordError("explanation records require input_ids")
        try:
            input_ids = record["input_ids"]
            attention = record.get("attention_mask")
            input_ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            attention = torch.ones_like(input_ids) if attention is None else attention
            attention = attention if isinstance(attention, torch.Tensor) else torch.tensor(attention, dtype=torch.long)
            if attention.ndim == 1:
                attention = attention.unsqueeze(0)
            if input_ids.shape != attention.shape or input_ids.size(0) != 1:
                raise ExplanationRecordError("explanation records must contain one [1, time] input and mask")
            return input_ids.to(dtype=torch.long), attention.to(dtype=torch.long)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ExplanationRecordError("explanation record has invalid token inputs") from exc

    def _inference_batch(self, records: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not records:
            raise ExplanationRuntimeError("cannot collate an empty explanation batch")
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for record in records:
            ids, mask = self._record_inputs(record)
            active = mask.squeeze(0).bool()
            if not bool(active.any()):
                raise ExplanationRecordError("explanation records must contain an active token")
            rows.append(ids.squeeze(0)[active])
            masks.append(torch.ones(int(active.sum()), dtype=torch.long))
        width = max(int(row.numel()) for row in rows)
        pad_id = int(getattr(self.tokenizer, "pad_token_id", getattr(self.tokenizer, "eos_token_id", 0)))
        input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for index, (row, mask) in enumerate(zip(rows, masks, strict=True)):
            start = width - row.numel()
            input_ids[index, start:] = row
            attention[index, start:] = mask
        return {"input_ids": input_ids, "attention_mask": attention}

    def _prepare_device_batch(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        moved = move_batch_to_model_device(batch, self.model, device=self.device)
        if not self._device_report_written:
            report = assert_runtime_device_contract(self.model, self.device, model_family="vistral_7b", batch=moved)
            write_device_report(self.artifact_root / "runtime_device_report.json", report)
            self._device_report_written = True
        return moved

    def _decode(self, ids: Any) -> str:
        if isinstance(ids, str):
            return ids.strip()
        if not callable(getattr(self.tokenizer, "decode", None)):
            raise ExplanationRuntimeError("explanation inference requires tokenizer.decode")
        try:
            return str(self.tokenizer.decode(ids, skip_special_tokens=True)).strip()
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise ExplanationRecordError("tokenizer could not decode rationale tokens") from exc

    def _row(self, split: str, record: Mapping[str, Any], decoded: torch.Tensor) -> dict[str, Any]:
        values = decoded.detach().cpu().tolist()
        eos = getattr(self.tokenizer, "eos_token_id", 2)
        eos_position = values.index(int(eos)) if eos is not None and int(eos) in values else None
        stopped = eos_position is not None
        if eos_position is not None:
            values = values[: eos_position + 1]
        reasoning = self._decode(torch.tensor(values, dtype=torch.long))
        truncated = bool(not stopped and len(values) >= int(self.request.config.decoder_max_tokens))
        return {
            "sample_id": str(record["sample_id"]),
            "split": split,
            "generated_reasoning": reasoning,
            "raw_generation": reasoning,
            "generation_status": "PASS" if reasoning else "INVALID",
            "failure_reason": None if reasoning else "empty_rationale",
            "truncated": truncated,
            "inference_output_source": self.request.config.inference_output_source,
            "engine_identity": self.engine_identity.as_dict(),
            "engine_fingerprint": self.engine_fingerprint,
            "source_checkpoint_key": self.source.source_checkpoint_key,
            "source_checkpoint_sha256": self.source.checkpoint_sha256,
            "protocol_id": self.engine_identity.protocol_id,
            "protocol_version": self.engine_identity.protocol_version,
            "batch_policy_id": self.engine_identity.batch_policy_id,
            "batch_policy_version": self.engine_identity.batch_policy_version,
        }

    def _failure_row(self, split: str, record: Mapping[str, Any], exc: Exception) -> dict[str, Any]:
        row = self._row(split, record, torch.empty(0, dtype=torch.long))
        row.update({"generated_reasoning": "", "raw_generation": "", "generation_status": "INVALID", "failure_reason": f"{type(exc).__name__}: {exc}", "truncated": False})
        return row

    def _generate_batch(self, split: str, records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        batch = self._prepare_device_batch(self._inference_batch(records))
        encoded = self.model.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        hidden = encoded["last_hidden_state"] if isinstance(encoded, Mapping) else getattr(encoded, "last_hidden_state", None)
        if not isinstance(hidden, torch.Tensor):
            raise ExplanationRuntimeError("source backbone did not return last_hidden_state")
        bos = int(getattr(self.tokenizer, "bos_token_id", 1))
        eos = int(getattr(self.tokenizer, "eos_token_id", 2))
        decoded = self.model.rationale_decoder.greedy_decode(
            hidden,
            batch["attention_mask"],
            bos,
            eos,
            int(self.request.config.decoder_max_tokens),
        )
        if not isinstance(decoded, torch.Tensor) or decoded.ndim != 2 or decoded.size(0) != len(records):
            raise ExplanationRuntimeError("rationale decoder returned an invalid batch")
        return [self._row(split, record, decoded[index]) for index, record in enumerate(records)]

    def _validate_committed(self, store: GenerationChunkStore, sample_ids: Sequence[str], split: str) -> list[dict[str, Any]]:
        rows = store.committed_rows()
        observed_ids = [str(row.get("sample_id", "")) for row in rows]
        expected_prefix = [str(value) for value in sample_ids[: len(rows)]]
        if observed_ids != expected_prefix:
            raise ExplanationRuntimeError(f"{split} committed chunks are not in stable input order")
        if len(observed_ids) != len(set(observed_ids)):
            raise ExplanationRuntimeError(f"{split} committed chunks contain duplicate samples")
        for row in rows:
            if str(row.get("engine_fingerprint")) != self.engine_fingerprint:
                raise ExplanationRuntimeError(f"{split} chunk engine identity mismatch")
            if str(row.get("source_checkpoint_sha256")) != self.source.checkpoint_sha256:
                raise ExplanationRuntimeError(f"{split} chunk source checkpoint binding mismatch")
            if str(row.get("source_checkpoint_key")) != self.source.source_checkpoint_key:
                raise ExplanationRuntimeError(f"{split} chunk source checkpoint key mismatch")
        return rows

    def _generation_contract(self, split: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        repo_root = Path(__file__).resolve().parents[3]
        sample_ids = [str(record["sample_id"]) for record in records]
        source = self.source.as_dict()
        return {
            "contract_version": GenerationChunkStore.GENERATION_CONTRACT_VERSION,
            "source_identity": source,
            "code_identity": {
                "commit": git_commit(repo_root),
                "source_fingerprint": git_source_fingerprint(repo_root),
                "engine_fingerprint": self.engine_fingerprint,
            },
            "model_identity": {
                "system_id": source["source_system_id"],
                "model_revision": source["model_revision"],
            },
            "tokenizer_identity": {"tokenizer_revision": source["tokenizer_revision"]},
            "checkpoint_identity": {
                "checkpoint_key": source["source_checkpoint_key"],
                "checkpoint_sha256": source["checkpoint_sha256"],
            },
            "config_identity": {
                "config_sha256": source["config_sha256"],
                "request_fingerprint": self.request.fingerprint,
                "config": self.request.config.as_dict(),
            },
            "dataset_identity": {
                "identity": self.request.dataset_identity,
                "data_hash": self.request.data_hash,
            },
            "split": str(split),
            "data_hash": self.request.data_hash,
            "input_record_digest": sha256_json([_record_identity(record) for record in records]),
            "record_order_digest": sha256_json(sample_ids),
            "seed": self.request.seed,
            "system_identity": {
                "system_id": EXPLANATION_SYSTEM_ID,
                "source_system_id": SOURCE_SYSTEM_ID,
            },
            "budget": "NOT_APPLICABLE",
        }

    def _store(
        self,
        split: str,
        sample_ids: Sequence[str],
        records: Sequence[Mapping[str, Any]] | None = None,
    ) -> GenerationChunkStore:
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ExplanationRuntimeError(f"{split} records must have unique sample IDs")
        if self.request.fixture_mode:
            return GenerationChunkStore(self.artifact_root, split, sample_ids, fixture_mode=True)
        if records is None:
            raise ExplanationRuntimeError(
                "production explanation downstream validation requires the original input records"
            )
        return GenerationChunkStore(
            self.artifact_root,
            split,
            sample_ids,
            generation_contract=self._generation_contract(split, records),
        )

    def generate_reasoning_split(
        self,
        split: str,
        records: Iterable[Mapping[str, Any]],
        *,
        resume: bool = True,
        batch_size: int | None = None,
        on_committed_chunk: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], None] | None = None,
    ) -> list[dict[str, Any]]:
        rows = [dict(record) for record in records]
        sample_ids = [str(record.get("sample_id", "")) for record in rows]
        if any(not value for value in sample_ids):
            raise ExplanationRuntimeError("explanation records require sample_id")
        if len(sample_ids) != len(set(sample_ids)):
            raise ExplanationRuntimeError(f"{split} records contain duplicate sample IDs")
        store = self._store(split, sample_ids, rows)
        committed = self._validate_committed(store, sample_ids, split) if resume else []
        if not resume and committed:
            raise ExplanationRuntimeError(f"cannot discard already committed {split} chunks")
        committed_ids = {str(row["sample_id"]) for row in committed}
        pending = [record for record in rows if str(record["sample_id"]) not in committed_ids]
        selected = self.selected_batch_size if batch_size is None else select_generation_batch_size(self.request.config.generation_profile, requested=batch_size)
        generated: list[dict[str, Any]] = []
        with reversible_inference_context(self.model):
            for start in range(0, len(pending), selected):
                batch_records = pending[start : start + selected]
                try:
                    chunk = self._generate_batch(split, batch_records)
                except ExplanationRecordError:
                    chunk = []
                    for record in batch_records:
                        try:
                            chunk.extend(self._generate_batch(split, [record]))
                        except ExplanationRecordError as exc:
                            chunk.append(self._failure_row(split, record, exc))
                entry = store.commit(chunk)
                # The callback is deliberately after GenerationChunkStore.commit.
                if on_committed_chunk is not None:
                    on_committed_chunk(entry, tuple(chunk))
                generated.extend(chunk)
        ordered_by_id = {str(row["sample_id"]): row for row in committed + generated}
        ordered = [ordered_by_id[sample_id] for sample_id in sample_ids]
        final_rows = self._validate_committed(store, sample_ids, split)
        if [str(row["sample_id"]) for row in final_rows] != sample_ids:
            raise ExplanationRuntimeError(f"{split} generation has missing samples")
        store.mark_complete()
        content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered)
        _write_idempotent_text(self.artifact_root / f"reasoning/{split}_reasoning.jsonl", content)
        self._write_state(ExplanationOnlyState(CONTRACT_VERSION, self.request.fingerprint, self.engine_fingerprint, self.source.checkpoint_sha256, split, tuple(sample_ids), tuple(sample_ids), True, True))
        return ordered

    def committed_rows_for_downstream(
        self,
        split: str,
        sample_ids: Sequence[str],
        records: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return only rows already committed by the canonical chunk store."""
        normalized_ids = [str(value) for value in sample_ids]
        store = self._store(split, normalized_ids, records)
        return self._validate_committed(store, sample_ids, split)

    def train(self, *_: Any, **__: Any) -> None:
        raise ExplanationRuntimeError("explanation-only runtime has no training path")

    train_generation = train

    def create_optimizer(self, *_: Any, **__: Any) -> None:
        raise ExplanationRuntimeError("explanation-only runtime cannot create an optimizer")

    def create_scheduler(self, *_: Any, **__: Any) -> None:
        raise ExplanationRuntimeError("explanation-only runtime cannot create a scheduler")


GenerationRuntimeError = ExplanationRuntimeError
ExplanationRuntime = ExplanationOnlyRuntime
ExplanationRuntimeConfig = ExplanationOnlyConfig


def resolve_explanation_source(root: str | Path, *, seed: int | str, source_checkpoint_key: str | None = None) -> ValidatedSourceCheckpointIdentity:
    """Resolve exactly the approved same-seed source; no alternate fallback is allowed."""
    expected = f"{SOURCE_SYSTEM_ID}:{seed}"
    if source_checkpoint_key not in (None, expected):
        raise ExplanationRuntimeError(f"unauthorized source checkpoint key {source_checkpoint_key!r}; expected {expected!r}")
    try:
        source = resolve_approved_full_vistral_source(root, {"seed": seed, "source_checkpoint_id": expected})
        validation = validate_source_checkpoint(root, source)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExplanationRuntimeError(f"could not resolve approved explanation source {expected}: {exc}") from exc
    if validation.get("status") != "PASS":
        raise ExplanationRuntimeError(f"approved explanation source failed validation: {validation.get('errors')}")
    # ``validate_source_checkpoint`` has already verified the exact resolved
    # path and approval bindings.  Preserve that verified boundary so request
    # construction and runtime validation do not hash the large checkpoint
    # again.
    return ValidatedSourceCheckpointIdentity(SourceCheckpointIdentity.from_approved_source(source))


def validate_three_seed_binding(requests: Sequence[ExplanationOnlyRequest]) -> str:
    """Require all three explanation seeds to share one frozen runtime identity."""
    if len(requests) != len(EXPECTED_EXPLANATION_SEEDS):
        raise ExplanationRuntimeError("exactly three explanation-only seed requests are required")
    observed = [str(request.seed) for request in requests]
    expected = [str(seed) for seed in EXPECTED_EXPLANATION_SEEDS]
    if set(observed) != set(expected):
        raise ExplanationRuntimeError(f"explanation-only seeds must be {EXPECTED_EXPLANATION_SEEDS}")
    fingerprints = {request.config.identity.fingerprint for request in requests}
    if len(fingerprints) != 1:
        raise ExplanationRuntimeError("all explanation-only seeds must bind to one frozen engine/protocol/batch-policy")
    for request in requests:
        request.validate()
    return next(iter(fingerprints))


bind_explanation_seeds = validate_three_seed_binding


__all__ = [
    "CONTRACT_VERSION",
    "EXPECTED_EXPLANATION_SEEDS",
    "EXPLANATION_ENGINE_ID",
    "EXPLANATION_ENGINE_VERSION",
    "EXPLANATION_SYSTEM_ID",
    "ExplanationContractError",
    "ExplanationOnlyConfig",
    "ExplanationOnlyRequest",
    "ExplanationOnlyRuntime",
    "ExplanationOnlyState",
    "ExplanationRuntime",
    "ExplanationRuntimeConfig",
    "ExplanationRuntimeError",
    "SharedInferenceIdentity",
    "SourceCheckpointIdentity",
    "ValidatedSourceCheckpointIdentity",
    "bind_explanation_seeds",
    "resolve_explanation_source",
    "validate_three_seed_binding",
]
