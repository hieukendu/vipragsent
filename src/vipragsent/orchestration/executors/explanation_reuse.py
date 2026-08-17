from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ...atomic import atomic_write_json, atomic_write_text
from ...evaluation.reasoning_judge import (
    ReasoningJudge,
    build_reasoning_prediction_row,
    compute_reasoning_metrics,
)
from ...hashing import sha256_file
from ...runtime.device import (
    assert_runtime_device_contract,
    move_batch_to_model_device,
    resolve_model_input_device,
    write_device_report,
)

FULL_VISTRAL_SYSTEM_ID = "vipragsent_full_vistral"
SOURCE_RECEIPT_VERSION = 1
SOURCE_RECEIPT_PATH = Path("source/validated_source_identity.json")
_SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")


class SourceReceiptError(RuntimeError):
    """A validated explanation source receipt is missing or no longer true."""


def _receipt_path(root_or_path: str | Path) -> Path:
    candidate = Path(root_or_path)
    return candidate if candidate.name == SOURCE_RECEIPT_PATH.name else candidate / SOURCE_RECEIPT_PATH


def _source_value(source: Any, name: str, default: Any = "") -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _source_checkpoint_key(source: Any) -> str:
    explicit = str(_source_value(source, "source_checkpoint_key", "") or "")
    if explicit:
        return explicit
    seed = _source_value(source, "seed", "")
    return f"{FULL_VISTRAL_SYSTEM_ID}:{seed}"


def _required_text(value: Any, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise SourceReceiptError(f"validated source identity is missing {field_name}")
    return normalized


def _is_placeholder(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    return not normalized or normalized in {"none", "null", "not_provided", "not provided", "not-provided", "fixture", "synthetic"} or normalized.startswith(("fixture-", "synthetic-", "placeholder"))


def _checkpoint_stat(path: str | Path) -> tuple[Path, dict[str, int]]:
    candidate = Path(path)
    try:
        resolved = candidate.resolve(strict=True)
        stat = resolved.stat()
    except (OSError, RuntimeError) as exc:
        raise SourceReceiptError(f"validated source checkpoint is unavailable: {candidate}") from exc
    if not resolved.is_file():
        raise SourceReceiptError(f"validated source checkpoint is not a file: {resolved}")
    return resolved, {
        "filesystem_device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _same_checkpoint_signature(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        int(left.get(field, -1)) == int(right.get(field, -2))
        for field in ("filesystem_device", "inode", "size", "mtime_ns")
    )


@dataclass(frozen=True)
class ValidatedSourceIdentity:
    """The immutable source receipt consumed by explanation-only artifacts."""

    receipt_version: int
    checkpoint_path: str
    checkpoint_sha256: str
    device: str
    filesystem_device: int
    inode: int
    size: int
    mtime_ns: int
    run_id: str
    seed: int | str
    source_system_id: str
    source_checkpoint_key: str
    model_revision: str
    tokenizer_revision: str
    config_sha256: str
    dataset_identity: str
    data_hash: str
    approval_sha256: str
    checksum_file_sha256: str
    review_summary_sha256: str
    variant_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": self.receipt_version,
            "status": "PASS",
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "filesystem_device": self.filesystem_device,
            "inode": self.inode,
            "size": self.size,
            "mtime_ns": self.mtime_ns,
            "run_id": self.run_id,
            "seed": self.seed,
            "source_system_id": self.source_system_id,
            "source_checkpoint_key": self.source_checkpoint_key,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "config_sha256": self.config_sha256,
            "dataset_identity": self.dataset_identity,
            "data_hash": self.data_hash,
            "approval_sha256": self.approval_sha256,
            "checksum_file_sha256": self.checksum_file_sha256,
            "review_summary_sha256": self.review_summary_sha256,
            "variant_fingerprint": self.variant_fingerprint,
            "checkpoint_signature": {
                "checkpoint_path": self.checkpoint_path,
                "checkpoint_sha256": self.checkpoint_sha256,
                "filesystem_device": self.filesystem_device,
                "inode": self.inode,
                "size": self.size,
                "mtime_ns": self.mtime_ns,
            },
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ValidatedSourceIdentity:
        if value.get("status") != "PASS":
            raise SourceReceiptError("validated source receipt is not marked PASS")
        checkpoint = value.get("checkpoint_signature")
        checkpoint = checkpoint if isinstance(checkpoint, Mapping) else {}
        source = value.get("source")
        source = source if isinstance(source, Mapping) else {}

        def pick(name: str, *aliases: str, default: Any = "") -> Any:
            for candidate in (name, *aliases):
                if candidate in value:
                    return value[candidate]
                if candidate in checkpoint:
                    return checkpoint[candidate]
                if candidate in source:
                    return source[candidate]
            return default

        try:
            receipt = cls(
                receipt_version=int(value.get("receipt_version", value.get("schema_version", 0))),
                checkpoint_path=str(pick("checkpoint_path", "path")),
                checkpoint_sha256=str(pick("checkpoint_sha256", "sha256")).upper(),
                device=str(pick("device")),
                filesystem_device=int(pick("filesystem_device", "st_dev", default=-1)),
                inode=int(pick("inode", "st_ino", default=-1)),
                size=int(pick("size", "st_size", default=-1)),
                mtime_ns=int(pick("mtime_ns", "st_mtime_ns", default=-1)),
                run_id=str(pick("run_id")),
                seed=pick("seed"),
                source_system_id=str(pick("source_system_id", "system_id", default=FULL_VISTRAL_SYSTEM_ID)),
                source_checkpoint_key=str(pick("source_checkpoint_key", "source_checkpoint_id", "checkpoint_key")),
                model_revision=str(pick("model_revision")),
                tokenizer_revision=str(pick("tokenizer_revision")),
                config_sha256=str(pick("config_sha256", "config_hash", "configuration_hash")),
                dataset_identity=str(pick("dataset_identity", "dataset")),
                data_hash=str(pick("data_hash", "dataset_hash", "data_fingerprint", "dataset_fingerprint")),
                approval_sha256=str(pick("approval_sha256", "source_approval_sha256")),
                checksum_file_sha256=str(pick("checksum_file_sha256", "artifact_checksum_file_sha256")),
                review_summary_sha256=str(pick("review_summary_sha256")),
                variant_fingerprint=str(pick("variant_fingerprint", "variant")),
            )
        except (TypeError, ValueError) as exc:
            raise SourceReceiptError("validated source identity has invalid field types") from exc
        if receipt.receipt_version != SOURCE_RECEIPT_VERSION:
            raise SourceReceiptError(f"unsupported validated source receipt version: {receipt.receipt_version}")
        for field_name in (
            "checkpoint_path",
            "checkpoint_sha256",
            "device",
            "run_id",
            "source_system_id",
            "source_checkpoint_key",
            "model_revision",
            "tokenizer_revision",
            "config_sha256",
            "dataset_identity",
            "data_hash",
            "approval_sha256",
            "checksum_file_sha256",
            "review_summary_sha256",
            "variant_fingerprint",
        ):
            _required_text(getattr(receipt, field_name), field_name)
        if not _SHA256_RE.fullmatch(receipt.checkpoint_sha256):
            raise SourceReceiptError("validated source identity has an invalid checkpoint SHA256")
        if not Path(receipt.checkpoint_path).is_absolute():
            raise SourceReceiptError("validated source checkpoint path is not absolute")
        if receipt.source_system_id != FULL_VISTRAL_SYSTEM_ID:
            raise SourceReceiptError("validated source identity is not a full Vistral source")
        _required_text(receipt.seed, "seed")
        if receipt.source_checkpoint_key != f"{FULL_VISTRAL_SYSTEM_ID}:{receipt.seed}":
            raise SourceReceiptError("validated source identity has an unauthorized checkpoint key")
        if receipt.filesystem_device < 0 or receipt.inode < 0 or receipt.size < 0 or receipt.mtime_ns < 0:
            raise SourceReceiptError("validated source identity has an invalid checkpoint signature")
        if _is_placeholder(receipt.data_hash) or _is_placeholder(receipt.dataset_identity):
            raise SourceReceiptError("validated source identity has a placeholder dataset identity")
        return receipt


ValidatedSourceReceipt = ValidatedSourceIdentity


def _load_receipt(path: Path) -> ValidatedSourceIdentity:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SourceReceiptError(f"invalid validated source receipt: {path}") from exc
    if not isinstance(raw, Mapping):
        raise SourceReceiptError(f"validated source receipt must be an object: {path}")
    return ValidatedSourceIdentity.from_mapping(raw)


def _compare_source_identity(receipt: ValidatedSourceIdentity, source: Any) -> None:
    expected_path, _ = _checkpoint_stat(_source_value(source, "checkpoint_path", ""))
    if expected_path != Path(receipt.checkpoint_path):
        raise SourceReceiptError("validated source checkpoint path mismatch")
    expected = {
        "checkpoint_sha256": str(_source_value(source, "checkpoint_sha256", "")).upper(),
        "run_id": str(_source_value(source, "run_id", "")),
        "seed": _source_value(source, "seed", ""),
        "source_checkpoint_key": _source_checkpoint_key(source),
        "model_revision": str(_source_value(source, "model_revision", "")),
        "tokenizer_revision": str(_source_value(source, "tokenizer_revision", "")),
        "config_sha256": str(_source_value(source, "config_sha256", "")),
        "approval_sha256": str(_source_value(source, "approval_sha256", "")),
        "checksum_file_sha256": str(_source_value(source, "checksum_file_sha256", "")),
        "review_summary_sha256": str(_source_value(source, "review_summary_sha256", "")),
        "variant_fingerprint": str(_source_value(source, "variant_fingerprint", "")),
    }
    for field_name, expected_value in expected.items():
        if expected_value and str(getattr(receipt, field_name)) != str(expected_value):
            raise SourceReceiptError(f"validated source identity mismatch: {field_name}")


def validate_validated_source_identity(
    root_or_path: str | Path | Mapping[str, Any],
    *,
    source: Any | None = None,
    expected_device: str | None = None,
    expected_dataset_identity: str | None = None,
    expected_data_hash: str | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Read a receipt and validate its physical checkpoint signature without hashing."""
    if isinstance(root_or_path, Mapping):
        receipt = ValidatedSourceIdentity.from_mapping(root_or_path)
    else:
        path = _receipt_path(root_or_path)
        receipt = _load_receipt(path)
    checkpoint_path, signature = _checkpoint_stat(receipt.checkpoint_path)
    if checkpoint_path.as_posix() != Path(receipt.checkpoint_path).as_posix() or not _same_checkpoint_signature(signature, receipt.as_dict()):
        raise SourceReceiptError("validated source checkpoint was changed or replaced since source verification")
    if source is not None:
        _compare_source_identity(receipt, source)
    if expected_device is not None and receipt.device != str(expected_device):
        raise SourceReceiptError("validated source device mismatch")
    if expected_dataset_identity is not None and receipt.dataset_identity != str(expected_dataset_identity):
        raise SourceReceiptError("validated source dataset identity mismatch")
    if expected_data_hash is not None and receipt.data_hash != str(expected_data_hash):
        raise SourceReceiptError("validated source data hash mismatch")
    if expected_checkpoint_sha256 is not None and receipt.checkpoint_sha256 != str(expected_checkpoint_sha256).upper():
        raise SourceReceiptError("validated source checkpoint hash binding mismatch")
    return receipt.as_dict()


def build_validated_source_identity(
    source: Any,
    *,
    device: str,
    dataset_identity: str | None = None,
    data_hash: str | None = None,
    checkpoint_signature: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a receipt from a hash already obtained by source verification."""
    if not bool(_source_value(source, "checkpoint_verified", False)):
        raise SourceReceiptError("validated source receipt requires source verification")
    checkpoint_path, observed_signature = _checkpoint_stat(_source_value(source, "checkpoint_path", ""))
    recorded_signature = _source_value(source, "checkpoint_stat", None)
    if isinstance(recorded_signature, Mapping) and not _same_checkpoint_signature(observed_signature, recorded_signature):
        raise SourceReceiptError("validated source checkpoint was changed or replaced after source verification")
    if checkpoint_signature is not None and not _same_checkpoint_signature(observed_signature, checkpoint_signature):
        raise SourceReceiptError("validated source checkpoint was changed or replaced after source verification")
    checkpoint_sha256 = str(_source_value(source, "checkpoint_sha256", "")).upper()
    if not _SHA256_RE.fullmatch(checkpoint_sha256):
        raise SourceReceiptError("validated source checkpoint SHA256 is unavailable")
    resolved_data_hash = str(data_hash or _source_value(source, "data_hash", "")).strip()
    resolved_dataset_identity = str(dataset_identity or _source_value(source, "dataset_identity", "") or resolved_data_hash).strip()
    resolved_data_hash = resolved_data_hash or resolved_dataset_identity
    identity = ValidatedSourceIdentity(
        receipt_version=SOURCE_RECEIPT_VERSION,
        checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha256,
        device=_required_text(device, "device"),
        filesystem_device=observed_signature["filesystem_device"],
        inode=observed_signature["inode"],
        size=observed_signature["size"],
        mtime_ns=observed_signature["mtime_ns"],
        run_id=_required_text(_source_value(source, "run_id", ""), "run_id"),
        seed=_source_value(source, "seed", ""),
        source_system_id=_required_text(_source_value(source, "source_system_id", FULL_VISTRAL_SYSTEM_ID), "source_system_id"),
        source_checkpoint_key=_required_text(_source_checkpoint_key(source), "source_checkpoint_key"),
        model_revision=_required_text(_source_value(source, "model_revision", ""), "model_revision"),
        tokenizer_revision=_required_text(_source_value(source, "tokenizer_revision", ""), "tokenizer_revision"),
        config_sha256=_required_text(_source_value(source, "config_sha256", ""), "config_sha256"),
        dataset_identity=_required_text(resolved_dataset_identity, "dataset_identity"),
        data_hash=_required_text(resolved_data_hash, "data_hash"),
        approval_sha256=_required_text(_source_value(source, "approval_sha256", ""), "approval_sha256"),
        checksum_file_sha256=_required_text(_source_value(source, "checksum_file_sha256", ""), "checksum_file_sha256"),
        review_summary_sha256=_required_text(_source_value(source, "review_summary_sha256", ""), "review_summary_sha256"),
        variant_fingerprint=_required_text(_source_value(source, "variant_fingerprint", ""), "variant_fingerprint"),
    )
    if _is_placeholder(identity.dataset_identity) or _is_placeholder(identity.data_hash):
        raise SourceReceiptError("validated source receipt requires a real dataset identity")
    return identity.as_dict()


def write_validated_source_identity(
    root: str | Path,
    source: Any,
    *,
    device: str,
    dataset_identity: str | None = None,
    data_hash: str | None = None,
    checkpoint_signature: Mapping[str, Any] | None = None,
    fixture_mode: bool = False,
) -> dict[str, Any] | None:
    """Write one source receipt, refusing to rewrite an existing binding."""
    if fixture_mode:
        return None
    path = _receipt_path(root)
    if path.exists():
        return validate_validated_source_identity(
            path,
            source=source,
            expected_device=device,
            expected_dataset_identity=dataset_identity,
            expected_data_hash=data_hash,
        )
    payload = build_validated_source_identity(
        source,
        device=device,
        dataset_identity=dataset_identity,
        data_hash=data_hash,
        checkpoint_signature=checkpoint_signature,
    )
    atomic_write_json(path, payload)
    return payload


@dataclass(frozen=True)
class ApprovedFullVistralSource:
    run_id: str
    run_root: Path
    checkpoint_path: Path
    checkpoint_sha256: str
    review_summary_sha256: str
    approval_sha256: str
    checksum_file_sha256: str
    config_sha256: str
    variant_fingerprint: str
    seed: int | str
    model_revision: str
    tokenizer_revision: str
    dataset_identity: str = ""
    data_hash: str = ""
    checkpoint_stat: Mapping[str, Any] | None = None
    checkpoint_verified: bool = False

    @property
    def source_system_id(self) -> str:
        return FULL_VISTRAL_SYSTEM_ID

    @property
    def source_checkpoint_key(self) -> str:
        return f"{FULL_VISTRAL_SYSTEM_ID}:{self.seed}"

    def as_dict(self, root: Path) -> dict[str, Any]:
        try:
            checkpoint_path = str(self.checkpoint_path.relative_to(root)).replace("\\", "/")
        except ValueError:
            checkpoint_path = str(self.checkpoint_path.resolve())
        return {
            "run_id": self.run_id,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "review_summary_sha256": self.review_summary_sha256,
            "approval_sha256": self.approval_sha256,
            "checksum_file_sha256": self.checksum_file_sha256,
            "config_sha256": self.config_sha256,
            "variant_fingerprint": self.variant_fingerprint,
            "seed": self.seed,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "source_system_id": self.source_system_id,
            "source_checkpoint_key": self.source_checkpoint_key,
            "dataset_identity": self.dataset_identity,
            "data_hash": self.data_hash,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, root: str | Path | None = None) -> ApprovedFullVistralSource:
        source = value.get("source") if isinstance(value.get("source"), Mapping) else value
        source_root = Path(root) if root is not None else Path.cwd()
        raw_path = source.get("checkpoint_path", source.get("path"))
        if raw_path is None:
            raise RuntimeError("approved source mapping has no checkpoint path")
        checkpoint_path = Path(str(raw_path))
        if not checkpoint_path.is_absolute():
            checkpoint_path = source_root / checkpoint_path
        run_id = str(source.get("run_id", ""))
        if not run_id:
            raise RuntimeError("approved source mapping has no run_id")
        seed = source.get("seed", "")
        source_system_id = str(source.get("source_system_id", FULL_VISTRAL_SYSTEM_ID))
        if source_system_id != FULL_VISTRAL_SYSTEM_ID:
            raise RuntimeError("approved source mapping is not a full Vistral source")
        source_key = str(source.get("source_checkpoint_key", f"{FULL_VISTRAL_SYSTEM_ID}:{seed}"))
        if source_key != f"{FULL_VISTRAL_SYSTEM_ID}:{seed}":
            raise RuntimeError("approved source mapping has an unauthorized checkpoint key")
        return cls(
            run_id=run_id,
            run_root=source_root / "results/runs" / run_id,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=str(source.get("checkpoint_sha256", source.get("sha256", ""))).upper(),
            review_summary_sha256=str(source.get("review_summary_sha256", "")),
            approval_sha256=str(source.get("approval_sha256", "")),
            checksum_file_sha256=str(source.get("checksum_file_sha256", "")),
            config_sha256=str(source.get("config_sha256", source.get("config_hash", ""))),
            variant_fingerprint=str(source.get("variant_fingerprint", "")),
            seed=seed,
            model_revision=str(source.get("model_revision", "")),
            tokenizer_revision=str(source.get("tokenizer_revision", "")),
            dataset_identity=str(source.get("dataset_identity", "")),
            data_hash=str(source.get("data_hash", source.get("dataset_hash", ""))),
            checkpoint_stat=source.get("checkpoint_stat") if isinstance(source.get("checkpoint_stat"), Mapping) else None,
            checkpoint_verified=False,
        )


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid source artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"source artifact must be an object: {path}")
    return payload


def _approved_index(root: Path) -> list[dict[str, Any]]:
    path = root / "results/approved_run_index.json"
    if not path.exists():
        return []
    payload = _load(path)
    return [dict(item) for item in payload.get("runs", []) if isinstance(item, Mapping)]


def _checkpoint_hash_and_stat(path: Path) -> tuple[str, dict[str, int]]:
    """Hash one verified checkpoint and reject writes/replacements during hashing."""
    _, before = _checkpoint_stat(path)
    digest = sha256_file(path)
    _, after = _checkpoint_stat(path)
    if not _same_checkpoint_signature(before, after):
        raise RuntimeError(f"source checkpoint changed or was replaced while hashing: {path}")
    return digest, after


def _first_source_value(*payloads: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for payload in payloads:
        for key in keys:
            value = str(payload.get(key, "") or "").strip()
            if value:
                return value
    return ""


def resolve_approved_full_vistral_source(
    root: str | Path,
    entry: Mapping[str, Any],
    *,
    receipt_root: str | Path | None = None,
    device: str | None = None,
    dataset_identity: str | None = None,
    data_hash: str | None = None,
    fixture_mode: bool = False,
) -> ApprovedFullVistralSource:
    """Resolve one exact approved source; no substring or first-match fallback."""
    root = Path(root)
    seed = entry.get("seed")
    source_key = str(entry.get("source_checkpoint_id") or entry.get("reusable_checkpoint_key") or f"vipragsent_full_vistral:{seed}")
    expected_key = f"vipragsent_full_vistral:{seed}"
    if source_key != expected_key:
        raise RuntimeError(f"explanation-only source key must be {expected_key}, got {source_key}")
    candidates: list[Path] = []
    index = _approved_index(root)
    if index:
        for row in index:
            if str(row.get("system")) == "vipragsent_full_vistral" and str(row.get("seed")) == str(seed) and str(row.get("run_id")):
                candidates.append(root / "results/runs" / str(row["run_id"]))
    else:
        candidates = [path.parent for path in sorted((root / "results/runs").glob("*/review_summary.json"))]
    matched: list[ApprovedFullVistralSource] = []
    for run_root in candidates:
        summary_path = run_root / "review_summary.json"
        approval_path = run_root / "approval_status.json"
        state_path = run_root / "state.json"
        manifest_path = run_root / "checkpoints/checkpoint_manifest.json"
        checksums_path = run_root / "checksums.sha256"
        if not all(path.exists() for path in (summary_path, approval_path, state_path, manifest_path, checksums_path)):
            continue
        summary = _load(summary_path)
        approval = _load(approval_path)
        state = _load(state_path)
        manifest = _load(manifest_path)
        if str(summary.get("system_id")) != "vipragsent_full_vistral" or str(summary.get("seed")) != str(seed):
            continue
        if str(summary.get("reusable_checkpoint_key") or summary.get("source_checkpoint_id")) != expected_key:
            continue
        if approval.get("status") != "APPROVED" or state.get("run_status") not in {"COMPLETED_PENDING_APPROVAL", "APPROVED"}:
            continue
        summary_hash = sha256_file(summary_path)
        approval_record = approval.get("record") if isinstance(approval.get("record"), Mapping) else {}
        approval_review_sha = approval.get("review_summary_sha256") or approval_record.get("review_summary_sha256")
        approval_checksums_sha = approval.get("artifact_checksum_file_sha256") or approval_record.get("artifact_checksum_file_sha256")
        if approval_review_sha != summary_hash:
            continue
        checksum_hash = sha256_file(checksums_path)
        if approval_checksums_sha != checksum_hash:
            continue
        checkpoint_value = manifest.get("best") or manifest.get("checkpoint_path")
        if not checkpoint_value:
            continue
        checkpoint_path = run_root / str(checkpoint_value)
        if not checkpoint_path.is_file():
            continue
        try:
            checkpoint_hash, checkpoint_stat = _checkpoint_hash_and_stat(checkpoint_path)
        except (OSError, RuntimeError, SourceReceiptError):
            continue
        if str(manifest.get("checkpoint_sha256")) != checkpoint_hash:
            continue
        config_path = run_root / "config_snapshot.yaml"
        expected_config_hash = _first_source_value(
            summary,
            manifest,
            state,
            keys=("config_sha256", "config_hash", "configuration_hash"),
        )
        config_hash = sha256_file(config_path) if config_path.exists() else expected_config_hash
        if expected_config_hash and config_hash != expected_config_hash:
            continue
        variant_fingerprint = str(manifest.get("variant_fingerprint") or summary.get("variant_fingerprint") or "")
        if not variant_fingerprint:
            continue
        model_revision = str(summary.get("model_revision") or manifest.get("model_revision") or "")
        tokenizer_revision = str(summary.get("tokenizer_revision") or manifest.get("tokenizer_revision") or "")
        if entry.get("model_revision") not in (None, "", model_revision) or entry.get("tokenizer_revision") not in (None, "", tokenizer_revision):
            continue
        data_hash = _first_source_value(
            summary,
            manifest,
            state,
            keys=("data_hash", "dataset_hash", "data_fingerprint", "dataset_fingerprint"),
        )
        dataset_identity = _first_source_value(
            summary,
            manifest,
            state,
            keys=("dataset_identity", "dataset", "data_hash", "dataset_hash", "data_fingerprint", "dataset_fingerprint"),
        ) or data_hash
        matched.append(
            ApprovedFullVistralSource(
                run_id=str(run_root.name),
                run_root=run_root,
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_hash,
                review_summary_sha256=summary_hash,
                approval_sha256=sha256_file(approval_path),
                checksum_file_sha256=checksum_hash,
                config_sha256=config_hash,
                variant_fingerprint=variant_fingerprint,
                seed=seed,
                model_revision=model_revision,
                tokenizer_revision=tokenizer_revision,
                dataset_identity=dataset_identity,
                data_hash=data_hash or dataset_identity,
                checkpoint_stat=checkpoint_stat,
                checkpoint_verified=True,
            )
        )
    if len(matched) != 1:
        raise RuntimeError(f"exactly one approved full Vistral source is required for {expected_key}; found {len(matched)}")
    source = matched[0]
    if receipt_root is not None:
        if not fixture_mode and device is None:
            raise SourceReceiptError("production source receipt creation requires a runtime device")
        write_validated_source_identity(
            receipt_root,
            source,
            device=str(device),
            dataset_identity=dataset_identity or source.dataset_identity or None,
            data_hash=data_hash or source.data_hash or None,
            checkpoint_signature=source.checkpoint_stat,
            fixture_mode=fixture_mode,
        )
    return source


def validate_source_checkpoint(
    root: str | Path,
    source: ApprovedFullVistralSource,
    *,
    receipt_root: str | Path | None = None,
    device: str | None = None,
    dataset_identity: str | None = None,
    data_hash: str | None = None,
    fixture_mode: bool = False,
) -> dict[str, Any]:
    """Validate an approved source through its receipt, never by re-hashing it."""
    root = Path(root)
    manifest = _load(source.run_root / "checkpoints/checkpoint_manifest.json")
    errors: list[str] = []
    receipt: dict[str, Any] | None = None
    if receipt_root is None:
        if fixture_mode:
            if not source.checkpoint_path.exists() or sha256_file(source.checkpoint_path) != source.checkpoint_sha256:
                errors.append("source checkpoint hash mismatch")
        else:
            errors.append("validated source receipt is required")
    else:
        try:
            receipt = validate_validated_source_identity(
                receipt_root,
                source=source,
                expected_device=device,
                expected_dataset_identity=dataset_identity,
                expected_data_hash=data_hash,
            )
        except SourceReceiptError as exc:
            errors.append(str(exc))
            receipt = None
    if manifest.get("checkpoint_sha256") != source.checkpoint_sha256:
        errors.append("checkpoint manifest hash binding mismatch")
    manifest_value = manifest.get("best") or manifest.get("checkpoint_path")
    if manifest_value:
        try:
            manifest_path = (source.run_root / str(manifest_value)).resolve(strict=True)
            if manifest_path != source.checkpoint_path.resolve(strict=True):
                errors.append("checkpoint manifest path binding mismatch")
        except (OSError, RuntimeError):
            errors.append("checkpoint manifest path is unavailable")
    if not source.variant_fingerprint:
        errors.append("source variant fingerprint is missing")
    report: dict[str, Any] = {"status": "PASS" if not errors else "FAIL", "errors": errors, "source": source.as_dict(root)}
    if receipt is not None:
        report["validated_source_identity"] = receipt
    return report


def _decode(tokenizer: Any, ids: Any) -> str:
    if isinstance(ids, str):
        return ids
    if not hasattr(tokenizer, "decode"):
        raise ValueError("rationale decoder inference requires tokenizer.decode")
    return str(tokenizer.decode(ids, skip_special_tokens=True)).strip()


class ExplanationReuseExecutor:
    """Rationale-decoder-only inference over an approved full Vistral model."""

    def __init__(
        self,
        root: str | Path,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        judge: ReasoningJudge,
        run_root: str | Path,
        source: ApprovedFullVistralSource,
        fixture_mode: bool = False,
        dataset_identity: str | None = None,
        data_hash: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.model = model
        self.tokenizer = tokenizer
        self.judge = judge
        self.run_root = Path(run_root)
        self.source = source
        self.fixture_mode = bool(fixture_mode)
        self.dataset_identity = str(dataset_identity or source.dataset_identity or "")
        self.data_hash = str(data_hash or source.data_hash or "")
        if getattr(model, "rationale_decoder", None) is None:
            raise RuntimeError("approved full model does not expose a rationale decoder")
        self.device = resolve_model_input_device(model)
        self._checkpoint_sha256 = str(source.checkpoint_sha256).upper()
        self._ensure_source_receipt()
        self._device_report_written = False

    @property
    def source_checkpoint_sha256(self) -> str:
        return self._checkpoint_sha256

    def _ensure_source_receipt(self) -> dict[str, Any] | None:
        if self.fixture_mode:
            return None
        receipt_path = _receipt_path(self.run_root)
        if not receipt_path.exists():
            write_validated_source_identity(
                self.run_root,
                self.source,
                device=str(self.device),
                dataset_identity=self.dataset_identity or None,
                data_hash=self.data_hash or None,
                checkpoint_signature=self.source.checkpoint_stat,
            )
        receipt = validate_validated_source_identity(
            self.run_root,
            source=self.source,
            expected_device=str(self.device),
            expected_dataset_identity=self.dataset_identity or None,
            expected_data_hash=self.data_hash or None,
        )
        self._checkpoint_sha256 = str(receipt["checkpoint_sha256"]).upper()
        self.dataset_identity = str(receipt["dataset_identity"])
        self.data_hash = str(receipt["data_hash"])
        return receipt

    def generate_reasoning_split(self, split: str, records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        self._ensure_source_receipt()
        rows: list[dict[str, Any]] = []
        self.model.eval()
        bos = int(getattr(self.tokenizer, "bos_token_id", 1))
        eos = int(getattr(self.tokenizer, "eos_token_id", 2))
        with torch.no_grad():
            for record in records:
                input_ids = record["input_ids"] if isinstance(record["input_ids"], torch.Tensor) else torch.tensor(record["input_ids"], dtype=torch.long)
                attention = record.get("attention_mask")
                if attention is None:
                    attention = torch.ones_like(input_ids)
                elif not isinstance(attention, torch.Tensor):
                    attention = torch.tensor(attention, dtype=torch.long)
                if input_ids.ndim == 1:
                    input_ids = input_ids.unsqueeze(0)
                if attention.ndim == 1:
                    attention = attention.unsqueeze(0)
                batch = move_batch_to_model_device({"input_ids": input_ids, "attention_mask": attention}, self.model, device=self.device)
                if not self._device_report_written:
                    report = assert_runtime_device_contract(self.model, self.device, model_family="vistral_7b", batch=batch)
                    write_device_report(self.run_root / "training/device_report.json", report)
                    self._device_report_written = True
                encoded = self.model.backbone(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                decoded = self.model.rationale_decoder.greedy_decode(encoded.last_hidden_state, batch["attention_mask"], bos, eos, 160)
                reasoning = _decode(self.tokenizer, decoded[0])
                rows.append({"sample_id": str(record["sample_id"]), "split": split, "generated_reasoning": reasoning, "raw_generation": reasoning, "generation_status": "PASS" if reasoning else "INVALID", "failure_reason": None if reasoning else "empty_rationale", "truncated": bool(len(decoded[0]) >= 160 and eos not in decoded[0].tolist()), "inference_output_source": "judge_of_rationale_decoder_output", "source_checkpoint_key": self.source.source_checkpoint_key, "source_checkpoint_sha256": self.source_checkpoint_sha256})
        atomic_write_text(self.run_root / f"reasoning/{split}_reasoning.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        return rows

    def judge_and_write(self, split: str, generated: Iterable[Mapping[str, Any]], gold: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        self._ensure_source_receipt()
        predictions: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for row in generated:
            decision = self.judge.judge(str(row.get("generated_reasoning", "")))
            decisions.append({"sample_id": row["sample_id"], **decision})
            predictions.append(build_reasoning_prediction_row(str(row["sample_id"]), gold[str(row["sample_id"])], str(row.get("generated_reasoning", "")), decision, truncated=bool(row.get("truncated"))))
        self.judge.write_artifacts(self.run_root, split, predictions, decisions)
        atomic_write_text(self.run_root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        metrics = compute_reasoning_metrics(predictions, diagnostics=self.judge.diagnostics) | {"status": "PASS", "split": split, "inference_output_source": "judge_of_rationale_decoder_output", "source_checkpoint_sha256": self.source_checkpoint_sha256, "source_checkpoint_key": self.source.source_checkpoint_key}
        atomic_write_json(self.run_root / f"metrics/{split}_reasoning_metrics.json", metrics)
        return metrics

    def compute_split_metrics(self, split: str, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """Compute artifact-only metrics only after revalidating the source receipt."""
        self._ensure_source_receipt()
        metrics = compute_reasoning_metrics(rows, diagnostics=self.judge.diagnostics) | {
            "status": "PASS",
            "split": split,
            "inference_output_source": "judge_of_rationale_decoder_output",
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_checkpoint_key": self.source.source_checkpoint_key,
        }
        atomic_write_json(self.run_root / f"metrics/{split}_reasoning_metrics.json", metrics)
        return metrics

    def write_source_provenance(self) -> dict[str, Any]:
        receipt = self._ensure_source_receipt()
        provenance = {"status": "PASS", "source": self.source.as_dict(self.root), "validated_source_identity": receipt, "source_system_id": "vipragsent_full_vistral", "same_seed_source": True, "additional_training": False, "optimizer_created": False, "scheduler_created": False, "backward_calls": 0, "direct_classification_outputs_used": False, "rationale_decoder_enabled_at_inference": True, "native_causal_lm_generation_used": False, "inference_output_source": "judge_of_rationale_decoder_output"}
        atomic_write_json(self.run_root / "source/source_provenance.json", provenance)
        return provenance
