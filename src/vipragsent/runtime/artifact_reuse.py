"""Pure, read-only decisions for reusing a previously materialized artifact.

This module deliberately has no filesystem, network, Hugging Face, or process
mutation dependencies.  Callers provide the locally observed and remotely
recorded metadata; reuse is allowed only when the identity fields and their
hashes agree exactly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from typing import Any, Protocol


class ReuseStatus(StrEnum):
    CANDIDATE = "candidate"
    BLOCKED = "blocked"
    VERIFIED = "verified"
    INVALID = "invalid"


IDENTITY_FIELDS = ("model", "tokenizer", "config", "data", "source")
HASH_FIELDS = tuple(f"{name}_hash" for name in IDENTITY_FIELDS)
LIVE_CODE_IDENTITY_UNCERTAIN = "LIVE_CODE_IDENTITY_UNCERTAIN"


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 binding for JSON-compatible metadata."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(encoded).hexdigest()


def _value(metadata: Mapping[str, Any], name: str) -> Any:
    """Accept either ``model`` or the common ``model_identity`` spelling."""
    if name in metadata:
        return metadata[name]
    return metadata.get(f"{name}_identity")


def _is_uncertain(value: Any) -> bool:
    return value is None or value == "" or value == "unknown" or value is False


@dataclass(frozen=True)
class ReuseDecision:
    status: ReuseStatus
    reasons: tuple[str, ...] = ()
    matched_fields: tuple[str, ...] = ()
    mismatched_fields: tuple[str, ...] = ()
    local_metadata_hash: str | None = None
    remote_metadata_hash: str | None = None

    @property
    def reusable(self) -> bool:
        return self.status is ReuseStatus.VERIFIED


def decide_reuse(
    local_metadata: Mapping[str, Any],
    remote_metadata: Mapping[str, Any],
    *,
    live_code_identity: str | None = None,
) -> ReuseDecision:
    """Classify a possible reuse without reading or changing anything.

    ``BLOCKED`` is reserved for missing/uncertain provenance, especially live
    code identity and source identity.  A known disagreement is ``INVALID``.
    The caller must provide the live code identity explicitly; it is never
    inferred from local or remote artifact metadata.
    """
    local_hash = canonical_hash(dict(local_metadata))
    remote_hash = canonical_hash(dict(remote_metadata))
    if _is_uncertain(live_code_identity):
        return ReuseDecision(
            ReuseStatus.BLOCKED,
            (LIVE_CODE_IDENTITY_UNCERTAIN,),
            local_metadata_hash=local_hash,
            remote_metadata_hash=remote_hash,
        )

    matched: list[str] = []
    mismatched: list[str] = []
    blocked: list[str] = []
    for name in IDENTITY_FIELDS:
        local = _value(local_metadata, name)
        remote = _value(remote_metadata, name)
        if _is_uncertain(local) or _is_uncertain(remote):
            if name == "source":
                blocked.append("SOURCE_IDENTITY_ABSENT")
            else:
                blocked.append(f"{name.upper()}_IDENTITY_ABSENT")
        elif local != remote:
            mismatched.append(name)
        else:
            matched.append(name)

        local_hash_value = local_metadata.get(f"{name}_hash")
        remote_hash_value = remote_metadata.get(f"{name}_hash")
        if local_hash_value is not None or remote_hash_value is not None:
            if _is_uncertain(local_hash_value) or _is_uncertain(remote_hash_value):
                blocked.append(f"{name.upper()}_HASH_ABSENT")
            elif local_hash_value != remote_hash_value:
                mismatched.append(f"{name}_hash")

    local_code = local_metadata.get("live_code_identity", live_code_identity)
    remote_code = remote_metadata.get("live_code_identity", live_code_identity)
    if local_code != live_code_identity or remote_code != live_code_identity:
        mismatched.append("live_code_identity")

    if blocked:
        status = ReuseStatus.BLOCKED
        reasons = tuple(dict.fromkeys(blocked))
    elif mismatched:
        status = ReuseStatus.INVALID
        reasons = tuple(f"IDENTITY_MISMATCH:{name}" for name in dict.fromkeys(mismatched))
    else:
        status = ReuseStatus.VERIFIED
        reasons = ()
    return ReuseDecision(status, reasons, tuple(matched), tuple(dict.fromkeys(mismatched)), local_hash, remote_hash)


class ReuseEvent(StrEnum):
    OBSERVE = "observe"
    VERIFY = "verify"
    INVALIDATE = "invalidate"
    BLOCK = "block"
    BACKUP_REQUESTED = "backup_requested"
    RESTORE_REQUESTED = "restore_requested"


@dataclass(frozen=True)
class ReuseState:
    status: ReuseStatus = ReuseStatus.CANDIDATE
    history: tuple[str, ...] = ()
    backup_requested: bool = False
    restore_requested: bool = False


def transition(state: ReuseState, event: ReuseEvent | str) -> ReuseState:
    """Pure state transition; repeated events are idempotent."""
    event = ReuseEvent(event)
    if event is ReuseEvent.VERIFY:
        next_status = ReuseStatus.VERIFIED
    elif event is ReuseEvent.INVALIDATE:
        next_status = ReuseStatus.INVALID
    elif event is ReuseEvent.BLOCK:
        next_status = ReuseStatus.BLOCKED
    else:
        next_status = state.status
    return replace(
        state,
        status=next_status,
        history=state.history if event.value in state.history else state.history + (event.value,),
        backup_requested=state.backup_requested or event is ReuseEvent.BACKUP_REQUESTED,
        restore_requested=state.restore_requested or event is ReuseEvent.RESTORE_REQUESTED,
    )


@dataclass(frozen=True)
class MutationForbiddenError(RuntimeError):
    """Raised by the optional adapter boundary when a write is attempted."""


class ReadOnlyArtifactAdapter(Protocol):
    def read_metadata(self) -> Mapping[str, Any]: ...

    def backup(self) -> None: ...

    def restore(self) -> None: ...


@dataclass(frozen=True)
class ArtifactOperation:
    operation: str
    allowed: bool = False
    reason: str = "read-only reuse contract"


def plan_backup() -> ArtifactOperation:
    return ArtifactOperation("backup", False)


def plan_restore() -> ArtifactOperation:
    return ArtifactOperation("restore", False)


__all__ = [
    "ArtifactOperation",
    "IDENTITY_FIELDS",
    "LIVE_CODE_IDENTITY_UNCERTAIN",
    "MutationForbiddenError",
    "ReadOnlyArtifactAdapter",
    "ReuseDecision",
    "ReuseEvent",
    "ReuseState",
    "ReuseStatus",
    "canonical_hash",
    "decide_reuse",
    "plan_backup",
    "plan_restore",
    "transition",
]
