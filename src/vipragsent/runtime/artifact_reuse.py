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


IDENTITY_FIELDS = ("model", "tokenizer", "config", "data", "source", "live_code_identity")
HASH_FIELDS = tuple(f"{name}_hash" for name in IDENTITY_FIELDS)
APPROVED_FIELDS = IDENTITY_FIELDS + HASH_FIELDS
LIVE_CODE_IDENTITY_UNCERTAIN = "LIVE_CODE_IDENTITY_UNCERTAIN"


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 binding for JSON-compatible metadata."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(encoded).hexdigest()


def _is_uncertain(value: Any) -> bool:
    return value is None or value == "" or value == "unknown" or value is False


def _schema_issues(metadata: Mapping[str, Any], side: str) -> list[str]:
    issues: list[str] = []
    keys = set(metadata)
    for name in sorted(set(APPROVED_FIELDS) - keys):
        issues.append(f"MISSING_FIELD:{side}:{name}")
        if name == "live_code_identity":
            issues.append(LIVE_CODE_IDENTITY_UNCERTAIN)
        elif name == "source":
            issues.append("SOURCE_IDENTITY_ABSENT")
    for name in sorted(keys - set(APPROVED_FIELDS), key=str):
        issues.append(f"EXTRA_FIELD:{side}:{name}")
    for name in APPROVED_FIELDS:
        if name not in metadata or not _is_uncertain(metadata[name]):
            continue
        if name == "live_code_identity":
            issues.append(LIVE_CODE_IDENTITY_UNCERTAIN)
        elif name.endswith("_hash"):
            issues.append(f"{name.removesuffix('_hash').upper()}_HASH_ABSENT")
        else:
            issues.append(f"{name.upper()}_IDENTITY_ABSENT")
    return issues


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

    @property
    def evidence_bearing(self) -> bool:
        """Whether this decision contains complete proof for a VERIFY event."""
        return (
            self.reusable
            and not self.reasons
            and not self.mismatched_fields
            and self.local_metadata_hash is not None
            and self.local_metadata_hash == self.remote_metadata_hash
            and set(self.matched_fields) == set(APPROVED_FIELDS)
        )


def decide_reuse(
    local_metadata: Mapping[str, Any],
    remote_metadata: Mapping[str, Any],
    *,
    live_code_identity: str | None = None,
) -> ReuseDecision:
    """Classify a possible reuse without reading or changing anything.

    ``BLOCKED`` is returned for incomplete or non-exact metadata schemas.  A
    known disagreement between complete records is ``INVALID``.  The optional
    ``live_code_identity`` argument is only an expected-value assertion; it is
    never used to fill either record.  Both records must contain their own
    live-code field and digest.
    """
    local_hash = canonical_hash(dict(local_metadata))
    remote_hash = canonical_hash(dict(remote_metadata))
    blocked = _schema_issues(local_metadata, "local") + _schema_issues(remote_metadata, "remote")
    matched: list[str] = []
    mismatched: list[str] = []
    for name in APPROVED_FIELDS:
        if name not in local_metadata or name not in remote_metadata:
            continue
        local = local_metadata[name]
        remote = remote_metadata[name]
        if _is_uncertain(local) or _is_uncertain(remote):
            continue
        if local != remote:
            mismatched.append(name)
        else:
            matched.append(name)

    if live_code_identity is not None:
        for metadata in (local_metadata, remote_metadata):
            recorded = metadata.get("live_code_identity")
            if not _is_uncertain(recorded) and recorded != live_code_identity:
                mismatched.append("live_code_identity")

    if blocked:
        status = ReuseStatus.BLOCKED
        reasons = tuple(dict.fromkeys(blocked))
    elif mismatched:
        status = ReuseStatus.INVALID
        reasons = tuple(
            f"{'DIGEST' if name.endswith('_hash') else 'IDENTITY'}_MISMATCH:{name}"
            for name in dict.fromkeys(mismatched)
        )
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
    evidence: ReuseDecision | None = None


def transition(
    state: ReuseState,
    event: ReuseEvent | str,
    *,
    evidence: ReuseDecision | None = None,
) -> ReuseState:
    """Pure state transition; VERIFY requires complete verified evidence."""
    event = ReuseEvent(event)
    next_evidence = state.evidence
    if event is ReuseEvent.VERIFY and evidence is not None and evidence.evidence_bearing:
        next_status = ReuseStatus.VERIFIED
        next_evidence = evidence
    elif event is ReuseEvent.INVALIDATE:
        next_status = ReuseStatus.INVALID
        next_evidence = None
    elif event is ReuseEvent.BLOCK:
        next_status = ReuseStatus.BLOCKED
        next_evidence = None
    else:
        next_status = state.status
    return replace(
        state,
        status=next_status,
        history=state.history if event.value in state.history else state.history + (event.value,),
        backup_requested=state.backup_requested or event is ReuseEvent.BACKUP_REQUESTED,
        restore_requested=state.restore_requested or event is ReuseEvent.RESTORE_REQUESTED,
        evidence=next_evidence,
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
    "APPROVED_FIELDS",
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
