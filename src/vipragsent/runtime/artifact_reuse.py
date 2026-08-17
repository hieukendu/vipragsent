"""Pure, read-only exact reuse and resume classification.

The classifier in this module is intentionally metadata-only.  A checkpoint
path is never evidence of either reuse or resume: callers must provide the
complete binding and, when relevant, the run state explicitly.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any, Protocol

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - exercised only on Python 3.10
    from enum import Enum

    class StrEnum(str, Enum):  # noqa: UP042
        """Small compatibility fallback for Python versions before 3.11."""

        def __str__(self) -> str:
            return self.value


class ReuseStatus(StrEnum):
    """The only normalized outcomes of exact artifact classification.

    ``VERIFIED``, ``INVALID``, and ``CANDIDATE`` are retained as enum aliases
    for callers of the earlier read-only contract.  They deliberately resolve
    to the normalized outcomes rather than reintroducing a fourth state.
    """

    REUSE = "REUSE"
    RESUME = "RESUME"
    BLOCKED = "BLOCKED"

    # Compatibility aliases.  The aliases have the normalized values above.
    VERIFIED = "REUSE"
    INVALID = "BLOCKED"
    CANDIDATE = "BLOCKED"

    @classmethod
    def _missing_(cls, value: object) -> ReuseStatus | None:
        """Accept serialized spellings from the pre-R4 contract."""
        if isinstance(value, str):
            legacy = {
                "verified": cls.REUSE,
                "invalid": cls.BLOCKED,
                "candidate": cls.BLOCKED,
                "reuse": cls.REUSE,
                "resume": cls.RESUME,
                "blocked": cls.BLOCKED,
            }
            return legacy.get(value.lower())
        return None


# These are the logical fields, not path fragments.  Every one is required
# on both sides of a comparison before a positive outcome is possible.
EXACT_BINDING_FIELDS = (
    "system",
    "run",
    "seed",
    "checkpoint_sha",
    "config_sha",
    "dataset_sha",
    "model",
    "tokenizer",
    "code",
    "source",
    "approval",
    "checkpoint_epoch",
    "checkpoint_kind",
)
REUSE_BINDING_FIELDS = EXACT_BINDING_FIELDS

# Compatibility names from the earlier provenance-only contract.  The R4
# classifier uses EXACT_BINDING_FIELDS; these exports remain useful to older
# callers that imported the constants.
IDENTITY_FIELDS = ("model", "tokenizer", "config", "data", "source", "live_code_identity")
HASH_FIELDS = tuple(f"{name}_hash" for name in IDENTITY_FIELDS)
APPROVED_FIELDS = EXACT_BINDING_FIELDS
LIVE_CODE_IDENTITY_UNCERTAIN = "LIVE_CODE_IDENTITY_UNCERTAIN"
_SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")


# A few runtime producers use *_id, *_sha256, or *_revision spellings.  They
# are explicit metadata aliases, never values derived from a filename.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "system": ("system", "system_id"),
    "run": ("run", "run_id", "experiment_id", "job_id"),
    "seed": ("seed",),
    "checkpoint_sha": ("checkpoint_sha", "checkpoint_sha256", "checkpoint_hash"),
    "config_sha": ("config_sha", "config_sha256", "config_hash"),
    "dataset_sha": (
        "dataset_sha",
        "dataset_sha256",
        "dataset_hash",
        "data_sha",
        "data_sha256",
        "data_hash",
        "data_fingerprint",
    ),
    "model": ("model", "model_revision", "model_id", "model_identity"),
    "tokenizer": ("tokenizer", "tokenizer_revision", "tokenizer_id", "tokenizer_identity"),
    "code": (
        "code",
        "code_sha",
        "code_sha256",
        "code_commit",
        "code_hash",
        "live_code_identity",
    ),
    "source": (
        "source",
        "source_id",
        "source_run_id",
        "source_checkpoint_id",
        "source_fingerprint",
        "source_sha",
        "source_sha256",
    ),
    "approval": (
        "approval",
        "approval_id",
        "approval_sha",
        "approval_sha256",
        "approval_hash",
        "review_summary_sha256",
    ),
    "checkpoint_epoch": ("checkpoint_epoch", "checkpoint_epoch_number", "epoch", "best_epoch"),
    "checkpoint_kind": ("checkpoint_kind", "checkpoint_type", "checkpoint_role", "kind"),
}

_CONTROL_FIELDS = frozenset(
    {
        "status",
        "run_status",
        "state",
        "state_status",
        "paused",
        "is_paused",
        "in_progress",
        "is_in_progress",
        "pause",
        "lifecycle",
        "progress",
        "mode",
        "operation",
        "intent",
        "resume",
        "resume_requested",
        "checkpoint_path",
        "path",
    }
)
_LEGACY_OPTIONAL_FIELDS = frozenset(
    {
        "model_hash",
        "tokenizer_hash",
        "config_hash",
        "data_hash",
        "source_hash",
        "live_code_identity_hash",
    }
)
_ALLOWED_METADATA_FIELDS = frozenset().union(
    *(set(aliases) for aliases in _FIELD_ALIASES.values()),
    _CONTROL_FIELDS,
    _LEGACY_OPTIONAL_FIELDS,
)

_PAUSED_STATUSES = frozenset({"PAUSED", "SAFELY_PAUSED", "INTERRUPTED", "PAUSE"})
_IN_PROGRESS_STATUSES = frozenset({"RUNNING", "IN_PROGRESS", "ACTIVE", "STARTED", "RUNNING_STALE"})
_RESUME_CHECKPOINT_KINDS = frozenset({"RESUME", "RESUMABLE", "INTERRUPTED", "PAUSED"})
_TERMINAL_STATUSES = frozenset(
    {
        "APPROVED",
        "COMPLETED",
        "COMPLETED_PENDING_APPROVAL",
        "DONE",
        "FINISHED",
        "PASS",
        "REUSED",
        "REUSE",
    }
)
_BLOCKING_STATUSES = frozenset({"BLOCKED", "FAIL", "FAILED", "REJECTED", "NOT_STARTED", "PENDING"})


def canonical_hash(value: Any) -> str:
    """Return a stable SHA-256 binding for JSON-compatible metadata."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(encoded).hexdigest()


def _is_uncertain(value: Any) -> bool:
    return value is None or value == "" or value == "unknown" or value is False


def _same_value(left: Any, right: Any) -> bool:
    """Compare values exactly, including the distinction between 1 and "1"."""
    return type(left) is type(right) and left == right


def _extract_binding(metadata: Mapping[str, Any], side: str) -> tuple[dict[str, Any], list[str]]:
    binding: dict[str, Any] = {}
    issues: list[str] = []
    keys = set(metadata)

    for name in EXACT_BINDING_FIELDS:
        aliases = _FIELD_ALIASES[name]
        present = [(alias, metadata[alias]) for alias in aliases if alias in metadata]
        if not present:
            issues.append(f"MISSING_BINDING_FIELD:{side}:{name}")
            if name == "code":
                issues.append(LIVE_CODE_IDENTITY_UNCERTAIN)
            elif name == "source":
                issues.append("SOURCE_IDENTITY_ABSENT")
            continue

        known = [(alias, value) for alias, value in present if not _is_uncertain(value)]
        if not known:
            issues.append(f"ABSENT_BINDING_FIELD:{side}:{name}")
            if name == "code":
                issues.append(LIVE_CODE_IDENTITY_UNCERTAIN)
            elif name == "source":
                issues.append("SOURCE_IDENTITY_ABSENT")
            continue

        selected_alias, selected_value = known[0]
        binding[name] = selected_value
        if len(known) > 1 and any(not _same_value(selected_value, value) for _, value in known[1:]):
            issues.append(f"CONFLICTING_BINDING_ALIASES:{side}:{name}")
        # A canonical field and an alias that disagree is a conflict even if
        # the alias appears later in the mapping.
        if selected_alias != name and name in metadata and not _is_uncertain(metadata[name]) and not _same_value(metadata[name], selected_value):
            issues.append(f"CONFLICTING_BINDING_ALIASES:{side}:{name}")

    for name in sorted(keys - _ALLOWED_METADATA_FIELDS, key=str):
        issues.append(f"EXTRA_FIELD:{side}:{name}")
    return binding, list(dict.fromkeys(issues))


@dataclass
class _StateFlags:
    present: bool = False
    paused: bool = False
    in_progress: bool = False
    invalid: bool = False
    unknown: bool = False


def _normalise_state_status(value: Any) -> str:
    return str(value).strip().upper().replace("-", "_").replace(" ", "_")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on", "paused", "running", "in_progress"}
    return bool(value)


def _state_flags(payload: Any, *, _depth: int = 0) -> _StateFlags:
    flags = _StateFlags()
    if payload is None:
        return flags
    if _depth > 3:
        flags.present = True
        flags.unknown = True
        return flags

    if isinstance(payload, str):
        flags.present = True
        statuses = (_normalise_state_status(payload),)
        nested: tuple[Mapping[str, Any], ...] = ()
    elif isinstance(payload, Mapping):
        state_keys = {
            "status",
            "run_status",
            "state_status",
            "paused",
            "is_paused",
            "in_progress",
            "is_in_progress",
            "state",
            "pause",
            "lifecycle",
            "progress",
        }
        if not state_keys.intersection(payload):
            return flags
        flags.present = True
        statuses = tuple(
            _normalise_state_status(payload[key])
            for key in ("status", "run_status", "state_status")
            if key in payload and not isinstance(payload[key], Mapping) and not _is_uncertain(payload[key])
        )
        nested = tuple(
            value
            for key in ("state", "pause", "lifecycle", "progress")
            for value in (payload.get(key),)
            if isinstance(value, Mapping)
        )
        for key in ("paused", "is_paused"):
            if key in payload and _truthy(payload[key]):
                flags.paused = True
        for key in ("in_progress", "is_in_progress"):
            if key in payload and _truthy(payload[key]):
                flags.in_progress = True
    else:
        flags.unknown = True
        return flags

    for status in statuses:
        if status in _PAUSED_STATUSES:
            flags.paused = True
        elif status in _IN_PROGRESS_STATUSES:
            flags.in_progress = True
        elif status in _TERMINAL_STATUSES:
            continue
        elif status in _BLOCKING_STATUSES:
            flags.invalid = True
        else:
            flags.unknown = True

    for child in nested:
        child_flags = _state_flags(child, _depth=_depth + 1)
        flags.paused |= child_flags.paused
        flags.in_progress |= child_flags.in_progress
        flags.invalid |= child_flags.invalid
        flags.unknown |= child_flags.unknown
    return flags


def _merge_state_flags(*payloads: Any) -> _StateFlags:
    merged = _StateFlags()
    for payload in payloads:
        current = _state_flags(payload)
        merged.present |= current.present
        merged.paused |= current.paused
        merged.in_progress |= current.in_progress
        merged.invalid |= current.invalid
        merged.unknown |= current.unknown
    return merged


def _normalise_mode(value: Any) -> ReuseStatus | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return ReuseStatus.RESUME if value else ReuseStatus.REUSE
    if isinstance(value, ReuseStatus):
        if value in {ReuseStatus.REUSE, ReuseStatus.RESUME}:
            return value
        return None
    text = str(value).strip().upper()
    if text in {"REUSE", "VERIFIED"}:
        return ReuseStatus.REUSE
    if text in {"RESUME", "RESUMABLE"}:
        return ReuseStatus.RESUME
    if text in {"", "AUTO", "UNSPECIFIED"}:
        return None
    return None


@dataclass(frozen=True)
class ReuseDecision:
    status: ReuseStatus
    reasons: tuple[str, ...] = ()
    matched_fields: tuple[str, ...] = ()
    mismatched_fields: tuple[str, ...] = ()
    local_metadata_hash: str | None = None
    remote_metadata_hash: str | None = None
    local_binding_hash: str | None = None
    remote_binding_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ReuseStatus(self.status))

    @property
    def reusable(self) -> bool:
        return self.status is ReuseStatus.REUSE

    @property
    def resumable(self) -> bool:
        return self.status is ReuseStatus.RESUME

    @property
    def exact(self) -> bool:
        return (
            self.status in {ReuseStatus.REUSE, ReuseStatus.RESUME}
            and not self.reasons
            and not self.mismatched_fields
            and self.local_binding_hash is not None
            and self.local_binding_hash == self.remote_binding_hash
            and set(self.matched_fields) == set(EXACT_BINDING_FIELDS)
        )

    @property
    def evidence_bearing(self) -> bool:
        """Whether this decision contains complete proof for a VERIFY event."""
        return (
            self.reusable
            and self.exact
        )


def decide_reuse(
    local_metadata: Mapping[str, Any],
    remote_metadata: Mapping[str, Any],
    *,
    live_code_identity: str | None = None,
    mode: str | ReuseStatus | None = None,
    operation: str | ReuseStatus | None = None,
    intent: str | ReuseStatus | None = None,
    resume: bool | None = None,
    state: Mapping[str, Any] | str | None = None,
    run_state: Mapping[str, Any] | str | None = None,
    local_state: Mapping[str, Any] | str | None = None,
    remote_state: Mapping[str, Any] | str | None = None,
    paused: bool | None = None,
    in_progress: bool | None = None,
) -> ReuseDecision:
    """Classify exact reuse/resume without reading or changing anything.

    Both records must bind ``EXACT_BINDING_FIELDS``.  A mismatch, omitted
    field, conflicting alias, unknown state, or active run is ``BLOCKED``.
    With no explicit mode, a paused state yields ``RESUME`` and a quiescent
    state yields ``REUSE``.  The mode and state are explicit inputs; a
    checkpoint filename is never inspected.
    """
    local_hash = canonical_hash(dict(local_metadata))
    remote_hash = canonical_hash(dict(remote_metadata))
    local_binding, local_issues = _extract_binding(local_metadata, "local")
    remote_binding, remote_issues = _extract_binding(remote_metadata, "remote")

    matched: list[str] = []
    mismatched: list[str] = []
    for name in EXACT_BINDING_FIELDS:
        if name not in local_binding or name not in remote_binding:
            continue
        if _same_value(local_binding[name], remote_binding[name]):
            matched.append(name)
        else:
            mismatched.append(name)

    if live_code_identity is not None:
        for binding in (local_binding, remote_binding):
            if "code" in binding and not _same_value(binding["code"], live_code_identity):
                mismatched.append("code")

    requested_values = [value for value in (mode, operation, intent) if value is not None]
    if resume is not None:
        requested_values.append(resume)
    requested_modes = [_normalise_mode(value) for value in requested_values]
    mode_issues: list[str] = []
    if any(value is None for value in requested_modes):
        mode_issues.append("INVALID_MODE")
    requested_modes = [value for value in requested_modes if value is not None]
    if requested_modes and any(value is not requested_modes[0] for value in requested_modes[1:]):
        mode_issues.append("CONFLICTING_MODE")
    requested_mode = requested_modes[0] if requested_modes and not mode_issues else None

    state_flags = _merge_state_flags(
        local_metadata,
        remote_metadata,
        state,
        run_state,
        local_state,
        remote_state,
        {"paused": paused} if paused is not None else None,
        {"in_progress": in_progress} if in_progress is not None else None,
    )
    reasons = list(dict.fromkeys(local_issues + remote_issues + mode_issues))
    if state_flags.paused and state_flags.in_progress:
        reasons.append("STATE_CONFLICT:PAUSED_AND_IN_PROGRESS")
    elif state_flags.in_progress:
        reasons.append("IN_PROGRESS_STATE")
    elif state_flags.invalid:
        reasons.append("BLOCKING_RUN_STATE")
    elif state_flags.unknown:
        reasons.append("UNKNOWN_RUN_STATE")

    if mismatched:
        for name in dict.fromkeys(mismatched):
            reasons.append(f"BINDING_MISMATCH:{name}")
    reasons = list(dict.fromkeys(reasons))

    local_binding_hash = canonical_hash(local_binding)
    remote_binding_hash = canonical_hash(remote_binding)
    checkpoint_kind = local_binding.get("checkpoint_kind")
    checkpoint_kind_implies_resume = isinstance(checkpoint_kind, str) and checkpoint_kind.strip().upper() in _RESUME_CHECKPOINT_KINDS
    complete = (
        not reasons
        and not mismatched
        and set(local_binding) == set(EXACT_BINDING_FIELDS)
        and set(remote_binding) == set(EXACT_BINDING_FIELDS)
    )

    if complete:
        if requested_mode is ReuseStatus.RESUME:
            if state_flags.present and not state_flags.paused:
                reasons.append("RESUME_REQUIRES_PAUSED_STATE")
            else:
                status = ReuseStatus.RESUME
        elif requested_mode is ReuseStatus.REUSE:
            if state_flags.paused:
                reasons.append("REUSE_CONFLICT:PAUSED_STATE")
            else:
                status = ReuseStatus.REUSE
        elif state_flags.paused:
            status = ReuseStatus.RESUME
        elif checkpoint_kind_implies_resume:
            status = ReuseStatus.RESUME
        else:
            status = ReuseStatus.REUSE
    if reasons or not complete:
        status = ReuseStatus.BLOCKED
    return ReuseDecision(
        status,
        tuple(dict.fromkeys(reasons)),
        tuple(dict.fromkeys(matched)),
        tuple(dict.fromkeys(mismatched)),
        local_hash,
        remote_hash,
        local_binding_hash,
        remote_binding_hash,
    )


def classify_reuse(*args: Any, **kwargs: Any) -> ReuseDecision:
    """Named classification alias for callers that do not use ``decide_reuse``."""
    return decide_reuse(*args, **kwargs)


def classify_artifact_reuse(*args: Any, **kwargs: Any) -> ReuseDecision:
    """Compatibility alias for the runtime artifact classifier."""
    return decide_reuse(*args, **kwargs)


class ReuseEvent(StrEnum):
    OBSERVE = "observe"
    VERIFY = "verify"
    RESUME = "resume"
    INVALIDATE = "invalidate"
    BLOCK = "block"
    BACKUP_REQUESTED = "backup_requested"
    RESTORE_REQUESTED = "restore_requested"


@dataclass(frozen=True)
class ReuseState:
    status: ReuseStatus = ReuseStatus.BLOCKED
    history: tuple[str, ...] = ()
    backup_requested: bool = False
    restore_requested: bool = False
    evidence: ReuseDecision | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ReuseStatus(self.status))


def transition(
    state: ReuseState,
    event: ReuseEvent | str,
    *,
    evidence: ReuseDecision | None = None,
) -> ReuseState:
    """Pure state transition; VERIFY/RESUME require exact evidence."""
    event = ReuseEvent(event)
    next_evidence = state.evidence
    if event is ReuseEvent.VERIFY and evidence is not None and evidence.evidence_bearing:
        next_status = ReuseStatus.REUSE
        next_evidence = evidence
    elif event is ReuseEvent.RESUME and evidence is not None and evidence.resumable and evidence.exact:
        next_status = ReuseStatus.RESUME
        next_evidence = evidence
    elif event is ReuseEvent.INVALIDATE:
        next_status = ReuseStatus.BLOCKED
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
    "APPROVED_FIELDS",
    "ArtifactOperation",
    "EXACT_BINDING_FIELDS",
    "IDENTITY_FIELDS",
    "LIVE_CODE_IDENTITY_UNCERTAIN",
    "MutationForbiddenError",
    "REUSE_BINDING_FIELDS",
    "ReadOnlyArtifactAdapter",
    "ReuseDecision",
    "ReuseEvent",
    "ReuseState",
    "ReuseStatus",
    "canonical_hash",
    "classify_artifact_reuse",
    "classify_reuse",
    "decide_reuse",
    "plan_backup",
    "plan_restore",
    "transition",
]
