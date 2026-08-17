from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json
from ..hashing import sha256_file
from ..training.generation_checkpoint import GenerationCheckpointError, read_generation_checkpoint_pointer
from .contracts import RunStatus
from .review import validate_review_summary
from .run_store import RunStore, git_commit

_SHA256_RE = re.compile(r"^[0-9A-Fa-f]{64}$")


def validate_approval_record(
    run_root: str | Path,
    *,
    expected_run_id: str | None = None,
) -> list[str]:
    """Validate the complete, hash-bound approval record for an approved run.

    A status flag alone is not an approval.  Consumers that reuse a source
    checkpoint call this helper before accepting it, so missing reviewer,
    timestamp, decision, run identity, or artifact bindings fail closed.
    """

    root = Path(run_root)
    approval_path = root / "approval_status.json"
    state_path = root / "state.json"
    summary_path = root / "review_summary.json"
    checksums_path = root / "checksums.sha256"
    errors: list[str] = []
    run_id = str(expected_run_id or root.name)
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"run state is unreadable: {exc}")
        state = None
    if not isinstance(state, Mapping):
        errors.append("run state must be a JSON object")
    else:
        if state.get("run_id") != run_id:
            errors.append("run state run_id does not match the source run")
        if state.get("run_status") != "APPROVED":
            errors.append("run state is not APPROVED")
        if state.get("approval_status") != "APPROVED":
            errors.append("run state approval_status is not APPROVED")
    try:
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"approval record is unreadable: {exc}"]
    if not isinstance(approval, Mapping):
        return ["approval record must be a JSON object"]

    # A modern trainable generation run binds approval to both tiny pointer
    # records and their canonical epoch payloads.  Keep the legacy physical
    # fallback available when no pointer files exist, but fail closed if a
    # partially migrated run exposes only an invalid modern pointer.
    pointer_paths = (
        root / "checkpoints/best_checkpoint.json",
        root / "checkpoints/latest_checkpoint.json",
    )
    if any(path.exists() for path in pointer_paths):
        for kind in ("best", "latest"):
            try:
                read_generation_checkpoint_pointer(root, kind, allow_legacy=True)
            except GenerationCheckpointError as exc:
                errors.append(f"{kind} generation checkpoint pointer is invalid: {exc}")

    if approval.get("run_id") != run_id:
        errors.append("approval record run_id does not match the source run")
    if approval.get("status") != "APPROVED":
        errors.append("approval record status is not APPROVED")
    approved_by = approval.get("approved_by")
    approved_at = approval.get("approved_at")
    if not isinstance(approved_by, str) or not approved_by.strip():
        errors.append("approval record lacks an explicit reviewer")
    if not isinstance(approved_at, str) or not approved_at.strip():
        errors.append("approval record lacks an approval timestamp")

    record = approval.get("record")
    if not isinstance(record, Mapping):
        errors.append("approval record lacks the complete decision record")
        return errors
    required = (
        "run_id",
        "decision",
        "review_note",
        "approved_or_rejected_by",
        "timestamp",
        "review_summary_sha256",
        "artifact_checksum_file_sha256",
    )
    missing = [key for key in required if key not in record]
    if missing:
        errors.append(f"approval decision record is missing fields: {missing}")
        return errors
    if record.get("run_id") != run_id:
        errors.append("approval decision record run_id does not match the source run")
    if record.get("decision") != "approve":
        errors.append("approved source decision record is not approve")
    if not isinstance(record.get("review_note"), str):
        errors.append("approval decision record review_note must be a string")
    reviewer = record.get("approved_or_rejected_by")
    timestamp = record.get("timestamp")
    if not isinstance(reviewer, str) or not reviewer.strip():
        errors.append("approval decision record lacks an explicit reviewer")
    if not isinstance(timestamp, str) or not timestamp.strip():
        errors.append("approval decision record lacks a timestamp")
    if isinstance(approved_by, str) and isinstance(reviewer, str) and approved_by != reviewer:
        errors.append("approval reviewer fields disagree")
    if isinstance(approved_at, str) and isinstance(timestamp, str) and approved_at != timestamp:
        errors.append("approval timestamp fields disagree")

    summary_hash = record.get("review_summary_sha256")
    checksum_hash = record.get("artifact_checksum_file_sha256")
    if not isinstance(summary_hash, str) or not _SHA256_RE.fullmatch(summary_hash):
        errors.append("approval decision record review-summary binding is not a SHA-256 digest")
    if not isinstance(checksum_hash, str) or not _SHA256_RE.fullmatch(checksum_hash):
        errors.append("approval decision record checksum binding is not a SHA-256 digest")
    if not summary_path.is_file() or not checksums_path.is_file():
        errors.append("approval decision record is missing its bound review summary or checksum file")
    else:
        if isinstance(summary_hash, str) and summary_hash != sha256_file(summary_path):
            errors.append("approval decision record does not bind the current review summary")
        if isinstance(checksum_hash, str) and checksum_hash != sha256_file(checksums_path):
            errors.append("approval decision record does not bind the current checksum file")
    return errors


def record_run_approval(
    root: str | Path,
    run_id: str,
    *,
    decision: str,
    review_note: str,
    reviewer: str,
) -> dict[str, Any]:
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    if not reviewer.strip():
        raise ValueError("an explicit reviewer label is required; identity is never inferred")
    root = Path(root)
    run_root = root / "results/runs" / run_id
    state_path = run_root / "state.json"
    if not state_path.exists():
        raise ValueError(f"run does not exist: {run_id}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("run_status") != RunStatus.COMPLETED_PENDING_APPROVAL.value:
        raise ValueError("approval requires run_status=COMPLETED_PENDING_APPROVAL")
    summary_path = run_root / "review_summary.json"
    checksums_path = run_root / "checksums.sha256"
    if not summary_path.exists() or not checksums_path.exists():
        raise ValueError("review summary and checksums are required before approval")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    errors = validate_review_summary(summary, completed=True)
    if errors:
        raise ValueError("review summary validation failed: " + "; ".join(errors))
    store_context = type("ApprovalContext", (), {"root": root, "entry": type("Entry", (), {"run_id": run_id, "is_azure": False})(), "fixture": False, "run_root": run_root})()
    store = RunStore(store_context)  # type: ignore[arg-type]
    checksum_errors = store.validate_checksums()
    if checksum_errors:
        raise ValueError("artifact checksum validation failed: " + "; ".join(checksum_errors))
    approval_path = run_root / "approval_status.json"
    previous = json.loads(approval_path.read_text(encoding="utf-8")) if approval_path.exists() else {}
    if previous.get("status") != "PENDING_USER_APPROVAL":
        raise ValueError("run approval status is no longer pending")
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    record = {
        "run_id": run_id,
        "decision": decision,
        "review_note": review_note,
        "approved_or_rejected_by": reviewer,
        "timestamp": timestamp,
        "review_summary_sha256": sha256_file(summary_path),
        "artifact_checksum_file_sha256": sha256_file(checksums_path),
        "code_commit": state.get("code_commit") or git_commit(root),
    }
    # The artifact files are not touched. Only the approval/state metadata changes.
    atomic_write_json(approval_path, {
        "run_id": run_id,
        "status": "APPROVED" if decision == "approve" else "REJECTED",
        "approved_by": reviewer,
        "approved_at": timestamp,
        "record": record,
    })
    state["run_status"] = RunStatus.APPROVED.value if decision == "approve" else RunStatus.REJECTED.value
    state["approval_status"] = "APPROVED" if decision == "approve" else "REJECTED"
    state["next_run_allowed"] = "NO"
    atomic_write_json(state_path, state)
    return record | {"status": state["approval_status"]}
