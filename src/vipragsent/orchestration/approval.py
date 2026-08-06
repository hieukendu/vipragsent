from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json
from ..hashing import sha256_file
from .contracts import RunStatus
from .review import validate_review_summary
from .run_store import RunStore, git_commit


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
