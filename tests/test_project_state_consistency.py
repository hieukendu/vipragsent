from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_project_state_matches_verified_backup_and_paused_run() -> None:
    state = _read_json(ROOT / "PROJECT_STATE.json")
    backup = _read_json(ROOT / "reports/vipragsent_safe_pause_backup.json")
    summary = backup["checkpoint_summary"]
    incremental = backup["incremental_upload"]

    assert backup["backup_status"] == "PASS"
    assert backup["pipeline_status"] == "SAFELY_PAUSED"
    assert backup["protocol_integrity"] == "UNCHANGED_BY_BACKUP"
    assert summary["expected_canonical_entries"] == 35
    assert summary["remote_verified_entries"] == 35
    assert summary["blocked_entries"] == 0
    assert incremental["still_missing"] == 0
    assert state["remote_backup_manifest"] == "reports/vipragsent_safe_pause_backup.json"
    assert state["remote_backup_status"] == backup["backup_status"]

    for repository in [backup["hf_experiment_artifacts"], *backup["hf_repositories"].values()]:
        assert repository["access_verified"] is True
        assert repository["visibility_before"] == "PUBLIC"
        assert repository["visibility_after"] == "PUBLIC"

    paused = backup["paused_job"]
    paused_run_id = paused["run_id"]
    assert state["current_scientific_job_status"] == backup["pipeline_status"]
    assert state["paused_run_id"] == paused_run_id
    assert state["paused_resume_boundary"] == paused["resume_boundary"]
    assert state["paused_resume_mode"] == paused["resume_mode"]
    assert state["next_action"] == f"Restore the paused {paused_run_id} run from its verified epoch-1 checkpoint boundary."

    paused_state = _read_json(ROOT / "results/runs" / paused_run_id / "state.json")
    assert paused_state["run_status"] == paused["run_status"]
    assert paused_state["stages"][paused["stage"]]["status"] == paused["stage_status"]
    assert paused_state["stages"]["preflight"]["status"] == "PASS"
    assert paused["stage_status"] == "INTERRUPTED"
    assert state["full_run_started"] is True

    boundary = ROOT / str(state["paused_resume_boundary"])
    boundary_entries = [
        item
        for item in backup["model_checkpoints"]
        if item["run_id"] == paused_run_id
        and item["local_path"] == state["paused_resume_boundary"]
        and item["status"] == "REMOTE_VERIFIED"
    ]
    assert boundary_entries
    if boundary.is_file():
        boundary_sha = hashlib.sha256(boundary.read_bytes()).hexdigest()
        assert any(item["sha256"] == boundary_sha for item in boundary_entries)

    canonical_run_ids = {item["run_id"] for item in backup["model_checkpoints"]}
    assert len(canonical_run_ids) == 19
    manifest_approved_run_ids = {
        item["run_id"]
        for item in backup["model_checkpoints"]
        if item["resume_eligibility"] == "REUSABLE_APPROVED_CHECKPOINT"
    }
    assert manifest_approved_run_ids == canonical_run_ids - {paused_run_id}
    records: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for state_path in sorted((ROOT / "results/runs").glob("*/state.json")):
        run_state = _read_json(state_path)
        run_id = run_state.get("run_id")
        if run_id in canonical_run_ids:
            records[run_id] = (run_state, _read_json(state_path.with_name("approval_status.json")))

    if records:
        assert set(records) == canonical_run_ids
        approved_run_ids = {
            run_id
            for run_id, (run_state, approval) in records.items()
            if run_state.get("run_status") == "APPROVED" and approval.get("status") == "APPROVED"
        }
        assert approved_run_ids == manifest_approved_run_ids
    else:
        # The model checkpoints and local run records are intentionally not
        # committed to GitHub; the canonical backup manifest is the tracked
        # provenance available in CPU-only CI.
        approved_run_ids = manifest_approved_run_ids
    assert state["real_run_count"] == len(canonical_run_ids)
    assert state["approved_run_count"] == len(approved_run_ids)
