from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, atomic_write_text
from ..phase import inspect_phase15_handoff
from .model_assets import read_family_status, resolve_local_snapshot


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _local_snapshot_evidence(root: Path, handoff: dict[str, Any], *, required: bool) -> dict[str, Any]:
    evidence = handoff.get("phase15_evidence", {})
    family = evidence.get("model_family")
    if not isinstance(family, str) or not family:
        return {"status": "BLOCKED", "available": False, "missing_count": 0}
    cache = read_family_status(root, family, "cache")
    snapshot = resolve_local_snapshot(root, cache.get("local_path"))
    if snapshot is None or not snapshot.is_dir():
        return {"status": "BLOCKED" if required else "NOT_RECHECKED", "available": False, "missing_count": 1}
    expected = [str(item) for item in cache.get("snapshot_files", []) if item]
    if "config.json" not in expected:
        expected.append("config.json")
    missing_count = sum(1 for relative in expected if not (snapshot / relative).is_file())
    return {
        "status": "PASS" if missing_count == 0 else "BLOCKED",
        "available": missing_count == 0,
        "missing_count": missing_count,
        "local_snapshot": "data/model_cache/" + family,
    }


def _replace_line(text: str, key: str, value: bool | int) -> str:
    replacement = f"{key}={str(value).lower() if isinstance(value, bool) else value}"
    pattern = rf"^{re.escape(key)}=.*$"
    if re.search(pattern, text, flags=re.MULTILINE):
        return re.sub(pattern, replacement, text, flags=re.MULTILINE)
    return text.rstrip() + f"\n{replacement}\n"


def _setup_ready_text(existing: str, state: dict[str, Any]) -> str:
    text = existing or "# Setup readiness\n"
    for key in (
        "PHASE15_RUNTIME_READY",
        "RUNTIME_ENVIRONMENT_READY",
        "WEIGHTS_DOWNLOADED",
        "REAL_EXPERIMENT_READY",
        "FINAL_AGGREGATION_READY",
        "REAL_RUN_COUNT",
        "APPROVED_RUN_COUNT",
    ):
        state_key = key.lower()
        text = _replace_line(text, key, state.get(state_key, False if key.isupper() else 0))
    blocker_start = "## Runtime blockers"
    next_start = "## Exact next action"
    if blocker_start in text:
        prefix = text.split(blocker_start, 1)[0]
        text = prefix + blocker_start + "\n"
        blockers = state.get("runtime_blockers", [])
        if blockers:
            text += "\n".join(f"- {item}" for item in blockers) + "\n\n"
        else:
            text += "None\n\n"
        text += next_start + "\n" + str(state.get("next_action", "")).strip() + "\n"
    return text.rstrip() + "\n"


def reconcile_phase15_state(root: str | Path, *, require_local_snapshot: bool = True) -> dict[str, Any]:
    """Persist only runtime flags proven by the Phase 15 evidence artifacts."""
    project_root = Path(root).resolve()
    handoff = inspect_phase15_handoff(project_root)
    local = _local_snapshot_evidence(project_root, handoff, required=require_local_snapshot)
    status = handoff["status"] if handoff["status"] != "PASS" else "PASS" if local["status"] in {"PASS", "NOT_RECHECKED"} else "BLOCKED"
    blockers = list(handoff.get("blockers", []))
    if status != "PASS" and local["status"] == "BLOCKED" and local["available"] is False:
        blockers.append("Phase 15 local snapshot is not available on this machine")
    blockers = list(dict.fromkeys(str(item) for item in blockers if item))

    state_path = project_root / "PROJECT_STATE.json"
    state = _read_json(state_path)
    if status == "PASS":
        manifest = _read_json(project_root / "data/model_cache_manifest.json")
        state.update(
            {
                "phase15_runtime_ready": True,
                "runtime_environment_ready": require_local_snapshot and local["available"],
                "weights_downloaded": bool(manifest.get("weights_downloaded", state.get("weights_downloaded", False))),
                "runtime_blockers": [],
                "phase15_model_family": handoff.get("phase15_evidence", {}).get("model_family"),
                "phase15_approval_basis": handoff.get("phase15_evidence", {}).get("approval_basis"),
                "next_action": "Select the first incomplete eligible scientific job after the approved Phase 15 model-preparation handoff.",
            }
        )
    else:
        state.update(
            {
                "phase15_runtime_ready": False,
                "runtime_environment_ready": False,
                "runtime_blockers": blockers,
            }
        )
    atomic_write_json(state_path, state)
    setup_path = project_root / "SETUP_READY.md"
    atomic_write_text(setup_path, _setup_ready_text(setup_path.read_text(encoding="utf-8") if setup_path.exists() else "", state))
    report = {
        "schema_version": 1,
        "status": status,
        "model_family": handoff.get("phase15_evidence", {}).get("model_family"),
        "handoff_status": handoff.get("status"),
        "handoff_tests_passed": handoff.get("tests_passed"),
        "handoff_blockers": blockers,
        "local_snapshot": local,
        "state_fields_updated": ["phase15_runtime_ready", "runtime_environment_ready", "weights_downloaded", "runtime_blockers", "phase15_model_family", "phase15_approval_basis", "next_action"],
        "scientific_execution_started": bool(state.get("full_run_started")),
        "approved_run_count": int(state.get("approved_run_count", 0)),
    }
    atomic_write_json(project_root / "reports/phase15_state_recovery.json", report)
    atomic_write_text(
        project_root / "reports/phase15_state_recovery.md",
        "\n".join(
            [
                "# Phase 15 state recovery",
                "",
                f"- Status: `{status}`",
                f"- Model family: `{report['model_family']}`",
                f"- Handoff status: `{report['handoff_status']}`",
                f"- Local snapshot verified: `{str(local['available']).lower()}`",
                f"- Scientific execution started: `{str(report['scientific_execution_started']).lower()}`",
                "",
                "## Blockers",
                "",
                *([f"- {item}" for item in blockers] or ["- None"]),
                "",
            ]
        ),
    )
    return report


__all__ = ["reconcile_phase15_state"]
