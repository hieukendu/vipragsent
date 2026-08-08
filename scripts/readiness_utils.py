from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.hashing import sha256_bytes, sha256_file, sha256_json
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.protocol import compare_frozen_hashes, validate_protocol_resolution

REPOSITORY = "hieukendu/vipragsent"
BRANCH = "codex/phase-14-5-production-repair"
RUNTIME_BLOCKERS = [
    "Phase 15 has not been executed on the target server",
    "Model-family runtime assets are not prepared",
    "GPU and Azure live integration have not been validated",
    "No real approved production run exists",
]
NEXT_ACTION = "Checkout the exact final repair SHA on the target server, run Phase 15 for exactly one lightweight model family, print the complete smoke report, and stop for user review."
PROTECTED_PATHS = (
    "src/vipragsent/models",
    "src/vipragsent/training",
    "src/vipragsent/runtime",
    "src/vipragsent/evaluation",
    "src/vipragsent/orchestration/executors",
    "src/vipragsent/orchestration/q1b_dependencies.py",
    "src/vipragsent/orchestration/q1b_predictor.py",
    "src/vipragsent/azure",
    "configs/experiments",
    "configs/models",
    "configs/runtime",
    "configs/losses.yaml",
    "configs/statistics.yaml",
    "configs/statistics",
    "prompts/protocols",
    "data/processed",
    "data/manifests",
)
REVIEW_SOURCE_PATHS = (
    "reports/final_cleanup_review_cycles.json",
    "reports/luna_max_review_cycles.json",
    "reports/runtime_self_review.json",
)


def read_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.exists():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def git_sha(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


def git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, descendant], cwd=root, check=False).returncode == 0


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _review_count(review: dict[str, Any], keys: tuple[str, ...], *, label: str, errors: list[str]) -> int:
    value: Any = None
    for key in keys:
        if key in review:
            value = review[key]
            break
    if value is None:
        errors.append(f"missing review field: {label}")
        return 0
    if label == "cycles" and isinstance(value, list):
        return len(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"invalid review field: {label}")
        return 0
    return value


def normalize_review(review: Any, *, source: str | None = None) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(review, dict):
        review = {}
        errors.append("review source is not a JSON object")
    cycles = _review_count(review, ("cycle_count", "sequence_count", "cycles"), label="cycles", errors=errors)
    rounds = _review_count(
        review,
        ("rounds_per_cycle", "completed_rounds_per_sequence", "required_rounds_per_sequence"),
        label="rounds_per_cycle",
        errors=errors,
    )
    clean = _review_count(
        review,
        ("consecutive_clean_cycles", "consecutive_no_new_defect_sequences", "consecutive_clean_sequences"),
        label="consecutive_clean_cycles",
        errors=errors,
    )

    raw_status = review.get("status")
    if not isinstance(raw_status, str) or not raw_status:
        errors.append("missing or invalid review field: status")
        status = "FAIL"
    else:
        status = raw_status

    no_new_value: Any = None
    for key in ("no_new_defects", "no_new_defects_in_two_complete_cycles"):
        if key in review:
            no_new_value = review[key]
            break
    if not isinstance(no_new_value, bool):
        errors.append("missing or invalid review field: no_new_defects")
        no_new_defects = False
    else:
        no_new_defects = no_new_value

    execution_mode = review.get("execution_mode", "HISTORICAL")
    if not isinstance(execution_mode, str) or not execution_mode:
        errors.append("invalid review field: execution_mode")
        execution_mode = "UNKNOWN"
    subagents_called = review.get("subagents_called")
    if subagents_called is not None and not isinstance(subagents_called, bool):
        errors.append("invalid review field: subagents_called")
        subagents_called = None

    historical_profile: Any = review.get("historical_subagent_profile_verification")
    if historical_profile is None:
        historical_profile = review.get("actual_profile_verification")
    if historical_profile is None:
        historical_profile = review.get("subagent_profile_verification")
    if historical_profile is None:
        historical_profile = review.get("profile_resolution", "NOT_VERIFIED")
    historical_profile = str(historical_profile)
    if historical_profile.startswith("NOT_VERIFIED"):
        historical_profile = "NOT_VERIFIED"

    valid = (
        not errors
        and status == "PASS"
        and cycles >= 2
        and rounds > 0
        and clean >= 2
        and no_new_defects is True
    )
    return {
        "source": source,
        "execution_mode": execution_mode,
        "subagents_called": subagents_called,
        "cycles": cycles,
        "rounds_per_cycle": rounds,
        "consecutive_clean_cycles": clean,
        "status": status,
        "no_new_defects": no_new_defects,
        "historical_subagent_profile_verification": historical_profile,
        "subagent_profile_verification": historical_profile,
        "valid": valid,
        "normalization_errors": errors,
    }


def load_review(root: Path) -> dict[str, Any]:
    source_errors: list[str] = []
    first_present: str | None = None
    for relative in REVIEW_SOURCE_PATHS:
        path = root / relative
        if not path.exists():
            continue
        if first_present is None:
            first_present = relative
        try:
            payload = read_json(path, None)
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            source_errors.append(f"{relative}: unreadable review source ({exc})")
            continue
        normalized = normalize_review(payload, source=relative)
        if normalized["valid"]:
            normalized["source_selection_errors"] = source_errors
            return normalized
        source_errors.extend(f"{relative}: {error}" for error in normalized["normalization_errors"])

    failed = normalize_review({"status": "FAIL"}, source=first_present)
    failed["source_selection_errors"] = source_errors
    failed["normalization_errors"] = source_errors or failed["normalization_errors"]
    failed["valid"] = False
    return failed


def iter_protected_files(root: Path) -> list[Path]:
    files: dict[str, Path] = {}
    for relative in PROTECTED_PATHS:
        path = root / relative
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix not in {".pyc"} and "__pycache__" not in child.parts:
                    files[child.relative_to(root).as_posix()] = child
    return [files[key] for key in sorted(files)]


def worktree_protected_manifest(root: Path) -> dict[str, Any]:
    files = [{"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)} for path in iter_protected_files(root)]
    return {"files": files, "manifest_sha256": sha256_json(files)}


def commit_protected_manifest(root: Path, commit: str) -> dict[str, Any]:
    names: set[str] = set()
    for relative in PROTECTED_PATHS:
        result = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit, "--", relative], cwd=root, capture_output=True, text=True, check=True)
        names.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    files: list[dict[str, str]] = []
    for name in sorted(names):
        payload = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=root, capture_output=True, check=True).stdout
        files.append({"path": name, "sha256": sha256_bytes(payload)})
    return {"files": files, "manifest_sha256": sha256_json(files)}


def validate_ci_evidence(evidence: dict[str, Any], *, expected_head: str | None = None) -> list[str]:
    errors: list[str] = []
    if evidence.get("repository") != REPOSITORY:
        errors.append("repository mismatch")
    if evidence.get("branch") != BRANCH:
        errors.append("branch mismatch")
    if evidence.get("workflow") != "cpu-ci":
        errors.append("workflow mismatch")
    if not isinstance(evidence.get("run_id"), int) or evidence.get("run_id", 0) <= 0:
        errors.append("missing run_id")
    if not isinstance(evidence.get("run_number"), int) or evidence.get("run_number", 0) <= 0:
        errors.append("missing run_number")
    if not evidence.get("head_sha"):
        errors.append("missing head_sha")
    if expected_head and evidence.get("head_sha") != expected_head:
        errors.append("head_sha does not match expected audited code SHA")
    if evidence.get("status") != "completed":
        errors.append("workflow status is not completed")
    if evidence.get("conclusion") != "success":
        errors.append("workflow conclusion is not success")
    if evidence.get("verification_source") not in {"github_api", "github_connector"}:
        errors.append("verification source is not GitHub evidence")
    return errors


def build_snapshot(root: Path, ci_evidence: dict[str, Any]) -> dict[str, Any]:
    parent_sha = git_sha(root)
    ci_errors = validate_ci_evidence(ci_evidence, expected_head=parent_sha)
    state = read_json(root / "PROJECT_STATE.json", {})
    inventory = build_expected_runs(root)
    frozen = compare_frozen_hashes(root)
    protocol = validate_protocol_resolution(root)
    review = load_review(root)
    guard = read_json(root / "reports/final_cleanup_protocol_guard.json", {})
    source_manifest = guard.get("after_manifest", worktree_protected_manifest(root))
    implementation = {
        "status": "PASS" if not state.get("implementation_blockers") and state.get("setup_implementation_ready") else "FAIL",
        "setup_implementation_ready": bool(state.get("setup_implementation_ready")),
        "setup_frozen": bool(state.get("setup_frozen")),
        "phase15_code_ready": bool(state.get("phase15_code_ready")),
        "sequential_runtime_code_ready": bool(state.get("sequential_runtime_code_ready")),
        "full_matrix_code_ready": bool(state.get("full_matrix_code_ready")),
        "implementation_blockers": list(state.get("implementation_blockers", [])),
    }
    runtime = {
        "LOCAL_CODE_READINESS": "PASS" if implementation["status"] == "PASS" else "FAIL",
        "SERVER_RUNTIME_READINESS": "NOT_RUN",
        "REAL_EXPERIMENT_READINESS": False,
        "phase15_runtime_ready": bool(state.get("phase15_runtime_ready")),
        "runtime_environment_ready": bool(state.get("runtime_environment_ready")),
        "weights_downloaded": bool(state.get("weights_downloaded")),
        "real_experiment_ready": bool(state.get("real_experiment_ready")),
        "final_aggregation_ready": bool(state.get("final_aggregation_ready")),
        "real_run_count": int(state.get("real_run_count", 0)),
        "approved_run_count": int(state.get("approved_run_count", 0)),
        "runtime_blockers": list(state.get("runtime_blockers", [])),
    }
    return {
        "schema_version": 1,
        "repository": REPOSITORY,
        "branch": BRANCH,
        "branch_head_before_refresh": parent_sha,
        "audited_code_commit": ci_evidence.get("head_sha", ""),
        "audited_source_manifest_sha256": source_manifest.get("manifest_sha256", ""),
        "report_generation_parent_sha": parent_sha,
        "ci": {**ci_evidence, "validation_errors": ci_errors},
        "review": review,
        "scientific": {
            "frozen_data_changed": not frozen["unchanged"],
            "scientific_protocol_conflicts": list(protocol["scientific_protocol_conflicts"]),
            "inventory_rows": len(inventory["rows"]),
            "inventory_hash": inventory["inventory_hash"],
        },
        "implementation": implementation,
        "runtime": runtime,
        "execution_policy": {
            "maximum_concurrent_gpu_jobs": int(state.get("maximum_concurrent_gpu_jobs", 1)),
            "automatic_next_run": bool(state.get("automatic_next_run")),
            "global_full_dag_enabled": bool(state.get("global_full_dag_enabled")),
            "user_review_required_after_every_completed_run": True,
        },
        "evidence_boundary": {
            "cpu_only": True,
            "network_free_local_tests": True,
            "azure_live_free": True,
            "model_download_free": True,
            "synthetic": True,
            "production_proof": False,
        },
        "report_only_commit_expected": True,
        "next_action": str(state.get("next_action", NEXT_ACTION)),
    }


def snapshot_markdown(snapshot: dict[str, Any]) -> str:
    ci = snapshot["ci"]
    review = snapshot["review"]
    scientific = snapshot["scientific"]
    runtime = snapshot["runtime"]
    return "\n".join([
        "# Final readiness snapshot",
        "",
        f"- Branch: `{snapshot['branch']}`",
        f"- Branch head before refresh: `{snapshot['branch_head_before_refresh']}`",
        f"- Audited code commit: `{snapshot['audited_code_commit']}`",
        f"- Report generation parent SHA: `{snapshot['report_generation_parent_sha']}`",
        f"- Audited source manifest: `{snapshot['audited_source_manifest_sha256']}`",
        f"- Report-only commit expected: `{str(snapshot['report_only_commit_expected']).lower()}`",
        "",
        "## CI",
        "",
        f"- Workflow: `{ci.get('workflow')}`",
        f"- Run: `{ci.get('run_id')}` (#{ci.get('run_number')})",
        f"- Head SHA: `{ci.get('head_sha')}`",
        f"- Status/conclusion: `{ci.get('status')}/{ci.get('conclusion')}`",
        f"- Verification source: `{ci.get('verification_source')}`",
        "",
        "## Review",
        "",
        f"- Status: `{review['status']}`",
        f"- Review source: `{review['source']}`",
        f"- Execution mode: `{review['execution_mode']}`",
        f"- Subagents called: `{str(review['subagents_called']).lower() if review['subagents_called'] is not None else 'unknown'}`",
        f"- Cycles: `{review['cycles']}`",
        f"- Rounds per cycle: `{review['rounds_per_cycle']}`",
        f"- Consecutive clean cycles: `{review['consecutive_clean_cycles']}`",
        f"- No new defects: `{str(review['no_new_defects']).lower()}`",
        f"- Historical subagent profile verification: `{review['historical_subagent_profile_verification']}`",
        "",
        "## Readiness",
        "",
        f"- Local code readiness: `{runtime['LOCAL_CODE_READINESS']}`",
        f"- Server runtime readiness: `{runtime['SERVER_RUNTIME_READINESS']}`",
        f"- Real experiment readiness: `{str(runtime['REAL_EXPERIMENT_READINESS']).lower()}`",
        f"- Inventory rows: `{scientific['inventory_rows']}`",
        f"- Scientific conflicts: `{len(scientific['scientific_protocol_conflicts'])}`",
        f"- Implementation blockers: `{len(snapshot['implementation']['implementation_blockers'])}`",
        f"- Runtime blockers: `{len(runtime['runtime_blockers'])}`",
        "",
        "## Evidence boundary",
        "",
        "CPU-only, network-free local tests; no live Azure, model download, Phase 15, GPU training, real predictions, approval, or full DAG execution.",
        "",
        "## Runtime blockers",
        "",
        *[f"- {item}" for item in runtime["runtime_blockers"]],
        "",
        "## Next action",
        "",
        snapshot["next_action"],
        "",
    ])


def snapshot_report_fields(snapshot: dict[str, Any]) -> dict[str, Any]:
    ci = snapshot["ci"]
    return {
        "branch_head_before_refresh": snapshot["branch_head_before_refresh"],
        "audited_code_commit": snapshot["audited_code_commit"],
        "audited_source_manifest_sha256": snapshot["audited_source_manifest_sha256"],
        "report_generation_parent_sha": snapshot["report_generation_parent_sha"],
        "ci_verified_head_sha": ci.get("head_sha"),
        "ci_run_id": ci.get("run_id"),
        "ci_run_number": ci.get("run_number"),
        "ci_status": ci.get("status"),
        "ci_conclusion": ci.get("conclusion"),
        "report_only_commit_expected": True,
        "CI_STATUS": "PASS" if ci.get("conclusion") == "success" else "UNVERIFIED_EXTERNAL",
        "review_summary": snapshot["review"],
        "LOCAL_CODE_READINESS": snapshot["runtime"]["LOCAL_CODE_READINESS"],
        "SERVER_RUNTIME_READINESS": snapshot["runtime"]["SERVER_RUNTIME_READINESS"],
        "REAL_EXPERIMENT_READINESS": snapshot["runtime"]["REAL_EXPERIMENT_READINESS"],
        "next_action": snapshot["next_action"],
    }


def snapshot_is_compatible(root: Path, snapshot: dict[str, Any]) -> bool:
    """Accept a readiness snapshot only for this code or a report-only descendant."""
    audited = str(snapshot.get("audited_code_commit", ""))
    if not audited:
        return False
    current = git_sha(root)
    if current == audited:
        return True
    if snapshot.get("report_only_commit_expected") is not True or not git_is_ancestor(root, audited, current):
        return False
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{audited}..{current}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    changed = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return bool(changed) and all(path.startswith("reports/") for path in changed)


def merge_snapshot_into_report(report: dict[str, Any], snapshot: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    if root is not None and not snapshot_is_compatible(root, snapshot):
        report["snapshot_merge_status"] = "SKIPPED_STALE"
        report["snapshot_merge_reason"] = "readiness snapshot is not bound to the current code or a report-only descendant"
        return report
    fields = snapshot_report_fields(snapshot)
    report.update(fields)
    report["snapshot_merge_status"] = "APPLIED"
    report["code_commit_at_audit"] = snapshot["audited_code_commit"]
    report["github_ci_status_at_report_generation"] = snapshot["ci"].get("conclusion")
    report["next_action"] = snapshot["next_action"]
    report["scientific_protocol_conflicts"] = snapshot["scientific"]["scientific_protocol_conflicts"]
    report["implementation_blockers"] = snapshot["implementation"]["implementation_blockers"]
    report["runtime_blockers"] = snapshot["runtime"]["runtime_blockers"]
    report["inventory_count"] = snapshot["scientific"]["inventory_rows"]
    report["inventory_hash"] = snapshot["scientific"]["inventory_hash"]
    report["self_review_summary"] = snapshot["review"]
    if isinstance(report.get("self_review"), dict):
        report["self_review"]["canonical_summary"] = snapshot["review"]
    readiness = report.setdefault("readiness", {})
    readiness.update({
        "SETUP_CODE_READY": snapshot["implementation"]["status"] == "PASS",
        "CI_STATUS": snapshot["ci"].get("status"),
        "runtime_status": "BLOCKED",
        "runtime_blockers": snapshot["runtime"]["runtime_blockers"],
        "scientific_protocol_conflicts": snapshot["scientific"]["scientific_protocol_conflicts"],
        "self_review": snapshot["review"],
    })
    return report


def write_snapshot(root: Path, snapshot: dict[str, Any]) -> None:
    atomic_write_json(root / "reports/final_readiness_snapshot.json", snapshot)
    atomic_write_text(root / "reports/final_readiness_snapshot.md", snapshot_markdown(snapshot))
