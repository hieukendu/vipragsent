from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    from _bootstrap import ROOT
    from readiness_utils import (
        BRANCH,
        NEXT_ACTION,
        REPOSITORY,
        REVIEW_SOURCE_PATHS,
        RUNTIME_BLOCKERS,
        git_is_ancestor,
        git_sha,
        load_review,
        normalize_review,
        read_json,
        snapshot_markdown,
        validate_ci_evidence,
        worktree_protected_manifest,
    )
except ModuleNotFoundError:
    from scripts._bootstrap import ROOT
    from scripts.readiness_utils import (
        BRANCH,
        NEXT_ACTION,
        REPOSITORY,
        REVIEW_SOURCE_PATHS,
        RUNTIME_BLOCKERS,
        git_is_ancestor,
        git_sha,
        load_review,
        normalize_review,
        read_json,
        snapshot_markdown,
        validate_ci_evidence,
        worktree_protected_manifest,
    )
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.orchestration.inventory import build_expected_runs


def _setup_values(text: str) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key, value in re.findall(r"^([A-Z0-9_]+)=(true|false|[0-9]+)$", text, flags=re.MULTILINE):
        values[key] = value == "true" if value in {"true", "false"} else int(value)
    values["runtime_blockers"] = re.findall(r"^\- (.+)$", text.split("## Runtime blockers", 1)[-1].split("## Exact next action", 1)[0], flags=re.MULTILINE)
    values["next_action"] = text.split("## Exact next action", 1)[-1].strip()
    return values


def cross_file_errors(
    snapshot: dict[str, Any],
    ci: dict[str, Any],
    state: dict[str, Any],
    setup: dict[str, Any],
    runtime: dict[str, Any],
    preexperiment: dict[str, Any],
    review: dict[str, Any],
    inventory_count: int,
    local_closure: dict[str, Any],
    authoritative_review: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    implementation = snapshot.get("implementation", {})
    runtime_snapshot = snapshot.get("runtime", {})
    scientific = snapshot.get("scientific", {})
    expected_state = {
        "setup_implementation_ready": True,
        "setup_frozen": True,
        "phase15_code_ready": True,
        "sequential_runtime_code_ready": True,
        "full_matrix_code_ready": True,
        "phase15_runtime_ready": False,
        "runtime_environment_ready": False,
        "weights_downloaded": False,
        "real_experiment_ready": False,
        "final_aggregation_ready": False,
        "real_run_count": 0,
        "approved_run_count": 0,
        "implementation_blockers": [],
        "scientific_protocol_conflicts": [],
    }
    for key, expected in expected_state.items():
        if state.get(key) != expected:
            errors.append(f"PROJECT_STATE mismatch: {key}")
    setup_map = {
        "SETUP_IMPLEMENTATION_READY": True,
        "SETUP_FROZEN": True,
        "PHASE15_CODE_READY": True,
        "SEQUENTIAL_RUNTIME_CODE_READY": True,
        "FULL_MATRIX_CODE_READY": True,
        "PHASE15_RUNTIME_READY": False,
        "RUNTIME_ENVIRONMENT_READY": False,
        "WEIGHTS_DOWNLOADED": False,
        "REAL_EXPERIMENT_READY": False,
        "FINAL_AGGREGATION_READY": False,
        "REAL_RUN_COUNT": 0,
        "APPROVED_RUN_COUNT": 0,
    }
    for key, expected in setup_map.items():
        if setup.get(key) != expected:
            errors.append(f"SETUP_READY mismatch: {key}")
    if setup.get("runtime_blockers") != RUNTIME_BLOCKERS or state.get("runtime_blockers") != RUNTIME_BLOCKERS:
        errors.append("runtime blockers mismatch")
    if setup.get("next_action") != NEXT_ACTION or state.get("next_action") != NEXT_ACTION:
        errors.append("next action mismatch")
    if snapshot.get("repository") != REPOSITORY or snapshot.get("branch") != BRANCH:
        errors.append("snapshot repository or branch mismatch")
    if snapshot.get("audited_code_commit") != ci.get("head_sha"):
        errors.append("snapshot audited code SHA does not match CI head SHA")
    if snapshot.get("report_generation_parent_sha") != snapshot.get("audited_code_commit"):
        errors.append("report generation parent is not the audited code commit")
    if snapshot.get("report_only_commit_expected") is not True:
        errors.append("report-only provenance flag is not true")
    if validate_ci_evidence(ci, expected_head=snapshot.get("audited_code_commit")):
        errors.append("CI evidence is not exact completed/success evidence")
    review_expected = {
        "source": "reports/final_cleanup_review_cycles.json",
        "execution_mode": "SINGLE_AGENT",
        "subagents_called": False,
        "cycles": 2,
        "rounds_per_cycle": 6,
        "consecutive_clean_cycles": 2,
        "status": "PASS",
        "no_new_defects": True,
        "historical_subagent_profile_verification": "NOT_VERIFIED",
    }
    for key, expected in review_expected.items():
        if review.get(key) != expected or snapshot.get("review", {}).get(key) != expected:
            errors.append(f"review mismatch: {key}")
    if review.get("valid") is not True:
        errors.append("selected review source is not valid")
    authoritative = authoritative_review if authoritative_review is not None else review
    for key, expected in review_expected.items():
        if authoritative.get(key) != expected:
            errors.append(f"authoritative cleanup review mismatch: {key}")
    if authoritative.get("valid") is not True:
        errors.append("authoritative cleanup review is not valid")
    if inventory_count != 162 or scientific.get("inventory_rows") != 162:
        errors.append("inventory count is not 162")
    if scientific.get("scientific_protocol_conflicts") != [] or state.get("scientific_protocol_conflicts") != []:
        errors.append("scientific conflicts are not empty")
    if implementation.get("implementation_blockers") != [] or state.get("implementation_blockers") != []:
        errors.append("implementation blockers are not empty")
    for report_name, report in (("runtime", runtime), ("preexperiment", preexperiment)):
        if report.get("audited_code_commit") != snapshot.get("audited_code_commit"):
            errors.append(f"{report_name} audited code SHA mismatch")
        if report.get("ci_verified_head_sha") != ci.get("head_sha") or report.get("ci_conclusion") != "success":
            errors.append(f"{report_name} CI provenance mismatch")
        for summary_name in ("review_summary", "self_review_summary"):
            summary = report.get(summary_name, {})
            for key, expected in review_expected.items():
                if summary.get(key) != expected:
                    errors.append(f"{report_name} {summary_name} mismatch: {key}")
        self_review = report.get("self_review")
        if isinstance(self_review, dict):
            for key, expected in review_expected.items():
                if self_review.get(key) != expected:
                    errors.append(f"{report_name} self_review mismatch: {key}")
        if report.get("runtime_blockers") != RUNTIME_BLOCKERS:
            errors.append(f"{report_name} runtime blockers mismatch")
    if runtime_snapshot.get("LOCAL_CODE_READINESS") != "PASS" or runtime_snapshot.get("SERVER_RUNTIME_READINESS") != "NOT_RUN" or runtime_snapshot.get("REAL_EXPERIMENT_READINESS") is not False:
        errors.append("snapshot code/runtime readiness separation mismatch")
    if runtime_snapshot.get("final_aggregation_ready") is not False or runtime_snapshot.get("real_run_count") != 0 or runtime_snapshot.get("approved_run_count") != 0:
        errors.append("snapshot real-run or aggregation state mismatch")
    if local_closure.get("production_proof") is not False or local_closure.get("synthetic_results_enter_production_aggregation") is not False:
        errors.append("synthetic evidence boundary mismatch")
    return errors


def _stale_current_matches(root: Path) -> list[str]:
    matches: list[str] = []
    current_files = (
        root / "reports/final_readiness_snapshot.md",
        root / "reports/final_runtime_integration_audit.md",
        root / "reports/final_preexperiment_closure.md",
        root / "reports/final_production_correctness_repair.md",
    )
    patterns = (
        re.compile(r"CI status: `NOT_RUN`"),
        re.compile(r"Self-review: `0 rounds"),
        re.compile(r"Self-review: `5 rounds"),
        re.compile(r"consecutive clean (?:sequences|cycles): `0"),
        re.compile(r"Code commit at audit: `487134bf0e1b0b3d5f3165f0e7a71785141d4c8d`"),
    )
    for path in current_files:
        if not path.exists():
            matches.append(f"missing {path.relative_to(root).as_posix()}")
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            if pattern.search(text):
                matches.append(f"{path.relative_to(root).as_posix()}: {pattern.pattern}")
    return matches


def _review_markdown_errors(root: Path, expected: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    expected_lines = {
        "Review source": f"`{expected['source']}`",
        "Execution mode": f"`{expected['execution_mode']}`",
        "Subagents called": "`false`",
        "No new defects": "`true`",
        "Historical subagent profile verification": "`NOT_VERIFIED`",
    }
    for relative in (
        "reports/final_readiness_snapshot.md",
        "reports/final_runtime_integration_audit.md",
        "reports/final_preexperiment_closure.md",
        "reports/final_production_correctness_repair.md",
    ):
        path = root / relative
        if not path.exists():
            errors.append(f"missing review Markdown: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for label, value in expected_lines.items():
            if not re.search(rf"^-\s*{re.escape(label)}:\s*{re.escape(value)}\s*$", text, flags=re.MULTILINE):
                errors.append(f"{relative} review Markdown mismatch: {label}")
    return errors


def _normalized_review_source(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    if not path.exists():
        return normalize_review(None, source=relative)
    try:
        payload = read_json(path, None)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return normalize_review(None, source=relative)
    return normalize_review(payload, source=relative)


def audit(root: Path = ROOT, *, write_report: bool = True) -> dict[str, Any]:
    snapshot = read_json(root / "reports/final_readiness_snapshot.json", {})
    ci = read_json(root / "reports/ci_verification.json", {})
    state = read_json(root / "PROJECT_STATE.json", {})
    runtime = read_json(root / "reports/final_runtime_integration_audit.json", {})
    preexperiment = read_json(root / "reports/final_preexperiment_closure.json", {})
    local_closure = read_json(root / "reports/local_production_correctness_closure.json", {})
    review = load_review(root)
    authoritative_review = _normalized_review_source(root, REVIEW_SOURCE_PATHS[0])
    setup_text = (root / "SETUP_READY.md").read_text(encoding="utf-8") if (root / "SETUP_READY.md").exists() else ""
    setup = _setup_values(setup_text)
    inventory_count = len(build_expected_runs(root)["rows"])
    errors: list[str] = []
    if snapshot.get("schema_version") != 1:
        errors.append("snapshot schema invalid")
    if not snapshot or not ci:
        errors.append("snapshot or CI evidence missing")
    errors.extend(cross_file_errors(snapshot, ci, state, setup, runtime, preexperiment, review, inventory_count, local_closure, authoritative_review))
    errors.extend(_review_markdown_errors(root, {
        "source": REVIEW_SOURCE_PATHS[0],
        "execution_mode": "SINGLE_AGENT",
        "subagents_called": False,
        "no_new_defects": True,
        "historical_subagent_profile_verification": "NOT_VERIFIED",
    }))
    snapshot_markdown_path = root / "reports/final_readiness_snapshot.md"
    if snapshot and (not snapshot_markdown_path.exists() or snapshot_markdown_path.read_text(encoding="utf-8") != snapshot_markdown(snapshot)):
        errors.append("snapshot JSON and Markdown disagree")
    if snapshot and not git_is_ancestor(root, str(snapshot.get("audited_code_commit", "")), git_sha(root)):
        errors.append("final HEAD is not descended from audited code commit")
    manifest = read_json(root / "reports/final_cleanup_protocol_guard.json", {})
    current_manifest = worktree_protected_manifest(root)
    if manifest.get("status") != "PASS" or manifest.get("after_manifest", {}).get("manifest_sha256") != current_manifest["manifest_sha256"]:
        errors.append("protected source manifest guard mismatch")
    if read_json(root / "reports/luna_max_subagent_manifest.json", {}).get("actual_profile_verification") != "NOT_VERIFIED":
        errors.append("historical Luna profile verification is not truthfully NOT_VERIFIED")
    if _stale_current_matches(root):
        errors.extend(_stale_current_matches(root))
    if (root / "FINAL_EXPERIMENT_MANIFEST.json").exists():
        errors.append("FINAL_EXPERIMENT_MANIFEST.json must not exist")
    checks = {
        "snapshot_schema": not (snapshot.get("schema_version") != 1),
        "ci_exact_completed_success": not validate_ci_evidence(ci, expected_head=snapshot.get("audited_code_commit")),
        "cross_file_values": not cross_file_errors(snapshot, ci, state, setup, runtime, preexperiment, review, inventory_count, local_closure, authoritative_review),
        "authoritative_cleanup_review": authoritative_review.get("valid") is True,
        "review_source_precedence": review.get("source") == REVIEW_SOURCE_PATHS[0],
        "review_markdown_values": not _review_markdown_errors(root, {
            "source": REVIEW_SOURCE_PATHS[0],
            "execution_mode": "SINGLE_AGENT",
            "subagents_called": False,
            "no_new_defects": True,
            "historical_subagent_profile_verification": "NOT_VERIFIED",
        }),
        "protected_source_unchanged": manifest.get("status") == "PASS" and manifest.get("after_manifest", {}).get("manifest_sha256") == current_manifest["manifest_sha256"],
        "historical_luna_not_verified": read_json(root / "reports/luna_max_subagent_manifest.json", {}).get("actual_profile_verification") == "NOT_VERIFIED",
        "no_stale_current_status": not _stale_current_matches(root),
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "repository": REPOSITORY,
        "branch": BRANCH,
        "current_head": git_sha(root),
        "checks": checks,
        "errors": sorted(set(errors)),
        "snapshot": snapshot,
        "ci_verification": ci,
        "review": review,
        "authoritative_review": authoritative_review,
        "review_source_precedence": list(REVIEW_SOURCE_PATHS),
        "inventory_rows": inventory_count,
        "runtime_blockers": RUNTIME_BLOCKERS,
        "next_action": NEXT_ACTION,
        "execution_boundary": {"phase15_executed": False, "model_downloaded": False, "azure_live_called": False, "gpu_training_executed": False, "real_predictions_generated": False, "production_proof": False},
    }
    if write_report:
        atomic_write_json(root / "reports/final_readiness_consistency_audit.json", report)
        atomic_write_text(root / "reports/final_readiness_consistency_audit.md", "\n".join([
            "# Final readiness consistency audit",
            "",
            f"- Status: `{report['status']}`",
            f"- Current HEAD: `{report['current_head']}`",
            f"- Audited code commit: `{snapshot.get('audited_code_commit', 'UNVERIFIED_EXTERNAL')}`",
            f"- CI: `{ci.get('status', 'UNVERIFIED_EXTERNAL')}/{ci.get('conclusion', 'UNVERIFIED_EXTERNAL')}` run `{ci.get('run_id', 0)}`",
            f"- Review: `{review.get('cycles', 0)} cycles x {review.get('rounds_per_cycle', 0)} rounds`; clean `{review.get('consecutive_clean_cycles', 0)}`",
            f"- Review source: `{review.get('source')}`",
            f"- Execution mode/subagents: `{review.get('execution_mode')}/{str(review.get('subagents_called')).lower()}`",
            f"- Historical profile verification: `{review.get('historical_subagent_profile_verification')}`",
            f"- Inventory: `{inventory_count}`",
            f"- Protected source unchanged: `{str(checks['protected_source_unchanged']).lower()}`",
            "",
            "## Checks",
            "",
            *[f"- {key}: `{str(value).lower()}`" for key, value in checks.items()],
            "",
            "## Errors",
            "",
            *([f"- {error}" for error in report["errors"]] or ["- None"]),
            "",
        ]))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit final readiness cross-file consistency")
    parser.add_argument("--ci-self-check", action="store_true", help="allow the CI job to run before external completed-run evidence exists")
    args = parser.parse_args()
    report = audit(ROOT, write_report=True)
    if args.ci_self_check and not (ROOT / "reports/ci_verification.json").exists():
        return 0
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
