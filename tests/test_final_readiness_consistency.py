from __future__ import annotations

from copy import deepcopy

import pytest
from scripts.audit_final_readiness_consistency import _stale_current_matches, cross_file_errors
from scripts.readiness_utils import NEXT_ACTION, REPOSITORY, RUNTIME_BLOCKERS, validate_ci_evidence

CODE_SHA = "a" * 40


def _fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    ci = {
        "repository": REPOSITORY,
        "branch": "codex/phase-14-5-production-repair",
        "workflow": "cpu-ci",
        "run_id": 1,
        "run_number": 1,
        "head_sha": CODE_SHA,
        "status": "completed",
        "conclusion": "success",
        "verification_source": "github_api",
    }
    review = {"cycles": 2, "rounds_per_cycle": 5, "consecutive_clean_cycles": 2, "status": "PASS", "subagent_profile_verification": "NOT_VERIFIED; see manifest routing limitation"}
    snapshot = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "branch": "codex/phase-14-5-production-repair",
        "audited_code_commit": CODE_SHA,
        "report_generation_parent_sha": CODE_SHA,
        "report_only_commit_expected": True,
        "review": review,
        "scientific": {"scientific_protocol_conflicts": [], "inventory_rows": 162},
        "implementation": {"status": "PASS", "implementation_blockers": []},
        "runtime": {"LOCAL_CODE_READINESS": "PASS", "SERVER_RUNTIME_READINESS": "NOT_RUN", "REAL_EXPERIMENT_READINESS": False, "final_aggregation_ready": False, "real_run_count": 0, "approved_run_count": 0, "runtime_blockers": RUNTIME_BLOCKERS},
    }
    state = {"setup_implementation_ready": True, "setup_frozen": True, "phase15_code_ready": True, "sequential_runtime_code_ready": True, "full_matrix_code_ready": True, "phase15_runtime_ready": False, "runtime_environment_ready": False, "weights_downloaded": False, "real_experiment_ready": False, "final_aggregation_ready": False, "real_run_count": 0, "approved_run_count": 0, "implementation_blockers": [], "scientific_protocol_conflicts": [], "runtime_blockers": RUNTIME_BLOCKERS, "next_action": NEXT_ACTION}
    setup = {"SETUP_IMPLEMENTATION_READY": True, "SETUP_FROZEN": True, "PHASE15_CODE_READY": True, "SEQUENTIAL_RUNTIME_CODE_READY": True, "FULL_MATRIX_CODE_READY": True, "PHASE15_RUNTIME_READY": False, "RUNTIME_ENVIRONMENT_READY": False, "WEIGHTS_DOWNLOADED": False, "REAL_EXPERIMENT_READY": False, "FINAL_AGGREGATION_READY": False, "REAL_RUN_COUNT": 0, "APPROVED_RUN_COUNT": 0, "runtime_blockers": RUNTIME_BLOCKERS, "next_action": NEXT_ACTION}
    runtime = {"audited_code_commit": CODE_SHA, "ci_verified_head_sha": CODE_SHA, "ci_conclusion": "success", "review_summary": review, "runtime_blockers": RUNTIME_BLOCKERS}
    preexperiment = deepcopy(runtime)
    local = {"production_proof": False, "synthetic_results_enter_production_aggregation": False}
    return snapshot, ci, state, setup, runtime, preexperiment, review, local


def _errors(mutator) -> list[str]:
    values = _fixture()
    mutator(values)
    snapshot, ci, state, setup, runtime, preexperiment, review, local = values
    return cross_file_errors(snapshot, ci, state, setup, runtime, preexperiment, review, 162, local)


@pytest.mark.parametrize(
    "mutator,needle",
    [
        (lambda v: v[1].update({"head_sha": "b" * 40}), "CI evidence"),
        (lambda v: v[2].update({"phase15_runtime_ready": True}), "PROJECT_STATE mismatch"),
        (lambda v: v[3].update({"FINAL_AGGREGATION_READY": True}), "SETUP_READY mismatch"),
        (lambda v: v[5].update({"review_summary": {"consecutive_clean_cycles": 0}}), "preexperiment review summary mismatch"),
        (lambda v: v[6].update({"cycles": 0}), "review mismatch"),
        (lambda v: v[0]["scientific"].update({"inventory_rows": 161}), "inventory count"),
        (lambda v: v[0]["scientific"].update({"scientific_protocol_conflicts": ["unexpected"]}), "scientific conflicts"),
        (lambda v: v[0]["implementation"].update({"implementation_blockers": ["unexpected"]}), "implementation blockers"),
        (lambda v: v[7].update({"production_proof": True}), "synthetic evidence"),
        (lambda v: v[0]["runtime"].update({"REAL_EXPERIMENT_READINESS": True}), "snapshot code/runtime"),
        (lambda v: v[0].update({"report_generation_parent_sha": "c" * 40}), "report generation parent"),
        (lambda v: v[4].update({"audited_code_commit": "d" * 40}), "runtime audited code SHA"),
    ],
)
def test_cross_file_audit_rejects_adversarial_metadata(mutator, needle: str) -> None:
    assert any(needle in error for error in _errors(mutator))


def test_ci_validation_rejects_in_progress_and_wrong_sha() -> None:
    snapshot, ci, *_ = _fixture()
    assert validate_ci_evidence(ci | {"head_sha": "b" * 40}, expected_head=snapshot["audited_code_commit"])
    assert validate_ci_evidence(ci | {"status": "in_progress"}, expected_head=CODE_SHA)


def test_stale_current_status_scan_rejects_json_markdown_disagreement(tmp_path) -> None:
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/final_runtime_integration_audit.md").write_text("CI status: `NOT_RUN`\n", encoding="utf-8")
    assert _stale_current_matches(tmp_path)
