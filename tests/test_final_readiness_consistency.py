from __future__ import annotations

import json
from copy import deepcopy

import pytest
from scripts.audit_final_readiness_consistency import (
    _review_markdown_errors,
    _stale_current_matches,
    cross_file_errors,
)
from scripts.readiness_utils import (
    NEXT_ACTION,
    REPOSITORY,
    RUNTIME_BLOCKERS,
    load_review,
    normalize_review,
    snapshot_markdown,
    validate_ci_evidence,
)

CODE_SHA = "a" * 40


def _cleanup_review() -> dict[str, object]:
    return {
        "schema_version": 1,
        "execution_mode": "SINGLE_AGENT",
        "subagents_called": False,
        "status": "PASS",
        "cycles": [{"cycle": 1}, {"cycle": 2}],
        "rounds_per_cycle": 6,
        "consecutive_clean_cycles": 2,
        "no_new_defects_in_two_complete_cycles": True,
    }


def _luna_review() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "PASS",
        "cycle_count": 2,
        "rounds_per_cycle": 5,
        "consecutive_clean_cycles": 2,
        "no_new_defects_in_two_complete_cycles": True,
        "profile_resolution": "NOT_VERIFIED; historical routing limitation",
    }


def _runtime_review() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "PASS",
        "sequence_count": 2,
        "completed_rounds_per_sequence": 25,
        "consecutive_clean_sequences": 2,
        "no_new_defects_in_two_complete_cycles": True,
    }


def _write_review(root, relative: str, payload: object) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
    review = {
        "source": "reports/final_cleanup_review_cycles.json",
        "execution_mode": "SINGLE_AGENT",
        "subagents_called": False,
        "cycles": 2,
        "rounds_per_cycle": 6,
        "consecutive_clean_cycles": 2,
        "status": "PASS",
        "no_new_defects": True,
        "historical_subagent_profile_verification": "NOT_VERIFIED",
        "subagent_profile_verification": "NOT_VERIFIED",
        "valid": True,
        "normalization_errors": [],
    }
    snapshot = {
        "schema_version": 1,
        "repository": REPOSITORY,
        "branch": "codex/phase-14-5-production-repair",
        "audited_code_commit": CODE_SHA,
        "report_generation_parent_sha": CODE_SHA,
        "report_only_commit_expected": True,
        "review": deepcopy(review),
        "scientific": {"scientific_protocol_conflicts": [], "inventory_rows": 162},
        "implementation": {"status": "PASS", "implementation_blockers": []},
        "runtime": {
            "LOCAL_CODE_READINESS": "PASS",
            "SERVER_RUNTIME_READINESS": "NOT_RUN",
            "REAL_EXPERIMENT_READINESS": False,
            "final_aggregation_ready": False,
            "real_run_count": 0,
            "approved_run_count": 0,
            "runtime_blockers": RUNTIME_BLOCKERS,
        },
    }
    state = {
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
        "runtime_blockers": RUNTIME_BLOCKERS,
        "next_action": NEXT_ACTION,
    }
    setup = {
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
        "runtime_blockers": RUNTIME_BLOCKERS,
        "next_action": NEXT_ACTION,
    }
    runtime = {
        "audited_code_commit": CODE_SHA,
        "ci_verified_head_sha": CODE_SHA,
        "ci_conclusion": "success",
        "review_summary": deepcopy(review),
        "self_review_summary": deepcopy(review),
        "self_review": deepcopy(review),
        "runtime_blockers": RUNTIME_BLOCKERS,
    }
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
        (lambda v: v[5].update({"review_summary": {"consecutive_clean_cycles": 0}}), "preexperiment review_summary mismatch"),
        (lambda v: v[6].update({"cycles": 0}), "review mismatch"),
        (lambda v: v[0]["scientific"].update({"inventory_rows": 161}), "inventory count"),
        (lambda v: v[0]["scientific"].update({"scientific_protocol_conflicts": ["unexpected"]}), "scientific conflicts"),
        (lambda v: v[0]["implementation"].update({"implementation_blockers": ["unexpected"]}), "implementation blockers"),
        (lambda v: v[7].update({"production_proof": True}), "synthetic evidence"),
        (lambda v: v[0]["runtime"].update({"REAL_EXPERIMENT_READINESS": True}), "snapshot code/runtime"),
        (lambda v: v[0].update({"report_generation_parent_sha": "c" * 40}), "report generation parent"),
        (lambda v: v[4].update({"audited_code_commit": "d" * 40}), "runtime audited code SHA"),
        (lambda v: v[0]["review"].update({"rounds_per_cycle": 5}), "review mismatch"),
        (lambda v: v[0]["review"].update({"source": "reports/luna_max_review_cycles.json"}), "review mismatch"),
        (lambda v: v[4]["review_summary"].update({"execution_mode": "WORKER"}), "runtime review_summary mismatch"),
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


def test_cleanup_review_takes_precedence_over_luna_fallback(tmp_path) -> None:
    _write_review(tmp_path, "reports/final_cleanup_review_cycles.json", _cleanup_review())
    _write_review(tmp_path, "reports/luna_max_review_cycles.json", _luna_review())
    review = load_review(tmp_path)
    assert review["source"] == "reports/final_cleanup_review_cycles.json"
    assert review["rounds_per_cycle"] == 6


def test_luna_review_is_used_when_cleanup_is_missing(tmp_path) -> None:
    _write_review(tmp_path, "reports/luna_max_review_cycles.json", _luna_review())
    review = load_review(tmp_path)
    assert review["source"] == "reports/luna_max_review_cycles.json"
    assert review["rounds_per_cycle"] == 5
    assert review["historical_subagent_profile_verification"] == "NOT_VERIFIED"


def test_runtime_review_is_used_when_other_sources_are_missing(tmp_path) -> None:
    _write_review(tmp_path, "reports/runtime_self_review.json", _runtime_review())
    review = load_review(tmp_path)
    assert review["source"] == "reports/runtime_self_review.json"
    assert review["rounds_per_cycle"] == 25


def test_invalid_cleanup_review_does_not_silently_pass(tmp_path) -> None:
    _write_review(tmp_path, "reports/final_cleanup_review_cycles.json", {"status": "PASS", "rounds_per_cycle": 6})
    review = load_review(tmp_path)
    assert review["status"] == "FAIL"
    assert review["valid"] is False
    assert review["normalization_errors"]


def test_invalid_cleanup_falls_back_only_to_valid_luna_source(tmp_path) -> None:
    _write_review(tmp_path, "reports/final_cleanup_review_cycles.json", {"status": "PASS", "rounds_per_cycle": 6})
    _write_review(tmp_path, "reports/luna_max_review_cycles.json", _luna_review())
    review = load_review(tmp_path)
    assert review["source"] == "reports/luna_max_review_cycles.json"
    assert review["valid"] is True
    assert review["source_selection_errors"]


def test_no_review_source_is_a_failing_review(tmp_path) -> None:
    review = load_review(tmp_path)
    assert review["source"] is None
    assert review["status"] == "FAIL"
    assert review["valid"] is False


def test_normalization_preserves_single_agent_and_six_round_metadata() -> None:
    review = normalize_review(_cleanup_review(), source="reports/final_cleanup_review_cycles.json")
    assert review["execution_mode"] == "SINGLE_AGENT"
    assert review["subagents_called"] is False
    assert review["rounds_per_cycle"] == 6
    assert review["cycles"] == 2
    assert review["consecutive_clean_cycles"] == 2
    assert review["no_new_defects"] is True
    assert review["valid"] is True


def test_normalization_rejects_string_boolean_instead_of_coercing() -> None:
    payload = _cleanup_review() | {"subagents_called": "false", "no_new_defects_in_two_complete_cycles": "true"}
    review = normalize_review(payload)
    assert review["valid"] is False
    assert "invalid review field: subagents_called" in review["normalization_errors"]
    assert "missing or invalid review field: no_new_defects" in review["normalization_errors"]


def test_snapshot_markdown_persists_authoritative_review_source_and_values() -> None:
    review = normalize_review(_cleanup_review(), source="reports/final_cleanup_review_cycles.json")
    snapshot = {
        "branch": "codex/phase-14-5-production-repair",
        "branch_head_before_refresh": CODE_SHA,
        "audited_code_commit": CODE_SHA,
        "report_generation_parent_sha": CODE_SHA,
        "audited_source_manifest_sha256": "manifest",
        "report_only_commit_expected": True,
        "ci": {},
        "review": review,
        "scientific": {"inventory_rows": 162, "scientific_protocol_conflicts": []},
        "runtime": {"LOCAL_CODE_READINESS": "PASS", "SERVER_RUNTIME_READINESS": "NOT_RUN", "REAL_EXPERIMENT_READINESS": False, "runtime_blockers": []},
        "implementation": {"implementation_blockers": []},
        "next_action": NEXT_ACTION,
    }
    text = snapshot_markdown(snapshot)
    assert "- Review source: `reports/final_cleanup_review_cycles.json`" in text
    assert "- Execution mode: `SINGLE_AGENT`" in text
    assert "- Subagents called: `false`" in text
    assert "- Rounds per cycle: `6`" in text
    assert "- No new defects: `true`" in text
    assert "- Historical subagent profile verification: `NOT_VERIFIED`" in text


def test_runtime_and_preexperiment_summaries_require_six_rounds() -> None:
    assert not _errors(lambda v: None)
    values = _fixture()
    values[4]["review_summary"]["rounds_per_cycle"] = 5
    snapshot, ci, state, setup, runtime, preexperiment, review, local = values
    errors = cross_file_errors(snapshot, ci, state, setup, runtime, preexperiment, review, 162, local)
    assert any("runtime review_summary mismatch: rounds_per_cycle" in error for error in errors)


def test_consistency_audit_rejects_wrong_selected_source() -> None:
    errors = _errors(lambda v: v[0]["review"].update({"source": "reports/luna_max_review_cycles.json"}))
    assert any("review mismatch: source" in error for error in errors)


def test_historical_profile_verification_remains_not_verified() -> None:
    review = normalize_review(_luna_review(), source="reports/luna_max_review_cycles.json")
    assert review["historical_subagent_profile_verification"] == "NOT_VERIFIED"
    assert review["subagent_profile_verification"] == "NOT_VERIFIED"


def test_review_markdown_audit_rejects_five_round_or_wrong_source(tmp_path) -> None:
    expected = {
        "source": "reports/final_cleanup_review_cycles.json",
        "execution_mode": "SINGLE_AGENT",
        "subagents_called": False,
        "no_new_defects": True,
        "historical_subagent_profile_verification": "NOT_VERIFIED",
    }
    lines = [
        "- Review source: `reports/final_cleanup_review_cycles.json`",
        "- Execution mode: `SINGLE_AGENT`",
        "- Subagents called: `false`",
        "- No new defects: `true`",
        "- Historical subagent profile verification: `NOT_VERIFIED`",
    ]
    for relative in (
        "reports/final_readiness_snapshot.md",
        "reports/final_runtime_integration_audit.md",
        "reports/final_preexperiment_closure.md",
        "reports/final_production_correctness_repair.md",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert not _review_markdown_errors(tmp_path, expected)
    (tmp_path / "reports/final_runtime_integration_audit.md").write_text(lines[0].replace("final_cleanup", "luna_max") + "\n".join(lines[1:]) + "\n", encoding="utf-8")
    assert _review_markdown_errors(tmp_path, expected)
