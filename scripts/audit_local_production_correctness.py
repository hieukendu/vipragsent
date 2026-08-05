from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.hashing import sha256_file, sha256_json
from vipragsent.orchestration.q1b_dependencies import write_q1b_dependency_report

DEFECT_TESTS: tuple[tuple[str, str], ...] = (
    ("Defect 1", "tests/test_component_production_runner.py::test_component_runner_consumes_all_training_examples"),
    ("Defect 1", "tests/test_component_production_runner.py::test_component_runner_runs_locked_epochs_or_early_stops"),
    ("Defect 1", "tests/test_component_production_runner.py::test_component_runner_writes_real_checkpoint"),
    ("Defect 2", "tests/test_luna_max_01_generation.py::test_cot_factory_uses_causal_lm_loader"),
    ("Defect 2", "tests/test_luna_max_01_generation.py::test_cot_model_has_generate"),
    ("Defect 3", "tests/test_luna_max_01_generation.py::test_generation_test_stage_loads_frozen_best_checkpoint"),
    ("Defect 3", "tests/test_luna_max_08_red_team.py::test_red_team_generation_final_test_requires_frozen_checkpoint_and_reloads_best"),
    ("Defect 4", "tests/test_checkpoint_device_contract.py::test_checkpoint_v2_round_trip"),
    ("Defect 4", "tests/test_checkpoint_device_contract.py::test_checkpoint_zero_matching_keys_fails"),
    ("Defect 4", "tests/test_checkpoint_device_contract.py::test_checkpoint_load_report_written"),
    ("Defect 4", "tests/test_luna_max_08_red_team.py::test_red_team_q1b_loader_rejects_zero_matching_keys"),
    ("Defect 5", "tests/test_checkpoint_device_contract.py::test_custom_executor_moves_batch_to_model_device"),
    ("Defect 5", "tests/test_checkpoint_device_contract.py::test_custom_executor_device_report"),
    ("Defect 5", "tests/test_luna_max_08_red_team.py::test_red_team_custom_q1b_predictor_moves_inputs_and_rejects_zero_matching_checkpoint"),
    ("Defect 6", "tests/test_q1b_dependencies.py::test_q1b_every_consumer_has_exact_producer"),
    ("Defect 6", "tests/test_q1b_dependencies.py::test_q1b_dependency_graph_is_acyclic"),
    ("Defect 6", "tests/test_q1b_dependencies.py::test_q1b_ordinary_single_task_same_seed_composition"),
    ("Defect 6", "tests/test_luna_max_08_red_team.py::test_red_team_q1b_inventory_has_real_trainable_producers_for_every_consumer"),
    ("Defect 7", "tests/test_table2_statistics.py::test_table2_uses_joint_hierarchical_interval"),
    ("Defect 7", "tests/test_table2_statistics.py::test_table2_does_not_average_seed_bounds"),
    ("Defect 7", "tests/test_table2_statistics.py::test_table2_ci_golden_counterexample"),
    ("Defect 7", "tests/test_luna_max_08_red_team.py::test_red_team_table2_generation_uses_joint_ci_and_all_zero_fallback"),
    ("Defect 8", "tests/test_azure.py::test_public_client_parses_nested_responses_payload_and_caches"),
    ("Defect 8", "tests/test_azure.py::test_public_client_retries_payload_status_and_uses_retry_after"),
    ("Defect 8", "tests/test_azure.py::test_public_client_caches_terminal_invalid_response_without_retry"),
    ("Defect 8", "tests/test_preexperiment_closure.py::test_reasoning_judge_is_reasoning_only_strict_cached_and_transport_retrying"),
    ("Defect 9", "tests/test_provenance_artifacts.py::test_explanation_manifest_truthful_rationale_inference"),
    ("Defect 9", "tests/test_provenance_artifacts.py::test_explanation_validator_accepts_truthful_provenance"),
    ("Defect 9", "tests/test_provenance_artifacts.py::test_cot_manifest_marks_native_causal_generation"),
    ("Defect 9", "tests/test_provenance_artifacts.py::test_generation_provenance_system_specific"),
    ("Defect 10", "tests/test_final_production_repair.py::test_synthetic_full_sequential_run_is_review_gated_and_hash_valid"),
    ("Defect 10", "tests/test_final_runtime_integration.py::test_generation_executor_trains_causally_and_records_invalid_parser_rows"),
)

CLOSURE_TESTS: tuple[tuple[str, str], ...] = (
    ("ordinary_classifier_training", "tests/test_training_engine.py::test_training_engine_selects_dev_checkpoint_and_freezes_thresholds"),
    ("six_and_eight_component_bundles", "tests/test_preexperiment_closure.py::test_component_bundle_production_shape_covers_six_eight_split_alignment_and_resume"),
    ("cot_only_generation", "tests/test_preexperiment_closure.py::test_cot_executor_trains_selects_on_dev_and_seals_test_until_after_selection"),
    ("explanation_only_reuse", "tests/test_preexperiment_closure.py::test_explanation_executor_reuses_source_and_uses_only_rationale_decoder"),
    ("q1b_polarity", "tests/test_q1b_dependencies.py::test_q1b_no_fake_non_applicable_predictions"),
    ("q1b_emotion_and_multitask", "tests/test_q1b_dependencies.py::test_q1b_every_consumer_has_exact_producer"),
    ("table2_three_seeds", "tests/test_table2_statistics.py::test_table2_ci_prediction_alignment"),
    ("reasoning_judge_retry_cache", "tests/test_preexperiment_closure.py::test_reasoning_judge_is_reasoning_only_strict_cached_and_transport_retrying"),
    ("review_gate_and_artifacts", "tests/test_final_production_repair.py::test_synthetic_full_sequential_run_is_review_gated_and_hash_valid"),
)


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _source_hash(test_name: str) -> str:
    relative = test_name.split("::", 1)[0]
    path = ROOT / relative
    return sha256_file(path)


def _run_test(test_name: str, *, scope: str, label: str) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", test_name, "--disable-warnings", "--maxfail=1"]
    source_hash = _source_hash(test_name)
    fixture_input = sha256_json({"scope": scope, "label": label, "test_name": test_name, "test_source_sha256": source_hash, "synthetic": True})
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240, check=False)
        stdout = result.stdout[-4000:]
        stderr = result.stderr[-4000:]
        return {
            "scope": scope,
            "label": label,
            "test_name": test_name,
            "command": command,
            "fixture_input_sha256": fixture_input,
            "golden_input_sha256": fixture_input if scope == "table2" else None,
            "test_source_sha256": source_hash,
            "observed_output_sha256": sha256_json({"returncode": result.returncode, "stdout": stdout, "stderr": stderr}),
            "returncode": result.returncode,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "synthetic": True,
            "production_proof": False,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "scope": scope,
            "label": label,
            "test_name": test_name,
            "command": command,
            "fixture_input_sha256": fixture_input,
            "golden_input_sha256": fixture_input if scope == "table2" else None,
            "test_source_sha256": source_hash,
            "observed_output_sha256": sha256_json({"error": str(exc)}),
            "returncode": 1,
            "status": "FAIL",
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "synthetic": True,
            "production_proof": False,
        }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local production correctness closure",
        "",
        f"Status: `{report['status']}`",
        f"Code SHA: `{report['code_commit_at_audit']}`",
        "",
        "This report is CPU-only, network-free, Azure-live-free, model-download-free synthetic evidence. It is not production proof.",
        "",
        "## Defect evidence",
        "",
        "| Defect | Test | Input hash | Output hash | Status |",
        "|---|---|---|---|---|",
    ]
    for item in report["evidence"]:
        lines.append(f"| {item.get('defect', item.get('label'))} | `{item['test_name']}` | `{item['fixture_input_sha256']}` | `{item['observed_output_sha256']}` | `{item['status']}` |")
    lines.extend([
        "",
        "## Synthetic closure",
        "",
        "The named ordinary, component-bundle, generation, explanation-reuse, Q1b, Table 2, judge, and review-gate cases were executed as temporary-directory CPU tests.",
        "",
        f"`RUN_STATUS={report['RUN_STATUS']}`",
        f"`USER_REVIEW_STATUS={report['USER_REVIEW_STATUS']}`",
        f"`NEXT_RUN_ALLOWED={report['NEXT_RUN_ALLOWED']}`",
        "",
        "## Safety boundary",
        "",
        "No Phase 15, model download, live Azure request, GPU training, real test prediction, approval, or production aggregation was executed.",
        "",
    ])
    return "\n".join(lines)


def _write_area_reports(evidence: list[dict[str, Any]], code_sha: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        grouped.setdefault(str(item.get("defect", "")), []).append(item)
    def status(names: tuple[str, ...]) -> str:
        return "PASS" if all(item["status"] == "PASS" for name in names for item in grouped.get(name, [])) else "FAIL"

    reports = {
        "reports/checkpoint_schema_audit.json": {
            "schema_version": 1,
            "status": status(("Defect 4",)),
            "code_commit_at_audit": code_sha,
            "checkpoint_schema_version": 2,
            "canonical_keys": ["schema_version", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "loss_aggregator_state_dict", "run_state", "rng_state", "metadata"],
            "round_trip_evidence": grouped.get("Defect 4", []),
            "missing_unexpected_key_policy": "strict preflight; only explicitly allow-listed keys may differ",
            "zero_match_policy": "BLOCKED",
            "legacy_fixture_compatibility": "explicit allow_legacy_fixture=True only",
            "production_proof": False,
        },
        "reports/device_contract_audit.json": {
            "schema_version": 1,
            "status": status(("Defect 5",)),
            "code_commit_at_audit": code_sha,
            "device_policy": "one selected device; batches move to model input device; no implicit sharding",
            "first_batch_device_report_required": True,
            "evidence": grouped.get("Defect 5", []),
            "production_proof": False,
        },
        "reports/component_training_audit.json": {
            "schema_version": 1,
            "status": status(("Defect 1",)),
            "code_commit_at_audit": code_sha,
            "full_train_split": True,
            "locked_optimization": True,
            "real_checkpoint_paths": True,
            "resume_hashes": True,
            "evidence": grouped.get("Defect 1", []),
            "production_proof": False,
        },
        "reports/reasoning_judge_contract_audit.json": {
            "schema_version": 1,
            "status": status(("Defect 8",)),
            "code_commit_at_audit": code_sha,
            "public_structured_client": True,
            "original_sentence_visible": False,
            "live_calls": False,
            "cache_retry_evidence": grouped.get("Defect 8", []),
            "production_proof": False,
        },
        "reports/generation_causal_lm_audit.json": {
            "schema_version": 1,
            "status": status(("Defect 2", "Defect 3")),
            "code_commit_at_audit": code_sha,
            "loader": "AutoModelForCausalLM",
            "native_generate": True,
            "checkpoint_reload_and_freeze": True,
            "evidence": grouped.get("Defect 2", []) + grouped.get("Defect 3", []),
            "production_proof": False,
        },
        "reports/provenance_truthfulness_audit.json": {
            "schema_version": 1,
            "status": status(("Defect 9", "Defect 10")),
            "code_commit_at_audit": code_sha,
            "explanation_only_source": "judge_of_rationale_decoder_output",
            "cot_only_source": "judge_of_generated_reasoning",
            "synthetic_closure_evidence": grouped.get("Defect 9", []) + grouped.get("Defect 10", []),
            "production_proof": False,
        },
    }
    for relative, report in reports.items():
        atomic_write_json(ROOT / relative, report)
    graph = write_q1b_dependency_report(ROOT)
    reports["reports/q1b_dependency_graph.json"] = graph
    return reports


def _write_provisional(report: dict[str, Any]) -> None:
    atomic_write_json(ROOT / "reports/local_production_correctness_closure.json", report)
    atomic_write_text(ROOT / "reports/local_production_correctness_closure.md", _markdown(report))


def main() -> int:
    code_sha = _git_sha()
    evidence: list[dict[str, Any]] = []
    for defect, test_name in DEFECT_TESTS:
        item = _run_test(test_name, scope="defect", label=defect)
        item["defect"] = defect
        evidence.append(item)
    closure: list[dict[str, Any]] = []
    for label, test_name in CLOSURE_TESTS:
        item = _run_test(test_name, scope="closure", label=label)
        item["defect"] = "synthetic_closure"
        closure.append(item)
    evidence.extend(closure)
    all_pass = all(item["status"] == "PASS" for item in evidence)
    base_report: dict[str, Any] = {
        "schema_version": 2,
        "status": "PASS" if all_pass else "FAIL",
        "code_commit_at_audit": code_sha,
        "evidence_scope": "all ten production defects plus named production-shaped synthetic closure",
        "defects_covered": [f"Defect {index}" for index in range(1, 11)],
        "evidence": evidence,
        "production_shaped_synthetic_closure": closure,
        "production_proof": False,
        "synthetic_results_enter_production_aggregation": False,
        "RUN_STATUS": "PASS" if all_pass else "FAIL",
        "USER_REVIEW_STATUS": "PENDING",
        "NEXT_RUN_ALLOWED": "NO",
        "execution_safety": {
            "phase15_executed": False,
            "model_downloaded": False,
            "azure_request_made": False,
            "gpu_training_executed": False,
            "real_predictions_generated": False,
            "approval_recorded": False,
            "experiment_started": False,
            "production_aggregation_received_synthetic_results": False,
        },
    }
    _write_area_reports(evidence, code_sha)
    _write_provisional(base_report)

    audit_test = _run_test(
        "tests/test_luna_max_08_red_team.py::test_red_team_audits_bind_executable_test_and_observed_hashes",
        scope="defect",
        label="Defect 10 audit evidence schema",
    )
    audit_test["defect"] = "Defect 10"
    evidence.append(audit_test)
    all_pass = all(item["status"] == "PASS" for item in evidence)
    base_report["status"] = "PASS" if all_pass else "FAIL"
    base_report["RUN_STATUS"] = "PASS" if all_pass else "FAIL"
    base_report["evidence"] = evidence
    base_report["production_shaped_synthetic_closure"] = closure
    base_report["code_commit_at_audit"] = _git_sha()
    _write_provisional(base_report)
    provenance = json.loads((ROOT / "reports/provenance_truthfulness_audit.json").read_text(encoding="utf-8"))
    provenance["status"] = "PASS" if all(item["status"] == "PASS" for item in evidence if item.get("defect") in {"Defect 9", "Defect 10"}) else "FAIL"
    provenance["code_commit_at_audit"] = base_report["code_commit_at_audit"]
    provenance["executable_closure_evidence"] = [item for item in evidence if item.get("defect") in {"Defect 9", "Defect 10"}]
    atomic_write_json(ROOT / "reports/provenance_truthfulness_audit.json", provenance)
    print(json.dumps(base_report, indent=2, ensure_ascii=False))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
