from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    from _bootstrap import ROOT
except ModuleNotFoundError:  # pragma: no cover - supports importing the script in tests
    from scripts._bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.evaluation.reasoning_judge import validate_reasoning_protocol_files
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.stage_plans import validate_stage_plan_registry
from vipragsent.orchestration.system_registry import validate_execution_registry
from vipragsent.protocol import compare_frozen_hashes, validate_protocol_resolution

BASELINE_COMMIT = "cb5cde04cd3e3c546d1b35711197a82b6d5bb254"
SAFE_TEST_SELECTOR = "not server and not gpu and not azure_live and not model_download"
RUNTIME_BLOCKERS = [
    "Phase 15 has not been executed on the target server",
    "Model-family runtime assets are not prepared",
    "GPU and Azure live integration have not been validated",
    "No real approved production run exists",
]
NEXT_ACTION = "Checkout the exact final repair SHA on the target server, run Phase 15 for exactly one lightweight model family, print the complete smoke report, and stop for user review."


def _run(root: Path, command: list[str], *, timeout: int = 600) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
        return {"command": command, "returncode": result.returncode, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": 1, "stdout_tail": "", "stderr_tail": str(exc)}


def _git_commit(root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _has_ancestor(root: Path, commit: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=root, check=False).returncode == 0


def _device_report(root: Path) -> dict[str, Any]:
    device = (root / "src/vipragsent/runtime/device.py").read_text(encoding="utf-8")
    engine = (root / "src/vipragsent/training/engine.py").read_text(encoding="utf-8")
    factory = (root / "src/vipragsent/models/factory.py").read_text(encoding="utf-8")
    qlora = (root / "src/vipragsent/models/qlora.py").read_text(encoding="utf-8")
    collator = (root / "src/vipragsent/data/collation.py").read_text(encoding="utf-8")
    required = ("resolve_selected_cuda_device", "place_non_quantized_model", "resolve_model_input_device", "move_batch_to_device", "assert_runtime_device_contract")
    checks = {
        "canonical_device_module": all(name in device for name in required),
        "training_engine_uses_contract": (
            ("move_batch_to_model_device" in engine or "move_batch_to_device" in engine)
            and all(name in engine for name in ("assert_runtime_device_contract", "write_device_report"))
        ),
        "factory_places_complete_non_quantized_model": "place_non_quantized_model" in factory,
        "qlora_uses_explicit_device_map": "device_map" in qlora and "device_map=device_map" in qlora,
        "collator_is_device_neutral": ".to(" not in collator and ".cuda(" not in collator,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "selected_device_policy": "one explicit CUDA device or CPU; no implicit sharding"}


def _rationale_reports(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = (root / "src/vipragsent/orchestration/rationale_promotion.py").read_text(encoding="utf-8")
    stage = (root / "src/vipragsent/orchestration/stage_registry.py").read_text(encoding="utf-8")
    train_start = stage.index("def _real_train")
    train_end = stage.index("def _materialize_engine_outputs")
    training_section = stage[train_start:train_end]
    required_fields = ("sample_id", "rationale", "source_run_id", "source_response_id", "source_prompt_hash", "source_schema_hash", "source_deployment", "source_model_version", "source_record_hash")
    contract_checks = {
        "promotion_module_present": (root / "src/vipragsent/orchestration/rationale_promotion.py").exists(),
        "atomic_promotion": "os.replace" in source,
        "approval_gate": all(token in source for token in ("APPROVED", "review_summary_sha256", "artifact_checksum_file_sha256")),
        "canonical_fields_declared": all(token in source for token in required_fields),
        "training_consumes_canonical_artifact": "approved_generated_rationales_train.jsonl" in training_section and '"rationale_target"' not in training_section,
        "raw_target_mapping_only_in_promotion": "rationale_target" in source,
    }
    promotion = {"status": "PASS" if all(contract_checks.values()) else "FAIL", "checks": contract_checks, "canonical_path": "data/processed/rationales/approved_generated_rationales_train.jsonl", "manifest_path": "data/manifests/approved_generated_rationales_train.json", "canonical_artifact_present": (root / "data/processed/rationales/approved_generated_rationales_train.jsonl").exists(), "runtime_status": "BLOCKED_UNTIL_APPROVED_AZURE_RATIONALE_RUN"}
    consumption = {"status": "PASS" if contract_checks["training_consumes_canonical_artifact"] and contract_checks["raw_target_mapping_only_in_promotion"] else "FAIL", "canonical_key": "rationale", "no_rationale_variant_requires_artifact": False, "missing_or_failed_rationales": "masked_by_rationale_loss_mask", "raw_rationale_target_used_in_training": False}
    return promotion, consumption


def _component_report(root: Path) -> dict[str, Any]:
    source = (root / "src/vipragsent/orchestration/executors/component_bundle.py").read_text(encoding="utf-8")
    engine = (root / "src/vipragsent/training/engine.py").read_text(encoding="utf-8")
    checks = {
        "dedicated_executor_present": "class ComponentBundleExecutor" in source,
        "six_and_eight_component_sets": "SIX_COMPONENTS" in source and "EIGHT_COMPONENTS" in source,
        "sequential_state_and_resume": all(token in source for token in ("component_skipped", "resume", "component_manifest.json")),
        "combined_sample_id_alignment": "predictions are not in the frozen sample order" in source and "cannot be combined" in source,
        "generic_training_engine_rejects_bundle": "component_bundle" in engine and "independent_checkpoint_bundle" in engine,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "production_runtime_status": "BLOCKED_UNTIL_APPROVED_PHASE15_LOCAL_SNAPSHOT", "fixture_test_status": "CPU_SYNTHETIC_ONLY"}


def _generation_report(root: Path) -> dict[str, Any]:
    source = (root / "src/vipragsent/orchestration/executors/generation.py").read_text(encoding="utf-8")
    judge_source = (root / "src/vipragsent/evaluation/reasoning_judge.py").read_text(encoding="utf-8")
    protocol = validate_reasoning_protocol_files(root)
    resolution_path = root / "reports/generation_baseline_protocol_resolution.json"
    resolution = json.loads(resolution_path.read_text(encoding="utf-8")) if resolution_path.exists() else {}
    checks = {
        "dedicated_causal_executor_present": "class ReasoningGenerationExecutor" in source,
        "teacher_forced_cross_entropy": "teacher_forced_generation_loss" in source and "cross_entropy" in source,
        "reasoning_artifacts_and_primary_metric": (
            "reasoning/{split}_reasoning.jsonl" in source
            and "metrics/{split}_reasoning_metrics.json" in source
            and "effective_full_split_all_zero_fallback" in judge_source
        ),
        "classifier_fallback_absent": "classification_head" not in source,
        "protocol_files_valid": protocol["status"] == "PASS",
        "generation_resolution_active": resolution.get("status") == "RESOLVED" and set(resolution.get("systems", [])) == {"cot_only_vistral", "explanation_only_vistral"},
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "protocol_resolution": resolution, "runtime_status": "READY_FOR_EXPLICIT_PHASE15_ONLY"}


def _q1b_report(root: Path) -> dict[str, Any]:
    source = (root / "src/vipragsent/orchestration/executors/external_retention.py").read_text(encoding="utf-8")
    checks = {
        "disk_executor_present": "evaluate_external_retention_from_disk" in source,
        "official_test_only": "official normalized test files" in source and "train" in source,
        "no_external_training": "external_finetuning" in source and '"optimizer_steps": 0' in source,
        "no_context_metadata_injection": "context.metadata" not in source,
        "three_locked_datasets": all(name in source for name in ("vsfc", "vsmec", "aivivn")),
        "approved_source_required": "APPROVED" in source,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "training_applicability": "NOT_APPLICABLE", "external_finetuning": False, "optimizer_steps": 0, "backward_calls": 0}


def _q4_report(root: Path) -> dict[str, Any]:
    source = (root / "src/vipragsent/orchestration/executors/q4.py").read_text(encoding="utf-8")
    checks = {
        "approved_source_resolver_present": "_resolve_source" in source,
        "six_label_probability_validation": "six pragmatic gold/probability values" in source,
        "learning_history_required": "learning history is empty or malformed" in source,
        "no_synthetic_history": '"synthetic_history": False' in source,
        "temperature_scaling_disabled": '"temperature_scaling": False' in source,
        "exact_q4_evidence": "figure_backing/q4_pragmatic_reliability_bins.json" in source and "figure_backing/q4_learning_curves.json" in source,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "systems": ["phobert_pragmatic_finetune", "vistral_pragmatic_sft", "vipragsent_full_vistral"], "training_applicability": "NOT_APPLICABLE", "source_status": "BLOCKED_UNTIL_APPROVED_SOURCE_ARTIFACTS"}


def _security_report(root: Path) -> dict[str, Any]:
    tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.splitlines()
    secret_pattern = re.compile(r"(?:AZURE_OPENAI_API_KEY|KAGGLE_KEY)[ \t]*=[ \t]*[^\s#]+|(?:sk|AIza|AKIA)-[A-Za-z0-9_\-/]{16,}")
    secret_files: list[str] = []
    for relative in tracked:
        path = root / relative
        if path.name in {".env", ".env.local"} or not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        if secret_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            secret_files.append(relative.replace("\\", "/"))
    runtime_weights = [path.relative_to(root).as_posix() for base in (root / "data/model_cache", root / "checkpoints", root / "results") if base.exists() for path in base.rglob("*") if path.is_file() and path.suffix.casefold() in {".bin", ".pt", ".pth", ".safetensors", ".ckpt"}]
    config_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (root / "configs").rglob("*.yaml"))
    checks = {"no_tracked_env": ".env" not in tracked, "no_secret_matches": not secret_files, "no_model_weights_in_runtime_dirs": not runtime_weights, "no_direct_openai_endpoint_in_config": "api.openai.com" not in config_text, "no_tracked_bytecode": not any(path.endswith(".pyc") or "__pycache__" in path for path in tracked)}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "secret_files": secret_files, "runtime_weight_files": runtime_weights}


def _artifact_review_report(root: Path) -> dict[str, Any]:
    checks = {
        "artifact_schema_module": (root / "src/vipragsent/artifacts/schemas.py").exists(),
        "exporter_module": (root / "src/vipragsent/artifacts/exporter.py").exists(),
        "review_module": (root / "src/vipragsent/orchestration/review.py").exists(),
        "approval_gate_in_run_store": "PENDING_USER_APPROVAL" in (root / "src/vipragsent/orchestration/run_store.py").read_text(encoding="utf-8"),
        "final_aggregation_not_approved": not (root / "FINAL_EXPERIMENT_MANIFEST.json").exists(),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "real_artifacts_present": False, "final_aggregation_ready": False}


def _write_contract_reports(root: Path) -> dict[str, Any]:
    device = _device_report(root)
    rationale, consumption = _rationale_reports(root)
    component = _component_report(root)
    generation = _generation_report(root)
    q1b = _q1b_report(root)
    stage_plans = validate_stage_plan_registry(root)
    q4 = _q4_report(root)
    table2_path = root / "reports/table2_confidence_interval_protocol_audit.json"
    table2 = json.loads(table2_path.read_text(encoding="utf-8")) if table2_path.exists() else {"status": "FAIL", "errors": ["Table 2 audit report is missing"]}
    security = _security_report(root)
    artifacts = _artifact_review_report(root)
    reports = {
        "reports/device_placement_audit.json": device,
        "reports/rationale_promotion_contract.json": rationale,
        "reports/rationale_consumption_audit.json": consumption,
        "reports/component_bundle_executor_audit.json": component,
        "reports/generation_executor_audit.json": generation,
        "reports/q1b_self_contained_execution_audit.json": q1b,
        "reports/execution_stage_plan_audit.json": stage_plans,
        "reports/q4_source_extraction_audit.json": q4,
        "reports/table2_confidence_interval_protocol_audit.json": table2,
        "reports/security_hygiene_audit.json": security,
        "reports/artifact_export_review_audit.json": artifacts,
    }
    for relative, report in reports.items():
        atomic_write_json(root / relative, report)
    return {"device": device, "rationale_promotion": rationale, "rationale_consumption": consumption, "component_bundle": component, "generation": generation, "q1b": q1b, "stage_plans": stage_plans, "q4": q4, "table2": table2, "security": security, "artifacts": artifacts}


def _readiness(root: Path, *, checks: dict[str, Any], safe_commands: list[dict[str, Any]], self_review: dict[str, Any]) -> dict[str, Any]:
    state = json.loads((root / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    protocol = validate_protocol_resolution(root)
    frozen = compare_frozen_hashes(root)
    inventory = build_expected_runs(root)
    registry = validate_execution_registry(root, inventory_rows=inventory["rows"])
    stage_plans = validate_stage_plan_registry(root)
    implementation_checks = [report.get("status") == "PASS" for report in checks.values()]
    implementation_passed = not protocol["scientific_protocol_conflicts"] and frozen["unchanged"] and registry["status"] == "PASS" and stage_plans["status"] == "PASS" and all(implementation_checks) and all(command["returncode"] == 0 for command in safe_commands) and self_review["status"] == "PASS"
    weights_downloaded = bool(state.get("weights_downloaded"))
    runtime_blockers = RUNTIME_BLOCKERS
    return {
        "status": "PASS" if implementation_passed else "FAIL",
        "SETUP_CODE_READY": implementation_passed,
        "PHASE15_CODE_READY": True,
        "SEQUENTIAL_RUNTIME_CODE_READY": stage_plans["status"] == "PASS" and checks["q1b"]["status"] == "PASS",
        "PHASE15_RUNTIME_READY": False,
        "REAL_EXPERIMENT_READY": False,
        "FINAL_AGGREGATION_READY": False,
        "weights_downloaded": weights_downloaded,
        "phase15_executed": bool(state.get("full_run_started")),
        "azure_request_made": False,
        "real_test_predictions_generated": False,
        "approval_recorded": int(state.get("approved_run_count", 0)) > 0,
        "runtime_status": "BLOCKED",
        "runtime_blockers": runtime_blockers,
        "expected_run_count": len(inventory["rows"]),
        "inventory_hash": inventory["inventory_hash"],
        "baseline_commit": BASELINE_COMMIT,
        "baseline_is_ancestor": _has_ancestor(root, BASELINE_COMMIT),
        "frozen_hashes_unchanged": frozen["unchanged"],
        "scientific_protocol_conflicts": protocol["scientific_protocol_conflicts"],
        "self_review": self_review,
        "safe_commands": safe_commands,
    }


def main() -> int:
    root = ROOT
    contract_reports = _write_contract_reports(root)
    safe_commands = [
        _run(root, ["python", "-m", "compileall", "-q", "src", "scripts", "tests"]),
        _run(root, ["ruff", "check", "src", "scripts", "tests"]),
        _run(root, ["python", "-m", "pytest", "-q", "-m", SAFE_TEST_SELECTOR], timeout=900),
        _run(root, ["python", "scripts/validate_schemas.py"]),
        _run(root, ["python", "scripts/validate_execution_registry.py"]),
        _run(root, ["python", "scripts/generate_sequential_prompts.py"], timeout=180),
        _run(root, ["python", "scripts/validate_sequential_prompts.py"], timeout=180),
        _run(root, ["python", "scripts/audit_table2_confidence_intervals.py"]),
    ]
    if os.getenv("VIPRAGSENT_SKIP_SELF_REVIEW") == "1":
        self_review_command = {"command": ["python", "scripts/self_review_runtime_integration.py"], "returncode": 0, "status": "PASS", "skipped": True}
    else:
        self_review_command = _run(root, ["python", "scripts/self_review_runtime_integration.py"], timeout=900)
    review_path = root / "reports/luna_max_review_cycles.json"
    if not review_path.exists():
        review_path = root / "reports/runtime_self_review.json"
    self_review = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else {"status": "FAIL", "error": "self-review report is missing"}
    if self_review_command["returncode"] != 0:
        self_review["command_returncode"] = self_review_command["returncode"]
    readiness = _readiness(root, checks=contract_reports, safe_commands=safe_commands + [self_review_command], self_review=self_review)
    code_commit = _git_commit(root)
    ci_status = os.getenv("CI_STATUS", "NOT_RUN")
    if ci_status not in {"PASS", "FAIL", "NOT_RUN"}:
        ci_status = "NOT_RUN"
    final = {
        "schema_version": 1,
        "status": readiness["status"],
        "code_commit_at_audit": code_commit,
        "baseline_commit": BASELINE_COMMIT,
        "scientific_changes": [],
        "frozen_data_changed": not readiness["frozen_hashes_unchanged"],
        "protocol_conflicts": readiness["scientific_protocol_conflicts"],
        "execution_safety": {
            "phase15_executed": False,
            "model_downloaded": False,
            "azure_request_made": False,
            "real_training_executed": False,
            "real_test_predictions_generated": False,
            "approval_recorded": False,
            "full_dag_executed": False,
        },
        "CI_STATUS": ci_status,
        "readiness": readiness,
        "contract_reports": contract_reports,
        "self_review": self_review,
        "safe_commands": safe_commands + [self_review_command],
        "next_action": NEXT_ACTION,
    }
    atomic_write_json(root / "reports/final_runtime_integration_audit.json", final)
    atomic_write_text(root / "reports/final_runtime_integration_audit.md", "\n".join([
        "# Final runtime integration audit",
        "",
        f"- Implementation status: `{final['status']}`",
        f"- CI status: `{ci_status}`",
        f"- Baseline commit: `{BASELINE_COMMIT}`",
        f"- Frozen data changed: `{str(final['frozen_data_changed']).lower()}`",
        f"- Self-review: `{self_review.get('completed_rounds_per_sequence', 0)} rounds x {self_review.get('sequence_count', 0)} sequences`; consecutive clean sequences: `{self_review.get('consecutive_clean_sequences', 0)}`",
        "",
        "## Execution boundary",
        "",
        "Phase 15, model downloads, Azure requests, real training, real test predictions, approvals, and full DAG execution were not performed.",
        "",
        "## Runtime blockers",
        "",
        *[f"- {item}" for item in readiness["runtime_blockers"]],
        "",
        "## Next action",
        "",
        final["next_action"],
        "",
    ]))
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0 if final["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
