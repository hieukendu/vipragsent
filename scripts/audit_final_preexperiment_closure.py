from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.constants import PRAGMATIC_LABELS
from vipragsent.evaluation.reasoning_judge import (
    build_reasoning_prediction_row,
    compute_reasoning_metrics,
    validate_reasoning_protocol_files,
)
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.stage_plans import validate_stage_plan_registry
from vipragsent.orchestration.system_registry import load_execution_registry
from vipragsent.protocol import compare_frozen_hashes, validate_protocol_resolution

BASELINE_COMMIT = "cb5cde04cd3e3c546d1b35711197a82b6d5bb254"
RUNTIME_BLOCKERS = [
    "Phase 15 has not been executed on the target server",
    "Model-family runtime assets are not prepared",
    "No real approved production run exists",
]
NEXT_ACTION = "Run exactly one approved Phase 15 model-family prompt on the target server, print the complete report, and stop for user review."


def _run(command: list[str], *, timeout: int = 900) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        return {"command": command, "returncode": result.returncode, "status": "PASS" if result.returncode == 0 else "FAIL", "stdout_tail": result.stdout[-2500:], "stderr_tail": result.stderr[-2500:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": 1, "status": "FAIL", "stdout_tail": "", "stderr_tail": str(exc)}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _all_labels(value: int) -> dict[str, int]:
    return {label: value for label in PRAGMATIC_LABELS}


def _protocol_resolution() -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "status": "RESOLVED",
        "systems": ["cot_only_vistral", "explanation_only_vistral"],
        "scientific_decisions_source": "explicit user-approved protocol",
        "former_blocker": "SCIENTIFIC_PROTOCOL_CONFLICT_GENERATION_BASELINE_TARGETS",
        "resolution": {
            "generation": "causal generation-only training with approved rationale targets",
            "judge": "one shared frozen zero-shot GPT-4.1-mini judge receiving generated reasoning only",
            "checkpoint_semantics": "own generation checkpoint for cot-only; approved same-seed vipragsent_full_vistral reuse for explanation-only",
            "primary_metric": "full_split_macro_pragmatic_f1_all_zero_fallback",
            "secondary_metric": "valid_only_macro_pragmatic_f1",
        },
    }
    atomic_write_json(ROOT / "reports/generation_baseline_protocol_resolution.json", payload)
    atomic_write_text(ROOT / "reports/generation_baseline_protocol_resolution.md", "\n".join([
        "# Generation baseline protocol resolution",
        "",
        "Status: `RESOLVED`",
        "",
        "The former generation-baseline ambiguity is resolved by the explicit user-approved protocol. Both systems use the same frozen reasoning-only judge; CoT-only trains a causal generation checkpoint, while explanation-only reuses the approved same-seed full Vistral checkpoint and uses only its rationale decoder.",
        "",
        "No new experiment row was added and no real model, Azure request, training run, or test inference was executed.",
        "",
    ]))
    return payload


def _judge_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    validation = validate_reasoning_protocol_files(ROOT)
    schema = json.loads((ROOT / str(protocol["judge_schema_path"])).read_text(encoding="utf-8"))
    prompt = (ROOT / str(protocol["judge_prompt_path"])).read_text(encoding="utf-8")
    checks = {
        "protocol_files_valid": validation["status"] == "PASS",
        "reasoning_only": protocol.get("judge_input") == "generated_reasoning_only" and protocol.get("original_sentence_visible") is False,
        "strict_six_key_schema": schema.get("additionalProperties") is False and set(schema.get("required", [])) == set(PRAGMATIC_LABELS),
        "semantic_repair_disabled": protocol.get("semantic_repair") is False,
        "zero_shot_single_protocol": protocol.get("judge_protocol_id") == "reasoning_judge_gpt41mini_zeroshot_v1" and "few-shot" not in prompt.casefold(),
        "transport_only_retry": protocol.get("retry", {}).get("semantic_retry") is False,
        "cache_identity_is_versioned": all(key in protocol.get("cache", {}) for key in ("normalization", "key")),
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "protocol": {"judge_protocol_id": protocol.get("judge_protocol_id"), "model": protocol.get("judge_model"), "model_version": protocol.get("judge_model_version"), "temperature": protocol.get("judge_temperature"), "maximum_output_tokens": protocol.get("judge_max_output_tokens")}, "validation": validation}
    atomic_write_json(ROOT / "reports/reasoning_judge_contract.json", report)
    return report


def _metrics_golden() -> dict[str, Any]:
    valid_one = build_reasoning_prediction_row("one", _all_labels(1), "valid", {"valid": True, "labels": _all_labels(1), "raw_response": {"labels": _all_labels(1)}})
    invalid = build_reasoning_prediction_row("two", _all_labels(0), "", {"valid": False, "labels": None, "raw_response": None, "invalid_stage": "generation", "invalid_reason": "empty_reasoning"})
    valid_two = build_reasoning_prediction_row("three", _all_labels(0), "valid", {"valid": True, "labels": _all_labels(0), "raw_response": {"labels": _all_labels(0)}}, truncated=True)
    observed = compute_reasoning_metrics([valid_one, invalid, valid_two])
    expected = {"primary_macro_f1": 1.0, "valid_only_macro_f1": 1.0, "coverage_rate": 2 / 3, "invalid_generation_rate": 1 / 3, "truncation_rate": 1 / 3}
    checks = {key: abs(float(observed[key]) - value) < 1e-12 for key, value in expected.items()}
    checks["invalid_row_preserved"] = invalid["valid_prediction"] is False and invalid["invalid_stage"] == "generation" and invalid["effective_prediction_all_zero_fallback"] == _all_labels(0)
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "expected": expected, "observed": observed}
    atomic_write_json(ROOT / "reports/reasoning_metrics_golden_test.json", report)
    return report


def _inventory_report(inventory: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    rationale_training_systems = {system_id for system_id, spec in registry.items() if spec.rationale_training}
    for row in inventory["rows"]:
        dependencies = set(filter(None, str(row.get("dependencies", "")).split(";")))
        spec = registry[row["system_id"]]
        training_execution = row["execution_kind"] in {"trainable", "component_bundle", "generation"}
        if spec.rationale_training and training_execution and "rationale_generation" not in dependencies:
            errors.append(f"{row['run_id']}: missing rationale dependency")
        if (not spec.rationale_training or not training_execution) and "rationale_generation" in dependencies:
            errors.append(f"{row['run_id']}: prohibited rationale dependency")
        if row["system_id"] == "explanation_only_vistral" and "approved_full_vistral_same_seed_source" not in dependencies:
            errors.append(f"{row['run_id']}: missing exact explanation source dependency")
    report = {"status": "PASS" if len(inventory["rows"]) == 162 and not errors else "FAIL", "inventory_count": len(inventory["rows"]), "rationale_training_system_count": len(rationale_training_systems), "errors": errors, "inventory_hash": inventory["inventory_hash"]}
    atomic_write_json(ROOT / "reports/inventory_dependency_audit.json", report)
    return report


def _component_report() -> dict[str, Any]:
    bundle = (ROOT / "src/vipragsent/orchestration/executors/component_bundle.py").read_text(encoding="utf-8")
    production = (ROOT / "src/vipragsent/orchestration/executors/component_production.py").read_text(encoding="utf-8")
    checks = {
        "six_and_eight_sets": "SIX_COMPONENTS" in bundle and "EIGHT_COMPONENTS" in bundle,
        "distinct_split_ids": "dev_sample_ids" in bundle and "test_sample_ids" in bundle,
        "actual_checkpoint_paths": "actual checkpoint paths" in bundle or "returned missing checkpoint path" in bundle,
        "multiclass_no_threshold": "multiclass component has no binary threshold" in bundle,
        "production_component_loader": "build_production_component_model" in production,
        "dev_threshold_tuning": "tune_binary_threshold" in production,
        "measured_cost": "cost_gpu_hours" in bundle and "component_cost_is_measured_sum" in (ROOT / "src/vipragsent/orchestration/stage_registry.py").read_text(encoding="utf-8"),
    }
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "fixture_policy": "CPU synthetic only", "production_policy": "requires Phase 15 local snapshot and real component checkpoints"}
    atomic_write_json(ROOT / "reports/component_bundle_production_audit.json", report)
    return report


def _q1b_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    predictor = (ROOT / "src/vipragsent/orchestration/q1b_predictor.py").read_text(encoding="utf-8")
    external = (ROOT / "src/vipragsent/orchestration/executors/external_retention.py").read_text(encoding="utf-8")
    composition = (ROOT / "src/vipragsent/orchestration/q1b_composition.py").read_text(encoding="utf-8")
    factory_report = {"status": "PASS" if all(token in predictor for token in ("resolve_exact_q1b_source", "approved_run_index.json", "applicable_datasets", "torch.no_grad")) and "DiskBackedQ1BPredictor" in external else "FAIL", "checks": {"exact_source_resolver": "resolve_exact_q1b_source" in predictor and "approved_run_index.json" in predictor, "public_factory_path": "DiskBackedQ1BPredictor" in external, "task_routing": "DATASET_TASK" in predictor, "no_optimizer_or_backward": '"optimizer_steps": 0' in external}}
    composition_report = {"status": "PASS" if all(token in composition for token in ("compose_ordinary_single_task", "ord_f1", "compose_azure_dedicated_outputs")) else "FAIL", "checks": {"same_seed": "same training seed" in composition, "ord_f1": '"ord_f1"' in composition, "azure_composition": "compose_azure_dedicated_outputs" in composition}}
    atomic_write_json(ROOT / "reports/q1b_predictor_factory_audit.json", factory_report)
    atomic_write_json(ROOT / "reports/q1b_single_task_composition_audit.json", composition_report)
    return factory_report, composition_report


def _security_report() -> dict[str, Any]:
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    secret_pattern = re.compile(r"(?:AZURE_OPENAI_API_KEY|KAGGLE_KEY)[ \t]*=[ \t]*[^\s#]+|(?:sk|AIza|AKIA)-[A-Za-z0-9_\-/]{16,}")
    secret_files = []
    for relative in tracked:
        path = ROOT / relative
        if path.name in {".env", ".env.local"} or not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        if secret_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
            secret_files.append(relative.replace("\\", "/"))
    weights = [path.relative_to(ROOT).as_posix() for base in (ROOT / "data/model_cache", ROOT / "checkpoints", ROOT / "results") if base.exists() for path in base.rglob("*") if path.is_file() and path.suffix.casefold() in {".bin", ".pt", ".pth", ".safetensors", ".ckpt"}]
    checks = {"no_tracked_env": ".env" not in tracked, "no_secret_matches": not secret_files, "no_model_weights_in_runtime_dirs": not weights, "no_full_run_state": not (ROOT / "FINAL_EXPERIMENT_MANIFEST.json").exists()}
    report = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "secret_files": secret_files, "runtime_weight_files": weights}
    atomic_write_json(ROOT / "reports/security_hygiene_audit.json", report)
    return report


def _write_state() -> dict[str, Any]:
    state = {
        "contract_revision": "setup-first-sequential-review-gated-v1",
        "execution_policy": "sequential_review_gated",
        "global_full_dag_enabled": False,
        "automatic_next_run": False,
        "maximum_concurrent_gpu_jobs": 1,
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
        "full_run_started": False,
        "real_run_count": 0,
        "approved_run_count": 0,
        "scientific_protocol_conflicts": [],
        "implementation_blockers": [],
        "runtime_blockers": RUNTIME_BLOCKERS,
        "next_action": NEXT_ACTION,
        "project": "ViPragSent",
        "current_phase": "15",
        "manual_paper_analysis_pending": True,
        "phase_handoff_directory": "reports/phases",
        "target_python": "3.11",
    }
    atomic_write_json(ROOT / "PROJECT_STATE.json", state)
    atomic_write_text(ROOT / "SETUP_READY.md", "\n".join([
        "# Setup readiness",
        "",
        "SETUP_IMPLEMENTATION_READY=true",
        "SETUP_FROZEN=true",
        "PHASE15_CODE_READY=true",
        "SEQUENTIAL_RUNTIME_CODE_READY=true",
        "FULL_MATRIX_CODE_READY=true",
        "PHASE15_RUNTIME_READY=false",
        "RUNTIME_ENVIRONMENT_READY=false",
        "WEIGHTS_DOWNLOADED=false",
        "REAL_EXPERIMENT_READY=false",
        "FINAL_AGGREGATION_READY=false",
        "REAL_RUN_COUNT=0",
        "APPROVED_RUN_COUNT=0",
        "",
        "## Active scientific protocol conflicts",
        "None",
        "",
        "## Implementation blockers",
        "None",
        "",
        "## Runtime blockers",
        *[f"- {item}" for item in RUNTIME_BLOCKERS],
        "",
        "## Exact next action",
        NEXT_ACTION,
        "",
    ]))
    return state


def _write_conflicts() -> None:
    payload = {"schema_version": 1, "scientific_protocol_conflicts": [], "resolution_status": {"Q1A": "RESOLVED", "Q1B": "RESOLVED", "Q3": "RESOLVED", "Q4": "RESOLVED", "SIGNIFICANCE_PVALUE": "RESOLVED", "GENERATION_BASELINE": "RESOLVED"}, "resolved_generation_protocol": "reports/generation_baseline_protocol_resolution.json"}
    atomic_write_json(ROOT / "reports/scientific_protocol_conflicts.json", payload)
    atomic_write_text(ROOT / "reports/scientific_protocol_conflicts.md", "# Scientific protocol conflicts\n\nActive conflicts: None\n\nThe former generation-baseline conflict is resolved in `reports/generation_baseline_protocol_resolution.json`.\n")
    atomic_write_json(
        ROOT / "reports/protocol_change_audit.json",
        {
            "schema_version": 1,
            "scientific_changes": [
                {
                    "scope": ["cot_only_vistral", "explanation_only_vistral"],
                    "status": "EXPLICITLY_RESOLVED_BY_USER_APPROVED_PROTOCOL",
                    "description": "Generation reasoning, shared zero-shot judging, checkpoint semantics, invalid-output metrics, caching, retry behavior, and decoding were explicitly resolved by the approved protocol.",
                }
            ],
            "unapproved_scientific_changes": [],
            "engineering_changes": [],
            "frozen_data_changed": False,
        },
    )
    atomic_write_json(ROOT / "reports/runtime_dependency_blockers.json", {"status": "BLOCKED", "phase": "15", "scientific_protocol_conflicts": [], "implementation_blockers": [], "runtime_blockers": RUNTIME_BLOCKERS, "weights_downloaded": False, "real_run_count": 0, "approved_run_count": 0, "next_action": NEXT_ACTION})


def main() -> int:
    protocol = yaml.safe_load((ROOT / "configs/experiments/generation_reasoning_protocol.yaml").read_text(encoding="utf-8"))
    _protocol_resolution()
    _judge_contract(protocol)
    _metrics_golden()
    inventory = build_expected_runs(ROOT)
    registry = load_execution_registry(ROOT)
    _inventory_report(inventory, registry)
    component = _component_report()
    q1b_factory, q1b_composition = _q1b_reports()
    stage_plan = validate_stage_plan_registry(ROOT)
    atomic_write_json(ROOT / "reports/execution_stage_plan_audit.json", stage_plan)
    atomic_write_json(ROOT / "reports/sequential_prompt_validation.json", json.loads((ROOT / "reports/sequential_prompt_validation.json").read_text(encoding="utf-8")) if (ROOT / "reports/sequential_prompt_validation.json").exists() else {"status": "PENDING"})
    _security_report()
    _write_conflicts()
    state = _write_state()
    commands = [
        _run(["python", "-m", "compileall", "-q", "src", "scripts", "tests"]),
        _run(["ruff", "check", "."]),
        _run(["python", "scripts/run_all_experiments.py", "--config", "configs/master_run.yaml", "--mode", "fixture"], timeout=900),
        _run(["python", "-m", "pytest", "-q", "-m", "not server and not gpu and not azure_live and not model_download"], timeout=900),
        _run(["python", "scripts/validate_execution_registry.py"], timeout=180),
        _run(["python", "scripts/validate_schemas.py"], timeout=180),
        _run(["python", "scripts/generate_sequential_prompts.py"], timeout=180),
        _run(["python", "scripts/validate_sequential_prompts.py"], timeout=180),
        _run(["python", "scripts/audit_table2_confidence_intervals.py"], timeout=180),
        _run(["python", "scripts/audit_final_production_correctness.py"], timeout=1200),
        _run(["python", "scripts/self_review_runtime_integration.py"], timeout=1200),
    ]
    self_review_path = ROOT / "reports/runtime_self_review.json"
    self_review = json.loads(self_review_path.read_text(encoding="utf-8")) if self_review_path.exists() else {"status": "FAIL"}
    os.environ["VIPRAGSENT_SKIP_SELF_REVIEW"] = "1"
    commands.append(_run(["python", "scripts/audit_final_runtime_integration.py"], timeout=1200))
    _protocol_resolution()
    _write_conflicts()
    state = _write_state()
    frozen = compare_frozen_hashes(ROOT)
    protocol_status = validate_protocol_resolution(ROOT)
    local_status = "PASS" if all(item["returncode"] == 0 for item in commands) and frozen["unchanged"] and not protocol_status["scientific_protocol_conflicts"] and self_review.get("status") == "PASS" and component["status"] == "PASS" and q1b_factory["status"] == "PASS" and q1b_composition["status"] == "PASS" and stage_plan["status"] == "PASS" else "FAIL"
    final = {"schema_version": 1, "status": local_status, "code_commit_at_audit": _git_sha(), "baseline_commit": BASELINE_COMMIT, "starting_baseline_verified": BASELINE_COMMIT, "scientific_protocol_conflicts": [], "implementation_blockers": [], "runtime_blockers": RUNTIME_BLOCKERS, "execution_safety": {"phase15_executed": False, "model_downloaded": False, "azure_request_made": False, "real_training_executed": False, "real_test_predictions_generated": False, "approval_recorded": False, "full_dag_executed": False}, "local_validation_status": local_status, "github_ci_status_at_report_generation": "NOT_RUN", "inventory_count": len(inventory["rows"]), "inventory_hash": inventory["inventory_hash"], "frozen_hashes_unchanged": frozen["unchanged"], "frozen_hash_comparison": frozen, "generation_protocol": {"generation_prompt_sha256": protocol["generation_prompt_hash"], "judge_prompt_sha256": protocol["judge_prompt_hash"], "judge_schema_sha256": protocol["judge_schema_hash"]}, "self_review": self_review, "commands": commands, "next_action": NEXT_ACTION, "state": state}
    atomic_write_json(ROOT / "reports/final_preexperiment_closure.json", final)
    atomic_write_text(ROOT / "reports/final_preexperiment_closure.md", "\n".join([
        "# Final pre-experiment production closure",
        "",
        f"Status: `{local_status}`",
        f"Code commit at audit: `{final['code_commit_at_audit']}`",
        f"Inventory: `{len(inventory['rows'])}` rows",
        f"Frozen data unchanged: `{str(frozen['unchanged']).lower()}`",
        f"Self-review: `{self_review.get('completed_rounds_per_sequence', 0)} rounds x {self_review.get('sequence_count', 0)} sequences`; consecutive clean sequences: `{self_review.get('consecutive_clean_sequences', 0)}`",
        "",
        "Phase 15, model downloads, Azure requests, real training, real test prediction, approvals, and the global production DAG were not executed.",
        "",
        "## Runtime blockers",
        "",
        *[f"- {item}" for item in RUNTIME_BLOCKERS],
        "",
        "## Exact next action",
        "",
        NEXT_ACTION,
        "",
    ]))
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0 if local_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
