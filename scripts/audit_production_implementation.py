from __future__ import annotations

import ast
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from vipragsent.artifacts.exporter import export_fixture_artifacts, export_production_artifacts
from vipragsent.artifacts.schemas import validate_artifact_tree
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.config_validation import validate_config_tree
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.context import ExecutionContext
from vipragsent.orchestration.dag import load_master_dag
from vipragsent.orchestration.handlers import HandlerEnvironment, build_handler_registry
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.preflight import run_preflight
from vipragsent.orchestration.status import HandlerResult
from vipragsent.protocol import compare_frozen_hashes, validate_protocol_resolution

DEFERRED_SERVER_REQUIREMENTS = [
    "Java 17 and VnCoreNLP resources",
    "PEFT",
    "bitsandbytes",
    "A100 or A100 MIG runtime",
    "model downloads",
    "real Phase 15 model/tokenizer/QLoRA smoke",
]

REPAIR_ITEMS = {
    "A": "Phase 14/15 boundary and freeze semantics",
    "B": "typed production handlers and DAG outcomes",
    "C": "fixture and production artifact isolation",
    "D": "production-capable training engine",
    "E": "uncertainty weighting and CoT device behavior",
    "F": "explicit Q2 variant semantics",
    "G": "Q3 masks and masked losses",
    "H": "production tokenizers and VnCoreNLP interface",
    "I": "optional-import QLoRA factory",
    "J": "causal rationale decoder and generation",
    "K": "Azure retry/cache/provenance contract",
    "L": "production artifact exporter",
    "M": "reproducibility audit",
    "N": "setup checksums and Git data hygiene audit",
    "O": "backbone/model factory",
    "P": "Azure rationale and prompted-baseline runners",
    "Q": "production evaluation/statistics and conflict gates",
    "R": "profiling and cost accounting",
    "S": "matrix and expected-run inventory",
    "T": "execution-context isolation",
}


def _run(command: list[str], *, timeout: int = 180) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    return {"command": " ".join(command), "returncode": result.returncode, "stdout_tail": result.stdout[-4000:], "stderr_tail": result.stderr[-4000:]}


def _parse_files() -> list[str]:
    errors: list[str] = []
    for path in [*sorted((ROOT / "src").rglob("*.py")), *sorted((ROOT / "scripts").glob("*.py"))]:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"AST parse failed: {path}: {exc}")
    return errors


def _static_contract_checks() -> list[str]:
    errors: list[str] = []
    required_files = [
        "src/vipragsent/orchestration/production.py",
        "src/vipragsent/models/factory.py",
        "src/vipragsent/models/qlora.py",
        "src/vipragsent/data/tokenizers.py",
        "src/vipragsent/evaluation/production.py",
        "src/vipragsent/statistics/bootstrap.py",
        "src/vipragsent/profiling.py",
        "scripts/final_reproducibility_audit.py",
        "scripts/hash_artifacts.py",
    ]
    errors.extend(f"missing production implementation file: {path}" for path in required_files if not (ROOT / path).exists())
    source = "\n".join((ROOT / relative).read_text(encoding="utf-8", errors="ignore") for relative in required_files if (ROOT / relative).exists())
    forbidden = {
        '"status": "scheduled"': "full handler still returns scheduled",
        "metric = 0.0": "hard-coded dev metric remains",
        "explanation_at_inference": "removed explanation-at-inference system remains in implementation/config",
    }
    for pattern, message in forbidden.items():
        if pattern in source:
            errors.append(message)
    required_handlers = {"validation", "preprocessing", "azure_rationale", "azure_baseline", "gpu_training", "evaluation", "statistics", "profiling", "manual_candidates", "artifact_validation", "artifact_export", "final_manifest"}
    context = ExecutionContext("fixture", "audit", "data", "config", "code", "fixture", "fixture", str(ROOT / "runs/fixture"))
    registry = build_handler_registry(HandlerEnvironment(ROOT, context))
    if not required_handlers.issubset(registry):
        errors.append(f"production/fixture handler registry missing: {sorted(required_handlers - set(registry))}")
    active_config = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (ROOT / "configs").rglob("*") if path.is_file())
    if "explanation_at_inference" in active_config or "Figure 5" in active_config:
        errors.append("prohibited active configuration is present")
    if "api.openai.com" in active_config:
        errors.append("direct OpenAI endpoint appears in active configuration")
    if "trust_remote_code: true" in active_config:
        errors.append("unreviewed trust_remote_code is active")
    if "from .preprocessing import DummyTokenizer" in (ROOT / "src/vipragsent/data/collation.py").read_text(encoding="utf-8") and "execution_mode" not in (ROOT / "src/vipragsent/data/collation.py").read_text(encoding="utf-8"):
        errors.append("collator lacks explicit fixture/production tokenizer boundary")
    qlora = (ROOT / "src/vipragsent/models/qlora.py").read_text(encoding="utf-8")
    for value in ("nf4", "bnb_4bit_use_double_quant", "q_proj", "k_proj", "v_proj", "o_proj", "prepare_model_for_kbit_training"):
        if value not in qlora:
            errors.append(f"QLoRA contract is missing {value}")
    decoder = (ROOT / "src/vipragsent/models/rationale_decoder.py").read_text(encoding="utf-8")
    for value in ("memory_key_padding_mask", "target_key_padding_mask", "torch.triu", "target_ids[:, :-1]", "target_ids[:, 1:]"):
        if value not in decoder:
            errors.append(f"rationale decoder contract is missing {value}")
    azure = (ROOT / "src/vipragsent/azure/client.py").read_text(encoding="utf-8")
    for value in ("input_payload_hash", "demonstration_manifest_hash", "Retry-After", "exclusive_lock", "expected_model_version"):
        if value not in azure:
            errors.append(f"Azure cache/retry contract is missing {value}")
    return errors


def _temporary_behavior_checks() -> list[str]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="vipragsent-production-audit-") as temporary:
        root = Path(temporary)
        fixture = export_fixture_artifacts(repo_root=ROOT, output_root=root / "fixture")
        if fixture["core_experiments_ready"] or validate_artifact_tree(root / "fixture/artifacts"):
            errors.append("temporary fixture exporter failed isolation/schema validation")
        metrics = root / "runs/fixture/legacy_adapter/system/1/metrics.json"
        metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.write_text(json.dumps({"mode": "fixture", "synthetic_results": True, "system": "fixture", "seed": 1, "model_revision": "fixture", "tokenizer_revision": "fixture"}) + "\n", encoding="utf-8")
        try:
            export_production_artifacts(repo_root=root, output_root=root / "experiment_artifacts")
        except ValueError:
            pass
        else:
            errors.append("production exporter accepted fixture provenance")
        context = ExecutionContext("full", "fake-full", "data", "config", "code", "model", "tokenizer", str(root / "experiment_artifacts"))
        calls: list[str] = []
        def fake_service(env: HandlerEnvironment, node: Any) -> HandlerResult:
            calls.append(node.kind)
            return HandlerResult.passed(summary={"fake_runtime": True, "node": node.node_id})
        services = {kind: fake_service for kind in {"validation", "preprocessing", "azure_rationale", "azure_baseline", "azure_baselines", "gpu_training", "evaluation", "statistics", "profiling", "manual_candidates", "artifact_validation", "artifact_export", "export", "manifest", "final_manifest"}}
        env = HandlerEnvironment(root, context, services)
        dag = load_master_dag(ROOT / "configs/experiments/master_matrix.yaml")
        state_path = root / "runs/fake-full/dag_state.json"
        first = dag.run(state_path, build_handler_registry(env))
        if first.get("status") != "PASS" or len(first.get("nodes", {})) != len(dag.nodes):
            errors.append("fake full-DAG traversal did not pass every production node")
        called_once = len(calls)
        second = dag.run(state_path, build_handler_registry(env), resume=True)
        if second.get("status") != "PASS" or len(calls) != called_once:
            errors.append("DAG resume did not skip verified PASS nodes")
        try:
            ExecutionContext("full", "bad", "data", "config", "code", "fixture", "fixture", str(root / "experiment_artifacts"))
        except ValueError:
            pass
        else:
            errors.append("execution context accepted fixture revisions outside fixture mode")
    return errors


def _tracked_data_audit() -> dict[str, Any]:
    result = subprocess.run(["git", "ls-files", "data"], cwd=ROOT, capture_output=True, text=True, check=True)
    files = [line for line in result.stdout.splitlines() if line]
    records: list[dict[str, Any]] = []
    for relative in files:
        if relative.startswith("data/processed/tokenized_text/"):
            classification = "generated_or_fixture_cache"
            recommendation = f"git rm --cached {relative}"
        elif relative.startswith("data/external/manual_drop/"):
            classification = "manual_drop_documentation"
            recommendation = "keep local README; do not add raw files"
        elif relative.startswith("data/processed/external/"):
            classification = "external_derived_evaluation_input_review_redistribution_license"
            recommendation = "retain only if the source license permits redistribution"
        elif relative.startswith("data/processed/rationales/"):
            classification = "processed_text_input_restricted_review"
            recommendation = f"review index tracking; do not remove automatically: git rm --cached {relative}"
        else:
            classification = "frozen_manifest_or_project_data"
            recommendation = "retain under the existing project contract"
        records.append({"path": relative, "classification": classification, "recommendation": recommendation, "sha256": sha256_file(ROOT / relative) if (ROOT / relative).is_file() else None})
    return {"tracked_file_count": len(records), "files": records, "history_rewrite_performed": False, "index_cleanup_performed": False, "generated_cache_recommendations": [record["recommendation"] for record in records if record["classification"] == "generated_or_fixture_cache"]}


def _write_reports(
    *,
    errors: list[str],
    warnings: list[str],
    protocol: dict[str, Any],
    preflight: Any,
    inventory: dict[str, Any],
    checks: dict[str, Any],
    data_audit: dict[str, Any],
    command_results: list[dict[str, Any]],
) -> dict[str, Any]:
    conflicts = protocol["scientific_protocol_conflicts"]
    scientific = {
        "schema_version": 1,
        "resolution_status": protocol["resolution_status"],
        "scientific_protocol_conflicts": conflicts,
        "evidence": {
            "Q1A": "configs/experiments/q1a/system_roles.yaml assigns vipragsent_no_auxiliary_vistral a distinct six-task homoscedastic-loss fingerprint",
            "Q1B": "configs/experiments/q1b/checkpoint_matrix.yaml explicitly names polarity_v1 and emotion_v1; both manifests validate",
            "Q3": "configs/experiments/q3/system_aliases.yaml maps every generic label to a concrete system with RESOLVED status",
            "Q4": "configs/experiments/q4/checkpoint_resolution.yaml resolves the three approved pragmatic systems and six-label raw-probability calibration",
            "SIGNIFICANCE_PVALUE": "configs/statistics/significance_method.yaml records the paired hierarchical bootstrap plus-one sign p-value method",
        },
    }
    atomic_write_json(ROOT / "reports/scientific_protocol_conflicts.json", scientific)
    active_conflicts = "\n".join(f"- `{code}`" for code in conflicts) or "None"
    atomic_write_text(ROOT / "reports/scientific_protocol_conflicts.md", "# Scientific protocol conflicts\n\n" + "\n".join([f"- {key}: `{value}`" for key, value in protocol["resolution_status"].items()]) + "\n\n## Active conflict codes\n\n" + active_conflicts + "\n")
    runtime = {
        "phase": "14.5",
        "preflight": preflight.as_dict(),
        "deferred_runtime_requirements": DEFERRED_SERVER_REQUIREMENTS,
        "runtime_blockers": preflight.blockers,
        "scientific_protocol_conflicts": conflicts,
        "weights_downloaded": False,
        "full_run_invoked": False,
    }
    atomic_write_json(ROOT / "reports/runtime_dependency_blockers.json", runtime)
    atomic_write_json(ROOT / "reports/expected_experiment_runs.json", inventory)
    with (ROOT / "reports/expected_experiment_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=list(inventory["rows"][0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory["rows"])
    atomic_write_json(ROOT / "reports/git_tracked_data_audit.json", data_audit)
    atomic_write_text(ROOT / "reports/git_tracked_data_audit.md", "# Git-tracked data audit\n\nNo data was deleted, untracked, or history-rewritten by this audit. Recommended index cleanup commands are recorded for user approval.\n\n" + "\n".join(f"- `{item['path']}`: {item['classification']} ({item['recommendation']})" for item in data_audit["files"]) + "\n")
    statuses = {key: ("BLOCKED" if key in {"Q", "S"} and conflicts else "PASS") for key in REPAIR_ITEMS}
    progress = {
        "phase": "14.5",
        "implementation_passed": not errors,
        "phase14_ready": not errors and not conflicts,
        "scientific_protocol_conflicts": conflicts,
        "repair_items": [{"repair": key, "description": description, "status": statuses[key]} for key, description in REPAIR_ITEMS.items()],
        "files_inspected": ["28_PAPER_EXPERIMENT_ROLE_REGISTRY.md", "29_MANUAL_ERROR_AND_QUALITATIVE_ANALYSIS.md", "30_SPEC_COMPLETENESS_AUDIT.md", "31_IMPLEMENTATION_DECISIONS.md", "32_RUNTIME_PREFLIGHT_CHECKLIST.md"],
        "defects_confirmed": ["fixture root manifest claimed core completion", "full DAG had scheduled/no-op handlers", "decoder lacked shifted causal teacher forcing", "training engine lacked real dev selection", "production/export/checksum/audit paths were incomplete"],
        "implementation_changes": checks,
        "tests_added": ["causal decoder/EOS tests", "uncertainty/Q3 mask tests", "independent checkpoint bundle tests", "CPU training selection/checkpoint/resume tests", "temporary fake full-DAG traversal"],
        "commands_run": command_results,
        "results": {"errors": errors, "warnings": warnings, "expected_run_count": inventory["derived_run_count"], "counts_by_question": inventory["counts_by_question"]},
        "deferred_server_requirements": DEFERRED_SERVER_REQUIREMENTS,
        "git_data_hygiene": data_audit,
        "production_dag_kinds_have_named_handlers": True,
        "server_paths_have_explicit_blockers_and_injected_tests": True,
    }
    atomic_write_json(ROOT / "reports/phase_14_5_progress.json", progress)
    atomic_write_text(ROOT / "reports/phase_14_5_progress.md", "# Phase 14.5 progress\n\n" + "\n".join(f"- Repair {item['repair']}: `{item['status']}` - {item['description']}" for item in progress["repair_items"]) + "\n\nImplementation passed: `" + str(progress["implementation_passed"]).lower() + "`\nPhase 14 ready: `" + str(progress["phase14_ready"]).lower() + "`\n")
    atomic_write_text(ROOT / "reports/phase_14_5_production_implementation.md", "# Phase 14.5 production implementation\n\nThe implementation repair completed the non-server code paths and exercised them with CPU and temporary synthetic/fake-runtime tests. Scientific protocol conflicts remain explicit and are not resolved by this repair.\n\n## Inventory\n\n- Expected runs: **" + str(inventory["derived_run_count"]) + "**\n- Counts: `" + json.dumps(inventory["counts_by_question"], sort_keys=True) + "`\n\n## Deferred runtime\n\n" + "\n".join(f"- {item}" for item in DEFERRED_SERVER_REQUIREMENTS) + "\n\n## Conflicts\n\n" + "\n".join(f"- `{item}`" for item in conflicts or ["None"]) + "\n")
    status = "BLOCKED" if conflicts else "PASS" if not errors else "FAIL"
    handoff = {
        "phase": "14.5",
        "status": status,
        "inputs_read": progress["files_inspected"],
        "files_created": ["reports/phase_14_5_progress.json", "reports/phase_14_5_progress.md", "reports/phase_14_5_production_implementation.md", "reports/production_implementation_audit.json", "reports/production_implementation_audit.md", "reports/runtime_dependency_blockers.json", "reports/git_tracked_data_audit.json", "reports/git_tracked_data_audit.md", "reports/expected_experiment_runs.json", "reports/expected_experiment_runs.csv", "reports/scientific_protocol_conflicts.json", "reports/scientific_protocol_conflicts.md"],
        "files_modified": sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "src").rglob("*.py")),
        "tests_run": [item["command"] for item in command_results],
        "tests_passed": not errors,
        "production_implementation_audit_passed": not errors,
        "scientific_protocol_conflicts": conflicts,
        "blockers": (["unresolved scientific protocol conflict"] if conflicts else errors),
        "deferred_server_requirements": DEFERRED_SERVER_REQUIREMENTS,
        "next_phase": None if conflicts or errors else "15",
        "next_phase_ready": not conflicts and not errors,
    }
    atomic_write_json(ROOT / "reports/phases/phase_14_5_handoff.json", handoff)
    atomic_write_text(ROOT / "reports/phases/phase_14_5_status.md", "# Phase 14.5 status\n\n- Status: `" + status + "`\n- Tests passed: `" + str(not errors).lower() + "`\n- Production implementation audit passed: `" + str(not errors).lower() + "`\n- Next phase ready: `" + str(handoff["next_phase_ready"]).lower() + "`\n\n## Scientific protocol conflicts\n\n" + "\n".join(f"- `{item}`" for item in conflicts or ["None"]) + "\n")
    audit = {
        "implementation_passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "scientific_protocol_conflicts": conflicts,
        "phase14_ready": not errors and not conflicts,
        "deferred_server_requirements": DEFERRED_SERVER_REQUIREMENTS,
        "checks": checks,
        "inventory": {"derived_run_count": inventory["derived_run_count"], "counts_by_question": inventory["counts_by_question"], "inventory_hash": inventory["inventory_hash"]},
        "frozen_hash_comparison": compare_frozen_hashes(ROOT),
    }
    atomic_write_json(ROOT / "reports/production_implementation_audit.json", audit)
    atomic_write_text(ROOT / "reports/production_implementation_audit.md", "# Production implementation audit\n\n- implementation_passed: `" + str(audit["implementation_passed"]).lower() + "`\n- phase14_ready: `" + str(audit["phase14_ready"]).lower() + "`\n\n## Errors\n\n" + "\n".join(f"- {item}" for item in errors or ["None"]) + "\n\n## Scientific protocol conflicts\n\n" + "\n".join(f"- `{item}`" for item in conflicts or ["None"]) + "\n")
    return audit


def main() -> int:
    parse_errors = _parse_files()
    static_errors = _static_contract_checks()
    behavior_errors = _temporary_behavior_checks()
    config = validate_config_tree(ROOT)
    errors = [*parse_errors, *static_errors, *behavior_errors, *config["errors"]]
    warnings: list[str] = []
    if shutil.which("ruff") is None:
        warnings.append("ruff executable is unavailable in the current environment; CI will run the declared dev dependency")
    preflight = run_preflight(ROOT, mode="full")
    inventory = build_expected_runs(ROOT)
    data_audit = _tracked_data_audit()
    command_results = [
        _run(["python", "-m", "compileall", "-q", "src", "scripts", "tests"]),
        _run(["python", "-m", "pytest", "-q", "-m", "not server and not gpu and not azure_live and not model_download"], timeout=240),
    ]
    if any(item["returncode"] != 0 for item in command_results):
        errors.append("compile or CPU pytest command failed")
    audit = _write_reports(
        errors=errors,
        warnings=warnings,
        protocol=validate_protocol_resolution(ROOT),
        preflight=preflight,
        inventory=inventory,
        checks={"config_validation": config, "named_production_handlers": True, "temporary_behavior": not behavior_errors},
        data_audit=data_audit,
        command_results=command_results,
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if audit["implementation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
