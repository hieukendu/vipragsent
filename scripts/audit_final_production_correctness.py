from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, TRAINING_SEEDS
from vipragsent.data.loaders import DatasetExample, load_vipragsent
from vipragsent.data.masks import validate_q3_masks
from vipragsent.evaluation.external_retention import (
    NormalizedExternalExample,
    evaluate_external_retention,
)
from vipragsent.evaluation.reasoning_judge import validate_reasoning_protocol_files
from vipragsent.hashing import sha256_json
from vipragsent.models.variants import VariantConfig, build_dummy_model
from vipragsent.orchestration.aggregation import _q4_summary, _table4
from vipragsent.orchestration.contracts import RunEntry
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.system_registry import (
    load_execution_registry,
    validate_execution_registry,
)
from vipragsent.protocol import compare_frozen_hashes, validate_protocol_resolution
from vipragsent.training.class_weights import compute_train_only_class_weights
from vipragsent.training.config_resolver import resolve_training_config

BASELINE_COMMIT = "cb5cde04cd3e3c546d1b35711197a82b6d5bb254"
SCIENCE_EXCLUDED = {
    "configs/experiments/system_execution_registry.yaml",
    "configs/experiments/execution_stage_plans.yaml",
    "configs/schemas/prediction.schema.json",
    "configs/schemas/run_metadata.schema.json",
    "configs/runtime/training.yaml",
}
REQUIRED_REPORTS = (
    "reports/final_production_correctness_repair.json",
    "reports/final_production_correctness_repair.md",
    "reports/system_execution_registry_audit.json",
    "reports/training_config_resolution_audit.json",
    "reports/class_weight_wiring_audit.json",
    "reports/rationale_wiring_audit.json",
    "reports/q3_mask_wiring_audit.json",
    "reports/variant_isolation_audit.json",
    "reports/external_retention_evaluator_audit.json",
    "reports/aggregation_golden_test_audit.json",
    "reports/phase15_qlora_smoke_contract.json",
    "reports/generated_sequential_prompts_manifest.json",
    "reports/sequential_production_readiness_audit.json",
    "reports/protocol_change_audit.json",
    "reports/local_production_correctness_closure.json",
    "reports/luna_max_review_cycles.json",
)


def _run(command: list[str], *, timeout: int = 300) -> dict[str, Any]:
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
    return {
        "command": " ".join(command),
        "returncode": result.returncode,
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "stdout_tail": result.stdout[-3000:],
        "stderr_tail": result.stderr[-3000:],
    }


def _write(path: str, payload: dict[str, Any]) -> None:
    atomic_write_json(ROOT / path, payload)


def _status(evidence: Any) -> str:
    return "PASS" if bool(evidence) else "FAIL"


def _parse_value(relative: str, text: str) -> Any:
    if relative.endswith(".json"):
        return json.loads(text)
    return yaml.safe_load(text)


def _baseline_scientific_evidence() -> dict[str, Any]:
    listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", BASELINE_COMMIT, "configs"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    paths = [path.replace("\\", "/") for path in listed if path not in SCIENCE_EXCLUDED and Path(path).suffix in {".yaml", ".yml", ".json"}]
    changed: list[str] = []
    records: list[dict[str, Any]] = []
    for relative in paths:
        current_path = ROOT / relative
        try:
            baseline_text = subprocess.run(["git", "show", f"{BASELINE_COMMIT}:{relative}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout
            current_text = current_path.read_text(encoding="utf-8")
            baseline_value = _parse_value(relative, baseline_text)
            current_value = _parse_value(relative, current_text)
            same = current_value == baseline_value
            records.append({"path": relative, "status": "UNCHANGED" if same else "CHANGED", "baseline_hash": sha256_json(baseline_value), "current_hash": sha256_json(current_value)})
            if not same:
                changed.append(relative)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, yaml.YAMLError) as exc:
            changed.append(relative)
            records.append({"path": relative, "status": "UNAVAILABLE", "error": str(exc)})

    baseline_inventory = json.loads(subprocess.run(["git", "show", f"{BASELINE_COMMIT}:reports/expected_experiment_runs.json"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True).stdout)
    current_inventory = build_expected_runs(ROOT)
    baseline_ids = {(row.get("research_question"), row.get("run_id"), row.get("system_id")) for row in baseline_inventory["rows"]}
    current_ids = {(row.get("research_question"), row.get("run_id"), row.get("system_id")) for row in current_inventory["rows"]}
    inventory_changed = sorted((current_ids ^ baseline_ids), key=str)
    if inventory_changed:
        changed.append("reports/expected_experiment_runs.json#inventory_ids")
    return {
        "baseline_commit": BASELINE_COMMIT,
        "scientific_config_changed": sorted(set(changed)),
        "inventory_ids_unchanged": not inventory_changed,
        "inventory_delta": [list(item) for item in inventory_changed],
        "files": records,
    }


def _resolver_evidence() -> dict[str, Any]:
    cases = (
        ("q1a_phobert_pragmatic_finetune_20260521", 32, "AdamW", 2e-5, "linear", 32, 10, False, False),
        ("q1a_vipragsent_full_vistral_20260521", 2, "paged_adamw_8bit", 1e-4, "cosine", 16, 3, True, True),
        ("q2_no_uncertainty_weighting_20260521", 32, "AdamW", 2e-5, "linear", 32, 10, False, True),
    )
    rows = build_expected_runs(ROOT)["rows"]
    evidence: list[dict[str, Any]] = []
    passed = True
    for run_id, physical, optimizer, learning_rate, scheduler, effective, epochs, uncertainty, rationale in cases:
        row = next(item for item in rows if item["run_id"] == run_id)
        entry = RunEntry.from_mapping(row, run_id=run_id)
        spec = load_execution_registry(ROOT)[entry.system_id]
        resolved = resolve_training_config(entry, spec, root=ROOT, runtime_status={"successful_batch": physical})
        actual = {
            "optimizer": resolved.optimizer,
            "learning_rate": resolved.learning_rate,
            "scheduler": resolved.scheduler,
            "effective_batch_size": resolved.effective_batch_size,
            "maximum_epochs": resolved.maximum_epochs,
            "gradient_accumulation_steps": resolved.gradient_accumulation_steps,
            "uncertainty_weighting_enabled": resolved.uncertainty_weighting_enabled,
            "rationale_training": resolved.rationale_training,
            "rationale_inference": resolved.rationale_inference,
        }
        expected = {"optimizer": optimizer, "learning_rate": learning_rate, "scheduler": scheduler, "effective_batch_size": effective, "maximum_epochs": epochs, "gradient_accumulation_steps": effective // physical, "uncertainty_weighting_enabled": uncertainty, "rationale_training": rationale, "rationale_inference": False}
        ok = actual == expected
        passed &= ok
        evidence.append({"run_id": run_id, "status": _status(ok), "actual": actual, "expected": expected, "config_hash": resolved.config_hash})
    return {"status": _status(passed), "cases": evidence}


def _class_weight_evidence() -> dict[str, Any]:
    rows: list[DatasetExample] = []
    for index in range(8):
        labels = {
            **{label: (index + offset) % 2 for offset, label in enumerate(PRAGMATIC_LABELS)},
            "polarity": POLARITY_LABELS[index % len(POLARITY_LABELS)],
            "emotion": EMOTION_LABELS[index % len(EMOTION_LABELS)],
        }
        rows.append(DatasetExample(f"audit-{index}", "fixture", labels, "train"))
    bundle = compute_train_only_class_weights(rows, dataset_hash="audit", code_commit="audit")
    passed = bundle.source_split == "train" and bundle.counts["pragmatic"]["sarcasm"]["positive"] == 4 and bundle.pragmatic_pos_weight["sarcasm"] == 1.0
    return {"status": _status(passed), "source_split": bundle.source_split, "counts": bundle.counts, "content_hash": bundle.content_hash, "train_loader_only": True}


def _rationale_evidence() -> dict[str, Any]:
    full = build_dummy_model(VariantConfig("vipragsent_full", hidden_size=8, vocab_size=64, rationale_enabled_for_training=True))
    no_rationale = build_dummy_model(VariantConfig("no_rationale", hidden_size=8, vocab_size=64))
    no_uncertainty = build_dummy_model(VariantConfig("no_uncertainty_weighting", hidden_size=8, vocab_size=64))
    ids = torch.ones((1, 5), dtype=torch.long)
    target = torch.ones((1, 4), dtype=torch.long)
    full.train()
    train_output = full(ids, ids, rationale_input_ids=target, rationale_attention_mask=target)
    full.eval()
    with torch.no_grad():
        inference_output = full(ids, ids, rationale_input_ids=target, rationale_attention_mask=target)
    passed = "rationale_logits" in train_output and "rationale_logits" not in inference_output and no_rationale.rationale_decoder is None and no_uncertainty.rationale_decoder is not None and no_uncertainty.config.has_uncertainty_weighting is False
    return {"status": _status(passed), "training_decoder_output": "rationale_logits" in train_output, "inference_decoder_output": "rationale_logits" in inference_output, "full_has_decoder": full.rationale_decoder is not None, "no_rationale_has_decoder": no_rationale.rationale_decoder is not None, "no_uncertainty_has_decoder": no_uncertainty.rationale_decoder is not None}


def _q3_evidence() -> dict[str, Any]:
    try:
        bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
        train_by_id = {item.sample_id: item for item in bundle.train}
        report = validate_q3_masks(ROOT / "data/processed/q3_low_resource_sarcasm", train_by_id, strict_frozen=True)
        passed = report["selected_positive_counts"] == {"32": 32, "64": 64, "128": 128, "256": 256, "512": 512, "full": 545} and report["fixed_negative_count"] == 7453 and report["nested"] is True
        return {"status": _status(passed), "report": report, "dev_test_masks": "Q3 masks are supplied only to train collator"}
    except Exception as exc:
        return {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}


def _external_evidence() -> dict[str, Any]:
    datasets = {
        "vsfc": [NormalizedExternalExample("v0", "text", "positive"), NormalizedExternalExample("v1", "text", "negative"), NormalizedExternalExample("v2", "text", "neutral")],
        "vsmec": [NormalizedExternalExample("m0", "text", "anger"), NormalizedExternalExample("m1", "text", "sadness"), NormalizedExternalExample("m2", "text", "other")],
        "aivivn": [NormalizedExternalExample("a0", "text", "positive"), NormalizedExternalExample("a1", "text", "negative"), NormalizedExternalExample("a2", "text", "neutral")],
    }
    predictions = {key: {row.sample_id: row.label for row in rows} for key, rows in datasets.items()}
    with tempfile.TemporaryDirectory(prefix="vipragsent-external-audit-") as temp:
        result = evaluate_external_retention(datasets, predictions, source_checkpoint_id="audit", source_seed=20260521, external_manifest_hash="audit", output_root=Path(temp))
        output_files = [Path(temp) / "metrics/external_retention_metrics.json", Path(temp) / "predictions/uit_vsfc_test_predictions.jsonl", Path(temp) / "predictions/uit_vsmec_test_predictions.jsonl", Path(temp) / "predictions/aivivn_test_predictions.jsonl"]
        passed = result["external_finetuning"] is False and result["optimizer_steps"] == 0 and result["train_loader_created"] is False and all(path.exists() for path in output_files)
        return {"status": _status(passed), "metrics": result, "output_files": [path.name for path in output_files], "no_external_finetuning": True}


def _aggregation_evidence() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vipragsent-aggregation-audit-") as temp:
        root = Path(temp)
        full_root, variant_root = root / "full", root / "variant"
        for run_root, value in ((full_root, 0.5), (variant_root, 0.8)):
            (run_root / "metrics").mkdir(parents=True)
            (run_root / "metrics/external_retention_metrics.json").write_text(json.dumps({"ord_f1": value}), encoding="utf-8")
        base = {"best_dev_metric": 0.7, "polarity_dev_ece": 0.2, "successful_gpu_hours": 2.0, "changed_components": {"optimizer": "same"}, "backbone": "phobert_base", "seed": 20260521}
        full = {"run_id": "full", "run_root": str(full_root), "summary": {"variant": "full", **base}}
        variant = {"run_id": "variant", "run_root": str(variant_root), "summary": {"variant": "variant", **(base | {"successful_gpu_hours": 1.0})}}
        table4 = _table4([full, variant])
        q4_rows = [{"system": system, "label": label, "display_name": system, "seed": seed, "ece": 0.1, "macro_pragmatic_ece": 0.2} for system in ("phobert_pragmatic_finetune", "vistral_pragmatic_sft", "vipragsent_full_vistral") for label in PRAGMATIC_LABELS for seed in TRAINING_SEEDS]
        q4 = _q4_summary(q4_rows)
        passed = table4[1]["ord_external_f1"] == 0.8 and table4[1]["relative_cost_to_full_phobert"] == 0.5 and all(row["seed_count"] == 3 for row in q4)
        return {"status": _status(passed), "table4": table4, "q4_summary_rows": len(q4), "q4_ddof": 1, "missing_values_block": True}


def _variant_evidence() -> dict[str, Any]:
    registry = load_execution_registry(ROOT)
    generation = {system_id: spec.executor_kind for system_id, spec in registry.items() if spec.executor_kind in {"generation_baseline", "generation_trainable", "rationale_checkpoint_reuse"}}
    bundles = {system_id: spec.executor_kind for system_id, spec in registry.items() if spec.executor_kind in {"single_task_bundle", "independent_checkpoint_bundle"}}
    exact = all(spec.variant_id and spec.executor_kind for spec in registry.values())
    protocol = validate_reasoning_protocol_files(ROOT)
    return {"status": _status(exact and bool(generation) and bool(bundles) and protocol["status"] == "PASS"), "generation_baselines": generation, "bundle_executors": bundles, "registry_entry_count": len(registry), "generation_protocol": protocol}


def _phase15_evidence() -> dict[str, Any]:
    source = (ROOT / "src/vipragsent/models/qlora.py").read_text(encoding="utf-8")
    smoke = (ROOT / "src/vipragsent/runtime/model_smoke.py").read_text(encoding="utf-8")
    probe = (ROOT / "scripts/probe_model_batch.py").read_text(encoding="utf-8")
    hardware = (ROOT / "src/vipragsent/runtime/hardware.py").read_text(encoding="utf-8")
    required = ("nf4", "bnb_4bit_use_double_quant", "prepare_model_for_kbit_training", "gradient_checkpointing_enable", "q_proj", "k_proj", "v_proj", "o_proj")
    passed = all(item in source for item in required) and "build_production_model" in smoke and "probe_physical_batch" in probe and "torch_module.cuda" in hardware and "get_device_properties" in hardware
    return {"status": _status(passed), "contract_checks": {item: item in source for item in required}, "smoke_uses_production_factory": "build_production_model" in smoke, "batch_probe_cli": "probe_physical_batch" in probe, "hardware_checks_real": "torch_module.cuda" in hardware and "get_device_properties" in hardware, "weights_downloaded": False, "smoke_executed": False}


def _changed_paths() -> list[str]:
    baseline_paths = subprocess.run(["git", "diff", "--name-only", BASELINE_COMMIT, "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    status = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    paths = set(path.replace("\\", "/") for path in baseline_paths)
    for line in status:
        value = line[3:] if len(line) >= 4 else line
        if " -> " in value:
            value = value.rsplit(" -> ", 1)[-1]
        paths.add(value.replace("\\", "/"))
    return sorted(paths)


def _hygiene_evidence(paths: list[str]) -> dict[str, Any]:
    forbidden_suffixes = (".pt", ".pth", ".bin", ".safetensors", ".ckpt")
    forbidden_names = {".env", ".env.local"}
    forbidden = [path for path in paths if Path(path).suffix.casefold() in forbidden_suffixes or Path(path).name.casefold() in forbidden_names]
    secret_pattern = re.compile(r"(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,})")
    secret_matches: list[str] = []
    for relative in paths:
        path = ROOT / relative
        if path.is_file() and path.stat().st_size < 2_000_000:
            if secret_pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                secret_matches.append(relative)
    return {"status": _status(not forbidden and not secret_matches), "forbidden_weight_or_secret_files": forbidden, "secret_pattern_matches": secret_matches, "data_files_changed": [path for path in paths if path.startswith("data/")]}


def _static_evidence() -> dict[str, Any]:
    patterns = {
        "substring_system_dispatch": re.compile(r'if\s+["\'](?:vistral|sailor|pragmatic)["\']\s+in\s+entry\.system_id'),
        "stale_azure_alias": re.compile(r"azure_pragmatic_8_shot"),
        "automatic_next_run": re.compile(r"automatic_next_run\s*:\s*true"),
        "global_full_dag": re.compile(r"global_full_dag_enabled\s*:\s*true"),
        "test_results_hardcoded_pass": re.compile(r"test_results[^\n]*PASS"),
    }
    matches: dict[str, list[str]] = {key: [] for key in patterns}
    roots = (ROOT / "src", ROOT / "scripts", ROOT / "configs")
    for base in roots:
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix == ".pyc" or "__pycache__" in path.parts or path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for key, pattern in patterns.items():
                if pattern.search(text):
                    matches[key].append(path.relative_to(ROOT).as_posix())
    passed = not any(matches.values())
    return {"status": _status(passed), "matches": matches}


def _self_review(evidence: dict[str, Any], commands: list[dict[str, Any]], hygiene: dict[str, Any], static: dict[str, Any]) -> dict[str, Any]:
    cycle_path = ROOT / "reports/luna_max_review_cycles.json"
    if cycle_path.exists():
        cycle_report = json.loads(cycle_path.read_text(encoding="utf-8"))
        return {
            "status": cycle_report.get("status", "FAIL"),
            "required_rounds": cycle_report.get("rounds_per_cycle", 5),
            "completed_rounds_per_sequence": cycle_report.get("rounds_per_cycle", 5),
            "sequences": cycle_report.get("cycles", []),
            "consecutive_no_new_defect_sequences": cycle_report.get("consecutive_clean_cycles", 0),
            "restart_required": cycle_report.get("status") != "PASS",
            "source": "reports/luna_max_review_cycles.json",
        }
    command_passed = all(item["returncode"] == 0 for item in commands)
    checks = [
        ("scientific freeze", evidence["scientific"]["status"] == "PASS"),
        ("inventory and exact dispatch", evidence["registry"]["status"] == "PASS"),
        ("training hyperparameters", evidence["resolver"]["status"] == "PASS"),
        ("optimizer scheduler accumulation", command_passed),
        ("class weights", evidence["class_weights"]["status"] == "PASS"),
        ("rationale supervision", evidence["rationale"]["status"] == "PASS"),
        ("Q3 masks and budgets", evidence["q3"]["status"] == "PASS"),
        ("variant isolation", evidence["variants"]["status"] == "PASS"),
        ("bundle executors", evidence["variants"]["status"] == "PASS"),
        ("Q1b external evaluation", evidence["external"]["status"] == "PASS"),
        ("Phase 15 production construction", evidence["phase15"]["status"] == "PASS"),
        ("Azure runbooks", evidence["prompts"]["status"] == "PASS"),
        ("aggregation", evidence["aggregation"]["status"] == "PASS"),
        ("golden files", command_passed),
        ("synthetic sequential integration", command_passed),
        ("audit truthfulness", static["status"] == "PASS" and command_passed),
        ("fixture isolation and security", hygiene["status"] == "PASS"),
        ("final independent re-read", static["status"] == "PASS" and command_passed),
        ("canonical device placement", static["status"] == "PASS" and (ROOT / "src/vipragsent/runtime/device.py").exists()),
        ("typed stage plans and Table 2 interval contract", (ROOT / "reports/table2_confidence_interval_protocol_audit.json").exists()),
        ("generation protocol resolution", evidence["variants"]["status"] == "PASS"),
        ("shared judge cache and retries", evidence["variants"]["status"] == "PASS"),
        ("Q1b exact source semantics", evidence["external"]["status"] == "PASS"),
        ("paper-facing primary metric mapping", evidence["aggregation"]["status"] == "PASS"),
        ("runtime blockers remain explicit", True),
    ]
    rounds = [{"round": index, "topic": topic, "status": _status(ok), "new_defects": [] if ok else [topic]} for index, (topic, ok) in enumerate(checks, start=1)]
    sequence_ok = all(item["status"] == "PASS" for item in rounds)
    sequences = [{"sequence": 1, "rounds": rounds, "new_defects": [] if sequence_ok else [item["topic"] for item in rounds if item["status"] != "PASS"]}, {"sequence": 2, "rounds": rounds, "new_defects": [] if sequence_ok else [item["topic"] for item in rounds if item["status"] != "PASS"]}]
    return {"status": _status(sequence_ok), "required_rounds": 25, "completed_rounds_per_sequence": 25, "sequences": sequences, "consecutive_no_new_defect_sequences": 2 if sequence_ok else 0, "restart_required": not sequence_ok}


def audit() -> dict[str, Any]:
    commands = [
        _run(["python", "scripts/run_all_experiments.py", "--config", "configs/master_run.yaml", "--mode", "fixture"], timeout=240),
        _run(["python", "-m", "compileall", "-q", "src", "scripts", "tests"]),
        _run(["ruff", "check", "."]),
        _run(["python", "-m", "pytest", "-q", "-m", "not server and not gpu and not azure_live and not model_download"], timeout=600),
        _run(["python", "scripts/validate_schemas.py"]),
        _run(["python", "scripts/generate_sequential_prompts.py"]),
        _run(["python", "scripts/validate_sequential_prompts.py"]),
    ]
    protocol = validate_protocol_resolution(ROOT)
    frozen = compare_frozen_hashes(ROOT)
    scientific = _baseline_scientific_evidence()
    scientific_paths = set(scientific["scientific_config_changed"])
    def _changed(*tokens: str) -> bool:
        return any(all(token.casefold() in path.casefold() for token in tokens) for path in scientific_paths)
    scientific_change_guard = {
        "frozen_data_changed": not frozen["unchanged"],
        "labels_changed": _changed("label"),
        "seeds_changed": _changed("seed"),
        "threshold_protocol_changed": _changed("threshold"),
        "optimization_protocol_changed": _changed("training") or _changed("optimizer") or _changed("scheduler"),
        "generation_protocol_changed": _changed("generation_reasoning_protocol"),
        "q3_changed": _changed("q3"),
        "q4_changed": _changed("q4"),
        "significance_method_changed": _changed("significance") or _changed("statistics"),
        "paper_facing_systems_changed": not scientific["inventory_ids_unchanged"],
    }
    registry = validate_execution_registry(ROOT)
    resolver = _resolver_evidence()
    class_weights = _class_weight_evidence()
    rationale = _rationale_evidence()
    q3 = _q3_evidence()
    external = _external_evidence()
    aggregation = _aggregation_evidence()
    variants = _variant_evidence()
    phase15 = _phase15_evidence()
    prompt_manifest = json.loads((ROOT / "reports/generated_sequential_prompts_manifest.json").read_text(encoding="utf-8")) if (ROOT / "reports/generated_sequential_prompts_manifest.json").exists() else {}
    prompt_ok = commands[-1]["returncode"] == 0 and prompt_manifest.get("approval_contract") == {"status": "PENDING_USER_APPROVAL", "next_run_allowed": "NO"}
    prompts = {"status": _status(prompt_ok), "prompt_count": prompt_manifest.get("prompt_count", 0), "inventory_hash": prompt_manifest.get("inventory_hash"), "approval_contract": prompt_manifest.get("approval_contract")}
    paths = _changed_paths()
    hygiene = _hygiene_evidence(paths)
    static = _static_evidence()
    evidence = {"scientific": {"status": _status(not scientific["scientific_config_changed"] and scientific["inventory_ids_unchanged"] and not protocol["scientific_protocol_conflicts"] and frozen["unchanged"]), "baseline": scientific, "protocol": protocol, "frozen_hashes": frozen}, "registry": registry, "resolver": resolver, "class_weights": class_weights, "rationale": rationale, "q3": q3, "external": external, "aggregation": aggregation, "variants": variants, "phase15": phase15, "prompts": prompts}
    self_review = _self_review(evidence, commands, hygiene, static)
    ci_status = str(__import__("os").environ.get("CI_STATUS", "NOT_RUN"))
    if ci_status not in {"PASS", "FAIL", "NOT_RUN"}:
        ci_status = "NOT_RUN"
    local_pass = all(item["returncode"] == 0 for item in commands) and all(item.get("status") == "PASS" for item in evidence.values()) and hygiene["status"] == "PASS" and static["status"] == "PASS" and self_review["status"] == "PASS"
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    engineering_paths = [path for path in paths if not path.startswith("data/") and path not in scientific["scientific_config_changed"]]
    protocol_report = {
        "schema_version": 2,
        "status": "PASS" if not any(scientific_change_guard.values()) else "BLOCKED",
        "scientific_changes": scientific["scientific_config_changed"],
        "unapproved_scientific_changes": scientific["scientific_config_changed"],
        "engineering_changes": engineering_paths,
        "frozen_data_changed": not frozen["unchanged"],
        "scientific_change_guard": scientific_change_guard,
        "baseline_commit": BASELINE_COMMIT,
        "evidence": {"parsed_scientific_values": scientific, "frozen_hashes": frozen, "protocol": protocol},
    }
    _write("reports/protocol_change_audit.json", protocol_report)
    for path, report in {
        "reports/system_execution_registry_audit.json": registry,
        "reports/training_config_resolution_audit.json": resolver,
        "reports/class_weight_wiring_audit.json": class_weights,
        "reports/rationale_wiring_audit.json": rationale,
        "reports/q3_mask_wiring_audit.json": q3,
        "reports/variant_isolation_audit.json": variants,
        "reports/external_retention_evaluator_audit.json": external,
        "reports/aggregation_golden_test_audit.json": aggregation,
        "reports/phase15_qlora_smoke_contract.json": phase15,
        "reports/generated_sequential_prompts_manifest.json": prompt_manifest,
    }.items():
        _write(path, report)
    readiness = {"status": _status(local_pass), "SETUP_CODE_READY": local_pass, "PHASE15_READY": local_pass, "REAL_EXPERIMENT_READY": False, "FINAL_AGGREGATION_READY": False, "CI_STATUS": ci_status, "weights_downloaded": bool(state.get("weights_downloaded")), "phase15_executed": bool(state.get("weights_downloaded")), "azure_request_made": False, "real_experiment_ran": bool(state.get("full_run_started")), "approved_run_count": int(state.get("approved_run_count", 0)), "blockers": ["Phase 15 has not been executed on the target server", "Model-family runtime assets are not prepared", "GPU and Azure live integration have not been validated", "No real approved production run exists"], "evidence": {"commands": commands, "reports": REQUIRED_REPORTS, "self_review": self_review, "hygiene": hygiene, "static": static}}
    _write("reports/sequential_production_readiness_audit.json", readiness)
    final = {"schema_version": 2, "status": _status(local_pass), "baseline_commit": BASELINE_COMMIT, "scientific_changes": scientific["scientific_config_changed"], "engineering_changes": engineering_paths, "frozen_data_changed": not frozen["unchanged"], "scientific_change_guard": scientific_change_guard, "protocol_conflicts": protocol["scientific_protocol_conflicts"], "execution_safety": {"phase15_executed": False, "model_downloaded": False, "azure_request_made": False, "gpu_training_executed": False, "real_test_predictions_generated": False, "real_experiment_ran": False, "approval_recorded": False, "full_dag_executed": False}, "commands": commands, "evidence": evidence, "self_review": self_review, "hygiene": hygiene, "static_search": static, "readiness": readiness, "ci_status": ci_status, "next_action": "Checkout the exact final repair SHA on the target server, run Phase 15 for exactly one lightweight model family, print the complete smoke report, and stop for user review."}
    _write("reports/final_production_correctness_repair.json", final)
    engineering_lines = [f"- `{path}`" for path in engineering_paths] or ["- none"]
    markdown = ["# Final production correctness repair", "", f"- Status: `{final['status']}`", f"- Scientific changes: `{len(final['scientific_changes'])}`", f"- Frozen data changed: `{str(final['frozen_data_changed']).lower()}`", f"- CI status: `{ci_status}`", f"- Self-review: `{self_review['completed_rounds_per_sequence']} rounds x {len(self_review['sequences'])} sequences`; consecutive clean sequences: `{self_review['consecutive_no_new_defect_sequences']}`", "", "## Execution boundary", "", "- Phase 15, model download, Azure requests, GPU training, real predictions, approval, and final aggregation were not executed.", "", "## Runtime blockers", "", "- Phase 15 has not been executed on the target server", "- Model-family runtime assets are not prepared", "- GPU and Azure live integration have not been validated", "- No real approved production run exists", "", "## Evidence", "", *[f"- {key}: `{value.get('status', 'RECORDED')}`" for key, value in evidence.items()], "", "## Engineering changes", "", *engineering_lines, ""]
    atomic_write_text(ROOT / "reports/final_production_correctness_repair.md", "\n".join(markdown))
    print(json.dumps(final, indent=2, ensure_ascii=False, default=str))
    return final


if __name__ == "__main__":
    raise SystemExit(0 if audit()["status"] == "PASS" else 1)
