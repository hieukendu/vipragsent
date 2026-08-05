from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from ..constants import TRAINING_SEEDS
from ..hashing import sha256_file, sha256_json
from ..protocol import validate_protocol_resolution

INVENTORY_COLUMNS = [
    "experiment_id", "run_id", "research_question", "system_id", "system", "display_name", "variant",
    "backbone", "seed", "budget", "task", "split", "dependencies", "required_phase15_assets",
    "checkpoint_role", "expected_outputs", "reusable_checkpoint_key", "selection_metric", "evaluation_protocol",
    "execution_kind", "model_repository", "model_revision", "tokenizer_revision", "preprocessing_name",
    "preprocessing_version", "source_checkpoint_id", "q3_mask_path", "q3_mask_hash", "training_applicability",
    "approval_required", "execution_status", "approval_status", "protocol_resolution_status", "resolution_status",
]
Q3_BUDGETS = ("32", "64", "128", "256", "512", "full")
DISPLAY_NAMES = {
    "phobert_pragmatic_single_task": "PhoBERT pragmatic single-task bundle",
    "phobert_pragmatic_finetune": "PhoBERT pragmatic fine-tune",
    "xlmr_pragmatic_finetune": "XLM-R pragmatic fine-tune",
    "sailor_pragmatic_sft": "Sailor-7B pragmatic SFT",
    "vistral_pragmatic_sft": "Vistral-7B pragmatic SFT",
    "vipragsent_no_auxiliary_vistral": "ViPragSent - no auxiliary losses",
    "cot_only_vistral": "Vistral CoT-only",
    "explanation_only_vistral": "Vistral explanation-only",
    "vipragsent_full_vistral": "Full ViPragSent Vistral",
    "vipragsent_full_phobert": "Full ViPragSent PhoBERT",
}


def _row(**values: Any) -> dict[str, Any]:
    values.setdefault("experiment_id", values.get("run_id"))
    values.setdefault("system_id", values.get("system"))
    values.setdefault("system", values.get("system_id"))
    values.setdefault("display_name", DISPLAY_NAMES.get(str(values.get("system_id")), str(values.get("system_id"))))
    values.setdefault("required_phase15_assets", "azure_deployment;prompt_manifest" if values.get("backbone") == "azure" else "model_weights;tokenizer;runtime_profile")
    question = str(values.get("research_question", "")).casefold()
    values.setdefault("selection_metric", {
        "q1a": "macro_prag_f1_dev",
        "q1b": "ord_external_f1",
        "q2": "macro_prag_f1_dev",
        "q3": "sarcasm_dev_f1",
        "q4": "macro_pragmatic_ece_test;dev_macro_pragmatic_f1_by_epoch",
        "backbone_sensitivity": "macro_prag_f1_test",
    }.get(question, "not_applicable"))
    values.setdefault("evaluation_protocol", {
        "q1a": "q1a_frozen_dev_threshold_v1",
        "q1b": "q1b_external_retention_v1",
        "q2": "q2_ablation_v1",
        "q3": "q3_low_resource_masked_v1",
        "q4": "q4_pragmatic_calibration_v1",
        "backbone_sensitivity": "backbone_sensitivity_v1",
    }.get(question, "setup_preflight_v1"))
    values.setdefault("approval_required", True)
    values.setdefault("execution_status", "NOT_STARTED")
    values.setdefault("approval_status", "PENDING_USER_APPROVAL")
    values.setdefault("protocol_resolution_status", values.get("resolution_status", "RESOLVED"))
    values.setdefault("resolution_status", "RESOLVED")
    if "execution_kind" not in values:
        if values.get("backbone") == "azure":
            values["execution_kind"] = "azure"
        elif str(values.get("research_question", "")).casefold() == "q4":
            values["execution_kind"] = "artifact_extraction"
        elif str(values.get("research_question", "")).casefold() == "q1b":
            values["execution_kind"] = "evaluation_only"
        elif "reused_predictions" in str(values.get("dependencies", "")):
            values["execution_kind"] = "checkpoint_reuse"
        else:
            values["execution_kind"] = "trainable"
    values.setdefault("training_applicability", "NOT_APPLICABLE" if values["execution_kind"] in {"evaluation_only", "checkpoint_reuse", "artifact_extraction", "azure"} else "APPLICABLE")
    values.setdefault("model_repository", "" if values.get("backbone") == "azure" else values.get("backbone", ""))
    values.setdefault("model_revision", "" if values.get("backbone") == "azure" else values.get("model_revision", ""))
    values.setdefault("tokenizer_revision", "" if values.get("backbone") == "azure" else values.get("tokenizer_revision", ""))
    values.setdefault("preprocessing_name", "azure_prompt_protocol" if values.get("backbone") == "azure" else "vncorenlp_rdrsegmenter")
    values.setdefault("preprocessing_version", "locked-v1")
    values.setdefault("source_checkpoint_id", values.get("reusable_checkpoint_key", ""))
    values.setdefault("q3_mask_path", "" if values.get("research_question") != "Q3" else f"data/processed/q3_low_resource_sarcasm/budget_{values.get('budget')}_masks.csv")
    values.setdefault("q3_mask_hash", "")
    return {column: values.get(column, "") for column in INVENTORY_COLUMNS}


def build_expected_runs(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    registry_path = root / "configs/models/model_registry.yaml"
    registry_payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    registry = registry_payload.get("models", {})
    rows: list[dict[str, Any]] = []
    for system, variant, backbone, task in (
        ("phobert_pragmatic_single_task", "single_task_bundle", "phobert_base", "pragmatic"),
        ("phobert_pragmatic_finetune", "pragmatic_finetune", "phobert_base", "pragmatic"),
        ("xlmr_pragmatic_finetune", "pragmatic_finetune", "xlmr_large", "pragmatic"),
        ("sailor_pragmatic_sft", "pragmatic_sft", "sailor_7b", "pragmatic"),
        ("vistral_pragmatic_sft", "pragmatic_sft", "vistral_7b", "pragmatic"),
        ("vipragsent_no_auxiliary_vistral", "no_auxiliary", "vistral_7b", "pragmatic"),
        ("cot_only_vistral", "cot_only", "vistral_7b", "pragmatic"),
        ("explanation_only_vistral", "explanation_only", "vistral_7b", "pragmatic"),
        ("vipragsent_full_vistral", "full", "vistral_7b", "pragmatic"),
    ):
        for seed in TRAINING_SEEDS:
            rows.append(_row(run_id=f"q1a_{system}_{seed}", research_question="Q1a", system=system, variant=variant, backbone=backbone, seed=seed, task=task, split="vipragsent_test", checkpoint_role=system, dependencies="preflight_validation;rationale_generation", expected_outputs="predictions;metrics;history", reusable_checkpoint_key=f"{system}:{seed}"))
    for system, variant in (("azure_gpt41_mini_zeroshot", "zero_shot"), ("azure_gpt41_mini_8shot", "eight_shot")):
        rows.append(_row(run_id=f"q1a_{system}", research_question="Q1a", system=system, variant=variant, backbone="azure", seed="", task="pragmatic", split="vipragsent_test", checkpoint_role=system, dependencies="preflight_validation", expected_outputs="predictions;usage", reusable_checkpoint_key=system))
    q1b_systems = (("phobert_pol_single", "phobert_base", "polarity"), ("phobert_emo_single", "phobert_base", "emotion"), ("phobert_multitask_8head", "phobert_base", "polarity;emotion"), ("xlmr_multitask_8head", "xlmr_large", "polarity;emotion"), ("sailor_multitask_8head", "sailor_7b", "polarity;emotion"), ("vistral_multitask_8head", "vistral_7b", "polarity;emotion"), ("vipragsent_full_phobert", "phobert_base", "polarity;emotion"))
    for system, backbone, task in q1b_systems:
        for seed in TRAINING_SEEDS:
            rows.append(_row(run_id=f"q1b_{system}_{seed}", research_question="Q1b", system=system, variant="table3_checkpoint", backbone=backbone, seed=seed, task=task, split="external_test", checkpoint_role=system, dependencies="table3_checkpoint_training", expected_outputs="external_predictions;metrics", reusable_checkpoint_key=f"{system}:{seed}"))
    rows.append(_row(run_id="q1b_azure_gpt41_mini", research_question="Q1b", system="azure_gpt41_mini", variant="dedicated_prompts", backbone="azure", seed=None, task="polarity;emotion", split="external_test", checkpoint_role="azure_gpt41_mini", dependencies="azure_prompted_baselines", expected_outputs="external_predictions;metrics", reusable_checkpoint_key="azure_gpt41_mini:dedicated_prompts", resolution_status="RESOLVED"))
    for variant in ("full", "no_emotion_auxiliary", "no_polarity_auxiliary", "no_rationale", "no_uncertainty_weighting"):
        for seed in TRAINING_SEEDS:
            rows.append(_row(run_id=f"q2_{variant}_{seed}", research_question="Q2", system=f"{variant}_phobert", variant=variant, backbone="phobert_base", seed=seed, task="pragmatic;polarity;emotion", split="dev;external_test", checkpoint_role=f"{variant}_phobert", dependencies="phobert_jobs;table3_checkpoint_training", expected_outputs="metrics;predictions;history", reusable_checkpoint_key=f"{variant}_phobert:{seed}"))
    for seed in TRAINING_SEEDS:
        rows.append(_row(run_id=f"q2_no_multitask_bundle_{seed}", research_question="Q2", system="no_multitask", variant="no_multitask", backbone="phobert_base", seed=seed, task="pragmatic;polarity;emotion", split="dev;external_test", checkpoint_role="independent_checkpoint_bundle", dependencies="phobert_jobs", expected_outputs="bundle_metrics;predictions", reusable_checkpoint_key=f"no_multitask_bundle:{seed}"))
    for system, variant, backbone in (("phobert_pragmatic_finetune", "q3_budgeted", "phobert_base"), ("xlmr_pragmatic_finetune", "q3_budgeted", "xlmr_large"), ("vistral_pragmatic_sft", "q3_budgeted", "vistral_7b"), ("vipragsent_full_vistral", "q3_budgeted", "vistral_7b")):
        for budget in Q3_BUDGETS:
            for seed in TRAINING_SEEDS:
                rows.append(_row(run_id=f"q3_{system}_{budget}_{seed}", research_question="Q3", system=system, variant=variant, backbone=backbone, seed=seed, budget=budget, task="sarcasm;rationale;other_tasks", split="dev;test", checkpoint_role=system, dependencies="q3_low_resource", expected_outputs="q3_metrics;thresholds;mask_provenance", reusable_checkpoint_key=f"{system}:{budget}:{seed}"))
    for budget in Q3_BUDGETS:
            rows.append(_row(run_id=f"q3_azure_gpt41_mini_8shot_{budget}", research_question="Q3", system="azure_gpt41_mini_8shot", variant="q3_eight_shot", backbone="azure", seed=None, budget=budget, task="sarcasm", split="test", checkpoint_role="azure_gpt41_mini_8shot", dependencies="azure_prompted_baselines", expected_outputs="q3_predictions;usage", reusable_checkpoint_key=f"azure_gpt41_mini_8shot:{budget}", resolution_status="RESOLVED"))
    q4_resolution = yaml.safe_load((root / "configs/experiments/q4/checkpoint_resolution.yaml").read_text(encoding="utf-8"))["q4_checkpoint_resolution"]
    for item in q4_resolution:
        if item["resolution_status"] != "RESOLVED":
            continue
        system = item["resolved_checkpoint_id"]
        for seed in TRAINING_SEEDS:
            rows.append(_row(run_id=f"q4_{system}_{seed}", research_question="Q4", system=system, display_name=DISPLAY_NAMES.get(system, system), variant="pragmatic_calibration", backbone=item["backbone"], seed=seed, task="pragmatic_ece;learning_curve", split="vipragsent_test;vipragsent_dev_history", checkpoint_role=system, dependencies="reused_predictions;reused_histories", expected_outputs="q4_per_seed;reliability_bins;learning_curve", reusable_checkpoint_key=f"{system}:{seed}"))
    for system, backbone in (("vipragsent_full_phobert", "phobert_base"), ("vipragsent_full_vistral", "vistral_7b")):
        for seed in TRAINING_SEEDS:
            rows.append(_row(run_id=f"backbone_sensitivity_{system}_{seed}", research_question="backbone_sensitivity", system=system, variant="full", backbone=backbone, seed=seed, task="pragmatic;ordinary;polarity_ece;profiling", split="test", checkpoint_role=system, dependencies="reused_predictions;reused_profiles", expected_outputs="backbone_sensitivity", reusable_checkpoint_key=f"{system}:{seed}"))
    for row in rows:
        spec = registry.get(row.get("backbone"), {})
        if row.get("backbone") != "azure":
            row["model_repository"] = spec.get("repo_id", row.get("model_repository", ""))
            row["model_revision"] = spec.get("revision", row.get("model_revision", ""))
            row["tokenizer_revision"] = spec.get("tokenizer_revision", row.get("tokenizer_revision", ""))
        if row.get("research_question") == "Q3":
            mask = root / str(row["q3_mask_path"])
            row["q3_mask_hash"] = sha256_file(mask) if mask.exists() else ""
    execution_registry_path = root / "configs/experiments/system_execution_registry.yaml"
    if execution_registry_path.exists():
        execution_registry = yaml.safe_load(execution_registry_path.read_text(encoding="utf-8")) or {}
        execution_specs = {str(item.get("system_id")): dict(item) for item in execution_registry.get("systems", [])}
        execution_by_system = {system_id: str(item.get("executor_kind")) for system_id, item in execution_specs.items()}
        for row in rows:
            system_id = str(row.get("system_id"))
            registry_spec = execution_specs.get(system_id, {})
            executor_kind = execution_by_system.get(system_id)
            if executor_kind in {"single_task_bundle", "independent_checkpoint_bundle"}:
                row["execution_kind"] = "component_bundle"
            elif executor_kind in {"generation_baseline", "generation_trainable"}:
                row["execution_kind"] = "generation"
            elif executor_kind == "rationale_checkpoint_reuse":
                row["execution_kind"] = "checkpoint_reuse"
            if row.get("research_question") == "Q1a":
                if bool(registry_spec.get("rationale_training", False)):
                    row["dependencies"] = "preflight_validation;rationale_generation"
                elif system_id == "explanation_only_vistral":
                    row["dependencies"] = "preflight_validation;approved_full_vistral_same_seed_source"
                else:
                    row["dependencies"] = "preflight_validation"
            elif row.get("research_question") == "Q1b":
                row["dependencies"] = "approved_azure_output" if row.get("backbone") == "azure" else "approved_source_checkpoint"
            elif row.get("research_question") == "Q4":
                row["dependencies"] = "approved_source_predictions;approved_source_training_history"
            dependencies = [item for item in str(row.get("dependencies", "")).split(";") if item]
            rationale_required = bool(registry_spec.get("rationale_training", False)) and row.get("execution_kind") in {"trainable", "component_bundle", "generation"}
            if rationale_required and "rationale_generation" not in dependencies:
                dependencies.append("rationale_generation")
            if not rationale_required:
                dependencies = [item for item in dependencies if item != "rationale_generation"]
            row["dependencies"] = ";".join(dependencies)
            row["training_applicability"] = "APPLICABLE" if bool(registry_spec.get("additional_training", row.get("execution_kind") in {"trainable", "component_bundle", "generation"})) else "NOT_APPLICABLE"
    protocol = validate_protocol_resolution(root)
    inventory = {"schema_version": 1, "source": "configs/experiments/master_matrix.yaml and locked supporting registry", "training_seeds": list(TRAINING_SEEDS), "q3_budgets": list(Q3_BUDGETS), "scientific_protocol_conflicts": protocol["scientific_protocol_conflicts"], "rows": rows, "counts_by_question": {question: sum(row["research_question"] == question for row in rows) for question in ("Q1a", "Q1b", "Q2", "Q3", "Q4", "backbone_sensitivity")}, "derived_run_count": len(rows), "inventory_hash": sha256_json(rows)}
    validate_inventory(inventory)
    return inventory


def validate_inventory(inventory: dict[str, Any]) -> None:
    rows = list(inventory.get("rows", []))
    required = {"experiment_id", "run_id", "research_question", "system_id", "system", "display_name", "variant", "backbone", "task", "split", "dependencies", "required_phase15_assets", "checkpoint_role", "expected_outputs", "reusable_checkpoint_key", "selection_metric", "evaluation_protocol", "execution_kind", "training_applicability", "approval_required", "execution_status", "approval_status", "protocol_resolution_status"}
    missing = [row.get("run_id", "<missing>") for row in rows if not required.issubset(row) or any(row.get(key) in {"", None} for key in required)]
    if missing:
        raise ValueError(f"Inventory rows are missing required semantic fields: {missing[:5]}")
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Inventory contains duplicate run IDs")
    semantic_keys: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["research_question"], row["system"], row["variant"], row["backbone"], row.get("seed"), row.get("budget"), row["task"], row["split"])
        semantic_keys.setdefault(key, []).append(row)
    duplicates = [items for items in semantic_keys.values() if len(items) > 1 and not any(item.get("resolution_status") == "CONFLICT" for item in items)]
    if duplicates:
        raise ValueError(f"Inventory contains semantic duplicates: {[item[0]['run_id'] for item in duplicates[:3]]}")
    reuse_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["research_question"], row.get("seed"), row.get("budget"), row["reusable_checkpoint_key"])
        reuse_groups.setdefault(key, []).append(row)
    unresolved_reuse = [items for items in reuse_groups.values() if len(items) > 1 and not any(item.get("resolution_status") == "CONFLICT" for item in items)]
    if unresolved_reuse:
        raise ValueError(f"Inventory reuses a checkpoint across unresolved rows: {[item[0]['run_id'] for item in unresolved_reuse[:3]]}")
    trainable = [row for row in rows if row["backbone"] != "azure"]
    if any(row.get("seed") not in TRAINING_SEEDS for row in trainable):
        raise ValueError("Every trainable inventory row must use one of the three locked training seeds")
    q3 = [row for row in rows if row["research_question"] == "Q3"]
    if any(row.get("budget") not in Q3_BUDGETS for row in q3):
        raise ValueError("Q3 inventory has an invalid budget")
    azure_q1b = [row for row in rows if row["research_question"] == "Q1b" and row["backbone"] == "azure"]
    if len(azure_q1b) != 1 or azure_q1b[0].get("seed") is not None:
        raise ValueError("Q1b must contain exactly one non-seeded Azure row")
    if any(row.get("execution_kind") not in {"trainable", "component_bundle", "generation", "checkpoint_reuse", "evaluation_only", "azure", "artifact_extraction"} for row in rows):
        raise ValueError("Inventory contains an unsupported execution_kind")
    for row in rows:
        dependencies = {item for item in str(row.get("dependencies", "")).split(";") if item}
        system_id = str(row.get("system_id"))
        if row.get("research_question") == "Q1a":
            if system_id == "cot_only_vistral" and "rationale_generation" not in dependencies:
                raise ValueError("cot_only_vistral must depend on the approved rationale source")
            if system_id == "explanation_only_vistral" and "approved_full_vistral_same_seed_source" not in dependencies:
                raise ValueError("explanation_only_vistral must depend on an exact same-seed full Vistral source")
            if system_id not in {"cot_only_vistral", "explanation_only_vistral", "full_phobert", "no_emotion_auxiliary_phobert", "no_polarity_auxiliary_phobert", "no_uncertainty_weighting_phobert", "vipragsent_full_vistral"} and "rationale_generation" in dependencies:
                raise ValueError(f"non-rationale Q1a system has a rationale dependency: {system_id}")
        if row.get("research_question") == "Q1b" and not ({"approved_source_checkpoint", "approved_azure_output"} & dependencies):
            raise ValueError(f"Q1b row lacks an approved source dependency: {row.get('run_id')}")
        if row.get("research_question") == "Q4" and not {"approved_source_predictions", "approved_source_training_history"}.issubset(dependencies):
            raise ValueError(f"Q4 row lacks approved source dependencies: {row.get('run_id')}")


def write_expected_runs(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    inventory = build_expected_runs(root)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "expected_experiment_runs.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    with (reports / "expected_experiment_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(inventory["rows"])
    return inventory
