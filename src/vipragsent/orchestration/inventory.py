from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from ..constants import TRAINING_SEEDS
from ..hashing import sha256_json
from ..protocol import validate_protocol_resolution


INVENTORY_COLUMNS = ["run_id", "research_question", "system", "variant", "backbone", "seed", "budget", "task", "split", "checkpoint_role", "dependencies", "expected_outputs", "reusable_checkpoint_key", "resolution_status"]
Q3_BUDGETS = ("32", "64", "128", "256", "512", "full")


def _row(**values: Any) -> dict[str, Any]:
    return {column: values.get(column, "") for column in INVENTORY_COLUMNS}


def build_expected_runs(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    rows: list[dict[str, Any]] = []
    for system, variant, backbone, task in (
        ("phobert_pragmatic_single_task", "single_task_bundle", "phobert_base", "pragmatic"),
        ("phobert_pragmatic_finetune", "pragmatic_finetune", "phobert_base", "pragmatic"),
        ("xlmr_pragmatic_finetune", "pragmatic_finetune", "xlmr_large", "pragmatic"),
        ("sailor_pragmatic_sft", "pragmatic_sft", "sailor_7b", "pragmatic"),
        ("vistral_pragmatic_sft", "pragmatic_sft", "vistral_7b", "pragmatic"),
        ("vistral_no_auxiliary", "no_auxiliary", "vistral_7b", "pragmatic"),
        ("cot_only_vistral", "cot_only", "vistral_7b", "pragmatic"),
        ("explanation_only_vistral", "explanation_only", "vistral_7b", "pragmatic"),
        ("vipragsent_full_vistral", "full", "vistral_7b", "pragmatic"),
    ):
        for seed in TRAINING_SEEDS:
            resolution = "CONFLICT" if system in {"vistral_pragmatic_sft", "vistral_no_auxiliary"} else "RESOLVED"
            reuse_system = "vistral_pragmatic_sft" if system == "vistral_no_auxiliary" else system
            rows.append(_row(run_id=f"q1a_{system}_{seed}", research_question="Q1a", system=system, variant=variant, backbone=backbone, seed=seed, task=task, split="vipragsent_test", checkpoint_role=system, dependencies="preflight_validation;rationale_generation", expected_outputs="predictions;metrics;history", reusable_checkpoint_key=f"{reuse_system}:{seed}", resolution_status=resolution))
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
        for seed in TRAINING_SEEDS:
            rows.append(_row(run_id=f"q4_{item['resolved_checkpoint_id']}_{seed}", research_question="Q4", system=item["display_label"], variant="calibration", backbone="vistral_7b" if "vistral" in item["resolved_checkpoint_id"] else "phobert_base", seed=seed, task="polarity_ece", split="test;dev_history", checkpoint_role=item["resolved_checkpoint_id"], dependencies="reused_predictions;reused_histories", expected_outputs="reliability_bins;learning_curve", reusable_checkpoint_key=f"{item['resolved_checkpoint_id']}:{seed}", resolution_status=item["resolution_status"]))
    for system, backbone in (("vipragsent_full_phobert", "phobert_base"), ("vipragsent_full_vistral", "vistral_7b")):
        for seed in TRAINING_SEEDS:
            rows.append(_row(run_id=f"backbone_sensitivity_{system}_{seed}", research_question="backbone_sensitivity", system=system, variant="full", backbone=backbone, seed=seed, task="pragmatic;ordinary;polarity_ece;profiling", split="test", checkpoint_role=system, dependencies="reused_predictions;reused_profiles", expected_outputs="backbone_sensitivity", reusable_checkpoint_key=f"{system}:{seed}"))
    protocol = validate_protocol_resolution(root)
    inventory = {"schema_version": 1, "source": "configs/experiments/master_matrix.yaml and locked supporting registry", "training_seeds": list(TRAINING_SEEDS), "q3_budgets": list(Q3_BUDGETS), "scientific_protocol_conflicts": protocol["scientific_protocol_conflicts"], "rows": rows, "counts_by_question": {question: sum(row["research_question"] == question for row in rows) for question in ("Q1a", "Q1b", "Q2", "Q3", "Q4", "backbone_sensitivity")}, "derived_run_count": len(rows), "inventory_hash": sha256_json(rows)}
    validate_inventory(inventory)
    return inventory


def validate_inventory(inventory: dict[str, Any]) -> None:
    rows = list(inventory.get("rows", []))
    required = {"run_id", "research_question", "system", "variant", "backbone", "task", "split", "checkpoint_role", "dependencies", "expected_outputs", "reusable_checkpoint_key"}
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
