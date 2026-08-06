from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .constants import ALL_LABEL_KEYS, EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from .protocol import validate_protocol_resolution


def validate_config_tree(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    errors: list[str] = []
    labels_path = root / "configs/labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if tuple(labels.get("canonical_keys", [])) != ALL_LABEL_KEYS:
        errors.append("configs/labels.json canonical keys differ")
    if tuple(labels.get("pragmatic_binary_fields", [])) != PRAGMATIC_LABELS:
        errors.append("configs/labels.json pragmatic keys differ")
    if tuple(labels.get("polarity", [])) != POLARITY_LABELS:
        errors.append("configs/labels.json polarity classes differ")
    if tuple(labels.get("emotion", [])) != EMOTION_LABELS:
        errors.append("configs/labels.json emotion classes differ")
    paper_roles = yaml.safe_load((root / "configs/paper_roles.yaml").read_text(encoding="utf-8"))
    if paper_roles["paper_roles"]["table_2_headline"]["backbone"] != "vistral_7b":
        errors.append("Table 2 role is not Vistral")
    if paper_roles["paper_roles"]["table_3_retention"]["backbone"] != "phobert_base":
        errors.append("Table 3 role is not PhoBERT")
    if paper_roles["paper_roles"]["table_4_ablation"]["backbone"] != "phobert_base":
        errors.append("Table 4 role is not PhoBERT")
    if paper_roles["paper_roles"]["q4_calibration"]["systems"] != ["phobert_pragmatic_finetune", "vistral_pragmatic_sft", "vipragsent_full_vistral"]:
        errors.append("Q4 role registry does not use the three approved pragmatic systems")
    master = yaml.safe_load((root / "configs/master_run.yaml").read_text(encoding="utf-8"))
    execution_policy_path = root / "configs/execution_policy.yaml"
    if not execution_policy_path.exists():
        errors.append("configs/execution_policy.yaml is missing")
    else:
        execution_policy = yaml.safe_load(execution_policy_path.read_text(encoding="utf-8"))
        if execution_policy.get("execution_policy") != "sequential_review_gated":
            errors.append("standalone execution policy is not sequential_review_gated")
    for key, expected in {
        "execution_policy": "sequential_review_gated",
        "global_full_dag_enabled": False,
        "maximum_concurrent_gpu_jobs": 1,
        "automatic_next_run": False,
        "require_user_approval_after_each_run": True,
    }.items():
        if master.get(key) != expected:
            errors.append(f"master execution policy {key} is not {expected!r}")
    registry = yaml.safe_load((root / "configs/models/model_registry.yaml").read_text(encoding="utf-8"))
    for name, model in registry["models"].items():
        if not model.get("repo_id") or not model.get("revision") or not model.get("tokenizer_revision"):
            errors.append(f"{name} is not pinned to an immutable revision")
    q1b = yaml.safe_load((root / "configs/experiments/q1b/checkpoint_matrix.yaml").read_text(encoding="utf-8"))
    for name in ("phobert_ordinary_single_task", "phobert_multitask", "xlmr_multitask", "sailor_multitask", "vistral_multitask", "vipragsent", "azure_gpt41_mini"):
        if name not in q1b["systems"]:
            errors.append(f"Table 3 checkpoint matrix missing {name}")
    active_config_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (root / "configs").rglob("*" ) if path.is_file())
    if "explanation_at_inference" in active_config_text or "Figure 5" in active_config_text:
        errors.append("prohibited component appears in active configuration")
    q1a = yaml.safe_load((root / "configs/experiments/q1a/system_roles.yaml").read_text(encoding="utf-8"))["q1a"]
    baseline = q1a["roles"]["vistral_baseline"]
    no_auxiliary = q1a["roles"]["no_auxiliary"]
    if no_auxiliary.get("system_id") != "vipragsent_no_auxiliary_vistral" or baseline.get("system_id") == no_auxiliary.get("system_id"):
        errors.append("Q1a no-auxiliary system is not a distinct identity")
    if no_auxiliary.get("loss_aggregation") != "homoscedastic_uncertainty" or no_auxiliary.get("trainable_uncertainty_parameters") != "six_independent_pragmatic":
        errors.append("Q1a no-auxiliary loss fingerprint is incomplete")
    q4 = yaml.safe_load((root / "configs/experiments/q4/protocol.yaml").read_text(encoding="utf-8"))["q4"]
    q4_calibration_path = root / "configs/experiments/q4/pragmatic_calibration.yaml"
    if not q4_calibration_path.exists():
        errors.append("Q4 pragmatic calibration config is missing")
    else:
        q4_calibration = yaml.safe_load(q4_calibration_path.read_text(encoding="utf-8"))["q4_pragmatic_calibration"]
        if q4_calibration.get("systems") != ["phobert_pragmatic_finetune", "vistral_pragmatic_sft", "vipragsent_full_vistral"]:
            errors.append("Q4 pragmatic calibration config has the wrong systems")
    if q4.get("calibration_head") != "six_pragmatic_binary_heads" or q4.get("probability_definition") != "raw_positive_class_probability_sigmoid":
        errors.append("Q4 does not use the approved six-pragmatic raw-probability calibration")
    protocol = validate_protocol_resolution(root)
    if "full_model" in yaml.safe_load((root / "configs/losses.yaml").read_text(encoding="utf-8")):
        full_loss = yaml.safe_load((root / "configs/losses.yaml").read_text(encoding="utf-8"))["full_model"]
        if full_loss.get("independent_uncertainty_parameters") != 8:
            errors.append("Full model must own eight uncertainty parameters")
    return {"passed": not errors, "errors": errors, "canonical_label_keys": list(ALL_LABEL_KEYS), "model_count": len(registry["models"]), "scientific_protocol_conflicts": protocol["scientific_protocol_conflicts"]}
