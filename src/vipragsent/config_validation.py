from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .constants import ALL_LABEL_KEYS, EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS


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
    return {"passed": not errors, "errors": errors, "canonical_label_keys": list(ALL_LABEL_KEYS), "model_count": len(registry["models"])}
