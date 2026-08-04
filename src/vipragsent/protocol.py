from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .hashing import sha256_file


CONFLICT_CODES = (
    "SCIENTIFIC_PROTOCOL_CONFLICT_Q1A_VISTRAL_NO_AUXILIARY",
    "SCIENTIFIC_PROTOCOL_CONFLICT_Q1B_AZURE_PROMPT",
    "SCIENTIFIC_PROTOCOL_CONFLICT_Q3",
    "SCIENTIFIC_PROTOCOL_CONFLICT_Q4",
    "SCIENTIFIC_PROTOCOL_CONFLICT_SIGNIFICANCE_PVALUE",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping: {path}")
    return value


def validate_protocol_resolution(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    errors: list[str] = []
    statuses: dict[str, str] = {}

    q1a = _load_yaml(root / "configs/experiments/q1a/system_roles.yaml")
    roles = q1a.get("q1a", {}).get("roles", {})
    baseline = {key: value for key, value in roles.get("vistral_baseline", {}).items() if key != "system_id"}
    no_auxiliary = {key: value for key, value in roles.get("no_auxiliary", {}).items() if key != "system_id"}
    if baseline == no_auxiliary:
        statuses["Q1A"] = "CONFLICT"
    else:
        statuses["Q1A"] = "RESOLVED"

    q1b = _load_yaml(root / "configs/experiments/q1b/checkpoint_matrix.yaml")
    azure = q1b.get("systems", {}).get("azure_gpt41_mini", {})
    q1b_ok = bool(azure.get("polarity_prompt_id") and azure.get("emotion_prompt_id"))
    for prompt_id, task in ((azure.get("polarity_prompt_id"), "polarity"), (azure.get("emotion_prompt_id"), "emotion")):
        prompt_path = root / "data/manifests/prompts" / f"{prompt_id}.json" if prompt_id else None
        if prompt_path is None or not prompt_path.exists():
            q1b_ok = False
        else:
            prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
            q1b_ok = q1b_ok and prompt.get("task") == task and len(prompt.get("sample_ids", [])) == 8
    statuses["Q1B"] = "RESOLVED" if q1b_ok else "CONFLICT"

    q3 = _load_yaml(root / "configs/experiments/q3/system_aliases.yaml").get("q3_system_aliases", [])
    statuses["Q3"] = "RESOLVED" if q3 and all(item.get("resolution_status") == "RESOLVED" for item in q3) and len({item.get("paper_label") for item in q3}) == len(q3) else "CONFLICT"

    q4 = _load_yaml(root / "configs/experiments/q4/checkpoint_resolution.yaml").get("q4_checkpoint_resolution", [])
    statuses["Q4"] = "RESOLVED" if q4 and all(item.get("resolution_status") == "RESOLVED" for item in q4) else "CONFLICT"

    significance = _load_yaml(root / "configs/statistics/significance_method.yaml")
    statuses["SIGNIFICANCE_PVALUE"] = "RESOLVED" if significance.get("resolution_status") == "RESOLVED" and significance.get("method_id") and significance.get("raw_p_value_definition") else "CONFLICT"

    if statuses["Q1A"] == "CONFLICT":
        errors.append(CONFLICT_CODES[0])
    if statuses["Q1B"] == "CONFLICT":
        errors.append(CONFLICT_CODES[1])
    if statuses["Q3"] == "CONFLICT":
        errors.append(CONFLICT_CODES[2])
    if statuses["Q4"] == "CONFLICT":
        errors.append(CONFLICT_CODES[3])
    if statuses["SIGNIFICANCE_PVALUE"] == "CONFLICT":
        errors.append(CONFLICT_CODES[4])
    return {"scientific_protocol_conflicts": errors, "resolution_status": statuses, "passed": not errors}


def compare_frozen_hashes(root: str | Path, baseline_path: str | Path = "reports/phase_14_5_frozen_hash_baseline.json") -> dict[str, Any]:
    root = Path(root)
    baseline = json.loads((root / baseline_path).read_text(encoding="utf-8"))
    current: dict[str, str | None] = {}
    changed: list[str] = []
    for relative, expected in baseline["files"].items():
        path = root / relative
        actual = sha256_file(path) if path.exists() else None
        current[relative] = actual
        if actual != expected:
            changed.append(relative)
    return {"baseline_commit": baseline.get("recorded_at_commit"), "baseline": baseline["files"], "current": current, "unchanged": not changed, "changed": changed}
