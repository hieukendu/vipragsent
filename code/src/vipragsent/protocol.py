from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .hashing import sha256_bytes

CONFLICT_CODES = (
    "SCIENTIFIC_PROTOCOL_CONFLICT_Q1A_VISTRAL_NO_AUXILIARY",
    "SCIENTIFIC_PROTOCOL_CONFLICT_Q1B_AZURE_PROMPT",
    "SCIENTIFIC_PROTOCOL_CONFLICT_Q3",
    "SCIENTIFIC_PROTOCOL_CONFLICT_Q4",
    "SCIENTIFIC_PROTOCOL_CONFLICT_SIGNIFICANCE_PVALUE",
)

_FROZEN_TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".txt", ".yaml", ".yml"}


def _frozen_hash_candidates(path: Path) -> set[str]:
    """Return raw and line-ending-equivalent hashes for frozen text files."""
    payload = path.read_bytes()
    candidates = {sha256_bytes(payload)}
    if path.suffix.casefold() in _FROZEN_TEXT_SUFFIXES:
        logical = payload.replace(b"\r\n", b"\n")
        candidates.add(sha256_bytes(logical))
        candidates.add(sha256_bytes(logical.replace(b"\n", b"\r\n")))
    return candidates


def _sha256_frozen_file(path: Path, expected: str | None = None) -> str:
    """Return the matching frozen hash without treating line endings as content changes."""
    candidates = _frozen_hash_candidates(path)
    if expected and expected in candidates:
        return expected
    return sha256_bytes(path.read_bytes())


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
    baseline = roles.get("vistral_baseline", {})
    q1a_ok = (
        baseline.get("system_id") == "vistral_pragmatic_sft"
        and baseline.get("loss_aggregation") == "equal_weight"
    )
    statuses["Q1A"] = "RESOLVED" if q1a_ok else "CONFLICT"

    q1b = _load_yaml(root / "configs/experiments/q1b/checkpoint_matrix.yaml")
    q1b_ok = set(q1b.get("systems", {})) == {"phobert_ordinary_single_task", "phobert_multitask", "xlmr_multitask", "sailor_multitask", "vistral_multitask"}
    statuses["Q1B"] = "RESOLVED" if q1b_ok else "CONFLICT"

    q3 = _load_yaml(root / "configs/experiments/q3/system_aliases.yaml").get("q3_system_aliases", [])
    statuses["Q3"] = "RESOLVED" if {item.get("resolved_system_id") for item in q3} == {"phobert_pragmatic_finetune", "vistral_pragmatic_sft"} and all(item.get("resolution_status") == "RESOLVED" for item in q3) else "CONFLICT"

    q4 = _load_yaml(root / "configs/experiments/q4/checkpoint_resolution.yaml").get("q4_checkpoint_resolution", [])
    q4_expected = {"vistral_pragmatic_sft"}
    q4_ids = {item.get("resolved_checkpoint_id") for item in q4}
    q4_protocol = _load_yaml(root / "configs/experiments/q4/protocol.yaml").get("q4", {})
    statuses["Q4"] = (
        "RESOLVED"
        if len(q4) == 1
        and q4_ids == q4_expected
        and all(item.get("resolution_status") == "RESOLVED" for item in q4)
        and all(item.get("required_output") == "six_pragmatic_probabilities" for item in q4)
        and q4_protocol.get("calibration_head") == "six_pragmatic_binary_heads"
        and q4_protocol.get("probability_definition") == "raw_positive_class_probability_sigmoid"
        and q4_protocol.get("bins") == 10
        and q4_protocol.get("temperature_scaling") is False
        else "CONFLICT"
    )

    significance = _load_yaml(root / "configs/statistics/significance_method.yaml")
    significance_required = {
        "method_id": "paired_hierarchical_bootstrap_sign_plus_one_v1",
        "difference_direction": "left_minus_right",
        "resamples": 10000,
        "bootstrap_seed": 20260525,
        "confidence_interval": "percentile_95",
        "finite_resample_correction": "plus_one",
        "multiple_comparisons": "holm_within_7_metric_family",
    }
    statuses["SIGNIFICANCE_PVALUE"] = (
        "RESOLVED"
        if significance.get("resolution_status") == "RESOLVED"
        and all(significance.get(key) == value for key, value in significance_required.items())
        and significance.get("raw_p_value_definition") == "two_sided_sign_test_plus_one"
        else "CONFLICT"
    )

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
        actual = _sha256_frozen_file(path, expected) if path.exists() else None
        current[relative] = actual
        if actual != expected:
            changed.append(relative)
    return {"baseline_commit": baseline.get("recorded_at_commit"), "baseline": baseline["files"], "current": current, "unchanged": not changed, "changed": changed}
