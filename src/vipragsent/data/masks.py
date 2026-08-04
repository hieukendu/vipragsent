from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from ..constants import EXPECTED_SPLIT_COUNTS


REQUIRED_MASK_COLUMNS = {
    "sample_id",
    "is_sarcasm_positive",
    "positive_selected_for_budget",
    "sarcasm_target_mask",
    "rationale_loss_mask",
}
EXPECTED_BUDGETS = ("32", "64", "128", "256", "512", "full")


def read_mask(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_q3_masks(q3_dir: str | Path, train_by_id: dict[str, Any]) -> dict[str, Any]:
    root = Path(q3_dir)
    masks: dict[str, dict[str, dict[str, str]]] = {}
    for budget in EXPECTED_BUDGETS:
        rows = read_mask(root / f"budget_{budget}_masks.csv")
        if len(rows) != len(train_by_id) or set(rows[0]) != REQUIRED_MASK_COLUMNS:
            raise ValueError(f"Invalid Q3 mask schema or row count for budget {budget}")
        by_id: dict[str, dict[str, str]] = {}
        for row in rows:
            sample_id = row["sample_id"]
            if sample_id in by_id or sample_id not in train_by_id:
                raise ValueError(f"Invalid Q3 mask sample ID: {sample_id}")
            by_id[sample_id] = row
            positive = int(row["is_sarcasm_positive"])
            selected = int(row["positive_selected_for_budget"])
            sarcasm_mask = int(row["sarcasm_target_mask"])
            rationale_mask = int(row["rationale_loss_mask"])
            if positive not in (0, 1) or selected not in (0, 1) or sarcasm_mask not in (0, 1) or rationale_mask not in (0, 1):
                raise ValueError(f"Non-binary Q3 mask value for {sample_id}")
            if positive == 0 and selected != 0:
                raise ValueError(f"Negative sample selected as Q3 positive: {sample_id}")
            if positive == 1 and ((selected == 1) != (sarcasm_mask == 1)):
                raise ValueError(f"Positive selection and sarcasm mask disagree: {sample_id}")
            if positive == 1 and ((selected == 1) != (rationale_mask == 1)):
                raise ValueError(f"Positive selection and rationale mask disagree: {sample_id}")
        masks[budget] = by_id
    positive_ids = {sample_id for sample_id, example in train_by_id.items() if int(example.labels["sarcasm"]) == 1}
    negative_ids = set(train_by_id) - positive_ids
    selected_sets: dict[str, set[str]] = {}
    for budget, rows in masks.items():
        selected_sets[budget] = {sample_id for sample_id, row in rows.items() if int(row["positive_selected_for_budget"]) == 1}
        if not selected_sets[budget].issubset(positive_ids):
            raise ValueError(f"Q3 {budget} contains a non-positive selected sample")
        if any(int(rows[sample_id]["sarcasm_target_mask"]) != 1 for sample_id in negative_ids):
            raise ValueError(f"Q3 {budget} does not retain all negative sarcasm targets")
    ordered = list(EXPECTED_BUDGETS)
    for left, right in zip(ordered, ordered[1:]):
        if not selected_sets[left].issubset(selected_sets[right]):
            raise ValueError(f"Q3 masks are not nested: {left} is not a subset of {right}")
    counts = {budget: len(selected_sets[budget]) for budget in EXPECTED_BUDGETS}
    return {
        "budgets": list(EXPECTED_BUDGETS),
        "selected_positive_counts": counts,
        "fixed_negative_count": len(negative_ids),
        "positive_count_full": len(positive_ids),
        "nested": True,
        "expected_train_rows": EXPECTED_SPLIT_COUNTS["train"],
    }
