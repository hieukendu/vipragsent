from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExternalExample:
    sample_id: str
    text: str
    label: str


def load_external_csv(path: str | Path, *, label_column: str) -> list[ExternalExample]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "text", label_column}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"External file is missing columns {sorted(required)}: {path}")
    return [ExternalExample(row["sample_id"], row["text"], row[label_column]) for row in rows]


def validate_external_labels(examples: list[ExternalExample], allowed: set[str]) -> dict[str, Any]:
    invalid = sorted({example.label for example in examples if example.label not in allowed})
    if invalid:
        raise ValueError(f"Invalid external labels: {invalid}")
    if len({example.sample_id for example in examples}) != len(examples):
        raise ValueError("External sample IDs must be unique")
    return {"rows": len(examples), "unique_sample_ids": True, "labels": sorted({e.label for e in examples})}
