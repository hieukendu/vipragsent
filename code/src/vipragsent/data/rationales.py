from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .labels import validate_label_dict


def iter_rationale_inputs(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                yield {
                    "sample_id": item["sample_id"],
                    "comment": item["comment"],
                    "gold_labels": validate_label_dict(item["gold_labels"]),
                }


def rationale_only_target(text: str) -> str:
    explanation = text.strip()
    if not explanation:
        raise ValueError("Rationale text cannot be empty")
    return f"<RATIONALE>\n{explanation}\n</RATIONALE>"


def rationale_plus_labels_target(text: str, labels: dict[str, Any]) -> str:
    normalized = validate_label_dict(labels)
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return f"<RATIONALE>\n{text.strip()}\n</RATIONALE>\n<LABELS>\n{payload}\n</LABELS>"
