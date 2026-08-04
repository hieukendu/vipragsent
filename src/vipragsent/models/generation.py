from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from ..data.labels import validate_label_dict


@dataclass(frozen=True)
class ParsedGeneration:
    rationale: str
    labels: dict[str, Any]
    repaired_punctuation: bool = False


def repair_json_punctuation(value: str) -> str:
    repaired = re.sub(r",\s*([}\]])", r"\1", value)
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    return repaired


def parse_cot_generation(text: str) -> ParsedGeneration:
    rationale_match = re.search(r"<RATIONALE>\s*(.*?)\s*</RATIONALE>", text, flags=re.DOTALL)
    labels_match = re.search(r"<LABELS>\s*(.*?)\s*</LABELS>", text, flags=re.DOTALL)
    if not rationale_match or not labels_match:
        raise ValueError("Generation must contain RATIONALE and LABELS blocks")
    raw_json = labels_match.group(1).strip()
    repaired_json = repair_json_punctuation(raw_json)
    repaired = repaired_json != raw_json
    try:
        parsed = json.loads(repaired_json)
    except json.JSONDecodeError as exc:
        raise ValueError("LABELS block is not valid JSON; semantic repair is prohibited") from exc
    labels = validate_label_dict(parsed)
    return ParsedGeneration(rationale=rationale_match.group(1).strip(), labels=labels, repaired_punctuation=repaired)
