from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ..hashing import sha256_json
from .schemas import strict_label_schema


@dataclass(frozen=True)
class Demonstration:
    sample_id: str
    text: str
    labels: dict[str, Any]


@dataclass(frozen=True)
class PromptSpec:
    task: str
    version: str
    text: str
    schema: dict[str, Any]
    demonstration_ids: tuple[str, ...]
    prompt_hash: str


def _row_label(row: Any, key: str) -> Any:
    labels = row.labels if hasattr(row, "labels") else row
    return labels[key]


def _demo(row: Any) -> Demonstration:
    return Demonstration(row.sample_id, row.text, dict(row.labels))


def build_demo_manifest(train: Iterable[Any], *, budget: str | None = None, eligible_ids: set[str] | None = None) -> dict[str, Any]:
    rows = list(train)
    if budget is not None:
        eligible = eligible_ids or {row.sample_id for row in rows if getattr(row, "q3_eligible", True)}
        rows = [row for row in rows if row.sample_id in eligible]
    chosen: list[Demonstration] = []
    used: set[str] = set()
    for key in PRAGMATIC_LABELS:
        candidates = [row for row in rows if _row_label(row, key) == 1 and row.sample_id not in used]
        if not candidates:
            raise ValueError(f"No eligible demonstration for {key}")
        row = sorted(candidates, key=lambda item: item.sample_id)[0]
        chosen.append(_demo(row))
        used.add(row.sample_id)
    controls_positive = [row for row in rows if all(_row_label(row, key) == 0 for key in PRAGMATIC_LABELS) and _row_label(row, "polarity") == "positive" and row.sample_id not in used]
    controls_negative = [row for row in rows if all(_row_label(row, key) == 0 for key in PRAGMATIC_LABELS) and _row_label(row, "polarity") == "negative" and row.sample_id not in used]
    if not controls_positive or not controls_negative:
        raise ValueError("Could not find ordinary positive and negative control demonstrations")
    chosen.extend([_demo(sorted(controls_positive, key=lambda item: item.sample_id)[0]), _demo(sorted(controls_negative, key=lambda item: item.sample_id)[0])])
    return {
        "composition": [*PRAGMATIC_LABELS, "ordinary_positive_control", "ordinary_negative_control"],
        "budget": budget,
        "sample_ids": [item.sample_id for item in chosen],
        "demonstrations": [{"sample_id": item.sample_id, "text": item.text, "labels": item.labels} for item in chosen],
        "source_split": "vipragsent_train_only",
    }


def validate_demo_manifest(manifest: Mapping[str, Any], *, q3_eligible_ids: set[str] | None = None) -> None:
    ids = list(manifest.get("sample_ids", []))
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("Every pragmatic 8-shot manifest must contain eight unique IDs")
    if q3_eligible_ids is not None and not set(ids).issubset(q3_eligible_ids):
        raise ValueError("Q3 demonstration is outside the eligible budget pool")
    demos = manifest.get("demonstrations", [])
    if len(demos) != 8:
        raise ValueError("Demonstration payload must contain eight examples")
    for key, demo in zip(PRAGMATIC_LABELS, demos[:6]):
        if int(demo["labels"][key]) != 1:
            raise ValueError(f"Designated demonstration is not positive for {key}")
    for demo, polarity in zip(demos[6:], ("positive", "negative")):
        if any(int(demo["labels"][key]) != 0 for key in PRAGMATIC_LABELS) or demo["labels"]["polarity"] != polarity:
            raise ValueError("Invalid ordinary control demonstration")


class PromptRegistry:
    def __init__(self, manifest: Mapping[str, Any]) -> None:
        validate_demo_manifest(manifest)
        self.manifest = dict(manifest)

    def _render(self, task: str, target_text: str, demonstrations: list[Mapping[str, Any]]) -> PromptSpec:
        blocks = []
        for demo in demonstrations:
            blocks.append(f"<DEMO id='{demo['sample_id']}'>\nTEXT: {demo['text']}\nLABELS: {json.dumps(demo['labels'], ensure_ascii=False, sort_keys=True)}\n</DEMO>")
        text = f"Task: classify the Vietnamese comment using strict JSON.\n\n{chr(10).join(blocks)}\n\nINPUT:\n{target_text}"
        schema = {"strict": True, "schema": strict_label_schema(task)}
        return PromptSpec(task, "v1", text, schema, tuple(demo["sample_id"] for demo in demonstrations), sha256_json({"text": text, "schema": schema}))

    def pragmatic(self, text: str) -> PromptSpec:
        return self._render("pragmatic", text, self.manifest["demonstrations"])

    def polarity(self, text: str, demonstrations: list[Mapping[str, Any]]) -> PromptSpec:
        return self._render("polarity", text, demonstrations)

    def emotion(self, text: str, demonstrations: list[Mapping[str, Any]]) -> PromptSpec:
        return self._render("emotion", text, demonstrations)
