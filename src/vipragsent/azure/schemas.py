from __future__ import annotations

from typing import Any

from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS


def strict_label_schema(task: str = "all") -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    if task in {"all", "pragmatic"}:
        for key in PRAGMATIC_LABELS:
            properties[key] = {"type": "integer", "enum": [0, 1]}
            required.append(key)
    if task in {"all", "polarity"}:
        properties["polarity"] = {"type": "string", "enum": list(POLARITY_LABELS)}
        required.append("polarity")
    if task in {"all", "emotion"}:
        properties["emotion"] = {"type": "string", "enum": list(EMOTION_LABELS)}
        required.append("emotion")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def validate_structured_output(value: dict[str, Any], task: str = "all") -> dict[str, Any]:
    schema = strict_label_schema(task)
    expected = set(schema["required"])
    if set(value) != expected:
        raise ValueError(f"Structured output keys differ from {task} schema")
    normalized = dict(value)
    for key in PRAGMATIC_LABELS:
        if key in normalized and normalized[key] not in (0, 1):
            raise ValueError(f"Invalid binary output for {key}")
    if "polarity" in normalized and normalized["polarity"] not in POLARITY_LABELS:
        raise ValueError("Invalid polarity output")
    if "emotion" in normalized and normalized["emotion"] not in EMOTION_LABELS:
        raise ValueError("Invalid emotion output")
    return normalized


def strict_rationale_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"rationale": {"type": "string"}},
        "required": ["rationale"],
        "additionalProperties": False,
    }


def validate_rationale_output(value: dict[str, Any]) -> str:
    if set(value) != {"rationale"} or not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise ValueError("Rationale Structured Output must contain exactly a non-empty rationale string")
    return value["rationale"].strip()
