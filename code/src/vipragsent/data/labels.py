from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..constants import ALL_LABEL_KEYS, EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS


def _as_binary(value: Any, key: str) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be binary, got {value!r}") from exc
    if number not in (0, 1):
        raise ValueError(f"{key} must be 0 or 1, got {value!r}")
    return number


def validate_label_dict(labels: Mapping[str, Any], *, require_all: bool = True) -> dict[str, Any]:
    expected = set(ALL_LABEL_KEYS)
    actual = set(labels)
    if require_all and actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Label keys differ from canonical schema; missing={missing}, extra={extra}")
    if not require_all and not actual.issubset(expected):
        raise ValueError(f"Unknown label keys: {sorted(actual - expected)}")
    result: dict[str, Any] = {}
    for key in PRAGMATIC_LABELS:
        if key in labels:
            result[key] = _as_binary(labels[key], key)
    if "polarity" in labels:
        value = str(labels["polarity"])
        if value not in POLARITY_LABELS:
            raise ValueError(f"Invalid polarity: {value!r}")
        result["polarity"] = value
    if "emotion" in labels:
        value = str(labels["emotion"])
        if value not in EMOTION_LABELS:
            raise ValueError(f"Invalid emotion: {value!r}")
        result["emotion"] = value
    return result


def encode_labels(labels: Mapping[str, Any]) -> tuple[int, ...]:
    normalized = validate_label_dict(labels)
    return tuple(normalized[key] for key in PRAGMATIC_LABELS) + (
        POLARITY_LABELS.index(normalized["polarity"]),
        EMOTION_LABELS.index(normalized["emotion"]),
    )


def decode_labels(encoded: Sequence[int]) -> dict[str, Any]:
    if len(encoded) != len(ALL_LABEL_KEYS):
        raise ValueError(f"Expected {len(ALL_LABEL_KEYS)} encoded labels, got {len(encoded)}")
    result = {key: _as_binary(encoded[index], key) for index, key in enumerate(PRAGMATIC_LABELS)}
    polarity_index = int(encoded[6])
    emotion_index = int(encoded[7])
    try:
        result["polarity"] = POLARITY_LABELS[polarity_index]
        result["emotion"] = EMOTION_LABELS[emotion_index]
    except IndexError as exc:
        raise ValueError("Encoded class index is out of range") from exc
    return result
