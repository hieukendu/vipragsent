from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json
from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ..hashing import sha256_json


@dataclass(frozen=True)
class ClassWeightBundle:
    pragmatic_pos_weight: dict[str, float]
    polarity_weight: dict[str, float]
    emotion_weight: dict[str, float]
    counts: dict[str, dict[str, int]]
    source_split: str
    dataset_hash: str
    code_commit: str

    @property
    def content_hash(self) -> str:
        return sha256_json(self._payload())

    def _payload(self) -> dict[str, Any]:
        return asdict(self)

    def as_dict(self) -> dict[str, Any]:
        return self._payload() | {"content_hash": self.content_hash}


def _inverse_frequency(labels: list[str], allowed: tuple[str, ...]) -> tuple[dict[str, int], dict[str, float]]:
    counts = Counter(labels)
    missing = [label for label in allowed if counts[label] == 0]
    if missing:
        raise ValueError(f"Training-only class counts have zero frequency: {missing}")
    total = len(labels)
    weights = {label: float(total / (len(allowed) * counts[label])) for label in allowed}
    return {label: int(counts[label]) for label in allowed}, weights


def compute_train_only_class_weights(
    examples: Iterable[Any],
    *,
    dataset_hash: str,
    code_commit: str,
    source_split: str = "train",
) -> ClassWeightBundle:
    rows = list(examples)
    if not rows:
        raise ValueError("Cannot compute class weights from an empty training split")
    if any(str(getattr(row, "split", source_split)) != source_split for row in rows):
        raise ValueError("Class weights may only be computed from the frozen training split")
    pragmatic_counts: dict[str, dict[str, int]] = {}
    pragmatic_weights: dict[str, float] = {}
    for label in PRAGMATIC_LABELS:
        positives = sum(int(row.labels[label]) for row in rows)
        negatives = len(rows) - positives
        if positives == 0 or negatives == 0:
            raise ValueError(f"Training-only pragmatic class has zero active count for {label}")
        pragmatic_counts[label] = {"negative": negatives, "positive": positives}
        pragmatic_weights[label] = float(negatives / positives)
    polarity_counts, polarity_weights = _inverse_frequency([str(row.labels["polarity"]) for row in rows], POLARITY_LABELS)
    emotion_counts, emotion_weights = _inverse_frequency([str(row.labels["emotion"]) for row in rows], EMOTION_LABELS)
    return ClassWeightBundle(
        pragmatic_pos_weight=pragmatic_weights,
        polarity_weight=polarity_weights,
        emotion_weight=emotion_weights,
        counts={"pragmatic": pragmatic_counts, "polarity": polarity_counts, "emotion": emotion_counts},
        source_split=source_split,
        dataset_hash=dataset_hash,
        code_commit=code_commit,
    )


def persist_class_weights(root: str | Path, run_root: str | Path, weights: ClassWeightBundle) -> dict[str, str]:
    root = Path(root)
    path = Path(run_root) / "training/class_weights.json"
    atomic_write_json(path, weights.as_dict())
    try:
        display_path = path.relative_to(root).as_posix()
    except ValueError:
        display_path = path.as_posix()
    return {"path": display_path, "sha256": weights.content_hash}


def synthetic_class_weights(*, dataset_hash: str = "synthetic", code_commit: str = "synthetic") -> ClassWeightBundle:
    """Return deterministic non-uniform weights for CPU-only fixture integration tests."""
    return ClassWeightBundle(
        pragmatic_pos_weight={label: 2.0 + index for index, label in enumerate(PRAGMATIC_LABELS)},
        polarity_weight={label: 1.0 + index / 10 for index, label in enumerate(POLARITY_LABELS)},
        emotion_weight={label: 1.0 + index / 10 for index, label in enumerate(EMOTION_LABELS)},
        counts={"pragmatic": {label: {"negative": 2, "positive": 1} for label in PRAGMATIC_LABELS}, "polarity": {label: 2 for label in POLARITY_LABELS}, "emotion": {label: 2 for label in EMOTION_LABELS}},
        source_split="train",
        dataset_hash=dataset_hash,
        code_commit=code_commit,
    )
