from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..constants import PRAGMATIC_LABELS
from .metrics import binary_macro_f1


def tune_binary_threshold(true: Sequence[int], probabilities: Sequence[float], *, start: float = 0.05, stop: float = 0.95, step: float = 0.01) -> float:
    y_true = np.asarray(true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    candidates = np.round(np.arange(start, stop + step / 2, step), 2)
    scores = [(binary_macro_f1(y_true, (probs >= threshold).astype(int)), float(threshold)) for threshold in candidates]
    scores.sort(key=lambda item: (-item[0], abs(item[1] - 0.5), item[1]))
    return scores[0][1]


def tune_pragmatic_thresholds(
    true: Mapping[str, Sequence[int]], probabilities: Mapping[str, Sequence[float]], **kwargs: float
) -> dict[str, float]:
    if set(true) != set(probabilities):
        raise ValueError("Threshold inputs must have identical label keys")
    return {key: tune_binary_threshold(true[key], probabilities[key], **kwargs) for key in PRAGMATIC_LABELS if key in true}
