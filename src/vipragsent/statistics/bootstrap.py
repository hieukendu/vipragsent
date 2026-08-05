from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

APPROVED_P_VALUE_METHOD = "paired_hierarchical_bootstrap_sign_plus_one_v1"


def _width(values: object) -> int:
    if isinstance(values, dict):
        first = next(iter(values.values()))
        return len(first)
    return len(values)  # type: ignore[arg-type]


def _slice(values: object, indices: np.ndarray) -> object:
    if isinstance(values, dict):
        return {key: [sequence[int(index)] for index in indices] for key, sequence in values.items()}
    sequence = values  # type: ignore[assignment]
    return [sequence[int(index)] for index in indices]


@dataclass(frozen=True)
class BootstrapResult:
    observed: float
    ci_low: float
    ci_high: float
    distribution: list[float]
    p_value: float | None = None


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile, method="linear"))


def sign_plus_one_two_sided_p_value(distribution: Sequence[float]) -> float:
    """Return the approved finite-resample-corrected two-sided sign p-value."""
    values = np.asarray(distribution, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("A non-empty one-dimensional bootstrap distribution is required")
    resamples = values.size
    p_lower = (1.0 + float(np.count_nonzero(values <= 0.0))) / (resamples + 1.0)
    p_upper = (1.0 + float(np.count_nonzero(values >= 0.0))) / (resamples + 1.0)
    return min(1.0, 2.0 * min(p_lower, p_upper))


def _p_value_for_method(distribution: Sequence[float], method: str | None) -> float | None:
    if method is None:
        return None
    if method != APPROVED_P_VALUE_METHOD:
        raise ValueError(f"Unknown or prohibited p-value method: {method}")
    return sign_plus_one_two_sided_p_value(distribution)


def hierarchical_bootstrap(
    seed_predictions: Sequence[tuple[Sequence[object], Sequence[object]]],
    metric: Callable[[Sequence[object], Sequence[object]], float],
    *,
    resamples: int = 1000,
    seed: int = 20260525,
) -> BootstrapResult:
    if not seed_predictions:
        raise ValueError("At least one seed run is required")
    width = _width(seed_predictions[0][0])
    if width == 0 or any(_width(true) != width or _width(pred) != width for true, pred in seed_predictions):
        raise ValueError("All seed predictions must have the same non-zero width")
    observed = float(np.mean([metric(true, pred) for true, pred in seed_predictions]))
    rng = np.random.default_rng(seed)
    distribution: list[float] = []
    for _ in range(resamples):
        seed_indices = rng.integers(0, len(seed_predictions), size=len(seed_predictions))
        example_indices = rng.integers(0, width, size=width)
        scores: list[float] = []
        for seed_index in seed_indices:
            true, pred = seed_predictions[int(seed_index)]
            scores.append(metric(_slice(true, example_indices), _slice(pred, example_indices)))
        distribution.append(float(np.mean(scores)))
    values = np.asarray(distribution)
    return BootstrapResult(observed, _percentile(values, 2.5), _percentile(values, 97.5), distribution)


def paired_bootstrap_comparison(
    left: Sequence[tuple[Sequence[object], Sequence[object]]],
    right: Sequence[tuple[Sequence[object], Sequence[object]]],
    metric: Callable[[Sequence[object], Sequence[object]], float],
    *,
    resamples: int = 1000,
    seed: int = 20260525,
    p_value_method: str | None = None,
) -> BootstrapResult:
    if len(left) != len(right) or not left:
        raise ValueError("Paired systems must have equal non-zero seed runs")
    left_width = _width(left[0][0])
    right_width = _width(right[0][0])
    if left_width == 0 or right_width != left_width:
        raise ValueError("Paired systems must have identical non-zero example widths")
    for (left_true, left_pred), (right_true, right_pred) in zip(left, right, strict=True):
        if _width(left_true) != left_width or _width(left_pred) != left_width or _width(right_true) != left_width or _width(right_pred) != left_width:
            raise ValueError("Paired bootstrap inputs are not aligned")
    observed = float(np.mean([metric(a_true, a_pred) - metric(b_true, b_pred) for (a_true, a_pred), (b_true, b_pred) in zip(left, right)]))
    width = left_width
    rng = np.random.default_rng(seed)
    distribution: list[float] = []
    for _ in range(resamples):
        seed_indices = rng.integers(0, len(left), size=len(left))
        example_indices = rng.integers(0, width, size=width)
        deltas = []
        for seed_index in seed_indices:
            a_true, a_pred = left[int(seed_index)]
            b_true, b_pred = right[int(seed_index)]
            deltas.append(
                metric(_slice(a_true, example_indices), _slice(a_pred, example_indices))
                - metric(_slice(b_true, example_indices), _slice(b_pred, example_indices))
            )
        distribution.append(float(np.mean(deltas)))
    values = np.asarray(distribution)
    p_value = _p_value_for_method(distribution, p_value_method)
    return BootstrapResult(observed, _percentile(values, 2.5), _percentile(values, 97.5), distribution, p_value)


def paired_bootstrap_trainable_vs_azure(
    trainable: Sequence[tuple[Sequence[object], Sequence[object]]],
    azure: tuple[Sequence[object], Sequence[object]],
    metric: Callable[[Sequence[object], Sequence[object]], float],
    *,
    resamples: int = 1000,
    seed: int = 20260525,
    p_value_method: str | None = None,
) -> BootstrapResult:
    if not trainable:
        raise ValueError("At least one trainable seed run is required")
    width = _width(trainable[0][0])
    if _width(azure[0]) != width:
        raise ValueError("Azure and trainable predictions must have the same width")
    observed = float(np.mean([metric(true, pred) for true, pred in trainable]) - metric(*azure))
    rng = np.random.default_rng(seed)
    distribution: list[float] = []
    for _ in range(resamples):
        seed_indices = rng.integers(0, len(trainable), size=len(trainable))
        example_indices = rng.integers(0, width, size=width)
        train_scores = [metric(_slice(trainable[int(index)][0], example_indices), _slice(trainable[int(index)][1], example_indices)) for index in seed_indices]
        azure_score = metric(_slice(azure[0], example_indices), _slice(azure[1], example_indices))
        distribution.append(float(np.mean(train_scores) - azure_score))
    values = np.asarray(distribution)
    p_value = _p_value_for_method(distribution, p_value_method)
    return BootstrapResult(observed, _percentile(values, 2.5), _percentile(values, 97.5), distribution, p_value)


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    if any(not 0.0 <= float(value) <= 1.0 for value in p_values):
        raise ValueError("P-values must be in [0, 1]")
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    for rank, (index, value) in enumerate(indexed):
        corrected = min(1.0, (len(p_values) - rank) * float(value))
        previous = max(previous, corrected)
        adjusted[index] = previous
    return adjusted
