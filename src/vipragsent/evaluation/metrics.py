from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..constants import PRAGMATIC_LABELS


def _f1_for_class(true: np.ndarray, pred: np.ndarray, positive: int) -> float:
    tp = int(np.sum((true == positive) & (pred == positive)))
    fp = int(np.sum((true != positive) & (pred == positive)))
    fn = int(np.sum((true == positive) & (pred != positive)))
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def binary_macro_f1(true: Sequence[int], pred: Sequence[int]) -> float:
    y_true = np.asarray(true, dtype=int)
    y_pred = np.asarray(pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError("true and pred must have identical shape")
    return (_f1_for_class(y_true, y_pred, 0) + _f1_for_class(y_true, y_pred, 1)) / 2.0


def multiclass_macro_f1(true: Sequence[Any], pred: Sequence[Any], labels: Sequence[Any]) -> float:
    y_true = np.asarray(true)
    y_pred = np.asarray(pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("true and pred must have identical shape")
    return float(np.mean([_f1_for_class(y_true, y_pred, label) for label in labels]))


def macro_pragmatic_f1(true: Mapping[str, Sequence[int]], pred: Mapping[str, Sequence[int]]) -> float:
    if set(true) != set(PRAGMATIC_LABELS) or set(pred) != set(PRAGMATIC_LABELS):
        raise ValueError("Pragmatic prediction keys differ")
    return float(np.mean([binary_macro_f1(true[key], pred[key]) for key in PRAGMATIC_LABELS]))


def pragmatic_reliability_bins(
    true: Mapping[str, Sequence[int]],
    probabilities: Mapping[str, Sequence[float]],
    *,
    bins: int = 10,
) -> dict[str, list[dict[str, float | int]]]:
    """Build raw-positive-probability reliability bins for each pragmatic label."""
    if set(true) != set(PRAGMATIC_LABELS) or set(probabilities) != set(PRAGMATIC_LABELS):
        raise ValueError("Pragmatic calibration requires all six canonical labels")
    if bins != 10:
        raise ValueError("The locked pragmatic ECE protocol uses exactly ten bins")
    edges = np.linspace(0.0, 1.0, bins + 1)
    output: dict[str, list[dict[str, float | int]]] = {}
    for label in PRAGMATIC_LABELS:
        y_true = np.asarray(true[label], dtype=int)
        probs = np.asarray(probabilities[label], dtype=float)
        if y_true.ndim != 1 or probs.ndim != 1 or y_true.shape != probs.shape:
            raise ValueError(f"Calibration inputs are not aligned for {label}")
        if np.any((probs < 0.0) | (probs > 1.0)):
            raise ValueError(f"Calibration probabilities are outside [0, 1] for {label}")
        rows: list[dict[str, float | int]] = []
        for index in range(bins):
            lower, upper = float(edges[index]), float(edges[index + 1])
            mask = (probs >= lower) & ((probs < upper) if index < bins - 1 else (probs <= upper))
            count = int(mask.sum())
            mean_confidence = float(probs[mask].mean()) if count else 0.0
            empirical_rate = float(y_true[mask].mean()) if count else 0.0
            rows.append({
                "bin_index": index,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "empirical_positive_rate": empirical_rate,
                "absolute_gap": abs(mean_confidence - empirical_rate) if count else 0.0,
            })
        output[label] = rows
    return output


def pragmatic_ece(
    true: Mapping[str, Sequence[int]],
    probabilities: Mapping[str, Sequence[float]],
    *,
    bins: int = 10,
) -> tuple[dict[str, float], float, dict[str, list[dict[str, float | int]]]]:
    """Compute per-label and macro pragmatic ECE without thresholding or pooling seeds."""
    reliability = pragmatic_reliability_bins(true, probabilities, bins=bins)
    ece_by_label: dict[str, float] = {}
    for label, rows in reliability.items():
        total = sum(int(row["count"]) for row in rows)
        ece_by_label[label] = float(sum(int(row["count"]) / total * float(row["absolute_gap"]) for row in rows if row["count"]) if total else 0.0)
    return ece_by_label, float(np.mean([ece_by_label[label] for label in PRAGMATIC_LABELS])), reliability


def reliability_bins(
    true: Sequence[int], probabilities: Sequence[Sequence[float]], *, bins: int = 10
) -> list[dict[str, float | int | None]]:
    y_true = np.asarray(true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    confidence = probs.max(axis=1)
    labels = probs.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    output: list[dict[str, float | int]] = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence >= lower) & ((confidence < upper) if index < bins - 1 else (confidence <= upper))
        count = int(mask.sum())
        output.append({
            "bin": index,
            "lower": float(lower),
            "upper": float(upper),
            "count": count,
            "mean_confidence": float(confidence[mask].mean()) if count else None,
            "accuracy": float((labels[mask] == y_true[mask]).mean()) if count else None,
        })
    return output


def expected_calibration_error(true: Sequence[int], probabilities: Sequence[Sequence[float]], *, bins: int = 10) -> float:
    rows = reliability_bins(true, probabilities, bins=bins)
    total = sum(int(row["count"]) for row in rows)
    if not total:
        return 0.0
    return float(sum(int(row["count"]) / total * abs(float(row["accuracy"]) - float(row["mean_confidence"])) for row in rows if row["count"]))


def missing_prediction_report(sample_ids: Sequence[str], predictions: Mapping[str, Any]) -> dict[str, Any]:
    if len(predictions) != len(set(predictions)):
        raise ValueError("Prediction IDs must be unique")
    missing = [sample_id for sample_id in sample_ids if sample_id not in predictions]
    return {"requested": len(sample_ids), "returned": len(predictions), "missing": len(missing), "missing_sample_ids": missing}


def align_prediction_rows(sample_ids: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    expected = list(sample_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("Reference sample IDs must be unique")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in by_id:
            raise ValueError(f"Duplicate or missing prediction sample ID: {sample_id!r}")
        by_id[sample_id] = row
    missing = [sample_id for sample_id in expected if sample_id not in by_id]
    extra = sorted(set(by_id) - set(expected))
    if missing or extra:
        raise ValueError(f"Prediction alignment mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return [by_id[sample_id] for sample_id in expected]


def q1a_pragmatic_metrics(true: Mapping[str, Sequence[int]], pred: Mapping[str, Sequence[int]]) -> dict[str, float]:
    return {**{f"{key}_f1": binary_macro_f1(true[key], pred[key]) for key in PRAGMATIC_LABELS}, "macro_prag_f1": macro_pragmatic_f1(true, pred)}
