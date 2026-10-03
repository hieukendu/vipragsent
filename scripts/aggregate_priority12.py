"""Aggregate structurally valid Priority 1/2 runs without outcome filtering."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.constants import PRAGMATIC_LABELS
from vipragsent.evaluation.metrics import binary_macro_f1, macro_pragmatic_f1, pragmatic_ece
from vipragsent.statistics.bootstrap import paired_bootstrap_comparison

SEEDS = (20260521, 20260522, 20260523)
Q1_VARIANTS = ("full", "no_emotion_auxiliary", "no_polarity_auxiliary", "no_rationale", "no_uncertainty_weighting")
Q3_BUDGETS = ("32", "64", "128", "256", "512", "full")
RESULT_ROOT = ROOT / "results/runs"
OUTPUT_ROOT = ROOT / "reports/priority12_v37"
Q1_RUN_PREFIX = "priority12_v37_q1"
FULL_SOURCE_ROOT = ROOT / "reports/q1a_matched_xlmr_ft/proposed_source/campaigns/vipragsent-v8-local-mig2g20gb"
FULL_SOURCE_PREFIX = "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregation_root(run_id: str) -> tuple[Path, bool]:
    local_root = RESULT_ROOT / run_id
    if local_root.exists():
        return local_root, False
    prefix = f"{Q1_RUN_PREFIX}_full_"
    if run_id.startswith(prefix):
        seed = run_id.removeprefix(prefix)
        source_root = FULL_SOURCE_ROOT / f"{FULL_SOURCE_PREFIX}_{seed}"
        if source_root.exists():
            return source_root, True
    return local_root, False


def _rows_from_root(root: Path) -> list[dict[str, Any]]:
    path = root / "predictions/test_predictions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _positive_class_metrics(rows: list[dict[str, Any]], label: str) -> dict[str, float | int]:
    true = np.asarray([int(row["gold"][label]) for row in rows], dtype=int)
    pred = np.asarray([int(row["predictions"][label]) for row in rows], dtype=int)
    tp = int(np.sum((true == 1) & (pred == 1)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _complete_source_metrics(metrics: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize the existing V37 full metrics to the current ablation schema."""
    per_label = {
        label: binary_macro_f1(
            [int(row["gold"][label]) for row in rows],
            [int(row["predictions"][label]) for row in rows],
        )
        for label in PRAGMATIC_LABELS
    }
    macro = macro_pragmatic_f1(
        {label: [int(row["gold"][label]) for row in rows] for label in PRAGMATIC_LABELS},
        {label: [int(row["predictions"][label]) for row in rows] for label in PRAGMATIC_LABELS},
    )
    probabilities = {
        label: [float(row["probabilities"][label]) for row in rows]
        for label in PRAGMATIC_LABELS
    }
    truths = {label: [int(row["gold"][label]) for row in rows] for label in PRAGMATIC_LABELS}
    ece_by_label, macro_ece, reliability = pragmatic_ece(truths, probabilities)
    completed = dict(metrics)
    completed["prediction_count"] = len(rows)
    completed["per_label_f1"] = per_label
    completed["macro_pragmatic_f1"] = float(macro)
    completed["sarcasm_binary_macro_f1"] = float(
        binary_macro_f1(truths["sarcasm"], [int(row["predictions"]["sarcasm"]) for row in rows])
    )
    completed["positive_class_metrics"] = {
        label: _positive_class_metrics(rows, label) for label in PRAGMATIC_LABELS
    }
    completed["pragmatic_ece"] = {"per_label": ece_by_label, "macro": macro_ece, "bins": reliability}
    return completed


def _mean_sd(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": float(np.mean(values)) if values else float("nan"),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _run_summary(run_id: str) -> dict[str, Any]:
    root, is_existing_full = _aggregation_root(run_id)
    rows = _rows_from_root(root)
    if is_existing_full:
        manifest = _read(root / "optimization_manifest.json")
        if manifest.get("status") != "PASS" or manifest.get("variant_id") != "vipragsent_full_xlmr_large":
            raise RuntimeError(f"Existing V37 full source is not structurally valid: {run_id}")
        metrics = _complete_source_metrics(_read(root / "metrics/test_metrics.json"), rows)
        return {
            "run_id": run_id,
            "seed": manifest["seed"],
            "metrics": metrics,
            "rows": rows,
            "source": str(root.relative_to(ROOT)),
        }
    manifest = _read(root / "manifest.json")
    validation = _read(root / "validation.json")
    if manifest.get("status") != "PASS" or validation.get("status") != "PASS":
        raise RuntimeError(f"Run is not structurally valid: {run_id}")
    metrics = _read(root / "metrics/test_metrics.json")
    return {"run_id": run_id, "seed": manifest["seed"], "metrics": metrics, "rows": rows}


def _metric_values(items: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for item in items:
        value: Any = item
        for key in path:
            value = value[key]
        values.append(float(value))
    return values


def _summary_table(items: list[dict[str, Any]]) -> dict[str, Any]:
    paths: dict[str, tuple[str, ...]] = {
        "macro_pragmatic_f1": ("metrics", "macro_pragmatic_f1"),
        "sarcasm_binary_macro_f1": ("metrics", "sarcasm_binary_macro_f1"),
        "pragmatic_ece_macro": ("metrics", "pragmatic_ece", "macro"),
    }
    for label in PRAGMATIC_LABELS:
        paths[f"{label}_f1"] = ("metrics", "per_label_f1", label)
        paths[f"{label}_positive_precision"] = ("metrics", "positive_class_metrics", label, "precision")
        paths[f"{label}_positive_recall"] = ("metrics", "positive_class_metrics", label, "recall")
        paths[f"{label}_positive_f1"] = ("metrics", "positive_class_metrics", label, "f1")
    for task in ("polarity_macro_f1", "emotion_macro_f1"):
        if all(task in item["metrics"] for item in items):
            paths[task] = ("metrics", task)
    return {name: _mean_sd(_metric_values(items, path)) for name, path in paths.items()}


def _paired(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: Callable[[Any, Any], float],
    *,
    name: str,
) -> dict[str, Any]:
    left_pairs = []
    right_pairs = []
    for a, b in zip(left, right, strict=True):
        if [row["sample_id"] for row in a["rows"]] != [row["sample_id"] for row in b["rows"]]:
            raise ValueError(f"Paired prediction order mismatch for {name}")
        left_pairs.append((
            [row["gold"] for row in a["rows"]],
            [row["predictions"] for row in a["rows"]],
        ))
        right_pairs.append((
            [row["gold"] for row in b["rows"]],
            [row["predictions"] for row in b["rows"]],
        ))
    result = paired_bootstrap_comparison(
        left_pairs,
        right_pairs,
        metric,
        resamples=1000,
        seed=20260525,
    )
    return {
        "metric": name,
        "left_minus_right": result.observed,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "resamples": 1000,
        "bootstrap_seed": 20260525,
    }


def _macro_metric(true: Any, pred: Any) -> float:
    """Compute macro F1 for either row records or columnar vectors.

    The paired-bootstrap helper resamples row-record sequences, while the
    public metric accepts a columnar ``label -> values`` mapping.  Normalize
    both representations here so the aggregate uses the same metric as the
    per-run reports after bootstrap resampling.
    """
    if isinstance(true, dict):
        true_by_label = {label: true[label] for label in PRAGMATIC_LABELS}
    else:
        true_by_label = {
            label: [int(row[label]) for row in true]
            for label in PRAGMATIC_LABELS
        }
    if isinstance(pred, dict):
        pred_by_label = {label: pred[label] for label in PRAGMATIC_LABELS}
    else:
        pred_by_label = {
            label: [int(row[label]) for row in pred]
            for label in PRAGMATIC_LABELS
        }
    return macro_pragmatic_f1(
        true_by_label,
        pred_by_label,
    )


def _sarcasm_metric(true: Any, pred: Any) -> float:
    if isinstance(true, dict):
        true_sarcasm = true["sarcasm"]
    else:
        true_sarcasm = [int(row["sarcasm"]) for row in true]
    if isinstance(pred, dict):
        pred_sarcasm = pred["sarcasm"]
    else:
        pred_sarcasm = [int(row["sarcasm"]) for row in pred]
    return binary_macro_f1(true_sarcasm, pred_sarcasm)


def _aggregate_priority1() -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for variant in Q1_VARIANTS:
        groups[variant] = [_run_summary(f"{Q1_RUN_PREFIX}_{variant}_{seed}") for seed in SEEDS]
    comparisons: dict[str, Any] = {}
    for variant in Q1_VARIANTS[1:]:
        comparisons[f"full_vs_{variant}"] = _paired(
            groups["full"],
            groups[variant],
            _macro_metric,
            name="macro_pragmatic_f1",
        )
    return {
        "status": "PASS",
        "protocol": "priority1_v37_clean_matched_xlmr_large_component_ablation_v1",
        "source_recipe": "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37",
        "seed_policy": "all three predeclared seeds retained when structurally valid; no test-outcome filtering",
        "groups": {variant: {"runs": [item["run_id"] for item in items], "summary": _summary_table(items)} for variant, items in groups.items()},
        "paired_bootstrap_full_minus_ablation": comparisons,
        "interpretation_targets_are_not_selection_filters": True,
    }


def _aggregate_priority2() -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for budget in Q3_BUDGETS:
        groups[budget] = [_run_summary(f"priority12_q3_corrected_{budget}_{seed}") for seed in SEEDS]
    curve = []
    for budget in Q3_BUDGETS:
        summary = _summary_table(groups[budget])
        curve.append({"budget": budget, "summary": summary})
    full = groups["full"]
    comparisons: dict[str, Any] = {}
    for budget in Q3_BUDGETS[:-1]:
        comparisons[f"full_vs_{budget}"] = _paired(
            full,
            groups[budget],
            _sarcasm_metric,
            name="sarcasm_binary_macro_f1",
        )
    return {
        "status": "PASS",
        "protocol": "priority2_corrected_positive_sarcasm_budget_curve_v1",
        "seed_policy": "all three predeclared seeds retained when structurally valid; no monotonicity forcing",
        "curve": curve,
        "paired_bootstrap_full_minus_budget": comparisons,
        "interpretation_targets_are_not_selection_filters": True,
        "corrected_mask_policy": "rationale active; only sarcasm target/loss masked out of budget",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate complete Priority 1/2 runs")
    parser.add_argument("--priority", choices=("1", "2", "all"), default="all")
    args = parser.parse_args()
    payload: dict[str, Any] = {"schema_version": 1}
    if args.priority in {"1", "all"}:
        payload["priority1"] = _aggregate_priority1()
    if args.priority in {"2", "all"}:
        payload["priority2"] = _aggregate_priority2()
    output = OUTPUT_ROOT / "aggregate_priority12.json"
    atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
