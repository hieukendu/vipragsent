"""Run Priority 3 positive-class analysis on frozen XLM-R predictions.

This analysis intentionally does not train or select models.  It validates and
compares the frozen XLM-R-large multi-task Q1a predictions against the frozen
matched pragmatic-only predictions on the same test examples for all three
seeds.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.constants import EXPECTED_SPLIT_COUNTS, PRAGMATIC_LABELS, TRAINING_SEEDS
from vipragsent.hashing import sha256_file, sha256_json
from vipragsent.statistics.bootstrap import paired_bootstrap_comparison

REPORT_ROOT = ROOT / "reports/priority3_positive_class"
FULL_RUN_PREFIX = (
    ROOT
    / "reports/q1a_matched_xlmr_ft/proposed_source/campaigns/vipragsent-v8-local-mig2g20gb/"
    "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_"
    "implicit101_sarcasm101_code104_v37_"
)
BASELINE_ROOT = ROOT / "reports/q1a_matched_xlmr_ft"
SEEDS = tuple(int(seed) for seed in TRAINING_SEEDS)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260525
MODEL_KEYS = ("proposed_multitask", "baseline_pragmatic_only")
METRIC_KEYS = ("positive_precision", "positive_recall", "positive_f1")
COUNT_KEYS = ("tp", "fp", "fn", "tn")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _code_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _full_run_root(seed: int) -> Path:
    return Path(f"{FULL_RUN_PREFIX}{seed}")


def _baseline_run_root(seed: int) -> Path:
    return BASELINE_ROOT / f"seed_{seed}"


def _prediction_path(model: str, run_root: Path, split: str) -> Path:
    if model == "proposed_multitask":
        return run_root / "predictions" / f"{split}_predictions.jsonl"
    return run_root / f"predictions_{split}.jsonl"


def _threshold_path(model: str, run_root: Path) -> Path:
    if model == "proposed_multitask":
        return run_root / "selection" / "thresholds.json"
    return run_root / "thresholds.json"


def _source_metadata(model: str, run_root: Path, seed: int) -> dict[str, Any]:
    threshold_path = _threshold_path(model, run_root)
    _require(threshold_path.exists(), f"{model} seed {seed}: missing threshold artifact {threshold_path}")
    threshold_payload = _read_json(threshold_path)

    if model == "proposed_multitask":
        manifest_path = run_root / "optimization_manifest.json"
        selection_path = run_root / "selection" / "selection_metric.json"
        _require(manifest_path.exists(), f"{model} seed {seed}: missing optimization manifest")
        _require(selection_path.exists(), f"{model} seed {seed}: missing selection metric")
        manifest = _read_json(manifest_path)
        selection = _read_json(selection_path)
        thresholds = {label: float(threshold_payload[label]) for label in PRAGMATIC_LABELS}
        final_dev = manifest.get("final_dev", {})
        _require(selection.get("name") == "dev_macro_pragmatic_f1", f"{model} seed {seed}: non-dev selection metric")
        _require(int(final_dev.get("prediction_count", -1)) == EXPECTED_SPLIT_COUNTS["dev"], f"{model} seed {seed}: dev count is not frozen")
        _require(final_dev.get("thresholds") == threshold_payload, f"{model} seed {seed}: threshold artifact differs from final dev metadata")
        threshold_source = "dev"
        provenance = {
            "manifest": manifest_path,
            "selection_metric": selection_path,
            "selection_metric_name": selection["name"],
            "selection_metric_value": float(selection["value"]),
            "threshold_source_split": threshold_source,
            "thresholds": thresholds,
        }
    else:
        manifest_path = run_root / "run_manifest.json"
        _require(manifest_path.exists(), f"{model} seed {seed}: missing run manifest")
        manifest = _read_json(manifest_path)
        thresholds = {label: float(threshold_payload["thresholds"][label]) for label in PRAGMATIC_LABELS}
        _require(threshold_payload.get("seed") == seed, f"{model} seed {seed}: threshold seed mismatch")
        _require(threshold_payload.get("source_split") == "dev", f"{model} seed {seed}: thresholds not selected on dev")
        _require(threshold_payload.get("selection_metric") == "dev_macro_pragmatic_f1", f"{model} seed {seed}: non-dev selection metric")
        checkpoint = manifest.get("checkpoint", {})
        _require(checkpoint.get("selection_split") == "dev", f"{model} seed {seed}: checkpoint selection split is not dev")
        _require(checkpoint.get("threshold_source_split") == "dev", f"{model} seed {seed}: threshold source is not dev")
        _require(checkpoint.get("test_used_for_selection") is False, f"{model} seed {seed}: test was used for selection")
        threshold_source = "dev"
        provenance = {
            "manifest": manifest_path,
            "selection_metric_name": threshold_payload["selection_metric"],
            "threshold_source_split": threshold_source,
            "thresholds": thresholds,
        }

    return {
        "threshold_path": threshold_path,
        "threshold_sha256": sha256_file(threshold_path),
        "thresholds": thresholds,
        "provenance": provenance,
    }


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    model: str,
    seed: int,
    split: str,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    expected_count = EXPECTED_SPLIT_COUNTS[split]
    _require(len(rows) == expected_count, f"{model} seed {seed} {split}: expected {expected_count}, got {len(rows)}")
    ids = [str(row.get("sample_id", "")) for row in rows]
    _require(all(ids), f"{model} seed {seed} {split}: missing sample ID")
    _require(len(ids) == len(set(ids)), f"{model} seed {seed} {split}: duplicate sample IDs")

    for row in rows:
        gold = row.get("gold")
        predictions = row.get("predictions")
        probabilities = row.get("probabilities")
        _require(isinstance(gold, Mapping), f"{model} seed {seed} {split}: missing gold labels")
        _require(isinstance(predictions, Mapping), f"{model} seed {seed} {split}: missing predictions")
        _require(isinstance(probabilities, Mapping), f"{model} seed {seed} {split}: missing probabilities")
        for label in PRAGMATIC_LABELS:
            _require(label in gold and label in predictions and label in probabilities, f"{model} seed {seed} {split}: missing {label}")
            gold_value = int(gold[label])
            prediction_value = int(predictions[label])
            probability = float(probabilities[label])
            _require(gold_value in (0, 1), f"{model} seed {seed} {split}: non-binary gold for {label}")
            _require(prediction_value in (0, 1), f"{model} seed {seed} {split}: non-binary prediction for {label}")
            _require(math.isfinite(probability) and 0.0 <= probability <= 1.0, f"{model} seed {seed} {split}: invalid probability for {label}")
            expected_prediction = int(probability >= float(thresholds[label]))
            _require(
                prediction_value == expected_prediction,
                f"{model} seed {seed} {split}: prediction does not match frozen threshold for {label}",
            )
    return {
        "count": len(rows),
        "sample_id_sha256": sha256_json(ids),
        "sample_ids_unique": True,
        "probabilities_finite_and_bounded": True,
        "predictions_match_frozen_thresholds": True,
    }


def _validate_sources() -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    loaded: dict[int, dict[str, Any]] = {}
    source_manifest: dict[str, Any] = {"models": {}, "seeds": list(SEEDS)}
    first_gold: list[tuple[str, tuple[int, ...]]] | None = None

    for model in MODEL_KEYS:
        source_manifest["models"][model] = {
            "description": "XLM-R-large multi-task Q1a frozen predictions" if model == "proposed_multitask" else "XLM-R-large pragmatic-only matched frozen predictions",
            "runs": {},
        }

    for seed in SEEDS:
        loaded[seed] = {}
        for model in MODEL_KEYS:
            run_root = _full_run_root(seed) if model == "proposed_multitask" else _baseline_run_root(seed)
            _require(run_root.exists(), f"{model} seed {seed}: missing run root {run_root}")
            metadata = _source_metadata(model, run_root, seed)
            per_split: dict[str, Any] = {}
            rows_by_split: dict[str, list[dict[str, Any]]] = {}
            for split in ("dev", "test"):
                path = _prediction_path(model, run_root, split)
                _require(path.exists(), f"{model} seed {seed}: missing {split} predictions {path}")
                rows = _read_jsonl(path)
                rows_by_split[split] = rows
                per_split[split] = {
                    "path": path,
                    "sha256": sha256_file(path),
                    **_validate_rows(rows, model=model, seed=seed, split=split, thresholds=metadata["thresholds"]),
                }

            test_signature = [
                (str(row["sample_id"]), tuple(int(row["gold"][label]) for label in PRAGMATIC_LABELS))
                for row in rows_by_split["test"]
            ]
            if first_gold is None:
                first_gold = test_signature
            _require(test_signature == first_gold, f"{model} seed {seed}: test IDs/gold differ from the frozen paired reference")

            loaded[seed][model] = {
                "run_root": run_root,
                "thresholds": metadata["thresholds"],
                "threshold_provenance": metadata["provenance"],
                "rows": rows_by_split["test"],
                "validation": per_split,
            }
            source_manifest["models"][model]["runs"][str(seed)] = {
                "run_root": str(run_root),
                "threshold_path": str(metadata["threshold_path"]),
                "threshold_sha256": metadata["threshold_sha256"],
                "threshold_source_split": "dev",
                "thresholds": metadata["thresholds"],
                "prediction_files": {
                    split: {
                        "path": str(per_split[split]["path"]),
                        "sha256": per_split[split]["sha256"],
                        "count": per_split[split]["count"],
                    }
                    for split in ("dev", "test")
                },
                "validation": per_split,
            }

    for seed in SEEDS:
        proposed_ids = [str(row["sample_id"]) for row in loaded[seed]["proposed_multitask"]["rows"]]
        baseline_ids = [str(row["sample_id"]) for row in loaded[seed]["baseline_pragmatic_only"]["rows"]]
        _require(proposed_ids == baseline_ids, f"seed {seed}: proposed and baseline test IDs are not paired")
        proposed_gold = loaded[seed]["proposed_multitask"]["rows"]
        baseline_gold = loaded[seed]["baseline_pragmatic_only"]["rows"]
        for proposed_row, baseline_row in zip(proposed_gold, baseline_gold, strict=True):
            for label in PRAGMATIC_LABELS:
                _require(
                    int(proposed_row["gold"][label]) == int(baseline_row["gold"][label]),
                    f"seed {seed}: proposed/baseline gold mismatch for {label}",
                )

    source_manifest["pairing"] = {
        "same_test_ids_and_gold_across_models": True,
        "same_test_ids_and_gold_across_seeds": True,
        "test_count": EXPECTED_SPLIT_COUNTS["test"],
        "dev_count": EXPECTED_SPLIT_COUNTS["dev"],
    }
    return loaded, source_manifest


def _confusion(gold: Sequence[int], predictions: Sequence[int]) -> dict[str, int]:
    _require(len(gold) == len(predictions), "confusion inputs must have equal length")
    return {
        "tp": sum(int(actual == 1 and predicted == 1) for actual, predicted in zip(gold, predictions, strict=True)),
        "fp": sum(int(actual == 0 and predicted == 1) for actual, predicted in zip(gold, predictions, strict=True)),
        "fn": sum(int(actual == 1 and predicted == 0) for actual, predicted in zip(gold, predictions, strict=True)),
        "tn": sum(int(actual == 0 and predicted == 0) for actual, predicted in zip(gold, predictions, strict=True)),
    }


def _positive_metrics(confusion: Mapping[str, int]) -> dict[str, float]:
    tp = int(confusion["tp"])
    fp = int(confusion["fp"])
    fn = int(confusion["fn"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    return {"positive_precision": precision, "positive_recall": recall, "positive_f1": f1}


def _per_seed_rows(loaded: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for seed in SEEDS:
        for model in MODEL_KEYS:
            for label in PRAGMATIC_LABELS:
                rows = loaded[seed][model]["rows"]
                gold = [int(row["gold"][label]) for row in rows]
                predictions = [int(row["predictions"][label]) for row in rows]
                confusion = _confusion(gold, predictions)
                metrics = _positive_metrics(confusion)
                output.append(
                    {
                        "model": model,
                        "seed": seed,
                        "label": label,
                        "n": len(rows),
                        "gold_positive": sum(gold),
                        "predicted_positive": sum(predictions),
                        **metrics,
                        **confusion,
                    }
                )
    return output


def _aggregate_rows(per_seed: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_KEYS:
        for label in PRAGMATIC_LABELS:
            matching = [row for row in per_seed if row["model"] == model and row["label"] == label]
            for quantity in (*METRIC_KEYS, *COUNT_KEYS):
                values = [float(row[quantity]) for row in matching]
                output.append(
                    {
                        "model": model,
                        "label": label,
                        "quantity": quantity,
                        **{f"seed_{seed}": next(float(row[quantity]) for row in matching if int(row["seed"]) == seed) for seed in SEEDS},
                        "mean": statistics.mean(values),
                        "sd": statistics.stdev(values),
                    }
                )
    return output


def _metric_value(metric: str, gold: Sequence[int], predictions: Sequence[int]) -> float:
    return _positive_metrics(_confusion(gold, predictions))[metric]


def _paired_rows(loaded: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> list[dict[str, Any]]:
    metric_functions: dict[str, Callable[[Sequence[int], Sequence[int]], float]] = {
        metric: (lambda gold, predictions, metric=metric: _metric_value(metric, gold, predictions))
        for metric in METRIC_KEYS
    }
    output: list[dict[str, Any]] = []
    for label in PRAGMATIC_LABELS:
        proposed_pairs: list[tuple[list[int], list[int]]] = []
        baseline_pairs: list[tuple[list[int], list[int]]] = []
        seed_deltas: dict[str, dict[str, float]] = {}
        for seed in SEEDS:
            proposed_rows = loaded[seed]["proposed_multitask"]["rows"]
            baseline_rows = loaded[seed]["baseline_pragmatic_only"]["rows"]
            gold = [int(row["gold"][label]) for row in proposed_rows]
            proposed_predictions = [int(row["predictions"][label]) for row in proposed_rows]
            baseline_predictions = [int(row["predictions"][label]) for row in baseline_rows]
            proposed_pairs.append((gold, proposed_predictions))
            baseline_pairs.append((gold, baseline_predictions))
            seed_deltas[str(seed)] = {
                metric: (_metric_value(metric, gold, proposed_predictions) - _metric_value(metric, gold, baseline_predictions)) * 100.0
                for metric in METRIC_KEYS
            }

        for metric in METRIC_KEYS:
            result = paired_bootstrap_comparison(
                proposed_pairs,
                baseline_pairs,
                metric_functions[metric],
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
                p_value_method="paired_hierarchical_bootstrap_sign_plus_one_v1",
            )
            deltas = [seed_deltas[str(seed)][metric] for seed in SEEDS]
            output.append(
                {
                    "label": label,
                    "metric": metric,
                    **{f"seed_{seed}_delta_pp": delta for seed, delta in zip(SEEDS, deltas, strict=True)},
                    "observed_delta_pp": result.observed * 100.0,
                    "ci_low_pp": result.ci_low * 100.0,
                    "ci_high_pp": result.ci_high * 100.0,
                    "ci_excludes_zero": bool(result.ci_low > 0.0 or result.ci_high < 0.0),
                    "seeds_favoring_proposed": sum(int(delta > 0.0) for delta in deltas),
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "bootstrap_seed": BOOTSTRAP_SEED,
                    "p_value": result.p_value,
                    "comparison": "proposed_multitask_minus_baseline_pragmatic_only",
                }
            )
    return output


def _find_row(rows: Sequence[Mapping[str, Any]], *, model: str, label: str, quantity: str) -> Mapping[str, Any]:
    return next(row for row in rows if row["model"] == model and row["label"] == label and row["quantity"] == quantity)


def _summary_markdown(
    *,
    per_seed: Sequence[Mapping[str, Any]],
    aggregate: Sequence[Mapping[str, Any]],
    paired: Sequence[Mapping[str, Any]],
    source_manifest: Mapping[str, Any],
    analysis_manifest: Mapping[str, Any],
) -> str:
    lines = [
        "# Priority 3 — Positive-class performance analysis",
        "",
        "Status: COMPLETE on frozen test predictions; no model was retrained and no seed was outcome-selected.",
        "",
        "## Scope and protocol",
        "",
        "This analysis addresses the rare-positive limitation by reporting positive precision, positive recall, positive F1, and TP/FP/FN/TN for all six pragmatic labels. The comparison is the frozen XLM-R-large multi-task Q1a system minus the frozen matched XLM-R-large pragmatic-only system.",
        "",
        f"Seeds: {', '.join(str(seed) for seed in SEEDS)}. Test rows per seed: {EXPECTED_SPLIT_COUNTS['test']}. Thresholds were already frozen from dev and were not re-tuned on test. Paired bootstrap: {BOOTSTRAP_RESAMPLES:,} resamples with seed {BOOTSTRAP_SEED}; intervals are 95% percentage-point intervals.",
        "",
        "Expected sarcasm/mocking gains from the limitation plan are interpretation targets only. All three scientifically valid seeds are retained regardless of whether a target is met.",
        "",
        "## Frozen-source validation",
        "",
        "- Proposed source: `reports/q1a_matched_xlmr_ft/proposed_source/.../q1a_vipragsent_full_XLM_R_large_optimization_...`.",
        "- Baseline source: `reports/q1a_matched_xlmr_ft/seed_<seed>/` matched pragmatic-only runs.",
        "- Every seed has 1,999 dev rows and 2,000 test rows; IDs and six-label gold values are paired across systems and seeds.",
        "- Every recorded test prediction matches its frozen dev-selected threshold; this script never changes a prediction.",
        "",
        "## Per-seed positive metrics",
        "",
        "Values are fractions for precision/recall/F1 and exact counts for confusion fields. The complete six-label table is in `per_seed_metrics.csv`.",
        "",
        "| Label | Model | Seed | Precision+ | Recall+ | F1+ | TP | FP | FN | TN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in PRAGMATIC_LABELS:
        for model in MODEL_KEYS:
            for seed in SEEDS:
                row = next(item for item in per_seed if item["label"] == label and item["model"] == model and int(item["seed"]) == seed)
                lines.append(
                    f"| {label} | {model} | {seed} | {float(row['positive_precision']):.6f} | {float(row['positive_recall']):.6f} | {float(row['positive_f1']):.6f} | {int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} | {int(row['tn'])} |"
                )

    lines.extend([
        "",
        "## Mean ± SD across seeds",
        "",
        "| Label | Model | Precision+ | Recall+ | F1+ | TP | FP | FN | TN |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for label in PRAGMATIC_LABELS:
        for model in MODEL_KEYS:
            precision = _find_row(aggregate, model=model, label=label, quantity="positive_precision")
            recall = _find_row(aggregate, model=model, label=label, quantity="positive_recall")
            f1 = _find_row(aggregate, model=model, label=label, quantity="positive_f1")
            counts = {quantity: _find_row(aggregate, model=model, label=label, quantity=quantity) for quantity in COUNT_KEYS}
            lines.append(
                f"| {label} | {model} | {float(precision['mean']):.6f} ± {float(precision['sd']):.6f} | {float(recall['mean']):.6f} ± {float(recall['sd']):.6f} | {float(f1['mean']):.6f} ± {float(f1['sd']):.6f} | "
                f"{float(counts['tp']['mean']):.2f} ± {float(counts['tp']['sd']):.2f} | {float(counts['fp']['mean']):.2f} ± {float(counts['fp']['sd']):.2f} | {float(counts['fn']['mean']):.2f} ± {float(counts['fn']['sd']):.2f} | {float(counts['tn']['mean']):.2f} ± {float(counts['tn']['sd']):.2f} |"
            )

    lines.extend([
        "",
        "## Paired proposed-minus-baseline confidence intervals",
        "",
        "Positive metrics are shown in percentage points. `CI excludes zero` is the requested paired 95% evidence flag; it is descriptive and was not used to remove or rerun a seed.",
        "",
        "| Label | Metric | Seed deltas (pp) | Mean delta (pp) | 95% CI (pp) | CI excludes zero | Seeds favoring proposed |",
        "| --- | --- | --- | ---: | --- | --- | ---: |",
    ])
    for row in paired:
        seed_deltas = ", ".join(f"{float(row[f'seed_{seed}_delta_pp']):+.3f}" for seed in SEEDS)
        lines.append(
            f"| {row['label']} | {row['metric']} | {seed_deltas} | {float(row['observed_delta_pp']):+.6f} | [{float(row['ci_low_pp']):+.6f}, {float(row['ci_high_pp']):+.6f}] | {row['ci_excludes_zero']} | {int(row['seeds_favoring_proposed'])}/3 |"
        )

    lines.extend(["", "## Sarcasm and mocking interpretation checks", ""])
    for label in ("sarcasm", "mocking"):
        f1 = next(row for row in paired if row["label"] == label and row["metric"] == "positive_f1")
        recall = next(row for row in paired if row["label"] == label and row["metric"] == "positive_recall")
        precision = next(row for row in paired if row["label"] == label and row["metric"] == "positive_precision")
        lines.append(
            f"- **{label}**: positive-F1 change {float(f1['observed_delta_pp']):+.3f} pp; 95% CI [{float(f1['ci_low_pp']):+.3f}, {float(f1['ci_high_pp']):+.3f}] pp; recall change {float(recall['observed_delta_pp']):+.3f} pp; precision change {float(precision['observed_delta_pp']):+.3f} pp; proposed wins {int(f1['seeds_favoring_proposed'])}/3 seeds."
        )

    lines.extend([
        "",
        "## Reproducibility",
        "",
        "- `per_seed_metrics.csv`: all requested per-seed metrics and confusion counts.",
        "- `aggregate_metrics.csv`: mean and sample SD across the three seeds.",
        "- `paired_confidence_intervals.csv`: paired proposed-minus-baseline intervals for positive precision, recall, and F1 for every label.",
        "- `seed_validation.json`: per-seed source, split, pairing, and threshold checks.",
        "- `analysis_manifest.json`: source hashes, protocol, and report hashes.",
        "",
        f"Code commit used for the analysis: `{analysis_manifest['code_commit']}`.",
        f"Source manifest hash: `{sha256_json(source_manifest)}`.",
    ])
    return "\n".join(lines) + "\n"


def run(output_dir: Path = REPORT_ROOT, *, resamples: int = BOOTSTRAP_RESAMPLES, bootstrap_seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    global BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED
    BOOTSTRAP_RESAMPLES = int(resamples)
    BOOTSTRAP_SEED = int(bootstrap_seed)
    _require(BOOTSTRAP_RESAMPLES > 0, "resamples must be positive")

    loaded, source_manifest = _validate_sources()
    source_manifest = _json_safe(source_manifest)
    per_seed = _per_seed_rows(loaded)
    aggregate = _aggregate_rows(per_seed)
    paired = _paired_rows(loaded)

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_dir / "per_seed_metrics.csv"
    aggregate_path = output_dir / "aggregate_metrics.csv"
    paired_path = output_dir / "paired_confidence_intervals.csv"
    validation_path = output_dir / "seed_validation.json"
    source_manifest_path = output_dir / "source_manifest.json"
    paired_json_path = output_dir / "paired_confidence_intervals.json"

    _write_csv(
        per_seed_path,
        per_seed,
        ["model", "seed", "label", "n", "gold_positive", "predicted_positive", *METRIC_KEYS, *COUNT_KEYS],
    )
    _write_csv(
        aggregate_path,
        aggregate,
        ["model", "label", "quantity", *[f"seed_{seed}" for seed in SEEDS], "mean", "sd"],
    )
    _write_csv(
        paired_path,
        paired,
        [
            "label",
            "metric",
            *[f"seed_{seed}_delta_pp" for seed in SEEDS],
            "observed_delta_pp",
            "ci_low_pp",
            "ci_high_pp",
            "ci_excludes_zero",
            "seeds_favoring_proposed",
            "resamples",
            "bootstrap_seed",
            "p_value",
            "comparison",
        ],
    )
    atomic_write_json(source_manifest_path, source_manifest)
    atomic_write_json(paired_json_path, {"status": "PASS", "rows": paired})

    validation = {
        "status": "PASS",
        "seed_count": len(SEEDS),
        "seeds": list(SEEDS),
        "models": list(MODEL_KEYS),
        "split_counts": dict(EXPECTED_SPLIT_COUNTS),
        "same_test_ids_and_gold_across_models_and_seeds": True,
        "dev_only_threshold_selection_verified": True,
        "predictions_match_frozen_thresholds": True,
        "outcome_based_seed_selection": False,
        "outcome_based_seed_deletion_or_rerun": False,
        "per_seed": {
            str(seed): {
                model: loaded[seed][model]["validation"]
                for model in MODEL_KEYS
            }
            for seed in SEEDS
        },
    }
    atomic_write_json(validation_path, validation)

    analysis_manifest: dict[str, Any] = {
        "status": "PASS",
        "analysis_id": "priority3_positive_class_20261002",
        "priority": 3,
        "scope": "positive-class performance analysis on frozen predictions",
        "code_commit": _code_commit(),
        "seeds": list(SEEDS),
        "labels": list(PRAGMATIC_LABELS),
        "models": list(MODEL_KEYS),
        "split": "test",
        "test_count_per_seed": EXPECTED_SPLIT_COUNTS["test"],
        "metrics": list(METRIC_KEYS),
        "confusion_fields": list(COUNT_KEYS),
        "bootstrap": {
            "method": "paired_hierarchical_bootstrap_sign_plus_one_v1",
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": 0.95,
            "scale": "percentage_points_for_deltas_and_intervals",
        },
        "scientific_policy": {
            "thresholds_selected_on": "dev",
            "test_threshold_selection": False,
            "retrained": False,
            "seed_selection_by_expected_outcome": False,
            "keep_all_scientifically_valid_seeds": True,
        },
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_json(source_manifest),
        "artifacts": {
            "per_seed_metrics": str(per_seed_path),
            "aggregate_metrics": str(aggregate_path),
            "paired_confidence_intervals": str(paired_path),
            "paired_confidence_intervals_json": str(paired_json_path),
            "seed_validation": str(validation_path),
        },
    }
    summary = _summary_markdown(
        per_seed=per_seed,
        aggregate=aggregate,
        paired=paired,
        source_manifest=source_manifest,
        analysis_manifest=analysis_manifest,
    )
    summary_path = output_dir / "summary.md"
    atomic_write_text(summary_path, summary)
    analysis_manifest["artifacts"]["summary"] = str(summary_path)
    analysis_manifest["report_hashes"] = {
        "per_seed_metrics": sha256_file(per_seed_path),
        "aggregate_metrics": sha256_file(aggregate_path),
        "paired_confidence_intervals": sha256_file(paired_path),
        "paired_confidence_intervals_json": sha256_file(paired_json_path),
        "seed_validation": sha256_file(validation_path),
        "summary": sha256_file(summary_path),
    }
    analysis_manifest_path = output_dir / "analysis_manifest.json"
    atomic_write_json(analysis_manifest_path, analysis_manifest)

    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "artifacts": analysis_manifest["artifacts"]}, indent=2))
    return analysis_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze positive-class performance from frozen ViPragSent predictions")
    parser.add_argument("--output-dir", type=Path, default=REPORT_ROOT)
    parser.add_argument("--resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    run(output_dir, resamples=args.resamples, bootstrap_seed=args.bootstrap_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
