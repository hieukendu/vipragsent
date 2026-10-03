"""Validate and aggregate the corrected Priority 2 Q3 campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = "priority2-q3-corrected-v2-20261002"
PREFIX = "priority2_q3_corrected_v2_"
SEEDS = (20260521, 20260522, 20260523)
BUDGETS = ("32", "64", "128", "256", "512", "full")
MODEL_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
LABELS = ("implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking")
METRICS = (
    "sarcasm_binary_macro_f1",
    "sarcasm_positive_precision",
    "sarcasm_positive_recall",
    "sarcasm_positive_f1",
    "macro_pragmatic_f1",
)


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def binary_scores(gold: list[int], pred: list[int]) -> dict[str, float]:
    tp = sum(a == b == 1 for a, b in zip(gold, pred, strict=True))
    tn = sum(a == b == 0 for a, b in zip(gold, pred, strict=True))
    fp = sum(a == 0 and b == 1 for a, b in zip(gold, pred, strict=True))
    fn = sum(a == 1 and b == 0 for a, b in zip(gold, pred, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    positive_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_precision = tn / (tn + fn) if tn + fn else 0.0
    negative_recall = tn / (tn + fp) if tn + fp else 0.0
    negative_f1 = (
        2 * negative_precision * negative_recall / (negative_precision + negative_recall)
        if negative_precision + negative_recall
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "positive_f1": positive_f1,
        "binary_macro_f1": (positive_f1 + negative_f1) / 2,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def calculate_metrics(path: Path) -> tuple[dict[str, float], int, int]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"duplicate sample_id values in {path}")
    per_label: dict[str, dict[str, float]] = {}
    for label in LABELS:
        gold = [int(row["gold"][label]) for row in rows]
        pred = [int(row["predictions"][label]) for row in rows]
        per_label[label] = binary_scores(gold, pred)
    sarcasm = per_label["sarcasm"]
    metrics = {
        "sarcasm_binary_macro_f1": sarcasm["binary_macro_f1"],
        "sarcasm_positive_precision": sarcasm["precision"],
        "sarcasm_positive_recall": sarcasm["recall"],
        "sarcasm_positive_f1": sarcasm["positive_f1"],
        "macro_pragmatic_f1": mean(value["binary_macro_f1"] for value in per_label.values()),
    }
    return metrics, len(rows), len(set(sample_ids))


def close_enough(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def expected_mask_hash(budget: str) -> str:
    return sha256_file(ROOT / "data/processed/q3_low_resource_sarcasm_corrected" / f"budget_{budget}_masks.csv")


def validate_run(run_id: str) -> dict[str, Any]:
    run_root = ROOT / "results/runs" / run_id
    errors: list[str] = []
    parts = run_id.removeprefix(PREFIX).rsplit("_", 1)
    if len(parts) != 2 or parts[0] not in BUDGETS:
        return {"run_id": run_id, "valid": False, "errors": ["run_id does not encode an expected budget and seed"]}
    budget, seed_text = parts
    seed = int(seed_text)
    if seed not in SEEDS:
        errors.append(f"unexpected seed: {seed}")
    state = load_json(run_root / "state.json", {}) or {}
    approval = load_json(run_root / "approval_status.json", {}) or {}
    manifest = load_json(run_root / "run_manifest.json", {}) or {}
    checkpoint = load_json(run_root / "checkpoints/checkpoint_manifest.json", {}) or {}
    test_saved = load_json(run_root / "metrics/test_metrics.json", {}) or {}
    if state.get("run_status") != "APPROVED" or state.get("approval_status") != "APPROVED":
        errors.append(f"approval state is not APPROVED: {state.get('run_status')}/{state.get('approval_status')}")
    if approval.get("status") != "APPROVED" or approval.get("run_id") != run_id:
        errors.append("approval_status.json is not bound to this run")
    stages = state.get("stages", {})
    required_stages = ("preflight", "train", "evaluate_dev", "evaluate_test", "freeze_selection", "validate_artifacts", "export_artifacts", "generate_review_summary")
    if any(stages.get(stage, {}).get("status") != "PASS" for stage in required_stages):
        errors.append("not all required stages are PASS")
    if manifest.get("model_revision") != MODEL_REVISION:
        errors.append("run manifest model revision mismatch")
    mask_hash = expected_mask_hash(budget)
    if manifest.get("q3_mask_hash") != mask_hash:
        errors.append("run manifest q3 mask hash mismatch")
    if checkpoint.get("q3_mask_hash") != mask_hash:
        errors.append("checkpoint q3 mask hash mismatch")
    if checkpoint.get("model_revision") != MODEL_REVISION:
        errors.append("checkpoint model revision mismatch")
    if test_saved.get("test_threshold_tuning") is not False:
        errors.append("test threshold tuning was not disabled")
    try:
        test_metrics, test_count, test_unique = calculate_metrics(run_root / "predictions/test_predictions.jsonl")
        dev_metrics, dev_count, dev_unique = calculate_metrics(run_root / "predictions/dev_predictions.jsonl")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f"prediction validation failed: {exc}")
        test_metrics = {metric: float("nan") for metric in METRICS}
        dev_metrics = {metric: float("nan") for metric in METRICS}
        test_count = test_unique = dev_count = dev_unique = 0
    for metric in ("macro_pragmatic_f1", "sarcasm_test_f1"):
        saved_key = "macro_pragmatic_f1" if metric == "macro_pragmatic_f1" else "sarcasm_test_f1"
        calculated_key = "macro_pragmatic_f1" if metric == "macro_pragmatic_f1" else "sarcasm_binary_macro_f1"
        if not close_enough(float(test_saved.get(saved_key, float("nan"))), test_metrics[calculated_key]):
            errors.append(f"saved test metric mismatch: {saved_key}")
    if test_count != test_unique or dev_count != dev_unique:
        errors.append("prediction IDs are not unique")
    return {
        "run_id": run_id,
        "budget": budget,
        "seed": seed,
        "valid": not errors,
        "errors": errors,
        "test_metrics": test_metrics,
        "dev_metrics": dev_metrics,
        "test_prediction_count": test_count,
        "dev_prediction_count": dev_count,
        "model_revision": MODEL_REVISION,
        "mask_hash": mask_hash,
        "hf_upload": None,
    }


def attach_upload_status(records: list[dict[str, Any]]) -> None:
    status = load_json(ROOT / "runtime/priority2_q3_corrected_hf_uploader_status.json", {}) or {}
    for record in records:
        item = (status.get("runs", {}) or {}).get(record["run_id"], {})
        record["hf_upload"] = {
            "status": item.get("status"),
            "uploaded_file_count": len(item.get("uploaded", {}) or {}),
            "error_count": len(item.get("errors", []) or []),
            "receipt": bool(item.get("receipt")),
            "artifact_repository": item.get("artifact_repository"),
            "checkpoint_repository": item.get("checkpoint_repository"),
        }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if record["valid"]]
    by_budget: dict[str, dict[str, Any]] = {}
    for budget in BUDGETS:
        rows = sorted((record for record in valid if record["budget"] == budget), key=lambda row: row["seed"])
        budget_metrics: dict[str, Any] = {}
        for metric in METRICS:
            values = [row["test_metrics"][metric] for row in rows]
            budget_metrics[metric] = {
                "seed_values": {str(row["seed"]): row["test_metrics"][metric] for row in rows},
                "n": len(values),
                "mean": mean(values) if values else None,
                "sd": stdev(values) if len(values) > 1 else (0.0 if values else None),
            }
        by_budget[budget] = {"run_count": len(rows), "metrics": budget_metrics}
    interpretations: dict[str, Any] = {}
    for metric in METRICS:
        means = {budget: by_budget[budget]["metrics"][metric]["mean"] for budget in BUDGETS}
        low_values = [means[budget] for budget in ("32", "64") if means[budget] is not None]
        high_values = [means[budget] for budget in ("512", "full") if means[budget] is not None]
        low = mean(low_values) if len(low_values) == 2 else None
        high = mean(high_values) if len(high_values) == 2 else None
        interpretations[metric] = {
            "low_mean_32_64": low,
            "high_mean_512_full": high,
            "high_minus_low": high - low if low is not None and high is not None else None,
            "high_minus_low_percentage_points": (high - low) * 100 if low is not None and high is not None else None,
            "saturation_512_vs_full_abs_percentage_points": abs(means["full"] - means["512"]) * 100 if means["full"] is not None and means["512"] is not None else None,
        }
    main = interpretations["sarcasm_binary_macro_f1"]
    delta_pp = main["high_minus_low_percentage_points"]
    main["gain_interpretation"] = (
        "strong" if delta_pp is not None and delta_pp >= 2.0
        else "moderate" if delta_pp is not None and delta_pp >= 1.0
        else "below_moderate_threshold" if delta_pp is not None
        else "incomplete"
    )
    saturation_pp = main["saturation_512_vs_full_abs_percentage_points"]
    main["saturation_at_or_below_1pp"] = saturation_pp is not None and saturation_pp <= 1.0
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN,
        "protocol": "configs/experiments/q3/corrected_protocol.yaml",
        "model_revision": MODEL_REVISION,
        "budgets": list(BUDGETS),
        "seeds": list(SEEDS),
        "scientific_policy": "retain_all_valid_seeds_without_outcome_based_selection",
        "records": records,
        "valid_run_count": len(valid),
        "expected_run_count": len(BUDGETS) * len(SEEDS),
        "by_budget": by_budget,
        "interpretations": interpretations,
    }


def write_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["budget", "seed", "run_id", "valid", *METRICS]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in report["records"]:
            writer.writerow({
                "budget": record.get("budget"),
                "seed": record.get("seed"),
                "run_id": record["run_id"],
                "valid": record["valid"],
                **{metric: record["test_metrics"].get(metric) for metric in METRICS},
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/priority2_q3_corrected/aggregate_report.json")
    parser.add_argument("--csv-output", type=Path, default=ROOT / "reports/priority2_q3_corrected/per_run_metrics.csv")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    run_root = ROOT / "results/runs"
    run_ids = sorted(path.parent.name for path in run_root.glob(f"{PREFIX}*/state.json"))
    records = [validate_run(run_id) for run_id in run_ids]
    attach_upload_status(records)
    report = aggregate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(report, args.csv_output)
    print(json.dumps({
        "output": str(args.output.relative_to(ROOT)),
        "csv_output": str(args.csv_output.relative_to(ROOT)),
        "valid_run_count": report["valid_run_count"],
        "expected_run_count": report["expected_run_count"],
        "invalid_runs": [record["run_id"] for record in records if not record["valid"]],
    }, indent=2))
    return 0 if args.allow_incomplete or report["valid_run_count"] == report["expected_run_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
