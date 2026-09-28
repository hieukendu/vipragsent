"""Validate one matched XLM-R-large Q1a seed before Hub upload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.constants import EXPECTED_SPLIT_COUNTS, PRAGMATIC_LABELS
from vipragsent.data.loaders import load_vipragsent
from vipragsent.evaluation.metrics import binary_macro_f1, macro_pragmatic_f1
from vipragsent.hashing import sha256_file

REPORT_ROOT = ROOT / "reports/q1a_matched_xlmr_ft"
RESULT_ROOT = ROOT / "results/runs"
PROPOSED_BASE = (
    REPORT_ROOT
    / "proposed_source/campaigns/vipragsent-v8-local-mig2g20gb"
    / "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37_"
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_rows(rows: list[dict[str, Any]], examples: list[Any], split: str) -> dict[str, Any]:
    _assert(len(rows) == len(examples), f"{split}: expected {len(examples)} rows, got {len(rows)}")
    expected_ids = [str(example.sample_id) for example in examples]
    actual_ids = [str(row.get("sample_id", "")) for row in rows]
    _assert(len(set(actual_ids)) == len(actual_ids), f"{split}: sample IDs are not unique")
    _assert(actual_ids == expected_ids, f"{split}: prediction IDs do not match frozen CSV order")
    for row, example in zip(rows, examples, strict=True):
        gold = row.get("gold")
        pred = row.get("predictions")
        probs = row.get("probabilities")
        logits = row.get("logits")
        _assert(isinstance(gold, dict) and set(PRAGMATIC_LABELS).issubset(gold), f"{split}: gold labels missing")
        _assert(isinstance(pred, dict) and set(PRAGMATIC_LABELS).issubset(pred), f"{split}: predictions missing")
        _assert(isinstance(probs, dict) and set(PRAGMATIC_LABELS).issubset(probs), f"{split}: probabilities missing")
        _assert(isinstance(logits, dict) and set(PRAGMATIC_LABELS).issubset(logits), f"{split}: logits missing")
        for label in PRAGMATIC_LABELS:
            _assert(int(gold[label]) == int(example.labels[label]), f"{split}: gold mismatch for {row['sample_id']}/{label}")
            _assert(int(pred[label]) in (0, 1), f"{split}: non-binary prediction for {label}")
            probability = float(probs[label])
            logit = float(logits[label])
            _assert(math.isfinite(probability) and 0.0 <= probability <= 1.0, f"{split}: invalid probability for {label}")
            _assert(math.isfinite(logit), f"{split}: invalid logit for {label}")
    true = {label: [int(row["gold"][label]) for row in rows] for label in PRAGMATIC_LABELS}
    pred = {label: [int(row["predictions"][label]) for row in rows] for label in PRAGMATIC_LABELS}
    return {
        "prediction_count": len(rows),
        "per_label_f1": {label: binary_macro_f1(true[label], pred[label]) for label in PRAGMATIC_LABELS},
        "macro_pragmatic_f1": macro_pragmatic_f1(true, pred),
    }


def validate(seed: int) -> dict[str, Any]:
    report_root = REPORT_ROOT / f"seed_{seed}"
    run_root = RESULT_ROOT / f"xlmr_large_ft_q1a_matched_{seed}"
    _assert(report_root.exists() and run_root.exists(), f"missing matched output for seed {seed}")
    bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
    _assert({key: len(value) for key, value in bundle.splits.items()} == EXPECTED_SPLIT_COUNTS, "frozen split counts changed")
    required = (
        "resolved_config.json",
        "thresholds.json",
        "metrics_train.json",
        "metrics_dev.json",
        "metrics_test.json",
        "predictions_train.jsonl",
        "predictions_dev.jsonl",
        "predictions_test.jsonl",
        "training_log.jsonl",
        "best_checkpoint_metadata.json",
    )
    for name in required:
        _assert((report_root / name).exists(), f"missing required report artifact: {report_root / name}")

    rows = {
        split: _read_jsonl(report_root / f"predictions_{split}.jsonl")
        for split in ("train", "dev", "test")
    }
    recomputed = {
        split: _validate_rows(rows[split], getattr(bundle, split), split)
        for split in ("train", "dev", "test")
    }
    for split in recomputed:
        recorded = _read_json(report_root / f"metrics_{split}.json")
        _assert(int(recorded["prediction_count"]) == recomputed[split]["prediction_count"], f"{split}: recorded count mismatch")
        for label in PRAGMATIC_LABELS:
            _assert(abs(float(recorded["per_label_f1"][label]) - recomputed[split]["per_label_f1"][label]) < 1e-12, f"{split}: F1 mismatch for {label}")
        _assert(abs(float(recorded["macro_pragmatic_f1"]) - recomputed[split]["macro_pragmatic_f1"]) < 1e-12, f"{split}: macro F1 mismatch")

    thresholds = _read_json(report_root / "thresholds.json")
    _assert(int(thresholds["seed"]) == seed, "threshold artifact seed mismatch")
    _assert(thresholds["source_split"] == "dev", "thresholds were not selected on dev")
    _assert(thresholds["selection_metric"] == "dev_macro_pragmatic_f1", "unexpected selection metric")
    _assert(set(thresholds["thresholds"]) == set(PRAGMATIC_LABELS), "threshold labels differ")
    for value in thresholds["thresholds"].values():
        _assert(0.05 <= float(value) <= 0.95 and abs(round(float(value) * 100) - float(value) * 100) < 1e-7, "threshold outside locked grid")

    manifest = _read_json(report_root / "run_manifest.json")
    _assert(manifest["test_evaluated_during_training"] is False, "test was used during training")
    _assert(manifest["test_evaluated_after_checkpoint_and_threshold_freeze"] is True, "test freeze gate missing")
    _assert(manifest["dataset_split_sizes"] == {key: int(value) for key, value in EXPECTED_SPLIT_COUNTS.items()}, "manifest split counts mismatch")
    resolved = _read_json(report_root / "resolved_config.json")
    _assert(resolved["experiment_name"] == "xlmr_large_ft_q1a_matched", "wrong experiment name")
    _assert(resolved["variant_id"] == "xlmr_pragmatic_finetune", "wrong baseline variant")
    _assert(resolved["active_heads"] == list(PRAGMATIC_LABELS), "active head set changed")
    _assert(resolved["auxiliary_heads"] is False and resolved["rationale_decoder"] is False, "extra heads enabled")
    training = resolved["training_config"]
    _assert(training["scheduler"] == "cosine" and float(training["warmup_ratio"]) == 0.2, "scheduler recipe mismatch")
    _assert(int(training["patience"]) == 10 and int(training["max_epochs"]) == 10, "stopping recipe mismatch")
    _assert(int(training["physical_batch_size"]) == 8 and int(training["gradient_accumulation_steps"]) == 4, "batch recipe mismatch")
    _assert(float(training["learning_rate"]) == 2e-5 and float(training["weight_decay"]) == 0.01, "optimizer recipe mismatch")
    _assert(resolved["loss"]["positive_weight_policy"] == "train_only_negative_over_positive", "class weight policy mismatch")

    checkpoint = run_root / "checkpoints/best/model.pt"
    metadata = _read_json(report_root / "best_checkpoint_metadata.json")
    _assert(checkpoint.exists(), "selected checkpoint is missing")
    _assert(sha256_file(checkpoint) == metadata["checkpoint_sha256"], "selected checkpoint hash mismatch")
    _assert(not (run_root / "_engine_output/test_predictions.jsonl").exists(), "engine evaluated test during training")

    proposed_path = Path(f"{PROPOSED_BASE}{seed}/predictions/test_predictions.jsonl")
    _assert(proposed_path.exists(), f"proposed prediction source is missing: {proposed_path}")
    proposed_rows = _read_jsonl(proposed_path)
    _assert([row["sample_id"] for row in proposed_rows] == [row["sample_id"] for row in rows["test"]], "proposed and matched test IDs are not paired")
    for matched, proposed in zip(rows["test"], proposed_rows, strict=True):
        for label in PRAGMATIC_LABELS:
            _assert(int(matched["gold"][label]) == int(proposed["gold"][label]), f"paired gold mismatch for {label}")

    result = {
        "status": "PASS",
        "seed": seed,
        "split_counts": {split: len(rows[split]) for split in rows},
        "ids_unique_and_frozen_order": True,
        "gold_matches_frozen_dataset": True,
        "paired_with_proposed_test_ids_and_gold": True,
        "probabilities_finite_and_bounded": True,
        "metric_reproducibility": True,
        "dev_only_selection": True,
        "test_rows": len(rows["test"]),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "metrics": recomputed,
    }
    atomic_write_json(report_root / "validation.json", result)
    atomic_write_json(run_root / "validation.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a matched XLM-R-large seed")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    validate(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
