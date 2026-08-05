from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.constants import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PRAGMATIC_LABELS,
    TRAINING_SEEDS,
)
from vipragsent.evaluation.confidence_intervals import (
    evaluate_q1a_confidence_intervals,
    prediction_rows_to_pair,
)
from vipragsent.evaluation.metrics import binary_macro_f1, macro_pragmatic_f1
from vipragsent.orchestration.aggregation import _table2
from vipragsent.statistics.bootstrap import hierarchical_bootstrap


def _golden_rows(seed_index: int, *, generation: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for example_index in range(8):
        gold = {label: (example_index + label_index) % 2 for label_index, label in enumerate(PRAGMATIC_LABELS)}
        predictions = {
            label: int(gold[label]) ^ int((example_index + seed_index + label_index) % 4 == 0)
            for label_index, label in enumerate(PRAGMATIC_LABELS)
        }
        row: dict[str, Any] = {"sample_id": f"sample-{example_index}", "gold": gold}
        if generation:
            row["effective_full_split_all_zero_fallback"] = predictions
            row["predictions"] = {label: 1 - value for label, value in predictions.items()}
        else:
            row["predictions"] = predictions
        rows.append(row)
    return rows


def _golden_group(root: Path, *, system: str, generation: bool) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    direct_rows: list[list[dict[str, Any]]] = []
    for seed_index, seed in enumerate(TRAINING_SEEDS):
        run_root = root / f"{system}-{seed}"
        (run_root / "predictions").mkdir(parents=True)
        rows = _golden_rows(seed_index, generation=generation)
        (run_root / "predictions/test_predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        (run_root / "config_snapshot.yaml").write_text("locked: true\n", encoding="utf-8")
        direct_rows.append([dict(row, predictions=row.get("effective_full_split_all_zero_fallback", row["predictions"])) for row in rows])
        records.append(
            {
                "run_id": f"{system}-{seed}",
                "run_root": str(run_root),
                "summary": {
                    "system_id": system,
                    "backbone": "vistral_7b" if generation else "phobert_base",
                    "seed": seed,
                    "code_commit": "locked-commit",
                    "invalid_output_rate": 0.0,
                    "invalid_generation_rate": 0.0,
                    "invalid_judge_output_rate": 0.0,
                    "test_confidence_intervals": {label: {"low": 0.01, "high": 0.02} for label in PRAGMATIC_LABELS},
                    "macro_confidence_interval": {"low": 0.01, "high": 0.02},
                },
            }
        )
    return records, direct_rows


def _golden_aggregation_evidence() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vipragsent-table2-joint-ci-") as temp:
        root = Path(temp)
        ordinary_records, ordinary_rows = _golden_group(root, system="ordinary", generation=False)
        generation_records, generation_rows = _golden_group(root, system="cot_only_vistral", generation=True)
        method_report = evaluate_q1a_confidence_intervals(ordinary_rows, prediction_hash="golden", config_hash="locked", code_commit="locked-commit")
        ordinary_pairs = [prediction_rows_to_pair(rows) for rows in ordinary_rows]
        generation_pairs = [prediction_rows_to_pair(rows) for rows in generation_rows]
        ordinary_label = hierarchical_bootstrap(
            [(true["implicit_sentiment"], predicted["implicit_sentiment"]) for true, predicted in ordinary_pairs],
            binary_macro_f1,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        ordinary_macro = hierarchical_bootstrap(
            ordinary_pairs,
            macro_pragmatic_f1,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        generation_label = hierarchical_bootstrap(
            [(true["sarcasm"], predicted["sarcasm"]) for true, predicted in generation_pairs],
            binary_macro_f1,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        ordinary_result = _table2(ordinary_records)[0]
        generation_result = _table2(generation_records)[0]
        checks = {
            "ordinary_direct_bootstrap_match": all(
                math.isclose(float(ordinary_result[key]), float(value), rel_tol=0.0, abs_tol=1e-12)
                for key, value in (
                    ("implicit_f1", ordinary_label.observed),
                    ("implicit_ci_low", ordinary_label.ci_low),
                    ("implicit_ci_high", ordinary_label.ci_high),
                )
            ),
            "ordinary_macro_direct_bootstrap_match": all(
                math.isclose(float(ordinary_result[key]), float(value), rel_tol=0.0, abs_tol=1e-12)
                for key, value in (
                    ("macro_prag_f1", ordinary_macro.observed),
                    ("macro_prag_ci_low", ordinary_macro.ci_low),
                    ("macro_prag_ci_high", ordinary_macro.ci_high),
                )
            ),
            "generation_direct_bootstrap_match": all(
                math.isclose(float(generation_result[key]), float(value), rel_tol=0.0, abs_tol=1e-12)
                for key, value in (
                    ("sarcasm_f1", generation_label.observed),
                    ("sarcasm_ci_low", generation_label.ci_low),
                    ("sarcasm_ci_high", generation_label.ci_high),
                )
            ),
            "generation_ci_is_numeric": generation_result["macro_prag_ci_low"] != "NOT_APPLICABLE" and generation_result["macro_prag_ci_high"] != "NOT_APPLICABLE",
            "bounds_are_not_averaged": ordinary_result["implicit_ci_low"] != 0.01 and ordinary_result["implicit_ci_high"] != 0.02,
            "locked_seed_count": ordinary_result["seed_count"] == len(TRAINING_SEEDS) == generation_result["seed_count"],
        }
        return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "method": method_report["method"], "systems": ["ordinary", "cot_only_vistral"]}


def audit(root: str | Path = ROOT, *, write_report: bool = True) -> dict[str, object]:
    root = Path(root)
    sources = [
        "configs/statistics.yaml",
        "configs/statistics/significance_method.yaml",
        "src/vipragsent/statistics/bootstrap.py",
        "src/vipragsent/evaluation/confidence_intervals.py",
        "src/vipragsent/orchestration/aggregation.py",
        "configs/schemas/result.schema.json",
    ]
    method = yaml.safe_load((root / "configs/statistics/significance_method.yaml").read_text(encoding="utf-8")) or {}
    required = {
        "method_id": "paired_hierarchical_bootstrap_sign_plus_one_v1",
        "resamples": 1000,
        "bootstrap_seed": 20260525,
        "confidence_interval": "percentile_95",
    }
    exact = all(method.get(key) == value for key, value in required.items())
    implementation = root / "src/vipragsent/evaluation/confidence_intervals.py"
    implementation_text = implementation.read_text(encoding="utf-8") if implementation.exists() else ""
    aggregation = root / "src/vipragsent/orchestration/aggregation.py"
    aggregation_text = aggregation.read_text(encoding="utf-8") if aggregation.exists() else ""
    golden = _golden_aggregation_evidence()
    checks = {
        "approved_method_exact": exact,
        "dedicated_evaluator": implementation.exists(),
        "per_label_and_macro": all(fragment in implementation_text for fragment in ("PRAGMATIC_LABELS", '"macro"', '"labels"')),
        "prediction_hash_binding": "prediction_hash" in implementation_text,
        "production_aggregator_uses_joint_evaluator": "_joint_table2_confidence_intervals" in aggregation_text and "evaluate_q1a_confidence_intervals" in aggregation_text,
        "no_mean_ci_fallback": "mean(interval[0]" not in aggregation_text and "mean(interval[1]" not in aggregation_text,
        "generation_ci_is_applicable": "row[f\"{short}_ci_low\"] = \"NOT_APPLICABLE\"" not in aggregation_text,
        "no_unapproved_bound_averaging": "bounds are not averaged" in implementation_text,
        "executable_golden_aggregation": golden["status"] == "PASS",
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sources_checked": sources,
        "exact_method_found": exact,
        "method_id": method.get("method_id"),
        "resampling_unit": "seed_then_test_example",
        "resampling_count": method.get("resamples"),
        "bootstrap_seed": method.get("bootstrap_seed"),
        "confidence_level": 0.95,
        "per_seed_behavior": "hierarchical seed-then-test-example resampling",
        "cross_seed_behavior": "compute the joint hierarchical interval; do not average interval bounds",
        "azure_fixed_prediction_behavior": "test-example resampling only",
        "source_references": ["configs/statistics/significance_method.yaml", "configs/statistics.yaml"],
        "conflicts": [],
        "checks": checks,
        "individual_q1a_ci_output": "metrics/test_confidence_intervals.json",
        "table2_aggregation_status": "READY_FOR_APPROVED_RUN_INPUTS",
        "final_aggregation_ready": False,
        "golden_aggregation": golden,
    }
    if write_report:
        atomic_write_json(root / "reports/table2_joint_ci_audit.json", {"status": report["status"], "checks": checks, "golden_aggregation": golden, "method": method})
        atomic_write_json(root / "reports/table2_confidence_interval_protocol_audit.json", report)
    return report


def main() -> int:
    report = audit()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
