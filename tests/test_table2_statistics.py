from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from vipragsent.constants import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    PRAGMATIC_LABELS,
    TRAINING_SEEDS,
)
from vipragsent.evaluation.metrics import binary_macro_f1, macro_pragmatic_f1
from vipragsent.orchestration.aggregation import _table2
from vipragsent.statistics.bootstrap import hierarchical_bootstrap

ROOT = Path(__file__).resolve().parents[1]
SHORT_NAMES = {
    "implicit_sentiment": "implicit",
    "sarcasm": "sarcasm",
    "irony": "irony",
    "idiom_figurative": "idiom",
    "code_switching": "code_switching",
    "mocking": "mocking",
}


def _rows(
    true: list[int],
    predicted: list[int],
    *,
    effective: list[int] | None = None,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    ids = sample_ids or [f"s{index}" for index in range(len(true))]
    rows: list[dict[str, Any]] = []
    for sample_id, gold_value, predicted_value, effective_value in zip(
        ids, true, predicted, effective or predicted, strict=True
    ):
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "gold": {label: gold_value for label in PRAGMATIC_LABELS},
            "predictions": {label: predicted_value for label in PRAGMATIC_LABELS},
        }
        if effective is not None:
            row["effective_full_split_all_zero_fallback"] = {
                label: effective_value for label in PRAGMATIC_LABELS
            }
            row["valid_prediction"] = False
            row["valid_prediction_labels"] = {label: predicted_value for label in PRAGMATIC_LABELS}
        rows.append(row)
    return rows


def _pair(rows: list[dict[str, Any]]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    true = {label: [] for label in PRAGMATIC_LABELS}
    predicted = {label: [] for label in PRAGMATIC_LABELS}
    for row in rows:
        values = row.get("effective_full_split_all_zero_fallback", row["predictions"])
        for label in PRAGMATIC_LABELS:
            true[label].append(int(row["gold"][label]))
            predicted[label].append(int(values[label]))
    return true, predicted


def _summary_from_rows(
    *,
    system: str,
    backbone: str,
    seed: int | str,
    rows: list[dict[str, Any]],
    generation: bool = False,
    interval: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    true, predicted = _pair(rows)
    per_label = {label: binary_macro_f1(true[label], predicted[label]) for label in PRAGMATIC_LABELS}
    macro = macro_pragmatic_f1(true, predicted)
    summary: dict[str, Any] = {
        "system_id": system,
        "backbone": backbone,
        "seed": seed,
        "code_commit": "fixture-commit",
        "invalid_output_rate": 0.0,
        "invalid_generation_rate": 0.0,
        "invalid_judge_output_rate": 0.0,
    }
    if generation:
        summary.update(
            {
                "primary_per_label_f1": per_label,
                "primary_macro_f1": macro,
                "valid_only_macro_f1": 1.0,
                "coverage_rate": 0.25,
            }
        )
    else:
        summary.update(
            {
                "per_label_test_metrics": {f"{label}_f1": per_label[label] for label in PRAGMATIC_LABELS},
                "macro_pragmatic_f1": macro,
                "test_confidence_intervals": {
                    label: {"low": interval[0], "high": interval[1]} for label in PRAGMATIC_LABELS
                },
                "macro_confidence_interval": {"low": interval[0], "high": interval[1]},
            }
        )
    return summary


def _record(
    tmp_path: Path,
    *,
    system: str,
    backbone: str,
    seed: int | str,
    rows: list[dict[str, Any]],
    generation: bool = False,
    interval: tuple[float, float] = (0.0, 1.0),
) -> dict[str, Any]:
    run_root = tmp_path / f"{system}_{seed}"
    (run_root / "predictions").mkdir(parents=True)
    (run_root / "predictions/test_predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run_root / "config_snapshot.yaml").write_text("fixture: true\n", encoding="utf-8")
    return {
        "run_id": f"{system}_{seed}",
        "run_root": run_root,
        "summary": _summary_from_rows(
            system=system,
            backbone=backbone,
            seed=seed,
            rows=rows,
            generation=generation,
            interval=interval,
        ),
    }


def _counterexample_seed_rows() -> list[list[dict[str, Any]]]:
    true = [0, 0, 1, 1]
    return [
        _rows(true, [0, 0, 0, 0]),
        _rows(true, [0, 0, 0, 0]),
        _rows(true, [0, 0, 0, 1]),
    ]


def _counterexample_records(tmp_path: Path) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    seed_rows = _counterexample_seed_rows()
    records: list[dict[str, Any]] = []
    for seed, rows in zip(TRAINING_SEEDS, seed_rows, strict=True):
        pair = _pair(rows)
        per_seed = hierarchical_bootstrap(
            [pair],
            macro_pragmatic_f1,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        records.append(
            _record(
                tmp_path,
                system="phobert_pragmatic_finetune",
                backbone="phobert_base",
                seed=seed,
                rows=rows,
                interval=(per_seed.ci_low, per_seed.ci_high),
            )
        )
    return records, seed_rows


def _row_for(records: list[dict[str, Any]], system: str) -> dict[str, Any]:
    return next(row for row in _table2(records) if row["system"] == system)


def _load_audit_module() -> Any:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "audit_table2_confidence_intervals",
        scripts / "audit_table2_confidence_intervals.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_table2_uses_joint_hierarchical_interval(tmp_path: Path) -> None:
    records, seed_rows = _counterexample_records(tmp_path)
    row = _row_for(records, "phobert_pragmatic_finetune")
    expected = hierarchical_bootstrap(
        [_pair(rows) for rows in seed_rows],
        macro_pragmatic_f1,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    assert row["seed_count"] == len(TRAINING_SEEDS)
    assert row["macro_prag_f1"] == pytest.approx(expected.observed)
    assert row["macro_prag_ci_low"] == pytest.approx(expected.ci_low)
    assert row["macro_prag_ci_high"] == pytest.approx(expected.ci_high)


def test_table2_does_not_average_seed_bounds(tmp_path: Path) -> None:
    records, seed_rows = _counterexample_records(tmp_path)
    row = _row_for(records, "phobert_pragmatic_finetune")
    per_seed = [
        hierarchical_bootstrap(
            [_pair(rows)],
            macro_pragmatic_f1,
            resamples=BOOTSTRAP_RESAMPLES,
            seed=BOOTSTRAP_SEED,
        )
        for rows in seed_rows
    ]
    averaged_high_bound = sum(result.ci_high for result in per_seed) / len(per_seed)

    assert row["macro_prag_ci_high"] != pytest.approx(averaged_high_bound)


def test_generation_rows_receive_confidence_intervals(tmp_path: Path) -> None:
    true = [0, 0, 1, 1]
    seed_rows = [
        _rows(true, [0, 0, 0, 0], effective=[0, 0, 0, 0]),
        _rows(true, [0, 0, 0, 0], effective=[0, 0, 0, 0]),
        _rows(true, [0, 0, 0, 1], effective=[0, 0, 0, 1]),
    ]
    records = [
        _record(
            tmp_path,
            system="cot_only_vistral",
            backbone="vistral_7b",
            seed=seed,
            rows=rows,
            generation=True,
        )
        for seed, rows in zip(TRAINING_SEEDS, seed_rows, strict=True)
    ]
    row = _row_for(records, "cot_only_vistral")

    assert row["macro_prag_ci_low"] != "NOT_APPLICABLE"
    assert row["macro_prag_ci_high"] != "NOT_APPLICABLE"
    for short in SHORT_NAMES.values():
        assert isinstance(row[f"{short}_ci_low"], float)
        assert isinstance(row[f"{short}_ci_high"], float)


def test_generation_ci_uses_all_zero_fallback_predictions(tmp_path: Path) -> None:
    true = [0, 1, 1, 1]
    seed_rows = [
        _rows(true, true, effective=[0, 0, 0, 0]),
        _rows(true, true, effective=[0, 0, 0, 0]),
        _rows(true, true, effective=[0, 0, 0, 0]),
    ]
    records = [
        _record(
            tmp_path,
            system="explanation_only_vistral",
            backbone="vistral_7b",
            seed=seed,
            rows=rows,
            generation=True,
        )
        for seed, rows in zip(TRAINING_SEEDS, seed_rows, strict=True)
    ]
    row = _row_for(records, "explanation_only_vistral")
    expected = hierarchical_bootstrap(
        [_pair(rows) for rows in seed_rows],
        macro_pragmatic_f1,
        resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )

    assert all(record["summary"]["valid_only_macro_f1"] == 1.0 for record in records)
    assert row["macro_prag_f1"] == pytest.approx(expected.observed)
    assert row["macro_prag_f1"] != pytest.approx(1.0)


def test_table2_ci_prediction_alignment(tmp_path: Path) -> None:
    true = [0, 0, 1, 1]
    seed_rows = [
        _rows(true, [0, 0, 0, 0]),
        _rows(true, [0, 0, 0, 0], sample_ids=["s1", "s0", "s2", "s3"]),
        _rows(true, [0, 0, 0, 1]),
    ]
    records = [
        _record(
            tmp_path,
            system="phobert_pragmatic_finetune",
            backbone="phobert_base",
            seed=seed,
            rows=rows,
        )
        for seed, rows in zip(TRAINING_SEEDS, seed_rows, strict=True)
    ]

    with pytest.raises(ValueError, match="aligned by sample order/gold labels"):
        _table2(records)


def test_table2_ci_golden_counterexample(tmp_path: Path) -> None:
    records, _ = _counterexample_records(tmp_path)
    row = _row_for(records, "phobert_pragmatic_finetune")

    assert row["macro_prag_f1"] == pytest.approx(0.4666666666666666)
    assert row["macro_prag_ci_low"] == pytest.approx(0.06666666666666667)
    assert row["macro_prag_ci_high"] == pytest.approx(0.8095238095238094)

    report = _load_audit_module().audit(ROOT, write_report=False)
    assert report["checks"]["executable_golden_aggregation"] is True
    assert report["golden_aggregation"]["checks"]["bounds_are_not_averaged"] is True
