from __future__ import annotations

import json
from pathlib import Path

from scripts.import_hf_sources import PRAGMATIC_LABELS, _write_metric_derived_predictions


def test_metric_derived_predictions_use_frozen_thresholds(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    (run_root / "metrics").mkdir(parents=True)
    (run_root / "selection").mkdir()
    probabilities = {label: [0.6, 0.4] for label in PRAGMATIC_LABELS}
    gold = {label: [0, 1] for label in PRAGMATIC_LABELS}
    thresholds = {label: 0.7 for label in PRAGMATIC_LABELS}
    (run_root / "metrics/test_metrics.json").write_text(
        json.dumps({"gold_pragmatic": gold, "raw_positive_probabilities": probabilities}),
        encoding="utf-8",
    )
    (run_root / "selection/thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")

    derivation = _write_metric_derived_predictions(run_root, "run", "metric-sha")
    rows = [
        json.loads(line)
        for line in (run_root / "predictions/test_predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert all(row["predictions"] == {label: 0 for label in PRAGMATIC_LABELS} for row in rows)
    assert derivation["threshold_source"] == "selection/thresholds.json"
    assert derivation["thresholds"] == thresholds
