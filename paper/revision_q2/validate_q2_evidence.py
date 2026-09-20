"""Offline assertions for the persisted Q2 evidence package; no network access."""

from __future__ import annotations

import json
from pathlib import Path


OUT = Path(__file__).resolve().parent


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    checkpoint = load_json(OUT / "q2_checkpoint.json")
    evidence = [json.loads(line) for line in (OUT / "run_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
    inventory = load_json(OUT / "source_inventory.json")
    smoke = load_json(OUT / "q2_smoke_report.json")

    assert checkpoint["status"] == "complete"
    assert checkpoint["evidence_count"] == 18 == len(evidence)
    assert all(row["source_ok"] for row in evidence)
    assert all(row["tree_status"] == 200 for row in inventory if row["kind"] == "artifact")

    no_polarity = [row for row in evidence if row["variant"] == "no_polarity_auxiliary"]
    assert len(no_polarity) == 3
    assert all(row["head_removed_confirmed"] for row in no_polarity)
    assert all(row["dev_prediction_count"] == 1999 and row["test_prediction_count"] == 2000 for row in no_polarity)
    assert all(all(value == 0 for value in row["polarity_presence_counts"].values()) for row in no_polarity)
    assert all(row["saved_dev_metrics_polarity_dev_ece"] is None and row["saved_test_metrics_polarity_dev_ece"] is None for row in no_polarity)
    assert all(row["head_diagnosis"] == "not applicable (polarity head removed)" for row in no_polarity)

    headed = [row for row in evidence if not row["head_removed_confirmed"]]
    assert len(headed) == 15
    assert all(row["dev_split_match"] and row["test_split_match"] for row in headed)
    assert max(row["dev_ece_abs_delta"] for row in headed) <= 1e-12
    assert max(row["test_ece_abs_delta"] for row in headed) <= 1e-12

    smoke_full = next(row for row in smoke["evidence"] if row["run"] == "xlmr_followup_q2_full_20260521")
    final_full = next(row for row in evidence if row["run"] == "xlmr_followup_q2_full_20260521")
    assert smoke_full["recomputed_dev_polarity_ece"] == final_full["recomputed_dev_polarity_ece"]
    assert smoke_full["recomputed_test_polarity_ece"] == final_full["recomputed_test_polarity_ece"]

    optional = {row["run"]: row["optional_missing_raw_files"] for row in evidence if row["optional_missing_raw_files"]}
    assert "config_snapshot.yaml" in optional["xlmr_followup_q2_no_rationale_20260523"]
    for seed in (20260521, 20260522, 20260523):
        row = next(row for row in evidence if row["run"] == f"xlmr_followup_q2_no_multitask_{seed}")
        assert row["manifest_execution_kind"] == "component_bundle"
        assert row["manifest_status_ok"] is True
        assert set(row["optional_missing_raw_files"]) == {
            "training/resolved_training_config.json",
            "selection/best_checkpoint.json",
            "selection/selection_metric.json",
        }

    raw_files = sum(1 for path in (OUT / "raw_sources").rglob("*") if path.is_file())
    assert raw_files >= 198
    print(json.dumps({"status": "offline_validation_pass", "evidence": 18, "headed_rows": 15, "no_polarity_rows": 3, "raw_files": raw_files}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
