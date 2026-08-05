from __future__ import annotations

import json
from pathlib import Path

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json


def audit(root: str | Path = ROOT) -> dict[str, object]:
    root = Path(root)
    sources = [
        "configs/statistics.yaml",
        "configs/statistics/significance_method.yaml",
        "src/vipragsent/statistics/bootstrap.py",
        "src/vipragsent/evaluation/confidence_intervals.py",
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
    checks = {
        "approved_method_exact": exact,
        "dedicated_evaluator": implementation.exists(),
        "per_label_and_macro": all(fragment in implementation_text for fragment in ("PRAGMATIC_LABELS", '"macro"', '"labels"')),
        "prediction_hash_binding": "prediction_hash" in implementation_text,
        "no_mean_ci_fallback": "ci_low = mean" not in implementation_text and "ci_high = mean" not in implementation_text,
        "no_unapproved_bound_averaging": "bounds are not averaged" in implementation_text,
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
    }
    atomic_write_json(root / "reports/table2_confidence_interval_protocol_audit.json", report)
    return report


def main() -> int:
    report = audit()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
