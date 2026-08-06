from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..orchestration.status import ProtocolConflict


def load_p_value_strategy(path: str | Path = "configs/statistics/significance_method.yaml") -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    required = {
        "resolution_status": "RESOLVED",
        "method_id": "paired_hierarchical_bootstrap_sign_plus_one_v1",
        "difference_direction": "left_minus_right",
        "resamples": 1000,
        "bootstrap_seed": 20260525,
        "confidence_interval": "percentile_95",
        "finite_resample_correction": "plus_one",
        "multiple_comparisons": "holm_within_7_metric_family",
        "raw_p_value_definition": "two_sided_sign_test_plus_one",
    }
    if any(config.get(key) != value for key, value in required.items()):
        raise ProtocolConflict("SCIENTIFIC_PROTOCOL_CONFLICT_SIGNIFICANCE_PVALUE")
    return config
