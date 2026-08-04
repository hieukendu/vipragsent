from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..orchestration.status import ProtocolConflict


def load_p_value_strategy(path: str | Path = "configs/statistics/significance_method.yaml") -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if config.get("resolution_status") != "RESOLVED" or not config.get("method_id") or not config.get("raw_p_value_definition"):
        raise ProtocolConflict("SCIENTIFIC_PROTOCOL_CONFLICT_SIGNIFICANCE_PVALUE")
    return config
