from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..constants import EMOTION_LABELS, POLARITY_LABELS
from ..evaluation.metrics import multiclass_macro_f1


def compose_ordinary_single_task(
    *,
    polarity_results: Mapping[str, Any],
    emotion_results: Mapping[str, Any],
    output_root: str | None = None,
) -> dict[str, Any]:
    """Compose same-seed polarity/emotion partial outputs into the paper row."""
    if str(polarity_results.get("seed")) != str(emotion_results.get("seed")):
        raise ValueError("ordinary single-task composition requires the same training seed")
    polarity_rows = polarity_results.get("predictions", {})
    emotion_rows = emotion_results.get("predictions", {})
    if not isinstance(polarity_rows, Mapping) or not isinstance(emotion_rows, Mapping):
        raise ValueError("ordinary single-task composition requires dataset-keyed predictions")
    metrics: dict[str, float] = {}
    composed: dict[str, list[dict[str, Any]]] = {}
    for dataset in ("vsfc", "aivivn"):
        rows = list(polarity_rows.get(dataset, []))
        if not rows:
            raise ValueError(f"ordinary single-task polarity output is missing {dataset}")
        gold = [str(row["gold"]) for row in rows]
        pred = [str(row["prediction"]) for row in rows]
        metrics[f"{dataset}_macro_f1"] = multiclass_macro_f1(gold, pred, POLARITY_LABELS)
        composed[dataset] = rows
    rows = list(emotion_rows.get("vsmec", []))
    if not rows:
        raise ValueError("ordinary single-task emotion output is missing vsmec")
    metrics["vsmec_macro_f1"] = multiclass_macro_f1([str(row["gold"]) for row in rows], [str(row["prediction"]) for row in rows], EMOTION_LABELS)
    composed["vsmec"] = rows
    result = {
        "status": "PASS",
        "system_id": "phobert_ordinary_single_task",
        "source_seed": polarity_results.get("seed"),
        "source_checkpoints": {"polarity": polarity_results.get("source_checkpoint"), "emotion": emotion_results.get("source_checkpoint")},
        "applicable_external_datasets": ["vsfc", "vsmec", "aivivn"],
        "predictions": composed,
        **metrics,
        "ord_f1": sum(metrics.values()) / 3.0,
        "external_finetuning": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "partial": False,
    }
    if output_root is not None:
        import json
        from pathlib import Path

        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "metrics").mkdir(parents=True, exist_ok=True)
        (root / "metrics/ordinary_single_task_composition.json").write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def compose_azure_dedicated_outputs(*, polarity_results: Mapping[str, Any], emotion_results: Mapping[str, Any]) -> dict[str, Any]:
    if str(polarity_results.get("seed")) != str(emotion_results.get("seed")):
        raise ValueError("Azure dedicated-output composition requires matching source metadata")
    return compose_ordinary_single_task(
        polarity_results={"seed": polarity_results.get("seed"), "source_checkpoint": "azure_dedicated_polarity", "predictions": {"vsfc": polarity_results.get("vsfc", []), "aivivn": polarity_results.get("aivivn", [])}},
        emotion_results={"seed": emotion_results.get("seed"), "source_checkpoint": "azure_dedicated_emotion", "predictions": {"vsmec": emotion_results.get("vsmec", [])}},
    )
