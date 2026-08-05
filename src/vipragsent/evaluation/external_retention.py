from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, atomic_write_text
from ..constants import EMOTION_LABELS, POLARITY_LABELS
from ..hashing import sha256_file
from .metrics import multiclass_macro_f1


@dataclass(frozen=True)
class NormalizedExternalExample:
    sample_id: str
    text: str
    label: str


@dataclass(frozen=True)
class ExternalRetentionResult:
    vsfc_macro_f1: float
    vsmec_macro_f1: float
    aivivn_macro_f1: float
    ord_f1: float
    source_checkpoint_id: str
    source_seed: int | str | None
    external_finetuning: bool
    external_manifest_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "vsfc_macro_f1": self.vsfc_macro_f1,
            "vsmec_macro_f1": self.vsmec_macro_f1,
            "aivivn_macro_f1": self.aivivn_macro_f1,
            "ord_f1": self.ord_f1,
            "source_checkpoint_id": self.source_checkpoint_id,
            "source_seed": self.source_seed,
            "external_finetuning": self.external_finetuning,
            "external_manifest_hash": self.external_manifest_hash,
        }


def _check_alignment(examples: Sequence[NormalizedExternalExample], predictions: Mapping[str, str]) -> list[str]:
    ids = [example.sample_id for example in examples]
    if len(ids) != len(set(ids)):
        raise ValueError("External sample IDs must be unique")
    missing = sorted(set(ids) - set(predictions))
    extra = sorted(set(predictions) - set(ids))
    if missing or extra:
        raise ValueError(f"External prediction alignment mismatch; missing={missing[:5]}, extra={extra[:5]}")
    return [predictions[sample_id] for sample_id in ids]


def evaluate_external_retention(
    datasets: Mapping[str, Sequence[NormalizedExternalExample]],
    predictions: Mapping[str, Mapping[str, str]],
    *,
    source_checkpoint_id: str,
    source_seed: int | str | None,
    external_manifest_hash: str,
    external_finetuning: bool = False,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    if external_finetuning:
        raise ValueError("Q1b external_finetuning must be false")
    required = {"vsfc", "vsmec", "aivivn"}
    if set(datasets) != required or set(predictions) != required:
        raise ValueError("Q1b requires exactly VSFC, VSMEC, and AIVIVN external datasets")
    vsfc = datasets["vsfc"]
    vsmec = datasets["vsmec"]
    aivivn = datasets["aivivn"]
    if any(row.label not in POLARITY_LABELS for row in vsfc) or any(row.label not in EMOTION_LABELS for row in vsmec) or any(row.label not in POLARITY_LABELS for row in aivivn):
        raise ValueError("External labels do not match the locked VSFC/VSMEC/AIVIVN label spaces")
    vsfc_pred = _check_alignment(vsfc, predictions["vsfc"])
    vsmec_pred = _check_alignment(vsmec, predictions["vsmec"])
    aivivn_pred = _check_alignment(aivivn, predictions["aivivn"])
    if any(label not in POLARITY_LABELS for label in vsfc_pred + aivivn_pred) or any(label not in EMOTION_LABELS for label in vsmec_pred):
        raise ValueError("External predictions do not match the locked label spaces")
    vsfc_score = multiclass_macro_f1([row.label for row in vsfc], vsfc_pred, POLARITY_LABELS)
    vsmec_score = multiclass_macro_f1([row.label for row in vsmec], vsmec_pred, EMOTION_LABELS)
    aivivn_score = multiclass_macro_f1([row.label for row in aivivn], aivivn_pred, POLARITY_LABELS)
    result = ExternalRetentionResult(vsfc_score, vsmec_score, aivivn_score, (vsfc_score + vsmec_score + aivivn_score) / 3.0, str(source_checkpoint_id), source_seed, False, str(external_manifest_hash))
    payload = result.as_dict() | {"external_preprocessing": "locked_normalized_test_only", "optimizer_steps": 0, "train_loader_created": False}
    if output_root is not None:
        root = Path(output_root)
        for key, examples in datasets.items():
            rows = [{"sample_id": row.sample_id, "text": row.text, "gold": row.label, "prediction": predictions[key][row.sample_id]} for row in examples]
            filename = {"vsfc": "uit_vsfc_test_predictions.jsonl", "vsmec": "uit_vsmec_test_predictions.jsonl", "aivivn": "aivivn_test_predictions.jsonl"}[key]
            atomic_write_text(root / "predictions" / filename, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        atomic_write_json(root / "metrics/external_retention_metrics.json", payload)
    return payload


def evaluate_external_from_rows(
    gold_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    predict: Callable[[str, str], str],
    *,
    source_checkpoint_id: str,
    source_seed: int | str | None,
    external_manifest_hash: str,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    datasets: dict[str, list[NormalizedExternalExample]] = {}
    predictions: dict[str, dict[str, str]] = {}
    for key, rows in gold_rows.items():
        datasets[key] = [NormalizedExternalExample(str(row["sample_id"]), str(row.get("text", "")), str(row["label"])) for row in rows]
        predictions[key] = {example.sample_id: predict(key, example.text) for example in datasets[key]}
    return evaluate_external_retention(datasets, predictions, source_checkpoint_id=source_checkpoint_id, source_seed=source_seed, external_manifest_hash=external_manifest_hash, output_root=output_root)


def external_manifest_hash(path: str | Path) -> str:
    return sha256_file(path)
