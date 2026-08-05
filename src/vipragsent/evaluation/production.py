from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS, TRAINING_SEEDS
from ..hashing import sha256_file
from ..protocol import validate_protocol_resolution
from ..orchestration.status import ProtocolConflict
from ..statistics.bootstrap import BootstrapResult, paired_bootstrap_comparison
from ..statistics.significance import load_p_value_strategy
from .metrics import align_prediction_rows, binary_macro_f1, expected_calibration_error, macro_pragmatic_f1, multiclass_macro_f1, pragmatic_ece, reliability_bins
from .thresholds import tune_pragmatic_thresholds


def read_prediction_jsonl(path: str | Path, sample_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return [dict(row) for row in align_prediction_rows(sample_ids, rows)]


def evaluate_q1a(
    sample_ids: Sequence[str],
    gold: Mapping[str, Sequence[int]],
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float] | None = None,
    *,
    prediction_file: str | Path | None = None,
) -> dict[str, Any]:
    aligned = align_prediction_rows(sample_ids, rows)
    probabilities = {key: [float(row.get("probabilities", {}).get(key, row.get("pragmatic_probability", {}).get(key, 0.0))) for row in aligned] for key in PRAGMATIC_LABELS}
    frozen = dict(thresholds or tune_pragmatic_thresholds(gold, probabilities))
    predictions = {key: [int(value >= frozen[key]) for value in probabilities[key]] for key in PRAGMATIC_LABELS}
    metrics = {f"{key}_f1": binary_macro_f1(gold[key], predictions[key]) for key in PRAGMATIC_LABELS}
    metrics["macro_prag_f1"] = macro_pragmatic_f1(gold, predictions)
    metrics["thresholds_frozen"] = True
    metrics["prediction_file_sha256"] = None if prediction_file is None else sha256_file(prediction_file)
    return metrics


def evaluate_q1b_external(run_manifest: Mapping[str, Any], *, vsfc: Sequence[str], vsmec: Sequence[str], aivivn: Sequence[str], predictions: Mapping[str, Sequence[str]]) -> dict[str, float]:
    if bool(run_manifest.get("external_finetuning")):
        raise ValueError("Q1b rejects external_finetuning=true")
    vsfc_score = multiclass_macro_f1(vsfc, predictions["vsfc"], POLARITY_LABELS)
    vsmec_score = multiclass_macro_f1(vsmec, predictions["vsmec"], EMOTION_LABELS)
    aivivn_score = multiclass_macro_f1(aivivn, predictions["aivivn"], POLARITY_LABELS)
    return {"vsfc_macro_f1": vsfc_score, "vsmec_macro_f1": vsmec_score, "aivivn_macro_f1": aivivn_score, "ord_f1": float(np.mean([vsfc_score, vsmec_score, aivivn_score]))}


def evaluate_q3(true: Sequence[int], probabilities: Sequence[float], *, threshold: float, selected_positive_count: int, fixed_negative_count: int, data_hash: str, mask_hash: str) -> dict[str, Any]:
    return {"sarcasm_macro_f1": binary_macro_f1(true, (np.asarray(probabilities) >= threshold).astype(int)), "selected_positive_count": selected_positive_count, "fixed_negative_count": fixed_negative_count, "dev_threshold": threshold, "pos_weight": fixed_negative_count / selected_positive_count, "data_hash": data_hash, "mask_hash": mask_hash}


def evaluate_q4_seed(probabilities: Mapping[str, Sequence[float]], true: Mapping[str, Sequence[int]], *, seed: int) -> dict[str, Any]:
    ece_by_label, macro_ece, bins = pragmatic_ece(true, probabilities, bins=10)
    return {
        "seed": seed,
        "split": "vipragsent_test",
        "temperature_scaling": False,
        "ece_by_label": ece_by_label,
        "macro_pragmatic_ece": macro_ece,
        "reliability_bins": bins,
    }


def evaluate_q4_seeds(per_seed: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not per_seed or {item.get("seed") for item in per_seed} != set(TRAINING_SEEDS):
        raise ValueError("Q4 trainable systems require exactly the three locked training seeds")
    labels = tuple(PRAGMATIC_LABELS)
    summary = {
        label: {
            "mean_ece": float(np.mean([item["ece_by_label"][label] for item in per_seed])),
            "std_ece": float(np.std([item["ece_by_label"][label] for item in per_seed], ddof=1)),
        }
        for label in labels
    }
    macro_values = [item["macro_pragmatic_ece"] for item in per_seed]
    return {
        "split": "vipragsent_test",
        "temperature_scaling": False,
        "per_label": summary,
        "mean_macro_pragmatic_ece": float(np.mean(macro_values)),
        "std_macro_pragmatic_ece": float(np.std(macro_values, ddof=1)),
        "per_seed": per_seed,
        "probability_aggregation": "none",
    }


def paired_significance(left: Sequence[tuple[Sequence[Any], Sequence[Any]]], right: Sequence[tuple[Sequence[Any], Sequence[Any]]], metric: Any, *, root: str | Path = ".", resamples: int = 1000) -> BootstrapResult:
    resolution = validate_protocol_resolution(root)
    if "SCIENTIFIC_PROTOCOL_CONFLICT_SIGNIFICANCE_PVALUE" in resolution["scientific_protocol_conflicts"]:
        raise ProtocolConflict("SCIENTIFIC_PROTOCOL_CONFLICT_SIGNIFICANCE_PVALUE")
    strategy = load_p_value_strategy(Path(root) / "configs/statistics/significance_method.yaml")
    return paired_bootstrap_comparison(
        left,
        right,
        metric,
        resamples=int(strategy["resamples"] if resamples == 1000 else resamples),
        seed=int(strategy["bootstrap_seed"]),
        p_value_method=strategy["method_id"],
    )
