"""Build compact reviewer-gap analysis artifacts from completed runs.

This script is CPU-only. It never changes a training run and skips a run until
its canonical test predictions are present and validated. It emits per-seed
metrics, positive-class counts, precision-recall data, calibration bins,
dataset overlap counts, a lightweight code-switching composition, and learned
uncertainty log-variances when a checkpoint is available.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import statistics
import sys
import time
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve

# Older system images ship NumPy 1.x, while some archived torch checkpoints
# were serialized with the NumPy 2.x module path ``numpy._core``.  Register
# the compatible legacy module aliases before torch unpickles those files.
try:
    import numpy._core  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("numpy._core", np.core)
    try:
        import numpy.core.multiarray as _numpy_multiarray

        sys.modules.setdefault("numpy._core.multiarray", _numpy_multiarray)
    except ModuleNotFoundError:
        pass

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.constants import PRAGMATIC_LABELS

SEEDS = (20260521, 20260522, 20260523)
CAMPAIGN_ID = os.environ.get("VIPRAGSENT_HF_CAMPAIGN", "reviewer-gaps-20261005")
ANALYSIS_RUN_ID = "reviewer_gap_analyses_20261005"
DEFAULT_OUTPUT = ROOT / "results/runs/reviewer_gap_analyses"
HF_QUEUE = ROOT / os.environ.get("VIPRAGSENT_HF_QUEUE", "runtime/reviewer_gaps_hf_upload_queue.jsonl")

REMOTE_FULL_REPO = "Thundergod2007/vipragsent-xlmr-checkpoints"
REMOTE_FULL_PREFIX = "campaigns/vipragsent-v8-local-mig2g20gb/q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37_{}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _f1(y_true: Iterable[int], y_pred: Iterable[int], positive: int) -> float:
    true = list(map(int, y_true))
    pred = list(map(int, y_pred))
    tp = sum(a == positive and b == positive for a, b in zip(true, pred))
    fp = sum(a != positive and b == positive for a, b in zip(true, pred))
    fn = sum(a == positive and b != positive for a, b in zip(true, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _positive_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _calibration(y_true: list[int], probability: list[float], bins: int = 10) -> dict[str, Any]:
    values = np.asarray(probability, dtype=float)
    truth = np.asarray(y_true, dtype=int)
    records: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (values >= lower) & ((values < upper) if index < bins - 1 else (values <= upper))
        count = int(mask.sum())
        mean_confidence = float(values[mask].mean()) if count else 0.0
        positive_rate = float(truth[mask].mean()) if count else 0.0
        gap = abs(mean_confidence - positive_rate)
        ece += (count / max(len(values), 1)) * gap
        records.append({
            "bin_index": index,
            "bin_lower": lower,
            "bin_upper": upper,
            "count": count,
            "mean_confidence": mean_confidence,
            "empirical_positive_rate": positive_rate,
            "absolute_gap": gap,
        })
    return {"ece": float(ece), "bins": records}


def _multiclass_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    if not rows or key not in rows[0].get("gold", {}) or key not in rows[0].get("predictions", {}):
        return None
    truth = [int(row["gold"][key]) for row in rows]
    pred = [int(row["predictions"][key]) for row in rows]
    labels = sorted(set(truth) | set(pred))
    per_class = {str(label): _f1(truth, pred, label) for label in labels}
    return {"accuracy": float(np.mean(np.asarray(truth) == np.asarray(pred))), "macro_f1": float(np.mean(list(per_class.values()))), "per_class_f1": per_class}


def _metrics_for_rows(rows: list[dict[str, Any]], metric_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    pr_curves: dict[str, Any] = {}
    for label in PRAGMATIC_LABELS:
        truth = [int(row["gold"][label]) for row in rows]
        pred = [int(row["predictions"][label]) for row in rows]
        probability = [float(row["probabilities"][label]) for row in rows]
        per_label[label] = {
            "macro_f1": float(np.mean([_f1(truth, pred, 0), _f1(truth, pred, 1)])),
            "positive_class": _positive_metrics(truth, pred),
        }
        calibration[label] = _calibration(truth, probability)
        precision, recall, thresholds = precision_recall_curve(truth, probability)
        pr_curves[label] = {
            "precision": [float(value) for value in precision],
            "recall": [float(value) for value in recall],
            "thresholds": [float(value) for value in thresholds],
        }
    macro = float(np.mean([value["macro_f1"] for value in per_label.values()]))
    result: dict[str, Any] = {
        "prediction_count": len(rows),
        "macro_pragmatic_f1": macro,
        "per_label": per_label,
        "calibration": calibration,
        "macro_ece": float(np.mean([value["ece"] for value in calibration.values()])),
        "thresholds": (metric_payload or {}).get("thresholds", {}),
        "pr_curves": pr_curves,
    }
    for key in ("polarity", "emotion"):
        value = _multiclass_metrics(rows, key)
        if value is not None:
            result[key] = value
    if metric_payload:
        for key in ("loss", "selection_metric", "macro_pragmatic_f1", "per_label_f1", "pragmatic_ece", "polarity_macro_f1", "emotion_macro_f1"):
            if key in metric_payload:
                result[f"persisted_{key}"] = metric_payload[key]
    return result


def _source_specs() -> dict[str, dict[str, Any]]:
    remote_audit = ROOT / "reports/hf_vipragsent_remote_audit_2026-09-15/raw/model/Thundergod2007__vipragsent-experiment-artifacts/server_20260808/results/runs"
    specs: dict[str, dict[str, Any]] = {
        "xlmr_full_existing": {
            "kind": "current_full_reference",
            "predictions": lambda seed: ROOT / f"reports/q1a_matched_xlmr_ft/seed_{seed}/predictions_test.jsonl",
            "metrics": lambda seed: ROOT / f"reports/q1a_matched_xlmr_ft/seed_{seed}/metrics_test.json",
        },
        "xlmr_pragmatic_only_existing": {
            "kind": "existing_pragmatic_only_reference",
            "predictions": lambda seed: remote_audit / f"q1a_xlmr_pragmatic_finetune_{seed}/predictions/test_predictions.jsonl",
            "metrics": lambda seed: remote_audit / f"q1a_xlmr_pragmatic_finetune_{seed}/metrics/test_metrics.json",
        },
        "visobert_baseline": {
            "kind": "reviewer_gap_experiment",
            "predictions": lambda seed: ROOT / f"results/runs/reviewer_visobert_baseline_{seed}/predictions/test_predictions.jsonl",
            "metrics": lambda seed: ROOT / f"results/runs/reviewer_visobert_baseline_{seed}/metrics/test_metrics.json",
        },
    }
    for variant in ("all_multipliers_1", "beta_01", "beta_05"):
        specs[f"xlmr_{variant}"] = {
            "kind": "reviewer_gap_experiment",
            "predictions": lambda seed, variant=variant: ROOT / f"results/runs/priority12_v37_q1_{variant}_{seed}/predictions/test_predictions.jsonl",
            "metrics": lambda seed, variant=variant: ROOT / f"results/runs/priority12_v37_q1_{variant}_{seed}/metrics/test_metrics.json",
        }
    return specs


def _collect_predictions() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    per_seed: dict[str, Any] = {}
    aggregates: dict[str, Any] = {}
    pr_curves: dict[str, Any] = {}
    for experiment, spec in _source_specs().items():
        records: list[dict[str, Any]] = []
        for seed in SEEDS:
            prediction_path = Path(spec["predictions"](seed))
            metrics_path = Path(spec["metrics"](seed))
            if not prediction_path.exists():
                continue
            rows = _read_jsonl(prediction_path)
            metric_payload = _read_json(metrics_path) if metrics_path.exists() else None
            metrics = _metrics_for_rows(rows, metric_payload)
            record = {"experiment": experiment, "seed": seed, "prediction_path": str(prediction_path.relative_to(ROOT)), "metrics": metrics}
            records.append(record)
            pr_curves[f"{experiment}/seed_{seed}"] = metrics.pop("pr_curves")
        if not records:
            continue
        per_seed[experiment] = records
        numeric_fields = ["macro_pragmatic_f1", "macro_ece"]
        aggregate: dict[str, Any] = {"seed_count": len(records), "seeds": [record["seed"] for record in records]}
        for field in numeric_fields:
            values = [float(record["metrics"][field]) for record in records]
            aggregate[field] = {"mean": float(statistics.mean(values)), "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0, "values": values}
        labels: dict[str, Any] = {}
        for label in PRAGMATIC_LABELS:
            values = [float(record["metrics"]["per_label"][label]["macro_f1"]) for record in records]
            positive = [record["metrics"]["per_label"][label]["positive_class"] for record in records]
            positive_summary = {}
            for metric_name in ("precision", "recall", "f1"):
                metric_values = [float(item[metric_name]) for item in positive]
                positive_summary[metric_name] = {
                    "mean": float(statistics.mean(metric_values)),
                    "sample_sd": float(statistics.stdev(metric_values)) if len(metric_values) > 1 else 0.0,
                    "values": metric_values,
                }
            labels[label] = {
                "macro_f1": {"mean": float(statistics.mean(values)), "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0, "values": values},
                "positive_class": positive,
                "positive_class_summary": positive_summary,
            }
        aggregate["per_label"] = labels
        aggregates[experiment] = aggregate
    return per_seed, aggregates, pr_curves


def _dataset_overlap() -> dict[str, Any]:
    vipragsent_paths = [ROOT / "data/processed/vipragsent" / f"{split}.csv" for split in ("train", "dev", "test")]
    vipragsent_sets: dict[str, set[str]] = {}
    all_vipragsent: set[str] = set()
    for path in vipragsent_paths:
        frame = pd.read_csv(path)
        values = {_normalise_text(value) for value in frame["text"].dropna() if _normalise_text(value)}
        split = path.stem
        vipragsent_sets[split] = values
        all_vipragsent.update(values)
    external_specs = {
        "UIT-VSFC": (ROOT / "data/processed/external/uit_vsfc/test.csv", "text"),
        "UIT-VSMEC": (ROOT / "data/processed/external/uit_vsmec/test.csv", "text"),
        "AIVIVN": (ROOT / "data/processed/external/aivivn_human_derived_3way/test.csv", "comment"),
    }
    result: dict[str, Any] = {"normalization": "Unicode NFC + whitespace collapse + casefold", "vipragsent_unique_texts": len(all_vipragsent), "vipragsent_split_unique_texts": {key: len(value) for key, value in vipragsent_sets.items()}, "datasets": {}}
    for name, (path, column) in external_specs.items():
        frame = pd.read_csv(path)
        external = {_normalise_text(value) for value in frame[column].dropna() if _normalise_text(value)}
        overlap = all_vipragsent.intersection(external)
        result["datasets"][name] = {
            "source": str(path.relative_to(ROOT)),
            "source_rows": int(len(frame)),
            "unique_normalized_texts": len(external),
            "exact_normalized_overlap_count": len(overlap),
            "overlap_rate_of_external": len(overlap) / max(len(external), 1),
            "overlap_rate_of_vipragsent": len(overlap) / max(len(all_vipragsent), 1),
            "overlap_by_vipragsent_split": {split: len(values.intersection(external)) for split, values in vipragsent_sets.items()},
        }
    return result


def _code_switching_composition() -> dict[str, Any]:
    latin = re.compile(r"[A-Za-z]")
    vietnamese_marks = re.compile(r"[ăâđêôơưáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", re.IGNORECASE)
    result: dict[str, Any] = {"method": "heuristic on code_switching=1 rows: latin_with_vietnamese_marks / latin_without_marks / non_latin / no_latin", "splits": {}}
    for split in ("train", "dev", "test"):
        frame = pd.read_csv(ROOT / "data/processed/vipragsent" / f"{split}.csv")
        positive = frame[frame["code_switching"].astype(int) == 1]
        counts = {"latin_with_vietnamese_marks": 0, "latin_without_marks": 0, "non_latin": 0, "no_latin": 0}
        for text in positive["text"].fillna("").astype(str):
            has_latin = bool(latin.search(text))
            if not has_latin:
                counts["no_latin"] += 1
            elif vietnamese_marks.search(text):
                counts["latin_with_vietnamese_marks"] += 1
            elif any(ord(char) > 127 for char in text):
                counts["non_latin"] += 1
            else:
                counts["latin_without_marks"] += 1
        result["splits"][split] = {"positive_rows": int(len(positive)), "counts": counts, "shares": {key: value / max(len(positive), 1) for key, value in counts.items()}}
    return result


def _extract_log_variances(checkpoint_path: Path) -> dict[str, float]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    candidates: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for key in ("loss_aggregator_state_dict", "loss_aggregator", "loss_aggregator_state"):
            value = payload.get(key)
            if isinstance(value, dict):
                candidates.append(value)
        candidates.append(payload)
    for candidate in candidates:
        found = {str(key).split("log_variances.", 1)[-1]: float(value.detach().cpu()) for key, value in candidate.items() if "log_variances" in str(key) and torch.is_tensor(value)}
        if found:
            return found
    raise KeyError("checkpoint does not expose learned loss-aggregator log_variances")


def _learned_log_variance(output_root: Path, download_remote: bool) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        local = ROOT / f"results/runs/priority12_v37_q1_all_multipliers_1_{seed}/checkpoints/best/model.pt"
        if local.exists():
            sources[f"xlmr_all_multipliers_1_seed_{seed}"] = {"source": str(local.relative_to(ROOT)), "path": local}
        for variant in ("beta_01", "beta_05"):
            local = ROOT / f"results/runs/priority12_v37_q1_{variant}_{seed}/checkpoints/best/model.pt"
            if local.exists():
                sources[f"xlmr_{variant}_seed_{seed}"] = {"source": str(local.relative_to(ROOT)), "path": local}
        local_full = ROOT / f"results/runs/reviewer_visobert_baseline_{seed}/checkpoints/best/model.pt"
        if local_full.exists():
            sources[f"visobert_baseline_seed_{seed}"] = {"source": str(local_full.relative_to(ROOT)), "path": local_full}
        remote_path = REMOTE_FULL_PREFIX.format(seed) + "/checkpoints/best/model.pt"
        sources[f"xlmr_full_existing_seed_{seed}"] = {"source": f"hf://{REMOTE_FULL_REPO}/{remote_path}", "remote_path": remote_path}
    result: dict[str, Any] = {"status": "PASS", "entries": {}}
    for name, item in sources.items():
        path = item.get("path")
        try:
            if path is None:
                if not download_remote:
                    result["entries"][name] = {"status": "NOT_DOWNLOADED", "source": item["source"]}
                    continue
                from huggingface_hub import hf_hub_download

                path = Path(hf_hub_download(
                    repo_id=REMOTE_FULL_REPO,
                    filename=item["remote_path"],
                    repo_type="model",
                    token=os.environ.get("HF_TOKEN"),
                    cache_dir=str(ROOT / "data/hf_checkpoint_cache"),
                ))
            result["entries"][name] = {"status": "PASS", "source": item["source"], "log_variances": _extract_log_variances(path)}
        except Exception as exc:
            result["entries"][name] = {"status": "ERROR", "source": item["source"], "error": f"{type(exc).__name__}: {exc}"}
    return result


def _queue_analysis() -> None:
    HF_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "schema_version": 1,
        "run_id": ANALYSIS_RUN_ID,
        "source_root": "results/runs/reviewer_gap_analyses",
        "backbone": "xlmr_large",
        "status": "queued",
        "campaign_id": CAMPAIGN_ID,
        "queued_at": time.time(),
    }
    with HF_QUEUE.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing = {json.loads(line).get("run_id") for line in handle if line.strip()}
        if ANALYSIS_RUN_ID not in existing:
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reviewer-gap analysis artifacts")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download-remote-log-variance", action="store_true")
    parser.add_argument("--queue", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    per_seed, aggregates, pr_curves = _collect_predictions()
    atomic_write_json(output_root / "per_seed_metrics.json", per_seed)
    atomic_write_json(output_root / "aggregate_metrics.json", aggregates)
    atomic_write_json(output_root / "pr_curves.json", pr_curves)
    calibration = {experiment: {str(record["seed"]): record["metrics"]["calibration"] for record in records} for experiment, records in per_seed.items()}
    atomic_write_json(output_root / "calibration.json", calibration)
    atomic_write_json(output_root / "dataset_overlap.json", _dataset_overlap())
    atomic_write_json(output_root / "code_switching_composition.json", _code_switching_composition())
    atomic_write_json(output_root / "learned_log_variance.json", _learned_log_variance(output_root, args.download_remote_log_variance))
    atomic_write_json(output_root / "analysis_manifest.json", {
        "schema_version": 1,
        "status": "PASS",
        "run_id": ANALYSIS_RUN_ID,
        "campaign_id": CAMPAIGN_ID,
        "created_at_unix": time.time(),
        "source_protocol": "configs/experiments/priority12_xlmr.yaml",
        "experiments_present": sorted(per_seed),
        "notes": [
            "Existing Full XLM-R predictions are the frozen q1a_matched_xlmr_ft reference.",
            "Exact overlap uses normalized text sets over all ViPragSent splits and external test/evaluation CSVs.",
            "Code-switching composition is a heuristic diagnostic, not a new label annotation.",
        ],
    })
    if args.queue:
        _queue_analysis()
    print(json.dumps({"status": "PASS", "output_root": str(output_root), "experiments_present": sorted(per_seed), "queued": args.queue}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
