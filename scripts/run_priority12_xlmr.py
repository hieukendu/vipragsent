"""Run the isolated Priority 1 and corrected Priority 2 XLM-R experiments.

This runner intentionally writes only to the ``priority12_*`` namespaces. It
does not reuse or overwrite the frozen Q1a roots. Test data are tokenized and
evaluated only after the engine freezes the dev-selected checkpoint and
thresholds.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.constants import (
    EMOTION_LABELS,
    EXPECTED_SPLIT_COUNTS,
    POLARITY_LABELS,
    PRAGMATIC_LABELS,
)
from vipragsent.data.collation import BatchCollator
from vipragsent.data.loaders import DatasetExample, load_vipragsent
from vipragsent.data.masks import load_validated_q3_masks
from vipragsent.data.preprocessing import PreprocessingSpec, TextPreprocessor
from vipragsent.data.tokenizers import create_tokenizer
from vipragsent.evaluation.metrics import (
    binary_macro_f1,
    macro_pragmatic_f1,
    multiclass_macro_f1,
    pragmatic_ece,
)
from vipragsent.hashing import sha256_file
from vipragsent.models.factory import build_production_model
from vipragsent.runtime.model_assets import read_family_status, resolve_local_snapshot
from vipragsent.training.class_weights import compute_train_only_class_weights
from vipragsent.training.engine import SelectionResult, TrainingConfig, TrainingEngine
from vipragsent.training.seeding import seed_everything

PROTOCOL_PATH = ROOT / "configs/experiments/priority12_xlmr.yaml"
FAMILY = "xlmr_large"
MODEL_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
MODEL_REPOSITORY = "FacebookAI/xlm-roberta-large"
SEEDS = (20260521, 20260522, 20260523)
Q1_VARIANTS = {
    "full": "vipragsent_full_xlmr_large",
    "no_emotion_auxiliary": "no_emotion_auxiliary",
    "no_polarity_auxiliary": "no_polarity_auxiliary",
    "no_rationale": "no_rationale",
    "no_uncertainty_weighting": "no_uncertainty_weighting",
    "all_multipliers_1": "vipragsent_full_xlmr_large",
    "beta_01": "vipragsent_full_xlmr_large",
    "beta_05": "vipragsent_full_xlmr_large",
}
Q1_RUN_ALL_VARIANTS = ("full", "no_emotion_auxiliary", "no_polarity_auxiliary", "no_rationale", "no_uncertainty_weighting")
Q3_BUDGETS = ("32", "64", "128", "256", "512", "full")
RESULT_ROOT = ROOT / "results/runs"
REPORT_ROOT = ROOT / "reports/priority12_v37"
CAMPAIGN_ID = os.environ.get("VIPRAGSENT_HF_CAMPAIGN", "vipragsent-priority12-v37-20261003")
HF_QUEUE = ROOT / os.environ.get("VIPRAGSENT_HF_QUEUE", "runtime/priority12_v37_hf_upload_queue.jsonl")


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _protocol() -> dict[str, Any]:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8")) or {}
    if payload.get("model", {}).get("revision") != MODEL_REVISION:
        raise RuntimeError("Priority 1/2 protocol is not pinned to the local XLM-R snapshot revision")
    expected = {str(key): int(value) for key, value in payload["data"]["expected_split_counts"].items()}
    if expected != {key: int(value) for key, value in EXPECTED_SPLIT_COUNTS.items()}:
        raise RuntimeError(f"Priority 1/2 split contract differs from the frozen dataset: {expected}")
    return dict(payload)


def _snapshot() -> Path:
    status = read_family_status(ROOT, FAMILY, "cache")
    if status.get("status") != "PASS":
        raise RuntimeError(f"Pinned {FAMILY} cache is not PASS: {status.get('status')}")
    snapshot = resolve_local_snapshot(ROOT, status.get("local_path"))
    required = ("config.json", "model.safetensors", "sentencepiece.bpe.model", "tokenizer.json")
    if snapshot is None or not snapshot.exists() or any(not (snapshot / name).exists() for name in required):
        raise RuntimeError("Pinned local XLM-R-large snapshot is unavailable")
    return snapshot


def _load_rationales(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Approved rationale artifact is unavailable: {path}")
    records: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[str(row["sample_id"])] = row
    if len(records) != len(path.read_text(encoding="utf-8").splitlines()):
        raise RuntimeError("Approved rationale artifact contains duplicate sample IDs")
    return records


def _training_config(
    protocol: dict[str, Any],
    *,
    primary_metric: str,
    use_uncertainty: bool,
    rationale_beta_override: float | None = None,
) -> TrainingConfig:
    training = protocol["training"]
    return TrainingConfig(
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        max_epochs=int(training["maximum_epochs"]),
        effective_batch_size=int(training["effective_batch_size"]),
        physical_batch_size=int(training["physical_batch_size"]),
        max_grad_norm=float(training["gradient_clipping"]),
        patience=int(training["patience"]),
        min_delta=float(training["minimum_delta"]),
        precision=str(training["precision"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        primary_metric=primary_metric,
        scheduler=str(training["scheduler"]),
        warmup_ratio=float(training["warmup_ratio"]),
        use_uncertainty_weighting=use_uncertainty,
        rationale_beta=float(training["rationale_beta"] if rationale_beta_override is None else rationale_beta_override),
        rationale_target_max_length=int(training["rationale_target_max_length"]),
        optimizer=str(training["optimizer"]),
        gradient_checkpointing=False,
        deterministic_algorithms="warn_only",
        qlora={"quantization": {"type": "none"}, "lora": {}},
    )


def _loss_multipliers(protocol: dict[str, Any], *, all_multipliers_one: bool = False) -> dict[str, float]:
    if all_multipliers_one:
        return {label: 1.0 for label in (*PRAGMATIC_LABELS, "polarity", "emotion")}
    loss = protocol["loss"]
    head_multipliers = loss.get("pragmatic_head_multipliers")
    if head_multipliers is not None:
        expected = set(PRAGMATIC_LABELS)
        if set(head_multipliers) != expected:
            raise RuntimeError("V37 pragmatic head multiplier map does not cover the canonical heads exactly")
        multipliers = {label: float(head_multipliers[label]) for label in PRAGMATIC_LABELS}
    else:
        multipliers = {label: float(loss["pragmatic_loss_multiplier"]) for label in PRAGMATIC_LABELS}
        # Legacy profiles use an explicit irony replacement, not a second
        # multiplication.
        multipliers["irony"] = float(loss["irony_loss_multiplier"])
    multipliers["polarity"] = float(loss["auxiliary_loss_multiplier"])
    multipliers["emotion"] = float(loss["auxiliary_loss_multiplier"])
    return multipliers


def _batches(
    rows: list[DatasetExample],
    *,
    tokenizer: Any,
    preprocessor: TextPreprocessor,
    weights: Any,
    physical_batch_size: int,
    rationale_records: dict[str, Any] | None,
    rationale_target_max_length: int,
    q3_masks: dict[str, dict[str, dict[str, str]]] | None = None,
    budget: str | None = None,
    mask_hash: str | None = None,
) -> list[dict[str, Any]]:
    collator = BatchCollator(
        tokenizer,
        preprocessor,
        q3_masks=q3_masks,
        budget=budget,
        mask_hash=mask_hash,
        class_weights=weights.as_dict() if hasattr(weights, "as_dict") else dict(weights),
        rationale_records=rationale_records,
        rationale_target_max_length=rationale_target_max_length,
    )
    return [
        collator(rows[index : index + physical_batch_size])
        for index in range(0, len(rows), physical_batch_size)
    ]


def _positive_metrics(true: list[Any], pred: list[Any]) -> dict[str, float | int]:
    y_true = np.asarray(true, dtype=int)
    y_pred = np.asarray(pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _metrics(selection: SelectionResult) -> dict[str, Any]:
    pragmatic_true = {label: selection.true[label] for label in PRAGMATIC_LABELS}
    pragmatic_pred = {label: selection.predictions[label] for label in PRAGMATIC_LABELS}
    pragmatic_prob = {label: selection.probabilities[label] for label in PRAGMATIC_LABELS}
    ece_by_label, macro_ece, reliability = pragmatic_ece(pragmatic_true, pragmatic_prob)
    result: dict[str, Any] = {
        "prediction_count": len(selection.true[PRAGMATIC_LABELS[0]]),
        "per_label_f1": {label: binary_macro_f1(pragmatic_true[label], pragmatic_pred[label]) for label in PRAGMATIC_LABELS},
        "positive_class_metrics": {label: _positive_metrics(pragmatic_true[label], pragmatic_pred[label]) for label in PRAGMATIC_LABELS},
        "macro_pragmatic_f1": macro_pragmatic_f1(pragmatic_true, pragmatic_pred),
        "sarcasm_binary_macro_f1": binary_macro_f1(pragmatic_true["sarcasm"], pragmatic_pred["sarcasm"]),
        "pragmatic_ece": {"per_label": ece_by_label, "macro": macro_ece, "bins": reliability},
        "thresholds": dict(selection.thresholds),
        "selection_metric": float(selection.metric),
        "loss": float(selection.total_loss),
    }
    if "polarity" in selection.true and "polarity" in selection.predictions:
        result["polarity_macro_f1"] = multiclass_macro_f1(
            selection.true["polarity"], selection.predictions["polarity"], range(len(POLARITY_LABELS))
        )
        result["polarity_accuracy"] = float(np.mean(np.asarray(selection.true["polarity"]) == np.asarray(selection.predictions["polarity"])))
    if "emotion" in selection.true and "emotion" in selection.predictions:
        result["emotion_macro_f1"] = multiclass_macro_f1(
            selection.true["emotion"], selection.predictions["emotion"], range(len(EMOTION_LABELS))
        )
        result["emotion_accuracy"] = float(np.mean(np.asarray(selection.true["emotion"]) == np.asarray(selection.predictions["emotion"])))
    return result


def _write_predictions(path: Path, engine: TrainingEngine, selection: SelectionResult, batches: list[dict[str, Any]]) -> None:
    sample_ids = [sample_id for batch in batches for sample_id in batch.get("sample_ids", [])]
    engine._write_prediction_jsonl(path, engine._prediction_rows(selection, sample_ids=sample_ids))


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _resource_snapshot(device: int) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "pid": os.getpid(),
        "cpu_count": os.cpu_count(),
        "device_index": device,
        "captured_at_unix": time.time(),
    }
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,comm=,%cpu=,%mem=,args="],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        rows: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            fields = line.strip().split(None, 4)
            if len(fields) < 5:
                continue
            try:
                rows.append({"pid": int(fields[0]), "command": fields[1], "cpu_percent": float(fields[2]), "mem_percent": float(fields[3]), "args": fields[4]})
            except ValueError:
                continue
        snapshot["top_cpu_processes"] = sorted(rows, key=lambda row: (-row["cpu_percent"], row["pid"]))[:12]
        snapshot["ps_status"] = result.returncode
    except (OSError, subprocess.TimeoutExpired) as exc:
        snapshot["ps_status"] = f"unavailable: {exc}"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        snapshot["nvidia_smi_status"] = result.returncode
        snapshot["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        snapshot["nvidia_smi_status"] = f"unavailable: {exc}"
    if torch.cuda.is_available():
        selected = torch.device(f"cuda:{device}")
        properties = torch.cuda.get_device_properties(selected)
        free, total = torch.cuda.mem_get_info(selected)
        snapshot["torch_cuda"] = {
            "device_name": torch.cuda.get_device_name(selected),
            "device_count": torch.cuda.device_count(),
            "compute_capability": list(torch.cuda.get_device_capability(selected)),
            "total_memory_gb": total / (1024**3),
            "free_memory_gb": free / (1024**3),
            "properties_total_memory_gb": properties.total_memory / (1024**3),
            "allocated_gb": torch.cuda.memory_allocated(selected) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(selected) / (1024**3),
        }
    else:
        snapshot["torch_cuda"] = {"available": False}
    return snapshot


def _gold(example: DatasetExample) -> dict[str, Any]:
    return {
        **{label: int(example.labels[label]) for label in PRAGMATIC_LABELS},
        "polarity": POLARITY_LABELS.index(example.labels["polarity"]),
        "emotion": EMOTION_LABELS.index(example.labels["emotion"]),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _validate_predictions(
    path: Path,
    rows: list[DatasetExample],
    *,
    active_auxiliaries: set[str],
) -> dict[str, Any]:
    predictions = _read_jsonl(path)
    expected_ids = [row.sample_id for row in rows]
    actual_ids = [str(item.get("sample_id", "")) for item in predictions]
    if actual_ids != expected_ids:
        raise ValueError(f"Prediction order/alignment mismatch for {path}")
    if len(actual_ids) != len(set(actual_ids)):
        raise ValueError(f"Duplicate prediction sample IDs for {path}")
    expected_prediction_keys = set(PRAGMATIC_LABELS)
    expected_supervised_keys = expected_prediction_keys | active_auxiliaries
    for index, (item, example) in enumerate(zip(predictions, rows, strict=True)):
        if (
            set(item.get("gold", {})) != expected_supervised_keys
            or set(item.get("predictions", {})) != expected_prediction_keys
            or set(item.get("probabilities", {})) != expected_supervised_keys
        ):
            raise ValueError(f"Prediction head keys mismatch at row {index} in {path}")
        if item["gold"] != {key: _gold(example)[key] for key in expected_supervised_keys}:
            raise ValueError(f"Prediction gold mismatch at row {index} in {path}")
        for label in PRAGMATIC_LABELS:
            probability = float(item["probabilities"][label])
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"Invalid pragmatic probability at row {index}, label {label}")
            if int(item["predictions"][label]) not in (0, 1):
                raise ValueError(f"Invalid pragmatic prediction at row {index}, label {label}")
        if "polarity" in active_auxiliaries:
            probs = np.asarray(item["probabilities"]["polarity"], dtype=float)
            if probs.shape != (len(POLARITY_LABELS),) or not np.all(np.isfinite(probs)) or not np.isclose(float(probs.sum()), 1.0, atol=1e-5):
                raise ValueError(f"Invalid polarity probabilities at row {index}")
        if "emotion" in active_auxiliaries:
            probs = np.asarray(item["probabilities"]["emotion"], dtype=float)
            if probs.shape != (len(EMOTION_LABELS),) or not np.all(np.isfinite(probs)) or not np.isclose(float(probs.sum()), 1.0, atol=1e-5):
                raise ValueError(f"Invalid emotion probabilities at row {index}")
    return {"status": "PASS", "path": str(path.relative_to(ROOT)), "count": len(predictions), "sample_id_order": "frozen"}


def _recompute_macro(rows: list[dict[str, Any]]) -> tuple[dict[str, float], float, float]:
    per_label = {
        label: binary_macro_f1([int(item["gold"][label]) for item in rows], [int(item["predictions"][label]) for item in rows])
        for label in PRAGMATIC_LABELS
    }
    macro = float(np.mean(list(per_label.values())))
    sarcasm = binary_macro_f1(
        [int(item["gold"]["sarcasm"]) for item in rows],
        [int(item["predictions"]["sarcasm"]) for item in rows],
    )
    return per_label, macro, sarcasm


def _validate_run(
    run_root: Path,
    manifest: dict[str, Any],
    bundle: Any,
    *,
    active_auxiliaries: set[str],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    prediction_checks: dict[str, Any] = {}
    persisted_metrics: dict[str, Any] = {}
    for split, rows in (("train", bundle.train), ("dev", bundle.dev), ("test", bundle.test)):
        # The public runner schema is ``<split>_predictions.jsonl``.  Keep
        # validation on that same schema so a completed run is not rejected
        # after test evaluation merely because of a validator filename typo.
        path = run_root / "predictions" / f"{split}_predictions.jsonl"
        prediction_checks[split] = _validate_predictions(path, rows, active_auxiliaries=active_auxiliaries)
        values = _read_jsonl(path)
        metrics_path = run_root / "metrics" / f"{split}_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        persisted_metrics[split] = str(metrics_path.relative_to(ROOT))
        per_label, macro, sarcasm = _recompute_macro(values)
        if not math.isclose(float(metrics["macro_pragmatic_f1"]), macro, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Persisted macro F1 mismatch for {split}")
        if not math.isclose(float(metrics["sarcasm_binary_macro_f1"]), sarcasm, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Persisted sarcasm macro F1 mismatch for {split}")
        for label, value in per_label.items():
            if not math.isclose(float(metrics["per_label_f1"][label]), value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"Persisted {label} F1 mismatch for {split}")
    checkpoint = run_root / "checkpoints/best/model.pt"
    checkpoint_manifest = json.loads((run_root / "checkpoints/checkpoint_manifest.json").read_text(encoding="utf-8"))
    checks["checkpoint_sha256"] = sha256_file(checkpoint) == checkpoint_manifest.get("checkpoint_sha256") == manifest.get("checkpoint_sha256")
    if not checks["checkpoint_sha256"]:
        raise ValueError("Best checkpoint hash mismatch")
    checks["test_evaluated_during_training"] = manifest.get("test_evaluated_during_training") is False
    checks["test_evaluated_after_dev_freeze"] = manifest.get("test_evaluated_after_checkpoint_and_threshold_freeze") is True
    checks["threshold_source"] = manifest.get("threshold_source_split") == "dev"
    checks["prediction_files"] = prediction_checks
    checks["metrics_files"] = persisted_metrics
    checks["status"] = "PASS"
    return checks


def _append_hf_queue(run_id: str) -> None:
    HF_QUEUE.parent.mkdir(parents=True, exist_ok=True)
    with HF_QUEUE.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        existing: set[str] = set()
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("run_id"):
                existing.add(str(item["run_id"]))
        if run_id not in existing:
            handle.seek(0, os.SEEK_END)
            handle.write(json.dumps({
                "schema_version": 1,
                "run_id": run_id,
                "source_root": f"results/runs/{run_id}",
                "backbone": FAMILY,
                "status": "queued",
                "campaign_id": CAMPAIGN_ID,
                "queued_at": time.time(),
            }, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_id(experiment: str, variant: str | None, budget: str | None, seed: int) -> str:
    if experiment == "priority1":
        return f"priority12_v37_q1_{variant}_{seed}"
    return f"priority12_q3_corrected_{budget}_{seed}"


def _roots(experiment: str, variant: str | None, budget: str | None, seed: int) -> tuple[str, Path, Path]:
    run_id = _run_id(experiment, variant, budget, seed)
    run_root = RESULT_ROOT / run_id
    name = variant if experiment == "priority1" else f"budget_{budget}"
    report_root = REPORT_ROOT / experiment / str(name) / f"seed_{seed}"
    return run_id, run_root, report_root


def _copy_report_bundle(run_root: Path, report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    for relative in (
        "manifest.json",
        "config_snapshot.yaml",
        "validation.json",
        "metrics/summary.json",
        "metrics/train_metrics.json",
        "metrics/dev_metrics.json",
        "metrics/test_metrics.json",
        "selection/thresholds.json",
        "checkpoints/checkpoint_manifest.json",
    ):
        source = run_root / relative
        if source.exists():
            target = report_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def run_one(
    *,
    experiment: str,
    variant: str | None,
    budget: str | None,
    seed: int,
    device: int,
    force: bool,
    all_multipliers_one: bool = False,
    rationale_beta_override: float | None = None,
) -> dict[str, Any]:
    protocol = _protocol()
    if seed not in SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    if experiment == "priority1":
        if variant not in Q1_VARIANTS:
            raise ValueError(f"unsupported Priority 1 variant: {variant}")
    elif experiment == "priority2":
        if budget not in Q3_BUDGETS:
            raise ValueError(f"unsupported Priority 2 budget: {budget}")
    else:
        raise ValueError(f"unsupported experiment: {experiment}")

    if experiment == "priority1" and variant == "all_multipliers_1":
        all_multipliers_one = True
    if experiment == "priority1" and variant == "beta_01":
        rationale_beta_override = 0.1
    if experiment == "priority1" and variant == "beta_05":
        rationale_beta_override = 0.5

    run_id, run_root, report_root = _roots(experiment, variant, budget, seed)
    if run_root.exists() or report_root.exists():
        if not force:
            raise FileExistsError(f"Priority 1/2 output exists; use --force for an explicit rerun: {run_root}")
        if run_root.exists():
            shutil.rmtree(run_root)
        if report_root.exists():
            shutil.rmtree(report_root)
    run_root.mkdir(parents=True, exist_ok=True)

    bundle = load_vipragsent(ROOT / protocol["data"]["dataset_root"])
    split_sizes = {key: len(value) for key, value in bundle.splits.items()}
    expected_sizes = {str(key): int(value) for key, value in protocol["data"]["expected_split_counts"].items()}
    if split_sizes != expected_sizes:
        raise RuntimeError(f"Frozen dataset split sizes differ from protocol: {split_sizes}")
    train_by_id = {example.sample_id: example for example in bundle.train}
    q3_masks: dict[str, dict[str, dict[str, str]]] | None = None
    q3_report: dict[str, Any] | None = None
    mask_hash: str | None = None
    if experiment == "priority2":
        q3_masks, q3_report = load_validated_q3_masks(
            ROOT / protocol["data"]["q3_mask_root"],
            train_by_id,
            strict_frozen=True,
            rationale_mask_policy="active",
        )
        mask_hash = q3_report["mask_hashes"][str(budget)]

    model_variant = Q1_VARIANTS[variant] if experiment == "priority1" else "vipragsent_full_xlmr_large"
    rationale_enabled = model_variant != "no_rationale"
    rationale_path = ROOT / "data/processed/rationales/approved_generated_rationales_train.jsonl"
    rationale_records = _load_rationales(rationale_path) if rationale_enabled else None
    primary_metric = "dev_macro_pragmatic_f1" if experiment == "priority1" else "dev_sarcasm_binary_macro_f1"
    use_uncertainty = model_variant != "no_uncertainty_weighting"
    config = _training_config(
        protocol,
        primary_metric=primary_metric,
        use_uncertainty=use_uncertainty,
        rationale_beta_override=rationale_beta_override,
    )
    snapshot = _snapshot()
    tokenizer = create_tokenizer(FAMILY, revision=MODEL_REVISION, local_path=snapshot, execution_mode="production")
    preprocessor = TextPreprocessor(
        PreprocessingSpec(
            FAMILY,
            "unicode_nfc",
            "locked-v1",
            max_length=int(protocol["model"]["max_sequence_length"]),
            tokenizer_revision=MODEL_REVISION,
            model_revision=MODEL_REVISION,
            execution_mode="production",
        )
    )
    git_commit = _git_commit()
    weights = compute_train_only_class_weights(bundle.train, dataset_hash=bundle.fingerprint, code_commit=git_commit)
    train_batches = _batches(
        bundle.train,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=config.physical_batch_size,
        rationale_records=rationale_records,
        rationale_target_max_length=config.rationale_target_max_length,
        q3_masks=q3_masks,
        budget=str(budget) if experiment == "priority2" else None,
        mask_hash=mask_hash,
    )
    dev_batches = _batches(
        bundle.dev,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=config.physical_batch_size,
        rationale_records=rationale_records,
        rationale_target_max_length=config.rationale_target_max_length,
    )

    seed_everything(seed)
    model, spec = build_production_model(
        FAMILY,
        model_variant,
        local_snapshot=snapshot,
        execution_mode="production",
        selected_device=device,
    )
    loss_multipliers = _loss_multipliers(protocol, all_multipliers_one=all_multipliers_one)
    resolved = {
        "experiment_name": protocol["experiment_name"],
        "experiment": experiment,
        "run_id": run_id,
        "seed": seed,
        "variant_key": variant,
        "budget": budget,
        "model_family": FAMILY,
        "variant_id": model_variant,
        "model_repository": spec.repo_id,
        "model_revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "max_sequence_length": int(protocol["model"]["max_sequence_length"]),
        "preprocessing": {"name": preprocessor.spec.preprocessing_name, "version": preprocessor.spec.preprocessing_version, "normalization": "NFC"},
        "active_tasks": sorted(getattr(model.config, "active_tasks", set())),
        "rationale_decoder": bool(getattr(model.config, "has_rationale_decoder", False)),
        "uncertainty_weighting": bool(getattr(model.config, "has_uncertainty_weighting", False)),
        "training_config": asdict(config),
        "loss_multipliers": loss_multipliers,
        "checkpoint_selection_rule": "best_checkpoint_selected_on_development_macro_pragmatic_f1",
        "uncertainty_log_variances": "learned" if use_uncertainty else "disabled",
        "sensitivity": {
            "all_task_multipliers_one": bool(all_multipliers_one),
            "rationale_beta_override": rationale_beta_override,
        },
        "gradient_strategy": protocol["training"]["gradient_strategy"],
        "batch_order": protocol["training"]["batch_order"],
        "dataset": {"root": protocol["data"]["dataset_root"], "fingerprint": bundle.fingerprint, "split_sizes": split_sizes, "sample_id_order": "frozen CSV order"},
        "q3": q3_report,
        "q3_mask_hash": mask_hash,
        "code_commit": git_commit,
        "local_snapshot": str(snapshot.relative_to(ROOT)),
    }
    engine = TrainingEngine(
        model,
        config,
        run_id="model",
        checkpoint_root=run_root / "_engine_checkpoints",
        class_weights=weights,
        resolved_config=resolved,
        selected_device=device,
        loss_multipliers=loss_multipliers,
        gradient_strategy=str(protocol["training"]["gradient_strategy"]),
        batch_order=str(protocol["training"]["batch_order"]),
    )
    initial_dev = engine._evaluate_dev(dev_batches)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "run_id": run_id,
        "experiment": experiment,
        "variant_key": variant,
        "variant_id": model_variant,
        "budget": budget,
        "seed": seed,
        "model_family": FAMILY,
        "backbone_architecture": "XLM-R-large encoder",
        "base_model_repository": spec.repo_id,
        "base_model_revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "local_snapshot": str(snapshot.relative_to(ROOT)),
        "dataset_fingerprint": bundle.fingerprint,
        "dataset_split_sizes": split_sizes,
        "rationale_artifact": str(rationale_path.relative_to(ROOT)) if rationale_enabled else None,
        "rationale_artifact_sha256": sha256_file(rationale_path) if rationale_enabled else None,
        "rationale_record_count": len(rationale_records or {}),
        "preprocessing_name": preprocessor.spec.preprocessing_name,
        "preprocessing_version": preprocessor.spec.preprocessing_version,
        "code_commit": git_commit,
        "training_config": asdict(config),
        "resolved_training_config": resolved,
        "model_parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "trainable_parameter_count": int(engine.optimizer_summary.get("trainable", 0)),
        "class_weights": weights.as_dict(),
        "loss_multipliers": loss_multipliers,
        "checkpoint_selection_rule": "best_checkpoint_selected_on_development_macro_pragmatic_f1",
        "uncertainty_log_variances_learned": bool(use_uncertainty),
        "sensitivity": {
            "all_task_multipliers_one": bool(all_multipliers_one),
            "rationale_beta_override": rationale_beta_override,
        },
        "selection_metric": primary_metric,
        "selection_split": "dev",
        "threshold_source_split": "dev",
        "thresholds_frozen_before_test": True,
        "test_evaluated_during_training": False,
        "test_evaluated_after_checkpoint_and_threshold_freeze": False,
        "hardware_before": _resource_snapshot(device),
        "batch_probe_reference": read_family_status(ROOT, FAMILY, "batch"),
        "initial_dev": _metrics(initial_dev),
        "scientific_integrity": protocol["scientific_integrity"],
    }
    atomic_write_json(run_root / "manifest.json", manifest)
    atomic_write_json(run_root / "training/class_weights.json", weights.as_dict())
    atomic_write_json(run_root / "training/resolved_training_config.json", resolved)
    atomic_write_json(run_root / "selection/initial_dev.json", _metrics(initial_dev))
    atomic_write_text(run_root / "config_snapshot.yaml", yaml.safe_dump({"source": str(PROTOCOL_PATH.relative_to(ROOT)), "resolved": resolved, "protocol": protocol}, sort_keys=False, allow_unicode=False))

    started = time.time()
    state = engine.train(
        train_batches,
        seed=seed,
        dev_batches=dev_batches,
        test_batches=None,
        resume=False,
        output_root=run_root / "_engine_output",
        run_metadata={"experiment": experiment, "run_id": run_id, "test_evaluated_during_training": False},
    )
    final_dev = engine._evaluate_dev(dev_batches, thresholds_override=state.thresholds)
    final_train = engine._evaluate_dev(train_batches, thresholds_override=state.thresholds)
    _write_predictions(run_root / "predictions/train_predictions.jsonl", engine, final_train, train_batches)
    _write_predictions(run_root / "predictions/dev_predictions.jsonl", engine, final_dev, dev_batches)
    atomic_write_json(run_root / "metrics/train_metrics.json", _metrics(final_train))
    atomic_write_json(run_root / "metrics/dev_metrics.json", _metrics(final_dev))
    thresholds = {"source_split": "dev", "selection_metric": primary_metric, "thresholds": dict(state.thresholds), "best_epoch": state.best_epoch}
    atomic_write_json(run_root / "selection/thresholds.json", thresholds)

    # Test batches are deliberately not created until after checkpoint/threshold
    # freezing. The gate is checked before any test forward pass.
    engine.assert_test_access()
    test_batches = _batches(
        bundle.test,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=config.physical_batch_size,
        rationale_records=rationale_records,
        rationale_target_max_length=config.rationale_target_max_length,
    )
    final_test = engine._evaluate_dev(test_batches, thresholds_override=state.thresholds)
    _write_predictions(run_root / "predictions/test_predictions.jsonl", engine, final_test, test_batches)
    atomic_write_json(run_root / "metrics/test_metrics.json", _metrics(final_test))
    atomic_write_json(run_root / "metrics/summary.json", {"train": _metrics(final_train), "dev": _metrics(final_dev), "test": _metrics(final_test)})

    best_engine_checkpoint = run_root / "_engine_checkpoints/model/best.pt"
    if not best_engine_checkpoint.exists():
        raise RuntimeError(f"TrainingEngine did not produce the selected checkpoint: {best_engine_checkpoint}")
    best_checkpoint = run_root / "checkpoints/best/model.pt"
    _link_or_copy(best_engine_checkpoint, best_checkpoint)
    checkpoint_hash = sha256_file(best_checkpoint)
    atomic_write_json(run_root / "checkpoints/checkpoint_manifest.json", {
        "status": "PASS",
        "best": "checkpoints/best/model.pt",
        "checkpoint_sha256": checkpoint_hash,
        "best_epoch": state.best_epoch,
        "selection_split": "dev",
        "threshold_source_split": "dev",
        "test_used_for_selection": False,
        "model_repository": spec.repo_id,
        "model_revision": spec.revision,
    })
    atomic_write_json(run_root / "selection/selection_metric.json", {"name": primary_metric, "value": state.best_metric, "best_epoch": state.best_epoch})
    atomic_write_json(run_root / "training/history.json", state.history)
    if (run_root / "training/device_report.json").exists() is False:
        raise RuntimeError("TrainingEngine did not write the first-step device report")
    peak_memory = max((float(row.get("peak_memory_gb", 0.0)) for row in state.history), default=0.0)
    wall_seconds = time.time() - started
    atomic_write_json(run_root / "training/resource_usage.json", {
        "fixture": False,
        "wall_seconds": wall_seconds,
        "successful_gpu_hours": wall_seconds / 3600.0,
        "failed_or_retried_gpu_hours": 0.0,
        "peak_vram_gb": peak_memory,
        "measurement_source": "TrainingEngine epoch timing and CUDA peak memory",
        "historical_memory_budget_gb": 20,
    })
    manifest.update({
        "status": state.status,
        "best_epoch": state.best_epoch,
        "best_dev_metric": state.best_metric,
        "selected_thresholds": dict(state.thresholds),
        "history": state.history,
        "final_train": _metrics(final_train),
        "final_dev": _metrics(final_dev),
        "final_test": _metrics(final_test),
        "checkpoint_path": "checkpoints/best/model.pt",
        "checkpoint_sha256": checkpoint_hash,
        "test_evaluated_after_checkpoint_and_threshold_freeze": True,
        "hardware_after": _resource_snapshot(device),
        "wall_seconds": wall_seconds,
        "peak_vram_gb": peak_memory,
        "completed_at_unix": time.time(),
    })
    atomic_write_json(run_root / "manifest.json", manifest)
    validation = _validate_run(run_root, manifest, bundle, active_auxiliaries=set(getattr(model.config, "active_tasks", set())) - {"pragmatic"})
    atomic_write_json(run_root / "validation.json", validation)
    manifest["validation"] = validation
    internal_cleanup: list[str] = []
    for relative in ("_engine_checkpoints", "_engine_output"):
        internal_path = run_root / relative
        if internal_path.exists():
            shutil.rmtree(internal_path)
            internal_cleanup.append(relative)
    manifest["internal_training_copies_removed_after_validation"] = internal_cleanup
    atomic_write_json(run_root / "manifest.json", manifest)
    _copy_report_bundle(run_root, report_root)
    _append_hf_queue(run_id)
    return {"status": "PASS", "run_id": run_id, "run_root": str(run_root), "report_root": str(report_root), "validation": validation, "test": _metrics(final_test), "best_epoch": state.best_epoch, "peak_vram_gb": peak_memory}


def _tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.run_all:
        if args.seed is None:
            raise ValueError("--seed is required unless --run-all is used")
        return [{"experiment": args.experiment, "variant": args.variant, "budget": args.budget, "seed": args.seed}]
    tasks: list[dict[str, Any]] = []
    for variant in Q1_RUN_ALL_VARIANTS:
        for seed in SEEDS:
            tasks.append({"experiment": "priority1", "variant": variant, "budget": None, "seed": seed})
    for budget in Q3_BUDGETS:
        for seed in SEEDS:
            tasks.append({"experiment": "priority2", "variant": None, "budget": budget, "seed": seed})
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run isolated Priority 1 and corrected Priority 2 XLM-R experiments")
    parser.add_argument("--experiment", choices=("priority1", "priority2"), default="priority1")
    parser.add_argument("--variant", choices=tuple(Q1_VARIANTS), default=None)
    parser.add_argument("--budget", choices=Q3_BUDGETS, default=None)
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--all-multipliers-one", action="store_true", help="Set all eight task loss multipliers to 1.0")
    parser.add_argument("--rationale-beta", type=float, default=None, help="Override rationale beta for a sensitivity run")
    args = parser.parse_args()
    results: list[dict[str, Any]] = []
    for task in _tasks(args):
        results.append(
            run_one(
                device=args.device,
                force=args.force,
                all_multipliers_one=args.all_multipliers_one,
                rationale_beta_override=args.rationale_beta,
                **task,
            )
        )
        print(json.dumps(results[-1], indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
