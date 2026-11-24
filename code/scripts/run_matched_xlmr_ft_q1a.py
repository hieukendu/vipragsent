"""Run one controlled matched XLM-R-large Q1a baseline seed.

The model variant deliberately reuses the existing six-head
``xlmr_pragmatic_finetune`` semantics.  The dedicated recipe is loaded from
``configs/experiments/q1a_matched_xlmr_ft.yaml`` and the test split is only
evaluated after the training engine freezes the development checkpoint and
thresholds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.constants import EXPECTED_SPLIT_COUNTS, PRAGMATIC_LABELS
from vipragsent.data.collation import BatchCollator
from vipragsent.data.loaders import DatasetExample, load_vipragsent
from vipragsent.data.preprocessing import PreprocessingSpec, TextPreprocessor
from vipragsent.data.tokenizers import create_tokenizer
from vipragsent.evaluation.metrics import binary_macro_f1, macro_pragmatic_f1
from vipragsent.hashing import sha256_file
from vipragsent.models.factory import build_production_model
from vipragsent.runtime.model_assets import read_family_status, resolve_local_snapshot
from vipragsent.training.class_weights import compute_train_only_class_weights
from vipragsent.training.engine import SelectionResult, TrainingConfig, TrainingEngine
from vipragsent.training.seeding import seed_everything

CONFIG_PATH = ROOT / "configs/experiments/q1a_matched_xlmr_ft.yaml"
REPORT_ROOT = ROOT / "reports/q1a_matched_xlmr_ft"
RESULT_ROOT = ROOT / "results/runs"
FAMILY = "xlmr_large"
VARIANT = "xlmr_pragmatic_finetune"
SEEDS = (20260521, 20260522, 20260523)


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


def _load_recipe() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    if payload.get("experiment_name") != "xlmr_large_ft_q1a_matched":
        raise RuntimeError("matched experiment config has an unexpected experiment name")
    return payload


def _snapshot(recipe: dict[str, Any]) -> Path:
    family = str(recipe["model"]["family"])
    revision = str(recipe["model"]["revision"])
    status = read_family_status(ROOT, family, "cache")
    if status.get("status") != "PASS":
        raise RuntimeError(f"pinned {family} cache is not PASS: {status.get('status')}")
    candidates = [resolve_local_snapshot(ROOT, status.get("local_path")), ROOT / "models/xlm-roberta-large"]
    required = ("config.json", "model.safetensors", "sentencepiece.bpe.model", "tokenizer.json")
    for snapshot in candidates:
        if snapshot is None or not snapshot.exists() or any(not (snapshot / name).exists() for name in required):
            continue
        metadata = snapshot / ".cache/huggingface/download/model.safetensors.metadata"
        if metadata.exists() and metadata.read_text(encoding="utf-8").splitlines()[0].strip() != revision:
            continue
        return snapshot
    raise RuntimeError(f"pinned local {family} snapshot is unavailable or has the wrong revision")


def _training_config(recipe: dict[str, Any]) -> TrainingConfig:
    training = recipe["training"]
    optimizer = recipe["optimizer"]
    scheduler = recipe["scheduler"]
    return TrainingConfig(
        learning_rate=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
        max_epochs=int(training["maximum_epochs"]),
        effective_batch_size=int(training["effective_batch_size"]),
        physical_batch_size=int(training["physical_batch_size"]),
        max_grad_norm=float(training["gradient_clipping"]),
        patience=int(training["patience"]),
        min_delta=float(training["minimum_delta"]),
        precision=str(training["precision"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        primary_metric=str(recipe["selection"]["metric"]),
        scheduler=str(scheduler["name"]),
        warmup_ratio=float(scheduler["warmup_ratio"]),
        use_uncertainty_weighting=False,
        rationale_beta=0.3,
        rationale_target_max_length=160,
        optimizer=str(optimizer["name"]),
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        deterministic_algorithms=str(training["deterministic_algorithms"]),
        qlora={"quantization": {"type": "none"}, "lora": {}},
    )


def _batches(
    rows: list[DatasetExample],
    *,
    tokenizer: Any,
    preprocessor: TextPreprocessor,
    weights: Any,
    physical_batch_size: int,
) -> list[dict[str, Any]]:
    collator = BatchCollator(
        tokenizer,
        preprocessor,
        class_weights=weights.as_dict() if hasattr(weights, "as_dict") else dict(weights),
    )
    return [
        collator(rows[index : index + physical_batch_size])
        for index in range(0, len(rows), physical_batch_size)
    ]


def _metrics(selection: SelectionResult) -> dict[str, Any]:
    true = {label: selection.true[label] for label in PRAGMATIC_LABELS}
    pred = {label: selection.predictions[label] for label in PRAGMATIC_LABELS}
    return {
        "prediction_count": len(selection.true[PRAGMATIC_LABELS[0]]),
        "per_label_f1": {label: binary_macro_f1(true[label], pred[label]) for label in PRAGMATIC_LABELS},
        "macro_pragmatic_f1": macro_pragmatic_f1(true, pred),
        "thresholds": dict(selection.thresholds),
        "selection_metric": float(selection.metric),
        "loss": float(selection.total_loss),
    }


def _write_predictions(
    path: Path,
    engine: TrainingEngine,
    selection: SelectionResult,
    batches: list[dict[str, Any]],
) -> None:
    sample_ids = [sample_id for batch in batches for sample_id in batch.get("sample_ids", [])]
    engine._write_prediction_jsonl(path, engine._prediction_rows(selection, sample_ids=sample_ids))


def _write_json_pair(run_root: Path, report_root: Path, relative: str, payload: Any) -> None:
    atomic_write_json(run_root / relative, payload)
    atomic_write_json(report_root / Path(relative).name, payload)


def _write_text_pair(run_root: Path, report_root: Path, relative: str, payload: str) -> None:
    atomic_write_text(run_root / relative, payload)
    atomic_write_text(report_root / Path(relative).name, payload)


def _copy_prediction_pair(
    run_root: Path,
    report_root: Path,
    split: str,
    engine: TrainingEngine,
    selection: SelectionResult,
    batches: list[dict[str, Any]],
) -> Path:
    run_path = run_root / "predictions" / f"predictions_{split}.jsonl"
    report_path = report_root / f"predictions_{split}.jsonl"
    _write_predictions(run_path, engine, selection, batches)
    shutil.copy2(run_path, report_path)
    return run_path


def run(seed: int, *, device: int, force: bool) -> int:
    if seed not in SEEDS:
        raise ValueError(f"unsupported training seed: {seed}")
    recipe = _load_recipe()
    expected = {str(key): int(value) for key, value in recipe["data"]["expected_split_counts"].items()}
    if expected != {key: int(value) for key, value in EXPECTED_SPLIT_COUNTS.items()}:
        raise RuntimeError(f"matched config split counts differ from the frozen contract: {expected}")

    run_id = f"xlmr_large_ft_q1a_matched_{seed}"
    run_root = RESULT_ROOT / run_id
    report_root = REPORT_ROOT / f"seed_{seed}"
    if run_root.exists() or report_root.exists():
        if not force:
            raise FileExistsError(f"matched seed output already exists; use --force: {run_root}")
        if run_root.exists():
            shutil.rmtree(run_root)
        if report_root.exists():
            shutil.rmtree(report_root)
    run_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    model_spec = recipe["model"]
    variant_spec = recipe["variant"]
    recipe_training = _training_config(recipe)
    snapshot = _snapshot(recipe)
    bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
    split_sizes = {key: len(value) for key, value in bundle.splits.items()}
    if split_sizes != expected:
        raise RuntimeError(f"frozen data split sizes differ from the contract: {split_sizes}")
    git_commit = _git_commit()
    weights = compute_train_only_class_weights(
        bundle.train,
        dataset_hash=bundle.fingerprint,
        code_commit=git_commit,
    )
    preprocessor = TextPreprocessor(
        PreprocessingSpec(
            FAMILY,
            "unicode_nfc",
            "locked-v1",
            max_length=int(model_spec["max_sequence_length"]),
            tokenizer_revision=str(model_spec["tokenizer_revision"]),
            model_revision=str(model_spec["revision"]),
            execution_mode="production",
        )
    )
    tokenizer = create_tokenizer(
        FAMILY,
        revision=str(model_spec["tokenizer_revision"]),
        local_path=snapshot,
        execution_mode="production",
    )
    train_batches = _batches(
        bundle.train,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=recipe_training.physical_batch_size,
    )
    dev_batches = _batches(
        bundle.dev,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=recipe_training.physical_batch_size,
    )
    test_batches = _batches(
        bundle.test,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=recipe_training.physical_batch_size,
    )

    resolved: dict[str, Any] = {
        "experiment_name": recipe["experiment_name"],
        "run_id": run_id,
        "seed": seed,
        "model_family": FAMILY,
        "variant_id": VARIANT,
        "model_repository": model_spec["repo_id"],
        "model_revision": model_spec["revision"],
        "tokenizer_revision": model_spec["tokenizer_revision"],
        "architecture": model_spec["architecture"],
        "max_sequence_length": int(model_spec["max_sequence_length"]),
        "preprocessing": {
            "name": preprocessor.spec.preprocessing_name,
            "version": preprocessor.spec.preprocessing_version,
            "normalization": "NFC",
        },
        "active_heads": list(PRAGMATIC_LABELS),
        "pooling": variant_spec["pooling"],
        "dropout": float(variant_spec["dropout"]),
        "auxiliary_heads": False,
        "rationale_decoder": False,
        "uncertainty_weighting": False,
        "loss": {
            "name": recipe["loss"]["name"],
            "aggregation": recipe["loss"]["aggregation"],
            "positive_weight_policy": recipe["loss"]["positive_weight_policy"],
            "active_task": "pragmatic",
        },
        "training_config": asdict(recipe_training),
        "selection": dict(recipe["selection"]),
        "dataset": {
            "root": "data/processed/vipragsent",
            "fingerprint": bundle.fingerprint,
            "split_sizes": split_sizes,
            "sample_id_order": "frozen CSV order",
        },
        "class_weights": weights.as_dict(),
        "code_commit": git_commit,
        "local_snapshot": str(snapshot.relative_to(ROOT)),
        "test_evaluated_during_training": False,
    }
    _write_json_pair(run_root, report_root, "resolved_config.json", resolved)
    atomic_write_text(
        run_root / "config_snapshot.yaml",
        yaml.safe_dump({"source": str(CONFIG_PATH.relative_to(ROOT)), "resolved": resolved}, sort_keys=False),
    )
    shutil.copy2(run_root / "config_snapshot.yaml", report_root / "config_snapshot.yaml")
    _write_json_pair(run_root, report_root, "training/class_weights.json", weights.as_dict())

    seed_everything(seed)
    model, spec = build_production_model(
        FAMILY,
        VARIANT,
        local_snapshot=snapshot,
        execution_mode="production",
        selected_device=device,
    )
    resolved["model_parameter_count"] = sum(int(parameter.numel()) for parameter in model.parameters())
    resolved["trainable_parameter_count"] = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    resolved["resolved_model_spec"] = {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
    }
    _write_json_pair(run_root, report_root, "resolved_config.json", resolved)
    atomic_write_text(
        run_root / "config_snapshot.yaml",
        yaml.safe_dump({"source": str(CONFIG_PATH.relative_to(ROOT)), "resolved": resolved}, sort_keys=False),
    )
    shutil.copy2(run_root / "config_snapshot.yaml", report_root / "config_snapshot.yaml")

    engine = TrainingEngine(
        model,
        recipe_training,
        run_id="model",
        checkpoint_root=run_root / "_engine_checkpoints",
        class_weights=weights,
        resolved_config=resolved,
        selected_device=device,
        executor_kind="single_model_trainable",
    )
    initial_dev = engine._evaluate_dev(dev_batches)
    _write_json_pair(run_root, report_root, "metrics/initial_dev.json", _metrics(initial_dev))

    state = engine.train(
        train_batches,
        seed=seed,
        dev_batches=dev_batches,
        test_batches=None,
        resume=False,
        output_root=run_root / "_engine_output",
        run_metadata={
            "experiment_name": recipe["experiment_name"],
            "run_id": run_id,
            "variant_id": VARIANT,
            "test_evaluated_during_training": False,
        },
    )
    final_dev = engine._evaluate_dev(dev_batches, thresholds_override=state.thresholds)
    final_train = engine._evaluate_dev(train_batches, thresholds_override=state.thresholds)
    engine.assert_test_access()
    final_test = engine._evaluate_dev(test_batches, thresholds_override=state.thresholds)

    metrics_by_split = {"train": _metrics(final_train), "dev": _metrics(final_dev), "test": _metrics(final_test)}
    prediction_paths = {
        "train": _copy_prediction_pair(run_root, report_root, "train", engine, final_train, train_batches),
        "dev": _copy_prediction_pair(run_root, report_root, "dev", engine, final_dev, dev_batches),
        "test": _copy_prediction_pair(run_root, report_root, "test", engine, final_test, test_batches),
    }
    for split, metrics in metrics_by_split.items():
        _write_json_pair(run_root, report_root, f"metrics_{split}.json", metrics)

    thresholds = {
        "seed": seed,
        "selection_metric": recipe["selection"]["metric"],
        "source_split": "dev",
        "threshold_grid": {
            "start": float(recipe["selection"]["threshold_grid_start"]),
            "stop": float(recipe["selection"]["threshold_grid_stop"]),
            "step": float(recipe["selection"]["threshold_grid_step"]),
        },
        "tie_break": recipe["selection"]["threshold_tie_break"],
        "thresholds": dict(state.thresholds),
    }
    _write_json_pair(run_root, report_root, "thresholds.json", thresholds)

    history_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in state.history)
    _write_text_pair(run_root, report_root, "training_log.jsonl", history_text)
    atomic_write_json(run_root / "training/history.json", state.history)
    shutil.copy2(run_root / "training/history.json", report_root / "history.json")
    for name in ("device_report.json", "optimizer_summary.json", "scheduler_summary.json"):
        source = run_root / "training" / name
        if source.exists():
            shutil.copy2(source, report_root / name)

    engine_checkpoint = run_root / "_engine_checkpoints/model/best.pt"
    if not engine_checkpoint.exists():
        raise RuntimeError(f"TrainingEngine did not produce the selected checkpoint: {engine_checkpoint}")
    final_checkpoint = run_root / "checkpoints/best/model.pt"
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(engine_checkpoint, final_checkpoint)
    checkpoint_hash = sha256_file(final_checkpoint)
    best_metadata = {
        "status": "PASS",
        "checkpoint_path": "checkpoints/best/model.pt",
        "checkpoint_sha256": checkpoint_hash,
        "best_epoch": state.best_epoch,
        "best_dev_macro_pragmatic_f1": state.best_metric,
        "selection_split": "dev",
        "threshold_source_split": "dev",
        "test_used_for_selection": False,
        "test_evaluated_after_freeze": True,
    }
    _write_json_pair(run_root, report_root, "best_checkpoint_metadata.json", best_metadata)
    _write_json_pair(
        run_root,
        report_root,
        "selection/selection_metric.json",
        {"name": recipe["selection"]["metric"], "value": state.best_metric, "best_epoch": state.best_epoch},
    )
    _write_json_pair(run_root, report_root, "selection/thresholds.json", thresholds)

    wall_seconds = sum(float(row.get("seconds", 0.0)) for row in state.history)
    peak_memory = max((float(row.get("peak_memory_gb", 0.0)) for row in state.history), default=0.0)
    manifest = {
        "status": state.status,
        "experiment_name": recipe["experiment_name"],
        "run_id": run_id,
        "seed": seed,
        "model_family": FAMILY,
        "variant_id": VARIANT,
        "resolved_config": "resolved_config.json",
        "dataset_split_sizes": split_sizes,
        "prediction_files": {split: str(path.relative_to(run_root)) for split, path in prediction_paths.items()},
        "metrics_files": {split: f"metrics_{split}.json" for split in metrics_by_split},
        "checkpoint": best_metadata,
        "test_evaluated_during_training": False,
        "test_evaluated_after_checkpoint_and_threshold_freeze": True,
        "history": state.history,
        "wall_seconds": wall_seconds,
        "peak_vram_gb": peak_memory,
        "resolved_thresholds": thresholds,
        "metrics": metrics_by_split,
    }
    _write_json_pair(run_root, report_root, "run_manifest.json", manifest)
    atomic_write_json(run_root / "metrics/summary.json", metrics_by_split)
    shutil.copy2(run_root / "metrics/summary.json", report_root / "summary.json")
    # The public run root retains only the selected checkpoint. Engine resume
    # copies include every epoch and are intentionally not part of the
    # reproducibility/upload artifact set.
    shutil.rmtree(run_root / "_engine_checkpoints", ignore_errors=True)
    print(
        json.dumps(
            {
                "status": state.status,
                "run_id": run_id,
                "seed": seed,
                "run_root": str(run_root),
                "report_root": str(report_root),
                "best_epoch": state.best_epoch,
                "best_dev_macro_pragmatic_f1": state.best_metric,
                "test_macro_pragmatic_f1": metrics_by_split["test"]["macro_pragmatic_f1"],
                "checkpoint_sha256": checkpoint_hash,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one matched XLM-R-large Q1a baseline seed")
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return run(args.seed, device=args.device, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
