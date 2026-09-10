"""Train and evaluate one full ViPragSent run with an XLM-R-large backbone.

This is an isolated Q1a follow-up run. It uses the locked encoder training
profile and keeps test access behind the engine's dev-selection gate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from vipragsent.data.collation import BatchCollator
from vipragsent.data.loaders import DatasetExample, load_vipragsent
from vipragsent.data.tokenizers import create_tokenizer
from vipragsent.evaluation.confidence_intervals import evaluate_q1a_confidence_intervals
from vipragsent.evaluation.metrics import binary_macro_f1, macro_pragmatic_f1, multiclass_macro_f1
from vipragsent.hashing import sha256_file
from vipragsent.models.factory import build_production_model
from vipragsent.orchestration.stage_registry import _build_production_preprocessor
from vipragsent.runtime.model_assets import read_family_status, resolve_local_snapshot
from vipragsent.training.class_weights import compute_train_only_class_weights
from vipragsent.training.engine import SelectionResult, TrainingConfig, TrainingEngine
from vipragsent.training.seeding import seed_everything

FAMILY = "xlmr_large"
VARIANT = "vipragsent_full_xlmr_large"
SEEDS = (20260521, 20260522, 20260523)
MODEL_REVISION = "c23d21b0620b635a76227c604d44e43a9f0ee389"
TOKENIZER_REVISION = MODEL_REVISION


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


def _snapshot() -> Path:
    status = read_family_status(ROOT, FAMILY, "cache")
    snapshot = resolve_local_snapshot(ROOT, status.get("local_path"))
    if snapshot is None or not snapshot.exists():
        raise RuntimeError("the pinned local XLM-R-large snapshot is unavailable")
    return snapshot


def _load_rationales(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"approved rationale artifact is unavailable: {path}")
    records: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            records[str(row["sample_id"])] = row
    return records


def _training_config(args: argparse.Namespace) -> tuple[TrainingConfig, dict[str, Any], dict[str, Any]]:
    training_path = ROOT / "configs/runtime/training.yaml"
    payload = yaml.safe_load(training_path.read_text(encoding="utf-8")) or {}
    encoder = dict(payload["encoder"])
    rationale = dict(payload["rationale"])
    batch_status = read_family_status(ROOT, FAMILY, "batch")
    physical = int(batch_status.get("successful_batch"))
    candidates = {int(value) for value in encoder["physical_batch_probe"][FAMILY]}
    if physical not in candidates:
        raise RuntimeError(f"locked XLM-R physical batch {physical} is absent from probe order {sorted(candidates)}")
    effective = int(encoder["effective_batch_size"])
    if effective % physical:
        raise RuntimeError(f"effective batch {effective} is not divisible by physical batch {physical}")
    config = TrainingConfig(
        learning_rate=float(args.learning_rate if args.learning_rate is not None else encoder["learning_rate"]),
        weight_decay=float(args.weight_decay if args.weight_decay is not None else encoder["weight_decay"]),
        max_epochs=int(args.max_epochs if args.max_epochs is not None else encoder["maximum_epochs"]),
        effective_batch_size=effective,
        physical_batch_size=physical,
        max_grad_norm=float(encoder["gradient_clipping"]),
        patience=int(args.patience if args.patience is not None else encoder["patience"]),
        min_delta=float(encoder["minimum_delta"]),
        precision=str(encoder["precision"]),
        gradient_accumulation_steps=effective // physical,
        primary_metric="dev_macro_pragmatic_f1",
        scheduler=str(args.scheduler if args.scheduler is not None else encoder["scheduler"]),
        warmup_ratio=float(args.warmup_ratio if args.warmup_ratio is not None else encoder["warmup_ratio"]),
        use_uncertainty_weighting=True,
        rationale_beta=float(args.rationale_beta if args.rationale_beta is not None else rationale["beta"]),
        rationale_target_max_length=max(int(value) for value in rationale["target_max_lengths"]),
        optimizer=str(encoder["optimizer"]),
        gradient_checkpointing=bool(encoder["gradient_checkpointing"]),
        deterministic_algorithms=str(payload["deterministic_algorithms"]),
        qlora={"quantization": {"type": "none"}, "lora": {}},
    )
    return config, payload, batch_status


def _preprocessing_kwargs() -> dict[str, str]:
    """Return the locked preprocessing identity for this encoder family."""
    return {
        "preprocessing_name": "unicode_nfc",
        "preprocessing_version": "locked-v1",
    }


def _trial_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not normalized:
        raise ValueError("trial name cannot be empty")
    return normalized


def _run_id(seed: int, trial: str) -> str:
    suffix = _trial_name(trial)
    if suffix == "baseline":
        return f"q1a_vipragsent_full_XLM_R_large_{seed}"
    return f"q1a_vipragsent_full_XLM_R_large_optimization_{suffix}_{seed}"


def _weights_payload(weights: Any, args: argparse.Namespace) -> dict[str, Any]:
    payload = dict(weights.as_dict())
    pragmatic = dict(payload.get("pragmatic_pos_weight", {}))
    pragmatic["sarcasm"] = float(pragmatic["sarcasm"]) * float(args.sarcasm_weight_multiplier)
    pragmatic["code_switching"] = float(pragmatic["code_switching"]) * float(args.code_switching_weight_multiplier)
    payload["pragmatic_pos_weight"] = pragmatic
    payload["optimization_adjustments"] = {
        "sarcasm_weight_multiplier": float(args.sarcasm_weight_multiplier),
        "code_switching_weight_multiplier": float(args.code_switching_weight_multiplier),
    }
    return payload


def _loss_multipliers(args: argparse.Namespace) -> dict[str, float]:
    """Build the loss profile while allowing a focused pragmatic-head override."""
    multipliers = {label: float(args.pragmatic_loss_multiplier) for label in PRAGMATIC_LABELS}
    for label in PRAGMATIC_LABELS:
        override = getattr(args, f"{label}_loss_multiplier", None)
        if override is not None:
            multipliers[label] = float(override)
    multipliers.update(
        {
            "polarity": float(args.auxiliary_loss_multiplier),
            "emotion": float(args.auxiliary_loss_multiplier),
        }
    )
    return multipliers


def _batches(
    rows: list[DatasetExample],
    *,
    tokenizer: Any,
    preprocessor: Any,
    weights: Any,
    physical_batch_size: int,
    rationale_records: dict[str, Any],
    rationale_target_max_length: int,
) -> list[dict[str, Any]]:
    collator = BatchCollator(
        tokenizer,
        preprocessor,
        class_weights=weights.as_dict() if hasattr(weights, "as_dict") else dict(weights),
        rationale_records=rationale_records,
        rationale_target_max_length=rationale_target_max_length,
    )
    return [
        collator(rows[index : index + physical_batch_size])
        for index in range(0, len(rows), physical_batch_size)
    ]


def _metrics(selection: SelectionResult) -> dict[str, Any]:
    per_label = {
        label: binary_macro_f1(selection.true[label], selection.predictions[label])
        for label in PRAGMATIC_LABELS
    }
    result: dict[str, Any] = {
        "prediction_count": len(selection.true[PRAGMATIC_LABELS[0]]),
        "per_label_f1": per_label,
        "macro_pragmatic_f1": macro_pragmatic_f1(
            {label: selection.true[label] for label in PRAGMATIC_LABELS},
            {label: selection.predictions[label] for label in PRAGMATIC_LABELS},
        ),
        "thresholds": dict(selection.thresholds),
        "selection_metric": selection.metric,
        "loss": selection.total_loss,
    }
    if "polarity" in selection.true and "polarity" in selection.predictions:
        result["polarity_macro_f1"] = multiclass_macro_f1(
            selection.true["polarity"], selection.predictions["polarity"], range(len(POLARITY_LABELS))
        )
        result["polarity_accuracy"] = sum(
            int(gold == pred)
            for gold, pred in zip(selection.true["polarity"], selection.predictions["polarity"])
        ) / len(selection.true["polarity"])
    if "emotion" in selection.true and "emotion" in selection.predictions:
        result["emotion_macro_f1"] = multiclass_macro_f1(
            selection.true["emotion"], selection.predictions["emotion"], range(len(EMOTION_LABELS))
        )
        result["emotion_accuracy"] = sum(
            int(gold == pred)
            for gold, pred in zip(selection.true["emotion"], selection.predictions["emotion"])
        ) / len(selection.true["emotion"])
    return result


def _write_predictions(path: Path, engine: TrainingEngine, selection: SelectionResult, batches: list[dict[str, Any]]) -> None:
    sample_ids = [sample_id for batch in batches for sample_id in batch.get("sample_ids", [])]
    engine._write_prediction_jsonl(path, engine._prediction_rows(selection, sample_ids=sample_ids))


def _copy_checkpoint(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _link_or_copy_checkpoint(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def run(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    if seed not in SEEDS:
        raise ValueError(f"unsupported training seed: {seed}")
    trial = _trial_name(args.trial)
    experiment_id = _run_id(seed, trial)
    run_root = ROOT / "results/runs" / experiment_id
    if run_root.exists() and not args.force:
        raise FileExistsError(f"candidate output already exists; use --force for an explicit rerun: {run_root}")
    if args.force and run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    config, training_payload, batch_status = _training_config(args)
    snapshot = _snapshot()
    bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
    rationale_path = ROOT / "data/processed/rationales/approved_generated_rationales_train.jsonl"
    rationale_records = _load_rationales(rationale_path)
    weights = compute_train_only_class_weights(
        bundle.train,
        dataset_hash=bundle.fingerprint,
        code_commit=_git_commit(),
    )
    weight_payload = _weights_payload(weights, args)
    loss_multipliers = _loss_multipliers(args)
    preprocessor_kwargs = _preprocessing_kwargs()
    preprocessor = _build_production_preprocessor(
        FAMILY,
        tokenizer_revision=TOKENIZER_REVISION,
        model_revision=MODEL_REVISION,
        **preprocessor_kwargs,
    )
    tokenizer = create_tokenizer(
        FAMILY,
        revision=TOKENIZER_REVISION,
        local_path=snapshot,
        execution_mode="production",
    )
    train_batches = _batches(
        bundle.train,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weight_payload,
        physical_batch_size=config.physical_batch_size,
        rationale_records=rationale_records,
        rationale_target_max_length=config.rationale_target_max_length,
    )
    dev_batches = _batches(
        bundle.dev,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weight_payload,
        physical_batch_size=config.physical_batch_size,
        rationale_records=rationale_records,
        rationale_target_max_length=config.rationale_target_max_length,
    )
    test_batches = _batches(
        bundle.test,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weight_payload,
        physical_batch_size=config.physical_batch_size,
        rationale_records=rationale_records,
        rationale_target_max_length=config.rationale_target_max_length,
    )

    seed_everything(seed)
    model, spec = build_production_model(
        FAMILY,
        VARIANT,
        local_snapshot=snapshot,
        execution_mode="production",
        selected_device=int(args.device),
    )
    resolved = {
        "system_id": experiment_id,
        "model_family": FAMILY,
        "variant_id": VARIANT,
        "model_revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "scheduler": config.scheduler,
        "warmup_ratio": config.warmup_ratio,
        "physical_batch_size": config.physical_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size": config.effective_batch_size,
        "maximum_epochs": config.max_epochs,
        "precision": config.precision,
        "gradient_clipping": config.max_grad_norm,
        "patience": config.patience,
        "minimum_delta": config.min_delta,
        "gradient_checkpointing": config.gradient_checkpointing,
        "deterministic_algorithms": config.deterministic_algorithms,
        "uncertainty_weighting_enabled": config.use_uncertainty_weighting,
        "active_uncertainty_tasks": list(PRAGMATIC_LABELS) + ["polarity", "emotion"],
        "rationale_training": True,
        "rationale_beta": config.rationale_beta,
        "rationale_target_max_length": config.rationale_target_max_length,
        "rationale_inference": False,
        "qlora": config.qlora,
        "selection_metric": config.primary_metric,
        "trial": trial,
        "weight_adjustments": weight_payload["optimization_adjustments"],
        "loss_multipliers": loss_multipliers,
        "gradient_strategy": args.gradient_strategy,
        "batch_order": args.batch_order,
    }
    engine = TrainingEngine(
        model,
        config,
        run_id="model",
        checkpoint_root=run_root / "_engine_checkpoints",
        class_weights=weight_payload,
        resolved_config=resolved,
        selected_device=int(args.device),
        loss_multipliers=loss_multipliers,
        gradient_strategy=args.gradient_strategy,
        batch_order=args.batch_order,
    )
    initial_dev = engine._evaluate_dev(dev_batches)
    model_parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "RUNNING",
        "run_id": experiment_id,
        "experiment_id": experiment_id,
        "seed": seed,
        "trial": trial,
        "model_family": FAMILY,
        "variant_id": VARIANT,
        "backbone_architecture": "XLM-R-large encoder",
        "initialization": "pinned_pretrained_base",
        "base_model_repository": spec.repo_id,
        "base_model_revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "local_snapshot": str(snapshot.relative_to(ROOT)),
        "dataset_fingerprint": bundle.fingerprint,
        "dataset_split_sizes": {key: len(value) for key, value in bundle.splits.items()},
        "rationale_artifact": str(rationale_path.relative_to(ROOT)),
        "rationale_artifact_sha256": sha256_file(rationale_path),
        "rationale_record_count": len(rationale_records),
        "preprocessing_name": preprocessor.spec.preprocessing_name,
        "preprocessing_version": preprocessor.spec.preprocessing_version,
        "code_commit": _git_commit(),
        "training_config": asdict(config),
        "resolved_training_config": resolved,
        "batch_probe": batch_status,
        "model_parameter_count": model_parameter_count,
        "trainable_parameter_count": engine.optimizer_summary.get("trainable"),
        "weight_adjustments": weight_payload["optimization_adjustments"],
        "loss_multipliers": loss_multipliers,
        "gradient_strategy": args.gradient_strategy,
        "batch_order": args.batch_order,
        "selection_protocol": "dev_only_frozen_threshold_v1",
        "test_evaluated_during_training": False,
        "initial_dev": _metrics(initial_dev),
    }
    atomic_write_json(run_root / "optimization_manifest.json", manifest)
    atomic_write_json(run_root / "training/class_weights.json", weights.as_dict())
    atomic_write_json(run_root / "training/resolved_training_config.json", resolved)
    atomic_write_text(
        run_root / "config_snapshot.yaml",
        yaml.safe_dump(
            {
                "configuration_source": "configs/runtime/training.yaml and pinned XLM-R full follow-up runner",
                "resolved_training_config": resolved,
                "training_profile": training_payload,
            },
            sort_keys=True,
            allow_unicode=False,
        ),
    )
    atomic_write_json(run_root / "selection/initial_dev.json", _metrics(initial_dev))

    state = engine.train(
        train_batches,
        seed=seed,
        dev_batches=dev_batches,
        test_batches=None,
        resume=False,
        output_root=run_root / "_engine_output",
        run_metadata={
            "mode": "fresh_full_xlmr_large_optimization" if trial != "baseline" else "fresh_full_xlmr_large",
            "experiment_id": experiment_id,
            "trial": trial,
            "test_evaluated_during_training": False,
        },
    )
    final_dev = engine._evaluate_dev(dev_batches, thresholds_override=state.thresholds)
    _write_predictions(run_root / "predictions/dev_predictions.jsonl", engine, final_dev, dev_batches)
    atomic_write_json(run_root / "metrics/dev_metrics.json", _metrics(final_dev))

    test_metrics: dict[str, Any] | None = None
    confidence: dict[str, Any] | None = None
    if not args.dev_only:
        engine.assert_test_access()
        test_selection = engine._evaluate_dev(test_batches, thresholds_override=state.thresholds)
        test_metrics = _metrics(test_selection)
        test_prediction_path = run_root / "predictions/test_predictions.jsonl"
        _write_predictions(test_prediction_path, engine, test_selection, test_batches)
        atomic_write_json(run_root / "metrics/test_metrics.json", test_metrics)
        confidence = evaluate_q1a_confidence_intervals(
            [[json.loads(line) for line in test_prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]],
            prediction_hash=sha256_file(test_prediction_path),
            config_hash=sha256_file(run_root / "config_snapshot.yaml"),
            code_commit=_git_commit(),
        )
        atomic_write_json(run_root / "metrics/test_confidence_intervals.json", confidence)

    best_engine_checkpoint = run_root / "_engine_checkpoints/model/best.pt"
    latest_candidates = sorted((run_root / "_engine_checkpoints/model").glob("epoch_*.pt"))
    if not best_engine_checkpoint.exists() or not latest_candidates:
        raise RuntimeError("TrainingEngine did not produce the expected checkpoint files")
    epoch_entries: list[dict[str, Any]] = []
    for epoch_checkpoint in latest_candidates:
        epoch_name = epoch_checkpoint.stem
        epoch_target = run_root / "checkpoints/epochs" / epoch_name / "model.pt"
        _link_or_copy_checkpoint(epoch_checkpoint, epoch_target)
        epoch_entries.append(
            {
                "epoch": int(epoch_name.removeprefix("epoch_")),
                "path": str(epoch_target.relative_to(run_root)),
                "bytes": epoch_checkpoint.stat().st_size,
            }
        )
    best_checkpoint = run_root / "checkpoints/best/model.pt"
    latest_checkpoint = run_root / "checkpoints/latest/model.pt"
    _link_or_copy_checkpoint(best_engine_checkpoint, best_checkpoint)
    _link_or_copy_checkpoint(latest_candidates[-1], latest_checkpoint)
    checkpoint_hash = sha256_file(best_checkpoint)
    atomic_write_json(
        run_root / "checkpoints/checkpoint_manifest.json",
        {
            "status": "PASS",
            "best": "checkpoints/best/model.pt",
            "latest": "checkpoints/latest/model.pt",
            "epochs": epoch_entries,
            "epoch_count": len(epoch_entries),
            "checkpoint_sha256": checkpoint_hash,
            "model_repository": spec.repo_id,
            "model_revision": spec.revision,
            "tokenizer_revision": spec.tokenizer_revision,
        },
    )
    atomic_write_json(
        run_root / "selection/best_checkpoint.json",
        {
            "status": "PASS",
            "path": "checkpoints/best/model.pt",
            "sha256": checkpoint_hash,
            "best_epoch": state.best_epoch,
            "best_dev_metric": state.best_metric,
        },
    )
    atomic_write_json(
        run_root / "selection/selection_metric.json",
        {"name": config.primary_metric, "value": state.best_metric, "best_epoch": state.best_epoch},
    )
    atomic_write_json(run_root / "selection/thresholds.json", state.thresholds)
    peak_memory = max((float(row.get("peak_memory_gb", 0.0)) for row in state.history), default=0.0)
    wall_seconds = sum(float(row.get("seconds", 0.0)) for row in state.history)
    atomic_write_json(
        run_root / "training/resource_usage.json",
        {
            "fixture": False,
            "successful_gpu_hours": wall_seconds / 3600.0,
            "failed_or_retried_gpu_hours": 0.0,
            "peak_vram_gb": peak_memory,
            "wall_seconds": wall_seconds,
            "memory_budget_gb": 20,
            "measurement_source": "TrainingEngine epoch timing and CUDA peak memory",
        },
    )
    manifest.update(
        {
            "status": state.status,
            "best_epoch": state.best_epoch,
            "best_dev_metric": state.best_metric,
            "selected_thresholds": dict(state.thresholds),
            "history": state.history,
            "final_dev": _metrics(final_dev),
            "test": test_metrics,
            "test_evaluated_during_training": False,
            "test_evaluated_after_dev_freeze": not args.dev_only,
            "peak_vram_gb": peak_memory,
            "wall_seconds": wall_seconds,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_path": "checkpoints/best/model.pt",
        }
    )
    atomic_write_json(run_root / "optimization_manifest.json", manifest)
    atomic_write_json(run_root / "metrics/summary.json", {"dev": _metrics(final_dev), "test": test_metrics, "confidence_intervals": confidence})
    print(
        json.dumps(
            {
                "status": state.status,
                "run_root": str(run_root),
                "best_epoch": state.best_epoch,
                "best_dev": _metrics(final_dev),
                "test": test_metrics,
                "peak_vram_gb": peak_memory,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the isolated full ViPragSent XLM-R-large Q1a follow-up")
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--trial", default="baseline", help="isolated candidate name; baseline uses the canonical seed path")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--scheduler", choices=("linear", "cosine"), default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--rationale-beta", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--sarcasm-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--code-switching-weight-multiplier", type=float, default=1.0)
    parser.add_argument("--pragmatic-loss-multiplier", type=float, default=1.0)
    parser.add_argument("--auxiliary-loss-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--gradient-strategy",
        choices=("sum", "pcgrad_sarcasm_irony", "pcgrad_shared_sarcasm_irony"),
        default="sum",
        help="opt-in gradient aggregation strategy for isolated optimization candidates",
    )
    parser.add_argument(
        "--batch-order",
        choices=("fixed", "deterministic_batch_shuffle"),
        default="fixed",
        help="opt-in deterministic per-epoch ordering for isolated optimization candidates",
    )
    for label in PRAGMATIC_LABELS:
        parser.add_argument(
            f"--{label.replace('_', '-')}-loss-multiplier",
            type=float,
            default=None,
            help=f"override the pragmatic loss multiplier for {label}",
        )
    parser.add_argument("--dev-only", action="store_true", help="stop after dev selection; do not access test")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
