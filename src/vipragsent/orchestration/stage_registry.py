from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..atomic import atomic_write_json, atomic_write_text
from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ..data.collation import BatchCollator
from ..data.loaders import DatasetExample, load_vipragsent
from ..data.preprocessing import PreprocessingSpec, TextPreprocessor
from ..evaluation.metrics import binary_macro_f1
from ..hashing import sha256_file, sha256_json
from ..models.variants import VariantConfig, build_dummy_model
from ..runtime.model_assets import read_family_status
from ..training.class_weights import (
    compute_train_only_class_weights,
    persist_class_weights,
    synthetic_class_weights,
)
from ..training.config_resolver import persist_resolved_training_config, resolve_training_config
from ..training.engine import TrainingConfig, TrainingEngine
from .contracts import (
    ExecutionKind,
    RunContext,
    RunEntry,
    StageOutcome,
)
from .executors.component_bundle import run_component_bundle
from .executors.external_retention import evaluate_external_retention_from_disk
from .executors.generation import generation_targets_available
from .executors.q4 import resolve_and_extract_q4_source
from .preflight_single import run_single_preflight
from .run_store import RunStore, artifact_hashes, git_commit, utc_now
from .system_registry import resolve_execution_spec

StageHandler = Callable[[], StageOutcome]


class _FixturePagedAdamW8bit(torch.optim.AdamW):
    """CPU-only stand-in used by synthetic integration tests, never production runs."""


class _FixtureBitsAndBytes:
    class optim:
        PagedAdamW8bit = _FixturePagedAdamW8bit


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    atomic_write_json(path, dict(payload))
    return path.as_posix()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _copy(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _execution_spec(root: Path, entry: RunEntry):
    return resolve_execution_spec(root, entry.system_id)


def _entry_variant(entry: RunEntry, root: Path | None = None) -> str:
    return _execution_spec(root or Path("."), entry).variant_id


def _active_tasks(entry: RunEntry, root: Path | None = None) -> set[str]:
    heads = set(resolve_execution_spec(root or Path("."), entry.system_id).active_heads)
    return {task for task, labels in (("pragmatic", set(PRAGMATIC_LABELS)), ("polarity", {"polarity"}), ("emotion", {"emotion"})) if heads & labels}


def _metric_name(entry: RunEntry) -> str:
    selection = str(entry.raw.get("selection_metric") or "")
    return {
        "macro_prag_f1_dev": "dev_macro_pragmatic_f1",
        "sarcasm_dev_f1": "dev_sarcasm_binary_macro_f1",
        "dev_macro_pragmatic_f1": "dev_macro_pragmatic_f1",
        "dev_sarcasm_macro_f1": "dev_sarcasm_macro_f1",
        "dev_sarcasm_binary_macro_f1": "dev_sarcasm_binary_macro_f1",
        "dev_polarity_macro_f1": "dev_polarity_macro_f1",
        "dev_emotion_macro_f1": "dev_emotion_macro_f1",
    }.get(selection, "dev_macro_pragmatic_f1")


def _fixture_batches(entry: RunEntry, split: str, *, batch_size: int = 4, include_rationale: bool = False, root: Path | None = None) -> list[dict[str, Any]]:
    tasks = _active_tasks(entry, root)
    batches: list[dict[str, Any]] = []
    for batch_index in range(2):
        ids = [f"fixture_{entry.run_id}_{split}_{batch_index}_{index}" for index in range(batch_size)]
        input_ids = torch.tensor([[1, 3 + ((index + batch_index * 3) % 20), 4 + index, 2] for index in range(batch_size)], dtype=torch.long)
        targets: dict[str, torch.Tensor] = {}
        if "pragmatic" in tasks:
            targets.update({key: torch.tensor([(index + batch_index + offset) % 2 for index in range(batch_size)], dtype=torch.float32) for offset, key in enumerate(PRAGMATIC_LABELS)})
        if "polarity" in tasks:
            targets["polarity"] = torch.tensor([(index + batch_index) % len(POLARITY_LABELS) for index in range(batch_size)], dtype=torch.long)
        if "emotion" in tasks:
            targets["emotion"] = torch.tensor([(index + batch_index) % len(EMOTION_LABELS) for index in range(batch_size)], dtype=torch.long)
        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "sample_ids": ids,
            "targets": targets,
        }
        if "pragmatic" in tasks:
            batch["pragmatic_pos_weight"] = {key: 1.0 for key in PRAGMATIC_LABELS}
        if include_rationale:
            rationale = torch.tensor([[1, 3 + ((index + batch_index) % 10), 4, 2] for index in range(batch_size)], dtype=torch.long)
            batch["rationale_input_ids"] = rationale
            batch["rationale_attention_mask"] = torch.ones_like(rationale)
            batch["rationale_targets"] = rationale.clone()
            batch["rationale_loss_mask"] = torch.ones(batch_size, dtype=torch.float32)
        batches.append(batch)
    return batches


def _csv_history(path: Path, history: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in history for key in row}) if history else ["epoch", "dev_macro_pragmatic_f1"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics_from_rows(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    output: dict[str, Any] = {"prediction_file": path.name, "prediction_count": len(rows), "invalid_prediction_count": 0}
    true: dict[str, list[int]] = {key: [] for key in PRAGMATIC_LABELS}
    pred: dict[str, list[int]] = {key: [] for key in PRAGMATIC_LABELS}
    probabilities: dict[str, list[float]] = {key: [] for key in PRAGMATIC_LABELS}
    polarity_true: list[int] = []
    polarity_probabilities: list[list[float]] = []
    for row in rows:
        gold = row.get("gold", {})
        predictions = row.get("predictions", {})
        probs = row.get("probabilities", {})
        for key in PRAGMATIC_LABELS:
            if key not in gold or key not in predictions:
                continue
            true[key].append(int(gold[key]))
            pred[key].append(int(predictions[key]))
            value = probs.get(key)
            if isinstance(value, list):
                value = value[-1]
            probabilities[key].append(float(value if value is not None else predictions[key]))
        if "polarity" in gold and isinstance(probs.get("polarity"), list):
            polarity_true.append(int(gold["polarity"]))
            polarity_probabilities.append([float(item) for item in probs["polarity"]])
    active = [key for key in PRAGMATIC_LABELS if true[key]]
    if active:
        output["per_label_f1"] = {key: binary_macro_f1(true[key], pred[key]) for key in active}
        output["macro_pragmatic_f1"] = float(np.mean(list(output["per_label_f1"].values())))
        output["raw_positive_probabilities"] = probabilities
        output["gold_pragmatic"] = true
    else:
        output["per_label_f1"] = {}
        output["macro_pragmatic_f1"] = "NOT_APPLICABLE"
    if polarity_true:
        from ..evaluation.metrics import expected_calibration_error

        output["polarity_dev_ece"] = expected_calibration_error(polarity_true, polarity_probabilities, bins=10)
    return output


def _fixture_generation_train(context: RunContext, entry: RunEntry) -> StageOutcome:
    """Synthetic generation-baseline adapter; it never constructs a classifier variant."""
    run_root = Path(context.run_root)
    spec = _execution_spec(context.root, entry)
    resolved = resolve_training_config(entry, spec, root=context.root, runtime_status={"successful_batch": 1})
    persist_resolved_training_config(context.root, run_root, resolved)
    persist_class_weights(context.root, run_root, synthetic_class_weights())
    for split in ("dev", "test"):
        rows: list[dict[str, Any]] = []
        generations: list[dict[str, Any]] = []
        for index in range(8):
            gold = {label: int((index + offset) % 2) for offset, label in enumerate(PRAGMATIC_LABELS)}
            probabilities = {label: 0.75 if gold[label] else 0.25 for label in PRAGMATIC_LABELS}
            sample_id = f"fixture_{entry.run_id}_{split}_{index}"
            raw_generation = "<RATIONALE>fixture cue</RATIONALE><LABELS>" + json.dumps(gold | {"polarity": "neutral", "emotion": "other"}, sort_keys=True) + "</LABELS>"
            generations.append({"sample_id": sample_id, "raw_generation": raw_generation, "parse_status": "PASS", "failure_reason": None})
            rows.append({"sample_id": sample_id, "gold": gold, "predictions": gold, "probabilities": probabilities, "generation_executor": spec.variant_id, "raw_generation": raw_generation, "parse_status": "PASS", "failure_reason": None})
        atomic_write_text(run_root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        atomic_write_text(run_root / f"generations/{split}_generations.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in generations))
        atomic_write_json(run_root / f"metrics/{split}_metrics.json", _metrics_from_rows(run_root / f"predictions/{split}_predictions.jsonl"))
    history = [{"epoch": float(epoch), "train_loss": 1.0 / epoch, "dev_loss": 1.0 / epoch, "dev_metric": 1.0, "dev_macro_pragmatic_f1": 1.0, "seconds": 0.0} for epoch in range(1, 4)]
    atomic_write_json(run_root / "training/history.json", history)
    _csv_history(run_root / "training/history.csv", history)
    atomic_write_json(run_root / "training/optimizer_summary.json", {"optimizer": "generation_executor", "learning_rate": resolved.learning_rate, "weight_decay": resolved.weight_decay, "groups": [], "trainable": 0, "frozen": 0, "executor_kind": spec.executor_kind})
    atomic_write_json(run_root / "training/scheduler_summary.json", {"scheduler": resolved.scheduler, "warmup_ratio": resolved.warmup_ratio, "warmup_steps": 0, "total_optimizer_steps": 0, "executor_kind": spec.executor_kind})
    atomic_write_json(run_root / "training/resource_usage.json", {"fixture": True, "successful_gpu_hours": 0.0, "failed_or_retried_gpu_hours": 0.0, "peak_vram_gb": 0.0})
    checkpoint = run_root / "checkpoints/best/model.pt"
    latest = run_root / "checkpoints/latest/model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    latest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"executor_kind": spec.executor_kind, "variant_id": spec.variant_id, "synthetic_results": True}, checkpoint)
    shutil.copy2(checkpoint, latest)
    digest = sha256_file(checkpoint)
    atomic_write_json(run_root / "checkpoints/checkpoint_manifest.json", {"status": "PASS", "best": _relative(checkpoint, run_root), "latest": _relative(latest, run_root), "checkpoint_sha256": digest, "model_revision": "fixture", "executor_kind": spec.executor_kind})
    atomic_write_json(run_root / "selection/best_checkpoint.json", {"path": _relative(checkpoint, run_root), "sha256": digest, "best_epoch": 3})
    atomic_write_json(run_root / "selection/selection_metric.json", {"name": "dev_macro_pragmatic_f1", "value": 1.0, "best_epoch": 3})
    atomic_write_json(run_root / "selection/thresholds.json", {label: 0.5 for label in PRAGMATIC_LABELS})
    atomic_write_json(run_root / "generation/parser_report.json", {"strict_parser": True, "semantic_repair": False, "dev": {"valid": 8, "invalid": 0}, "test": {"valid": 8, "invalid": 0}})
    return StageOutcome.passed(summary={"mode": "fixture", "synthetic_results": True, "executor_kind": spec.executor_kind, "best_epoch": 3, "best_dev_metric": 1.0, "checkpoint_path": _relative(checkpoint, run_root), "checkpoint_sha256": digest}, expected_files=("training/history.csv", "training/history.json", "training/optimizer_summary.json", "training/scheduler_summary.json", "training/resource_usage.json", "checkpoints/checkpoint_manifest.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json", "generations/dev_generations.jsonl", "generations/test_generations.jsonl", "generation/parser_report.json"))


def _fixture_component_bundle(context: RunContext, entry: RunEntry) -> StageOutcome:
    spec = _execution_spec(context.root, entry)
    sample_ids = [f"fixture_{entry.run_id}_test_{index}" for index in range(8)]
    manifest = run_component_bundle(
        context.run_root,
        executor_kind=spec.executor_kind,
        sample_ids=sample_ids,
        seed=int(entry.seed or 20260521),
        config_hash=sha256_json({"entry": entry.run_id, "variant": spec.variant_id}),
        data_hash="fixture-data",
        model_hash="fixture-model",
    )
    atomic_write_json(Path(context.run_root) / "training/resource_usage.json", {"fixture": True, "successful_gpu_hours": manifest["cost_gpu_hours"], "failed_or_retried_gpu_hours": 0.0, "component_cost_is_measured_sum": True})
    return StageOutcome.passed(
        summary={"executor_kind": spec.executor_kind, "component_count": len(manifest["component_names"]), "cost_gpu_hours": manifest["cost_gpu_hours"], "synthetic_results": True},
        expected_files=("components/state.json", "components/events.jsonl", "components/component_manifest.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json", "training/resource_usage.json"),
    )


def _execute_components(context: RunContext, entry: RunEntry) -> StageOutcome:
    if context.fixture:
        return _fixture_component_bundle(context, entry)
    return StageOutcome.blocked("component bundle requires the approved Phase 15 local snapshot and production component loader")


def _combine_component_predictions(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    manifest = _load_mapping(run_root / "components/component_manifest.json")
    if manifest.get("status") != "PASS" or not manifest.get("component_checkpoint_sha256"):
        return StageOutcome.blocked("component bundle manifest is missing independent checkpoint hashes")
    required = ("predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json")
    if any(not (run_root / path).exists() for path in required):
        return StageOutcome.blocked("component predictions were not combined by exact sample ID")
    atomic_write_json(run_root / "components/combined_prediction_manifest.json", {"status": "PASS", "component_names": manifest.get("component_names"), "combined_order_hash": manifest.get("combined_prediction_order_sha256")})
    return StageOutcome.passed(summary={"component_count": len(manifest.get("component_names", [])), "combined": True}, expected_files=required + ("components/combined_prediction_manifest.json",))


def _freeze_component_selection(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    manifest = _load_mapping(run_root / "components/component_manifest.json")
    if manifest.get("status") != "PASS":
        return StageOutcome.blocked("component selection requires a complete component manifest")
    atomic_write_json(run_root / "selection/freeze_manifest.json", {"frozen": True, "component_checkpoint_sha256": manifest.get("component_checkpoint_sha256"), "combined_prediction_order_sha256": manifest.get("combined_prediction_order_sha256"), "config_hash": manifest.get("config_hash")})
    return StageOutcome.passed(summary={"frozen": True, "component_count": len(manifest.get("component_names", []))}, expected_files=("selection/freeze_manifest.json",))


def _fixture_train(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    family = "causal" if entry.backbone in {"sailor_7b", "vistral_7b"} else "encoder"
    spec = _execution_spec(context.root, entry)
    config = VariantConfig(name=spec.variant_id, backbone_family=family, hidden_size=12, vocab_size=32, rationale_enabled_for_training=spec.rationale_training)
    model = build_dummy_model(config)
    fixture_batch = 1 if family == "causal" else 8 if entry.backbone == "xlmr_large" else 32
    resolved = resolve_training_config(entry, spec, root=context.root, runtime_status={"successful_batch": fixture_batch})
    resolved_payload = resolved.as_dict() | {"fixture": True}
    persist_resolved_training_config(context.root, run_root, resolved)
    weights = synthetic_class_weights()
    persist_class_weights(context.root, run_root, weights)
    output_root = run_root / "_engine_output"
    engine = TrainingEngine(
        model,
        TrainingConfig.from_resolved(resolved),
        run_id="model",
        checkpoint_root=run_root / "_engine_checkpoints",
        class_weights=weights,
        resolved_config=resolved_payload,
        optimizer_module=_FixtureBitsAndBytes if family == "causal" else None,
    )
    train_batches = _fixture_batches(entry, "train", include_rationale=spec.rationale_training, root=context.root)
    dev_batches = _fixture_batches(entry, "dev", include_rationale=spec.rationale_training, root=context.root)
    test_batches = _fixture_batches(entry, "test", include_rationale=spec.rationale_training, root=context.root)
    state = engine.train(
        train_batches,
        seed=int(entry.seed or 20260521),
        dev_batches=dev_batches,
        test_batches=test_batches,
        output_root=output_root,
        run_metadata={"mode": "fixture", "synthetic_results": True, "model_revision": "fixture", "tokenizer_revision": "fixture"},
    )
    history = list(state.history)
    atomic_write_json(run_root / "training/history.json", history)
    _csv_history(run_root / "training/history.csv", history)
    atomic_write_json(run_root / "training/optimizer_summary.json", engine.optimizer_summary | {"fixture": True})
    atomic_write_json(run_root / "training/scheduler_summary.json", engine.scheduler_summary | {"fixture": True})
    atomic_write_json(run_root / "training/resource_usage.json", {"fixture": True, "successful_gpu_hours": 0.0, "failed_or_retried_gpu_hours": 0.0, "peak_vram_gb": 0.0})
    engine_checkpoint = run_root / "_engine_checkpoints/model/best.pt"
    best_checkpoint = run_root / "checkpoints/best/model.pt"
    latest_checkpoint = run_root / "checkpoints/latest/model.pt"
    _copy(engine_checkpoint, best_checkpoint)
    epoch_checkpoints = sorted((run_root / "_engine_checkpoints/model").glob("epoch_*.pt"))
    if not epoch_checkpoints:
        return StageOutcome.failed("training engine did not persist an epoch checkpoint")
    _copy(epoch_checkpoints[-1], latest_checkpoint)
    checkpoint_hash = sha256_file(best_checkpoint)
    q3_data = {"budget": entry.budget, "selected_positive_count": min(int(entry.budget), 4) if entry.research_question == "Q3" and str(entry.budget) != "full" else 4, "fixed_negative_count": 4, "pos_weight": 1.0, "mask_hash": "fixture"} if entry.research_question == "Q3" else {}
    atomic_write_json(run_root / "checkpoints/checkpoint_manifest.json", {"status": "PASS", "best": _relative(best_checkpoint, run_root), "latest": _relative(latest_checkpoint, run_root), "checkpoint_sha256": checkpoint_hash, "model_revision": "fixture", "variant_fingerprint": sha256_json({"system_id": entry.system_id, "variant": spec.variant_id, "tasks": sorted(_active_tasks(entry, context.root))}), "executor_kind": spec.executor_kind, "q3_mask_hash": q3_data.get("mask_hash", "NOT_APPLICABLE"), **q3_data})
    _copy(output_root / "dev_predictions.jsonl", run_root / "predictions/dev_predictions.jsonl")
    _copy(output_root / "test_predictions.jsonl", run_root / "predictions/test_predictions.jsonl")
    dev_metrics = _metrics_from_rows(run_root / "predictions/dev_predictions.jsonl")
    test_metrics = _metrics_from_rows(run_root / "predictions/test_predictions.jsonl")
    atomic_write_json(run_root / "metrics/dev_metrics.json", dev_metrics)
    atomic_write_json(run_root / "metrics/test_metrics.json", test_metrics)
    thresholds = json.loads((output_root / "thresholds.json").read_text(encoding="utf-8"))
    atomic_write_json(run_root / "selection/best_checkpoint.json", {"path": _relative(best_checkpoint, run_root), "sha256": checkpoint_hash, "best_epoch": state.best_epoch})
    atomic_write_json(run_root / "selection/selection_metric.json", {"name": _metric_name(entry), "value": state.best_metric, "best_epoch": state.best_epoch})
    atomic_write_json(run_root / "selection/thresholds.json", thresholds)
    return StageOutcome.passed(
        summary={"mode": "fixture", "synthetic_results": True, "best_epoch": state.best_epoch, "best_dev_metric": state.best_metric, "checkpoint_path": _relative(best_checkpoint, run_root), "checkpoint_sha256": checkpoint_hash},
        artifacts=[_relative(best_checkpoint, run_root), _relative(run_root / "predictions/dev_predictions.jsonl", run_root), _relative(run_root / "predictions/test_predictions.jsonl", run_root)],
        expected_files=("training/history.csv", "training/history.json", "checkpoints/best/model.pt", "checkpoints/latest/model.pt", "checkpoints/checkpoint_manifest.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json"),
    )


def _real_train(context: RunContext, entry: RunEntry) -> StageOutcome:
    from ..data.tokenizers import create_tokenizer
    from ..models.factory import build_production_model

    root = context.root
    spec_entry = _execution_spec(root, entry)
    family = spec_entry.model_family
    cache = read_family_status(root, family, "cache")
    snapshot = cache.get("local_path")
    if not snapshot:
        return StageOutcome.blocked(f"Phase 15 local snapshot is unavailable for {family}")
    model, spec = build_production_model(family, spec_entry.variant_id, local_snapshot=snapshot, execution_mode="production")
    tokenizer = create_tokenizer(family, revision=spec.tokenizer_revision, local_path=snapshot, execution_mode="production")
    bundle = load_vipragsent(root / "data/processed/vipragsent")
    runtime_status = read_family_status(root, family, "batch")
    resolved = resolve_training_config(entry, spec_entry, root=root, runtime_status=runtime_status)
    persist_resolved_training_config(root, context.run_root, resolved)
    weights = compute_train_only_class_weights(
        bundle.train,
        dataset_hash=bundle.fingerprint,
        code_commit=git_commit(root),
    )
    persist_class_weights(root, context.run_root, weights)
    rationale_records: dict[str, Any] | None = None
    if spec_entry.rationale_training:
        rationale_path = root / "data/processed/rationales/approved_generated_rationales_train.jsonl"
        if not rationale_path.exists():
            return StageOutcome.blocked(f"Approved generated rationale artifact is unavailable: {rationale_path}")
        rationale_records = {}
        for line in rationale_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rationale_records[str(row["sample_id"])] = row
    q3_masks = None
    q3_mask_hash = None
    if entry.research_question == "Q3":
        from ..data.masks import load_validated_q3_masks

        q3_path = root / "data/processed/q3_low_resource_sarcasm"
        q3_masks, q3_report = load_validated_q3_masks(q3_path, {item.sample_id: item for item in bundle.train}, strict_frozen=True)
        q3_mask_hash = q3_report["mask_hashes"][str(entry.budget)]
    preprocessor = TextPreprocessor(PreprocessingSpec(family, entry.preprocessing_name or "vncorenlp_rdrsegmenter", entry.preprocessing_version or "locked-v1", tokenizer_revision=spec.tokenizer_revision, model_revision=spec.revision, execution_mode="production"))
    collator = BatchCollator(tokenizer, preprocessor, q3_masks=q3_masks, budget=str(entry.budget) if entry.research_question == "Q3" else None, mask_hash=q3_mask_hash, class_weights=weights.as_dict(), rationale_records=rationale_records, rationale_target_max_length=resolved.rationale_target_max_length)
    evaluation_collator = BatchCollator(tokenizer, preprocessor, class_weights=weights.as_dict(), rationale_records=rationale_records, rationale_target_max_length=resolved.rationale_target_max_length)
    batch_size = resolved.physical_batch_size
    def batches(examples: list[DatasetExample]) -> list[dict[str, Any]]:
        return [collator(examples[index:index + batch_size]) for index in range(0, len(examples), batch_size)]
    train_batches = batches(bundle.train)
    dev_batches = [evaluation_collator(bundle.dev[index:index + batch_size]) for index in range(0, len(bundle.dev), batch_size)]
    test_batches = [evaluation_collator(bundle.test[index:index + batch_size]) for index in range(0, len(bundle.test), batch_size)]
    config = TrainingConfig.from_resolved(resolved)
    engine = TrainingEngine(model, config, run_id="model", checkpoint_root=Path(context.run_root) / "_engine_checkpoints", class_weights=weights, resolved_config=resolved.as_dict())
    state = engine.train(train_batches, seed=int(entry.seed), dev_batches=dev_batches, test_batches=test_batches, output_root=Path(context.run_root) / "_engine_output", run_metadata={"mode": "full", "model_revision": spec.revision, "tokenizer_revision": spec.tokenizer_revision, "model_repository": spec.repo_id})
    peak_vram = max((float(row.get("peak_memory_gb", 0.0)) for row in state.history), default=0.0)
    wall_seconds = sum(float(row.get("seconds", 0.0)) for row in state.history)
    atomic_write_json(Path(context.run_root) / "training/resource_usage.json", {"fixture": False, "successful_gpu_hours": wall_seconds / 3600.0 if torch.cuda.is_available() else 0.0, "failed_or_retried_gpu_hours": 0.0, "peak_vram_gb": peak_vram, "wall_seconds": wall_seconds, "measurement_source": "TrainingEngine epoch timing and CUDA peak memory"})
    # Production uses the same engine outputs and checkpoint contract as the fixture adapter.
    return _materialize_engine_outputs(context, entry, state, model_revision=spec.revision, tokenizer_revision=spec.tokenizer_revision, resolved=resolved.as_dict(), q3_mask_hash=q3_mask_hash)


def _materialize_engine_outputs(context: RunContext, entry: RunEntry, state: Any, *, model_revision: str, tokenizer_revision: str, resolved: Mapping[str, Any] | None = None, q3_mask_hash: str | None = None) -> StageOutcome:
    run_root = Path(context.run_root)
    output_root = run_root / "_engine_output"
    _copy(output_root / "dev_predictions.jsonl", run_root / "predictions/dev_predictions.jsonl")
    _copy(output_root / "test_predictions.jsonl", run_root / "predictions/test_predictions.jsonl")
    atomic_write_json(run_root / "training/history.json", state.history)
    _csv_history(run_root / "training/history.csv", state.history)
    engine_manifest = output_root / "run_manifest.json"
    if engine_manifest.exists():
        root_manifest = _load_mapping(run_root / "run_manifest.json")
        root_manifest.update(json.loads(engine_manifest.read_text(encoding="utf-8")))
        atomic_write_json(run_root / "run_manifest.json", root_manifest)
    if not (run_root / "training/optimizer_summary.json").exists():
        return StageOutcome.failed("training engine did not persist the resolved optimizer summary")
    if not (run_root / "training/scheduler_summary.json").exists():
        return StageOutcome.failed("training engine did not persist the resolved scheduler summary")
    resource_path = run_root / "training/resource_usage.json"
    if not resource_path.exists():
        atomic_write_json(resource_path, {"fixture": False, "successful_gpu_hours": 0.0, "failed_or_retried_gpu_hours": 0.0, "peak_vram_gb": 0.0})
    checkpoint = run_root / "_engine_checkpoints/model/best.pt"
    latest = sorted((run_root / "_engine_checkpoints/model").glob("epoch_*.pt"))[-1]
    best = run_root / "checkpoints/best/model.pt"
    latest_target = run_root / "checkpoints/latest/model.pt"
    _copy(checkpoint, best)
    _copy(latest, latest_target)
    digest = sha256_file(best)
    atomic_write_json(run_root / "checkpoints/checkpoint_manifest.json", {"status": "PASS", "best": _relative(best, run_root), "latest": _relative(latest_target, run_root), "checkpoint_sha256": digest, "model_revision": model_revision, "tokenizer_revision": tokenizer_revision, "resolved_training_config_hash": (resolved or {}).get("config_hash", ""), "q3_mask_hash": q3_mask_hash or "NOT_APPLICABLE"})
    for name in ("dev_metrics.json", "test_metrics.json"):
        source = run_root / "metrics" / name
        if not source.exists():
            prediction = run_root / "predictions" / name.replace("_metrics.json", "_predictions.jsonl")
            atomic_write_json(source, _metrics_from_rows(prediction))
    thresholds = json.loads((output_root / "thresholds.json").read_text(encoding="utf-8"))
    atomic_write_json(run_root / "selection/best_checkpoint.json", {"path": _relative(best, run_root), "sha256": digest, "best_epoch": state.best_epoch})
    atomic_write_json(run_root / "selection/selection_metric.json", {"name": _metric_name(entry), "value": state.best_metric, "best_epoch": state.best_epoch})
    atomic_write_json(run_root / "selection/thresholds.json", thresholds)
    return StageOutcome.passed(summary={"best_epoch": state.best_epoch, "best_dev_metric": state.best_metric, "checkpoint_path": _relative(best, run_root), "checkpoint_sha256": digest}, expected_files=("training/history.csv", "training/history.json", "checkpoints/best/model.pt", "checkpoints/latest/model.pt", "checkpoints/checkpoint_manifest.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json"))


def _reuse_or_extract(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    source = entry.raw.get("source_checkpoint_path") or entry.raw.get("source_checkpoint")
    if entry.execution_kind == ExecutionKind.ARTIFACT_EXTRACTION.value:
        source_run = entry.raw.get("source_run_id")
        if source_run:
            source = context.root / "results/runs" / str(source_run)
        if not source:
            source = _approved_source_run(context.root, entry)
        if not source and context.fixture:
            return _fixture_extract(context, entry)
    if not source:
        if context.fixture:
            if entry.research_question == "Q1b":
                atomic_write_json(run_root / "checkpoint_reference.json", {"source": "fixture-approved-source", "source_sha256": sha256_json({"source": entry.system_id, "seed": entry.seed}), "source_approval_required": True, "source_status": "FIXTURE_SOURCE_ONLY", "training_applicability": "NOT_APPLICABLE", "not_applicable_reason": "Q1b evaluates official external tests from an approved upstream source; it does not train."})
                return StageOutcome.passed(summary={"training_applicability": "NOT_APPLICABLE", "not_applicable_reason": "Q1b official external retention evaluation; no training was run."}, expected_files=("checkpoint_reference.json",))
            if entry.execution_kind in {ExecutionKind.EVALUATION_ONLY.value, ExecutionKind.CHECKPOINT_REUSE.value, ExecutionKind.ARTIFACT_EXTRACTION.value}:
                return _fixture_extract(context, entry)
            return _fixture_train(context, entry)
        return StageOutcome.blocked("approved source checkpoint or prediction dependency is missing")
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = context.root / source_path
    if not source_path.exists():
        return StageOutcome.blocked(f"approved source dependency is missing: {source_path}")
    atomic_write_json(run_root / "checkpoint_reference.json", {"source": _relative(source_path, context.root), "source_sha256": sha256_file(source_path) if source_path.is_file() else "directory", "source_approval_required": True, "training_applicability": "NOT_APPLICABLE", "not_applicable_reason": "This inventory entry reuses an approved upstream checkpoint or prediction set."})
    if source_path.is_dir():
        for name in ("dev_predictions.jsonl", "test_predictions.jsonl"):
            candidate = source_path / "predictions" / name
            if candidate.exists():
                _copy(candidate, run_root / "predictions" / name)
    else:
        target = run_root / "predictions/test_predictions.jsonl"
        if source_path.suffix == ".jsonl":
            _copy(source_path, target)
    if not (run_root / "predictions/test_predictions.jsonl").exists():
        return StageOutcome.blocked("source dependency contains no approved prediction file")
    atomic_write_json(run_root / "metrics/test_metrics.json", _metrics_from_rows(run_root / "predictions/test_predictions.jsonl"))
    return StageOutcome.passed(summary={"training_applicability": "NOT_APPLICABLE", "not_applicable_reason": "Approved source dependency is reused."}, expected_files=("checkpoint_reference.json", "predictions/test_predictions.jsonl", "metrics/test_metrics.json"))


def _approved_source_run(root: Path, entry: RunEntry) -> Path | None:
    """Resolve an immutable checkpoint key to an already approved run."""
    if not entry.source_checkpoint_id:
        return None
    inventory_path = root / "reports/expected_experiment_runs.json"
    if not inventory_path.exists():
        return None
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    candidates = [
        row for row in payload.get("rows", [])
        if str(row.get("reusable_checkpoint_key")) == str(entry.source_checkpoint_id)
        and str(row.get("experiment_id") or row.get("run_id")) != entry.run_id
        and str(row.get("research_question")) != "Q4"
    ]
    for row in candidates:
        run_id = str(row.get("experiment_id") or row.get("run_id"))
        run_root = root / "results/runs" / run_id
        state = _load_mapping(run_root / "state.json")
        approval = _load_mapping(run_root / "approval_status.json")
        if state.get("run_status") == "APPROVED" and approval.get("status") == "APPROVED":
            return run_root
    return None


def _load_mapping(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _fixture_extract(context: RunContext, entry: RunEntry) -> StageOutcome:
    return StageOutcome.blocked("Q4 requires approved disk-backed source predictions and learning history; synthetic extraction is prohibited")


def _preflight(context: RunContext, entry: RunEntry) -> StageOutcome:
    report = run_single_preflight(context.root, entry, kind="azure" if entry.is_azure else "experiment", run_id=entry.run_id, fixture=context.fixture, dry_run=context.dry_run)
    atomic_write_json(Path(context.run_root) / "preflight.json", report)
    if report["passed"]:
        return StageOutcome.passed(summary=report, expected_files=("preflight.json",))
    return StageOutcome.blocked(*report["blockers"])


def _generation_stage(context: RunContext, entry: RunEntry, stage: str) -> StageOutcome:
    if not context.fixture:
        if not generation_targets_available(context.root):
            return StageOutcome.blocked("SCIENTIFIC_PROTOCOL_CONFLICT_GENERATION_BASELINE_TARGETS")
        return StageOutcome.blocked("production generation records require the exact approved target/template source")
    if not (Path(context.run_root) / "training/history.json").exists():
        _fixture_generation_train(context, entry)
    run_root = Path(context.run_root)
    expected = {
        "train_generation": ("training/history.json", "training/optimizer_summary.json", "training/scheduler_summary.json", "training/resource_usage.json", "checkpoints/checkpoint_manifest.json"),
        "generate_dev": ("generations/dev_generations.jsonl",),
        "parse_dev": ("predictions/dev_predictions.jsonl", "metrics/dev_metrics.json", "generation/parser_report.json"),
        "freeze_selection": ("selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json"),
        "generate_test": ("generations/test_generations.jsonl",),
        "parse_test": ("predictions/test_predictions.jsonl", "metrics/test_metrics.json", "generation/parser_report.json"),
    }.get(stage, ())
    missing = [path for path in expected if not (run_root / path).exists()]
    if missing:
        return StageOutcome.failed("generation stage outputs are missing: " + ", ".join(missing))
    return StageOutcome.passed(summary={"executor_kind": "generation_baseline", "stage": stage, "strict_parser": True, "classifier_fallback": False}, expected_files=expected)


def _q4_resolve_source(context: RunContext, entry: RunEntry) -> StageOutcome:
    try:
        report = resolve_and_extract_q4_source(context.root, entry.raw, output_root=context.run_root)
    except Exception as exc:
        if context.fixture:
            return StageOutcome.blocked(str(exc))
        return StageOutcome.blocked(str(exc))
    return StageOutcome.passed(summary=report, expected_files=("source/source_provenance.json", "paper_artifacts/q4_pragmatic_calibration_per_seed.json", "figure_backing/q4_pragmatic_reliability_bins.json", "figure_backing/q4_learning_curves.json"))


def _q4_validate_source(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    provenance = _load_mapping(run_root / "source/source_provenance.json")
    q4 = _load_mapping(run_root / "paper_artifacts/q4_pragmatic_calibration_per_seed.json")
    if provenance.get("status") != "PASS" or provenance.get("synthetic_history") is True:
        return StageOutcome.blocked("Q4 source provenance is not approved and non-synthetic")
    if set(q4.get("per_label_pragmatic_ece", {})) != set(PRAGMATIC_LABELS):
        return StageOutcome.blocked("Q4 source calibration does not contain six pragmatic ECE values")
    return StageOutcome.passed(summary={"status": "PASS", "training_applicability": "NOT_APPLICABLE"}, expected_files=("source/source_provenance.json", "paper_artifacts/q4_pragmatic_calibration_per_seed.json"))


def _q4_extract_stage(context: RunContext, entry: RunEntry, *, history: bool = False) -> StageOutcome:
    run_root = Path(context.run_root)
    path = run_root / ("figure_backing/q4_learning_curves.json" if history else "figure_backing/q4_pragmatic_reliability_bins.json")
    if not path.exists():
        return StageOutcome.blocked("Q4 source-backed figure data is missing")
    rows = _load_mapping(path)
    if not rows:
        return StageOutcome.blocked("Q4 source-backed figure data is empty")
    return StageOutcome.passed(summary={"rows": len(rows), "training_applicability": "NOT_APPLICABLE"}, expected_files=(path.relative_to(run_root).as_posix(),))


def _evaluate_reused_test(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    prediction = run_root / "predictions/test_predictions.jsonl"
    if not prediction.exists():
        return StageOutcome.blocked("approved checkpoint-reuse test predictions are missing")
    metrics = _metrics_from_rows(prediction)
    atomic_write_json(run_root / "metrics/test_metrics.json", metrics)
    return StageOutcome.passed(summary=metrics, expected_files=("predictions/test_predictions.jsonl", "metrics/test_metrics.json"))


def _evaluate_dev(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    path = run_root / "predictions/dev_predictions.jsonl"
    if not path.exists():
        return StageOutcome.blocked("dev predictions are missing")
    metrics = _metrics_from_rows(path)
    if entry.research_question == "Q3":
        metrics["sarcasm_dev_f1"] = (metrics.get("per_label_f1") or {}).get("sarcasm")
    atomic_write_json(run_root / "metrics/dev_metrics.json", metrics)
    return StageOutcome.passed(summary=metrics, expected_files=("predictions/dev_predictions.jsonl", "metrics/dev_metrics.json"))


def _evaluate_q1b_external(context: RunContext, entry: RunEntry) -> StageOutcome:
    from ..evaluation.external_retention import (
        NormalizedExternalExample,
        evaluate_external_retention,
    )

    run_root = Path(context.run_root)
    if context.fixture:
        datasets = {
            "vsfc": [NormalizedExternalExample("vsfc_0", "fixture", "positive"), NormalizedExternalExample("vsfc_1", "fixture", "negative"), NormalizedExternalExample("vsfc_2", "fixture", "neutral")],
            "vsmec": [NormalizedExternalExample("vsmec_0", "fixture", "anger"), NormalizedExternalExample("vsmec_1", "fixture", "sadness"), NormalizedExternalExample("vsmec_2", "fixture", "other")],
            "aivivn": [NormalizedExternalExample("aivivn_0", "fixture", "positive"), NormalizedExternalExample("aivivn_1", "fixture", "negative"), NormalizedExternalExample("aivivn_2", "fixture", "neutral")],
        }
        predictions = {key: {row.sample_id: row.label for row in rows} for key, rows in datasets.items()}
        manifest_hash = "fixture"
    else:
        try:
            result = evaluate_external_retention_from_disk(context.root, entry.raw, output_root=run_root)
        except Exception as exc:
            return StageOutcome.blocked(str(exc))
        atomic_write_json(run_root / "metrics/test_metrics.json", result)
        return StageOutcome.passed(summary=result, expected_files=("predictions/uit_vsfc_test_predictions.jsonl", "predictions/uit_vsmec_test_predictions.jsonl", "predictions/aivivn_test_predictions.jsonl", "metrics/external_retention_metrics.json", "metrics/test_metrics.json", "external/external_evaluation_manifest.json"))
    result = evaluate_external_retention(
        datasets,
        predictions,
        source_checkpoint_id=str(entry.source_checkpoint_id or entry.system_id),
        source_seed=entry.seed,
        external_manifest_hash=manifest_hash,
        output_root=run_root,
    )
    dev_metrics = _load_mapping(run_root / "metrics/dev_metrics.json")
    result = result | {"polarity_dev_ece": dev_metrics.get("polarity_dev_ece", "NOT_APPLICABLE")}
    atomic_write_json(run_root / "metrics/test_metrics.json", result)
    atomic_write_json(run_root / "external/external_evaluation_manifest.json", {"status": "PASS", "source_run_id": "fixture", "external_finetuning": False, "optimizer_steps": 0, "backward_calls": 0, "normalized_test_only": True})
    return StageOutcome.passed(summary=result, expected_files=("predictions/uit_vsfc_test_predictions.jsonl", "predictions/uit_vsmec_test_predictions.jsonl", "predictions/aivivn_test_predictions.jsonl", "metrics/external_retention_metrics.json", "metrics/test_metrics.json", "external/external_evaluation_manifest.json"))


def _freeze_selection(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    checkpoint = run_root / "selection/best_checkpoint.json"
    thresholds = run_root / "selection/thresholds.json"
    metric = run_root / "selection/selection_metric.json"
    if not all(path.exists() for path in (checkpoint, thresholds, metric)):
        return StageOutcome.blocked("best checkpoint, selection metric, and thresholds must exist before freeze")
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    threshold_payload = json.loads(thresholds.read_text(encoding="utf-8"))
    if not payload.get("sha256") or not threshold_payload:
        return StageOutcome.failed("selection freeze contains incomplete checkpoint or threshold values")
    atomic_write_json(run_root / "selection/freeze_manifest.json", {"frozen": True, "frozen_at": utc_now(), "checkpoint": payload, "thresholds": threshold_payload, "config_hash": sha256_file(run_root / "config_snapshot.yaml"), "dev_prediction_hash": sha256_file(run_root / "predictions/dev_predictions.jsonl") if (run_root / "predictions/dev_predictions.jsonl").exists() else ""})
    return StageOutcome.passed(summary={"frozen": True, "best_checkpoint": payload.get("path"), "threshold_count": len(threshold_payload)}, expected_files=("selection/freeze_manifest.json",))


def _evaluate_test(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    freeze = run_root / "selection/freeze_manifest.json"
    prediction = run_root / "predictions/test_predictions.jsonl"
    if not freeze.exists():
        return StageOutcome.blocked("test evaluation is prohibited before selection freeze")
    if not prediction.exists():
        return StageOutcome.blocked("test predictions are missing")
    if entry.research_question == "Q1b":
        return _evaluate_q1b_external(context, entry)
    metrics = _metrics_from_rows(prediction)
    if entry.research_question == "Q3":
        metrics["sarcasm_test_f1"] = (metrics.get("per_label_f1") or {}).get("sarcasm")
    metrics["thresholds_source"] = "selection/freeze_manifest.json"
    metrics["test_threshold_tuning"] = False
    atomic_write_json(run_root / "metrics/test_metrics.json", metrics)
    return StageOutcome.passed(summary=metrics, expected_files=("predictions/test_predictions.jsonl", "metrics/test_metrics.json"))


def _export_artifacts(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    if entry.research_question == "Q4":
        required_q4 = (
            run_root / "paper_artifacts/q4_pragmatic_calibration_per_seed.json",
            run_root / "figure_backing/q4_pragmatic_reliability_bins.json",
            run_root / "figure_backing/q4_learning_curves.json",
        )
        missing_q4 = [str(path.relative_to(run_root)) for path in required_q4 if not path.exists()]
        if missing_q4:
            return StageOutcome.blocked("Q4 export requires approved source-backed artifacts: " + ", ".join(missing_q4))
    if entry.research_question == "Q1a" and not entry.is_azure:
        try:
            from ..evaluation.confidence_intervals import write_q1a_confidence_intervals

            write_q1a_confidence_intervals(run_root, root=context.root)
        except Exception as exc:
            return StageOutcome.blocked(f"Table 2 confidence intervals could not be computed: {exc}")
    manifest_path = run_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update({
        "mode": "fixture" if context.fixture else "full",
        "synthetic_results": bool(context.fixture),
        "research_question": entry.research_question,
        "experiment_id": entry.run_id if not entry.is_azure else None,
        "azure_job_id": entry.run_id if entry.is_azure else None,
        "system": entry.system_id,
        "system_id": entry.system_id,
        "display_name": entry.display_name,
        "variant": entry.variant,
        "backbone": entry.backbone,
        "seed": entry.seed,
        "budget": entry.budget,
        "execution_kind": entry.execution_kind,
        "model_repository": entry.model_repository or ("fixture" if context.fixture else ""),
        "model_revision": entry.model_revision or ("fixture" if context.fixture else ""),
        "tokenizer_revision": entry.tokenizer_revision or ("fixture" if context.fixture else ""),
        "preprocessing_name": entry.preprocessing_name or ("fixture_unicode_nfc" if context.fixture else "vncorenlp_rdrsegmenter"),
        "preprocessing_version": entry.preprocessing_version or ("fixture-v1" if context.fixture else "locked-v1"),
        "data_fingerprint": sha256_file(context.root / "data/manifests/dataset_manifest.json") if (context.root / "data/manifests/dataset_manifest.json").exists() else "fixture",
        "config_hash": sha256_file(run_root / "config_snapshot.yaml"),
        "code_commit": git_commit(context.root),
        "prediction_files": [_relative(path, context.root) for path in sorted((run_root / "predictions").glob("*.jsonl"))],
        "paper_artifacts": [_relative(path, context.root) for path in sorted((run_root / "paper_artifacts").rglob("*")) if path.is_file()],
        "figure_backing": [_relative(path, context.root) for path in sorted((run_root / "figure_backing").rglob("*")) if path.is_file()],
        "external_finetuning": False,
        "inference_output_source": "classification_heads",
        "rationale_decoder_enabled_at_inference": False,
    })
    checkpoint_manifest = _load_mapping(run_root / "checkpoints/checkpoint_manifest.json")
    resolved_config = _load_mapping(run_root / "training/resolved_training_config.json")
    class_weight_path = run_root / "training/class_weights.json"
    manifest.update({
        "resolved_training_config_hash": resolved_config.get("config_hash", "NOT_APPLICABLE"),
        "class_weights_path": _relative(class_weight_path, context.root) if class_weight_path.exists() else "NOT_APPLICABLE",
        "class_weights_sha256": sha256_file(class_weight_path) if class_weight_path.exists() else "NOT_APPLICABLE",
        "q3_mask_hash": checkpoint_manifest.get("q3_mask_hash", "NOT_APPLICABLE"),
        "q3_budget": entry.budget if entry.research_question == "Q3" else "NOT_APPLICABLE",
    })
    atomic_write_json(manifest_path, manifest)
    metrics_path = run_root / "metrics.json"
    metrics = _load_mapping(metrics_path)
    metrics.update({
        "run_id": entry.run_id,
        "status": "PASS",
        "mode": "fixture" if context.fixture else "full",
        "synthetic_results": bool(context.fixture),
        "research_question": entry.research_question,
        "execution_kind": entry.execution_kind,
        "dev": _load_mapping(run_root / "metrics/dev_metrics.json"),
        "test": _load_mapping(run_root / "metrics/test_metrics.json"),
        "azure_usage": _load_mapping(run_root / "azure/usage.json"),
    })
    atomic_write_json(metrics_path, metrics)
    artifact_manifest = {"run_id": entry.run_id, "artifact_paths": sorted(artifact_hashes(run_root)), "artifact_sha256": artifact_hashes(run_root), "provenance": {"code_commit": git_commit(context.root), "config_hash": manifest["config_hash"]}}
    atomic_write_json(run_root / "provenance.json", artifact_manifest)
    expected = ("state.json", "stage_events.jsonl", "preflight.json", "run_manifest.json", "config_snapshot.yaml", "environment.json", "metrics.json", "approval_status.json", "provenance.json")
    return StageOutcome.passed(summary={"artifact_count": len(artifact_manifest["artifact_paths"])}, expected_files=expected)


def _validate_artifacts(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    common = ["state.json", "stage_events.jsonl", "preflight.json", "run_manifest.json", "config_snapshot.yaml", "environment.json", "metrics.json", "approval_status.json", "provenance.json"]
    missing = [name for name in common if not (run_root / name).exists()]
    if entry.is_azure:
        required = _azure_required_files(entry)
    elif entry.execution_kind == ExecutionKind.COMPONENT_BUNDLE.value:
        required = ["components/state.json", "components/events.jsonl", "components/component_manifest.json", "components/combined_prediction_manifest.json", "training/resource_usage.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json"]
    elif entry.execution_kind == ExecutionKind.GENERATION.value:
        required = ["training/history.csv", "training/history.json", "training/optimizer_summary.json", "training/scheduler_summary.json", "training/resource_usage.json", "checkpoints/checkpoint_manifest.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json", "generations/dev_generations.jsonl", "generations/test_generations.jsonl", "generation/parser_report.json"]
    elif entry.research_question == "Q1b":
        required = ["predictions/uit_vsfc_test_predictions.jsonl", "predictions/uit_vsmec_test_predictions.jsonl", "predictions/aivivn_test_predictions.jsonl", "metrics/external_retention_metrics.json", "metrics/test_metrics.json", "external/external_evaluation_manifest.json"]
    elif entry.research_question == "Q4":
        required = ["source/source_provenance.json", "paper_artifacts/q4_pragmatic_calibration_per_seed.json", "figure_backing/q4_pragmatic_reliability_bins.json", "figure_backing/q4_learning_curves.json"]
    elif entry.execution_kind in {ExecutionKind.EVALUATION_ONLY.value, ExecutionKind.CHECKPOINT_REUSE.value, ExecutionKind.ARTIFACT_EXTRACTION.value}:
        required = ["checkpoint_reference.json", "predictions/test_predictions.jsonl", "metrics/test_metrics.json"]
    else:
        required = ["training/history.csv", "training/history.json", "training/optimizer_summary.json", "training/scheduler_summary.json", "training/resource_usage.json", "checkpoints/checkpoint_manifest.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json"]
    if entry.research_question == "Q1a" and not entry.is_azure:
        required.append("metrics/test_confidence_intervals.json")
    missing.extend(name for name in required if not (run_root / name).exists())
    manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8")) if (run_root / "run_manifest.json").exists() else {}
    if not context.fixture:
        if manifest.get("mode") != "full":
            missing.append("run_manifest.mode=full")
        if manifest.get("synthetic_results") is True:
            missing.append("synthetic_results=true in a production run")
        if manifest.get("external_finetuning") is True:
            missing.append("external_finetuning=true")
        if manifest.get("rationale_decoder_enabled_at_inference") is True:
            missing.append("rationale decoder enabled at inference")
    approval = json.loads((run_root / "approval_status.json").read_text(encoding="utf-8")) if (run_root / "approval_status.json").exists() else {}
    if approval.get("status") != "PENDING_USER_APPROVAL":
        missing.append("approval is not pending")
    prediction_errors = _validate_prediction_files(run_root)
    missing.extend(prediction_errors)
    checksum_errors = RunStore(context).validate_checksums()
    missing.extend(checksum_errors)
    if missing:
        return StageOutcome.failed("artifact validation failed: " + "; ".join(missing))
    return StageOutcome.passed(summary={"validation_status": "PASS", "fixture": context.fixture}, expected_files=tuple(common) + tuple(required))


def _review_summary(context: RunContext, entry: RunEntry, state: Mapping[str, Any]) -> StageOutcome:
    from .review import build_review_summary, validate_review_summary

    summary = build_review_summary(context, entry, state)
    errors = validate_review_summary(summary, completed=True)
    if errors:
        return StageOutcome.failed(*errors)
    run_root = Path(context.run_root)
    atomic_write_json(run_root / "review_summary.json", summary)
    lines = ["# Sequential Run Review Summary", "", f"RUN_STATUS: {summary['RUN_STATUS']}", f"USER_REVIEW_STATUS: {summary['USER_REVIEW_STATUS']}", f"NEXT_RUN_ALLOWED: {summary['NEXT_RUN_ALLOWED']}", ""]
    for key in ("run_id", "research_question", "system_id", "execution_kind", "best_dev_metric", "checkpoint_path", "macro_pragmatic_f1", "artifact_paths", "artifact_sha256", "warnings", "blockers"):
        lines.extend([f"## {key}", json.dumps(summary.get(key), ensure_ascii=False, sort_keys=True) if isinstance(summary.get(key), (dict, list)) else str(summary.get(key)), ""])
    atomic_write_text(run_root / "review_summary.md", "\n".join(lines))
    return StageOutcome.passed(summary={"validation_status": "PASS", "review_summary_sha256": sha256_file(run_root / "review_summary.json")}, expected_files=("review_summary.json", "review_summary.md", "approval_status.json"))


def _azure_required_files(entry: RunEntry) -> tuple[str, ...]:
    files = ["azure/request_manifest.json", "azure/response_manifest.json", "azure/usage.json", "azure/invalid_outputs.jsonl", "azure/cache_manifest.json"]
    if entry.variant == "rationale_generation":
        files.extend(("azure/rationale.jsonl", "azure/rationale_failures.json"))
    else:
        files.append("predictions/test_predictions.jsonl")
    return tuple(files)


def _validate_prediction_files(run_root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted((run_root / "predictions").glob("*.jsonl")):
        seen: set[str] = set()
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.relative_to(run_root)}:{line_number}: invalid JSON ({exc})")
                continue
            sample_id = str(row.get("sample_id", ""))
            if not sample_id or sample_id in seen:
                errors.append(f"{path.relative_to(run_root)}:{line_number}: sample IDs must be present and unique")
            seen.add(sample_id)
            for value in (row.get("probabilities") or {}).values():
                values = value if isinstance(value, list) else [value]
                for probability in values:
                    try:
                        numeric = float(probability)
                    except (TypeError, ValueError):
                        errors.append(f"{path.relative_to(run_root)}:{line_number}: probability is not numeric")
                        continue
                    if not 0.0 <= numeric <= 1.0:
                        errors.append(f"{path.relative_to(run_root)}:{line_number}: probability is outside [0, 1]")
    return errors


def _azure_prompt_manifest(root: Path, entry: RunEntry) -> dict[str, Any]:
    if entry.research_question == "Q3":
        path = root / "data/manifests/prompts" / f"q3_budget_{entry.budget}_v1.json"
    else:
        path = root / "data/manifests/prompts" / f"{entry.task}_v1.json"
    if not path.exists():
        raise FileNotFoundError(f"Azure prompt manifest is missing: {path}")
    return _load_mapping(path)


def _render_azure_prompt(task: str, text: str, demonstrations: list[Mapping[str, Any]], schema: Mapping[str, Any]) -> str:
    blocks = [f"<DEMO id='{demo['sample_id']}'>\nTEXT: {demo['text']}\nLABELS: {json.dumps(demo['labels'], ensure_ascii=False, sort_keys=True)}\n</DEMO>" for demo in demonstrations]
    prefix = "Task: classify the Vietnamese comment using strict JSON."
    return f"{prefix}\n\n{chr(10).join(blocks)}\n\nINPUT:\n{text}\n\nOUTPUT_SCHEMA:\n{json.dumps(schema['schema'], ensure_ascii=False, sort_keys=True)}"


def _write_azure_manifests(
    run_root: Path,
    entry: RunEntry,
    requested: int,
    successful: int,
    invalid: int,
    usage_records: list[Mapping[str, Any]],
    settings: Any,
    *,
    synthetic: bool,
    failures: list[Mapping[str, Any]] | None = None,
) -> None:
    failures = failures or []
    retry_count = sum(int(record.get("retry_count", 0) or 0) for record in usage_records)
    input_tokens = sum(int(record.get("input_tokens", 0) or 0) for record in usage_records)
    output_tokens = sum(int(record.get("output_tokens", 0) or 0) for record in usage_records)
    cache_hits = sum(1 for record in usage_records if record.get("cache_hit") is True)
    atomic_write_json(run_root / "azure/request_manifest.json", {"job_id": entry.run_id, "job_type": entry.variant, "deployment": settings.deployment, "batch_deployment": settings.batch_deployment, "temperature": 0, "strict_schema": True, "requested": requested, "synthetic_results": synthetic})
    atomic_write_json(run_root / "azure/response_manifest.json", {"requested": requested, "successful": successful, "invalid": invalid, "missing": 0, "failed": len(failures) - invalid, "filtered": 0, "retried": retry_count, "synthetic_results": synthetic})
    atomic_write_json(run_root / "azure/usage.json", {"request_count": requested, "input_tokens": input_tokens, "output_tokens": output_tokens, "cache_hits": cache_hits, "cache_misses": max(0, successful - cache_hits), "failed_requests": len(failures) - invalid, "retried_requests": retry_count, "invalid_output_rate": invalid / requested if requested else 0.0, "synthetic_results": synthetic})
    atomic_write_text(run_root / "azure/invalid_outputs.jsonl", "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in failures))
    cache_files = sorted(path.name for path in (run_root / "azure/cache").glob("*.json")) if (run_root / "azure/cache").exists() else []
    atomic_write_json(run_root / "azure/cache_manifest.json", {"cache_entries": len(cache_files), "entries": cache_files, "synthetic_results": synthetic})


def _azure_execute(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    if context.fixture:
        request = {"job_id": entry.run_id, "job_type": entry.variant, "model": "gpt-4.1-mini", "temperature": 0, "strict_schema": True, "request_count": 4, "synthetic_results": True}
        atomic_write_json(run_root / "azure/request_manifest.json", request)
        atomic_write_json(run_root / "azure/response_manifest.json", {"requested": 4, "successful": 4, "invalid": 0, "missing": 0, "failed": 0, "filtered": 0, "retried": 0, "synthetic_results": True})
        atomic_write_json(run_root / "azure/usage.json", {"request_count": 4, "input_tokens": 0, "output_tokens": 0, "cache_hits": 0, "cache_misses": 4, "failed_requests": 0, "retried_requests": 0, "synthetic_results": True})
        atomic_write_text(run_root / "azure/invalid_outputs.jsonl", "")
        atomic_write_json(run_root / "azure/cache_manifest.json", {"cache_entries": 4, "request_hashes": [sha256_json({"job_id": entry.run_id, "index": i}) for i in range(4)], "synthetic_results": True})
        if entry.variant == "rationale_generation":
            atomic_write_text(run_root / "azure/rationale.jsonl", "")
            atomic_write_json(run_root / "azure/rationale_failures.json", [])
        else:
            rows = [{"sample_id": f"fixture_{entry.run_id}_{index}", "split": "test", "system_id": entry.system_id, "seed": entry.seed, "gold": {}, "probabilities": {}, "predictions": {}, "invalid_status": False, "failure_reason": None} for index in range(4)]
            atomic_write_text(run_root / "predictions/test_predictions.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        return StageOutcome.passed(summary={"azure_request_count": 4, "azure_input_tokens": 0, "azure_output_tokens": 0, "azure_cache_hits": 0, "azure_cache_misses": 4}, expected_files=_azure_required_files(entry))
    from ..azure.client import AzureCache, AzureResponsesClient, AzureSettings
    from ..azure.prompts import validate_task_demo_manifest
    from ..data.loaders import read_csv

    try:
        settings = AzureSettings.from_env()
    except ValueError as exc:
        return StageOutcome.blocked(str(exc))
    transport = context.metadata.get("azure_transport")
    if transport is not None and not callable(transport):
        return StageOutcome.failed("injected azure_transport must be callable")
    client = AzureResponsesClient(settings, transport=transport, cache=AzureCache(run_root / "azure/cache"))
    task = str(entry.task or "pragmatic")
    if entry.variant == "rationale_generation":
        input_path = context.root / "data/processed/rationales/azure_rationale_input_train.jsonl"
        if not input_path.exists():
            return StageOutcome.blocked("rationale input manifest is missing")
        inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        schema = {"strict": True, "schema": __import__("vipragsent.azure.schemas", fromlist=["strict_rationale_schema"]).strict_rationale_schema()}
        records, failures, usage = [], [], []
        for item in inputs:
            try:
                result = client.create_structured(prompt=f"Generate a rationale for this Vietnamese comment:\n{item['comment']}", task="rationale", schema=schema, max_output_tokens=256, sample_id=str(item["sample_id"]), input_payload=item)
                records.append({"sample_id": item["sample_id"], "rationale_target": result["labels"]["rationale"], **{key: result.get(key) for key in ("prompt_hash", "schema_hash", "response_id", "deployment", "observed_model", "observed_model_version", "usage")}})
                usage.append({**dict(result.get("usage", {})), "retry_count": result.get("retry_count", 0), "cache_hit": result.get("cache_hit", False)})
            except Exception as exc:
                failures.append({"sample_id": item.get("sample_id"), "status": "FAILED", "error": str(exc)})
        atomic_write_text(run_root / "azure/rationale.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records))
        atomic_write_json(run_root / "azure/rationale_failures.json", failures)
        _write_azure_manifests(run_root, entry, len(inputs), len(records), len(failures), usage, settings, synthetic=False, failures=failures)
        return StageOutcome.passed(summary={"azure_request_count": len(inputs), "azure_invalid_output_rate": len(failures) / len(inputs) if inputs else 0.0}, expected_files=tuple(_azure_required_files(entry)))
    prompt_manifest = _azure_prompt_manifest(context.root, entry)
    validate_task_demo_manifest(prompt_manifest, "pragmatic" if task == "sarcasm" else task)
    rows = read_csv(context.root / "data/processed/vipragsent/test.csv")
    if entry.variant == "pragmatic_zero_shot":
        demonstrations: list[Mapping[str, Any]] = []
    else:
        demonstrations = list(prompt_manifest.get("demonstrations", []))
    prompt_task = "pragmatic" if task == "sarcasm" else task
    schema = {"strict": True, "schema": __import__("vipragsent.azure.schemas", fromlist=["strict_label_schema"]).strict_label_schema(prompt_task)}
    records, failures, usage = [], [], []
    for row in rows:
        prompt = _render_azure_prompt(prompt_task, str(row.get("text", "")), demonstrations, schema)
        try:
            result = client.create_structured(prompt=prompt, task=prompt_task, schema=schema, max_output_tokens=128 if prompt_task == "pragmatic" else 32, sample_id=str(row["sample_id"]), input_payload=row)
            labels = result["labels"]
            records.append({"sample_id": row["sample_id"], "split": "test", "system_id": entry.system_id, "seed": entry.seed, "gold": {key: row[key] for key in labels if key in row}, "predictions": labels, "probabilities": {}, "invalid_status": False, "failure_reason": None})
            usage.append({**dict(result.get("usage", {})), "retry_count": result.get("retry_count", 0), "cache_hit": result.get("cache_hit", False)})
        except Exception as exc:
            failures.append({"sample_id": row.get("sample_id"), "status": "INVALID", "error": str(exc)})
    atomic_write_text(run_root / "predictions/test_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records))
    _write_azure_manifests(run_root, entry, len(rows), len(records), len(failures), usage, settings, synthetic=False, failures=failures)
    return StageOutcome.passed(summary={"azure_request_count": len(rows), "azure_invalid_output_rate": len(failures) / len(rows) if rows else 0.0}, expected_files=tuple(_azure_required_files(entry)))


def _azure_validate(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    required = [run_root / name for name in _azure_required_files(entry)]
    missing = [str(path.relative_to(run_root)) for path in required if not path.exists()]
    if missing:
        return StageOutcome.failed("Azure response validation missing: " + "; ".join(missing))
    response = json.loads((run_root / "azure/response_manifest.json").read_text(encoding="utf-8"))
    if int(response.get("requested", 0)) != int(response.get("successful", 0)) + int(response.get("invalid", 0)) + int(response.get("missing", 0)) + int(response.get("failed", 0)):
        return StageOutcome.failed("Azure response accounting does not close over requested requests")
    return StageOutcome.passed(summary=response, expected_files=tuple(str(path.relative_to(run_root)) for path in required))


def build_single_experiment_stage_registry(root: str | Path, entry_mapping: Mapping[str, Any] | RunEntry, context: RunContext | None = None) -> dict[str, StageHandler]:
    root = Path(root)
    entry = entry_mapping if isinstance(entry_mapping, RunEntry) else RunEntry.from_mapping(entry_mapping)
    context = context or RunContext(root, entry)
    return {
        "preflight": lambda: _preflight(context, entry),
        "train": lambda: _train_or_reuse(context, entry),
        "train_or_reuse": lambda: _train_or_reuse(context, entry),
        "execute_components": lambda: _execute_components(context, entry),
        "combine_component_predictions": lambda: _combine_component_predictions(context, entry),
        "evaluate_dev": lambda: _evaluate_dev(context, entry),
        "freeze_component_selection": lambda: _freeze_component_selection(context, entry),
        "freeze_selection": lambda: _freeze_selection(context, entry),
        "evaluate_test": lambda: _evaluate_test(context, entry),
        "train_generation": lambda: _generation_stage(context, entry, "train_generation"),
        "generate_dev": lambda: _generation_stage(context, entry, "generate_dev"),
        "parse_dev": lambda: _generation_stage(context, entry, "parse_dev"),
        "generate_test": lambda: _generation_stage(context, entry, "generate_test"),
        "parse_test": lambda: _generation_stage(context, entry, "parse_test"),
        "resolve_approved_source": lambda: _q4_resolve_source(context, entry) if entry.research_question == "Q4" else _reuse_or_extract(context, entry),
        "evaluate_external_tests": lambda: _evaluate_q1b_external(context, entry),
        "validate_source_predictions": lambda: _q4_validate_source(context, entry),
        "extract_pragmatic_calibration": lambda: _q4_extract_stage(context, entry),
        "extract_learning_history": lambda: _q4_extract_stage(context, entry, history=True),
        "evaluate_reused_test": lambda: _evaluate_reused_test(context, entry),
        "export_artifacts": lambda: _export_artifacts(context, entry),
        "validate_artifacts": lambda: _validate_artifacts(context, entry),
        "generate_review_summary": lambda: StageOutcome.passed(summary={"deferred": True}),
    }


def _train_or_reuse(context: RunContext, entry: RunEntry) -> StageOutcome:
    if entry.execution_kind == ExecutionKind.COMPONENT_BUNDLE.value:
        return _execute_components(context, entry)
    if entry.execution_kind == ExecutionKind.GENERATION.value:
        return _generation_stage(context, entry, "train_generation")
    if entry.execution_kind == ExecutionKind.TRAINABLE.value:
        return _fixture_train(context, entry) if context.fixture else _real_train(context, entry)
    return _reuse_or_extract(context, entry)


def build_single_azure_stage_registry(root: str | Path, entry_mapping: Mapping[str, Any] | RunEntry, context: RunContext | None = None) -> dict[str, StageHandler]:
    root = Path(root)
    entry = entry_mapping if isinstance(entry_mapping, RunEntry) else RunEntry.from_mapping(entry_mapping)
    context = context or RunContext(root, entry)
    return {
        "preflight": lambda: _preflight(context, entry),
        "execute_api_job": lambda: _azure_execute(context, entry),
        "validate_responses": lambda: _azure_validate(context, entry),
        "export_artifacts": lambda: _export_artifacts(context, entry),
        "validate_artifacts": lambda: _validate_artifacts(context, entry),
        "generate_review_summary": lambda: StageOutcome.passed(summary={"deferred": True}),
    }
