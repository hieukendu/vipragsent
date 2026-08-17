from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..atomic import atomic_write_json, atomic_write_text, exclusive_lock
from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ..data.collation import BatchCollator
from ..data.loaders import DatasetExample, load_vipragsent
from ..data.preprocessing import PreprocessingSpec, TextPreprocessor, VnCoreNLPSegmenter
from ..evaluation.metrics import binary_macro_f1
from ..evaluation.reasoning_judge import (
    ReasoningJudge,
    build_reasoning_prediction_row,
    compute_reasoning_metrics,
)
from ..hashing import sha256_file, sha256_json
from ..models.variants import VariantConfig, build_dummy_model
from ..profiling import (
    AZURE_COST_ACCOUNTING_METHOD,
    AZURE_COST_VERIFICATION_STATUS,
    AZURE_USER_SUPPLIED_RATES_USD_PER_1M,
    azure_successful_usage_cost,
)
from ..runtime.device import (
    assert_runtime_device_contract,
    resolve_model_input_device,
    write_device_report,
)
from ..runtime.hardware import validate_hardware
from ..runtime.model_assets import read_family_status, resolve_local_snapshot
from ..training.checkpoints import infer_required_head_prefixes, load_checkpoint
from ..training.class_weights import (
    compute_train_only_class_weights,
    persist_class_weights,
    synthetic_class_weights,
)
from ..training.config_resolver import persist_resolved_training_config, resolve_training_config
from ..training.engine import TrainingConfig, TrainingEngine
from ..training.generation_checkpoint import (
    GENERATION_CHECKPOINT_POINTER_KINDS,
    GENERATION_SELECTION_METRIC_NAME,
    is_real_dataset_hash,
    read_generation_checkpoint_pointer,
    resolve_generation_checkpoint_pointer,
    save_generation_checkpoint,
    write_generation_checkpoint_pointer,
)
from ..training.optimizers import build_optimizer
from ..training.schedulers import build_scheduler
from .approval import validate_approval_record
from .contracts import (
    ExecutionKind,
    RunContext,
    RunEntry,
    StageOutcome,
)
from .executors.component_bundle import run_component_bundle
from .executors.explanation_reuse import (
    resolve_approved_full_vistral_source,
    validate_source_checkpoint,
)
from .executors.external_retention import evaluate_external_retention_from_disk
from .executors.generation import (
    GenerationCheckpointError,
    ReasoningGenerationExecutor,
    _encode_text,
    build_cot_training_records,
    generation_optimizer_steps_per_epoch,
    generation_targets_available,
)
from .executors.q4 import resolve_and_extract_q4_source
from .explanation_runtime import (
    ExplanationOnlyConfig,
    ExplanationOnlyRequest,
    ExplanationOnlyRuntime,
    ExplanationRuntimeError,
    SharedInferenceIdentity,
    ValidatedSourceCheckpointIdentity,
)
from .generation_persistence import GenerationChunkStore
from .preflight_single import run_single_preflight
from .provenance import expected_inference_provenance, validate_inference_provenance
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


def _build_production_preprocessor(
    family: str,
    *,
    preprocessing_name: str,
    preprocessing_version: str,
    tokenizer_revision: str,
    model_revision: str,
) -> TextPreprocessor:
    segmenter = VnCoreNLPSegmenter.from_env() if family == "phobert_base" else None
    return TextPreprocessor(
        PreprocessingSpec(
            family,
            preprocessing_name,
            preprocessing_version,
            tokenizer_revision=tokenizer_revision,
            model_revision=model_revision,
            execution_mode="production",
        ),
        segmenter=segmenter,
    )


def _resolve_production_device(root: Path) -> tuple[int | None, str | None]:
    hardware = validate_hardware(root)
    if hardware.get("status") != "PASS":
        blockers = "; ".join(str(item) for item in hardware.get("blockers", [])) or "validated GPU runtime is unavailable"
        return None, "GPU training hardware preflight failed: " + blockers
    selected = hardware.get("selected_device_index")
    if selected is None:
        return None, "GPU training hardware preflight did not select a device"
    return int(selected), None


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
        "full_split_macro_pragmatic_f1_all_zero_fallback_dev": "full_split_macro_pragmatic_f1_all_zero_fallback_dev",
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


def _production_generation_profile(context: RunContext) -> Mapping[str, Any] | list[Mapping[str, Any]]:
    """Resolve generation-only profiling evidence, with a safe batch-one default."""
    profile = context.metadata.get("generation_profile")
    profile_path = context.metadata.get("generation_profile_path")
    if profile is None and profile_path:
        profile = Path(str(profile_path))
    if profile is None:
        profile = Path(context.run_root) / "profiling/generation_profile.json"
    if isinstance(profile, str | Path):
        path = Path(profile)
        if not path.is_absolute():
            path = context.root / path
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                loaded = None
            if isinstance(loaded, Mapping | list):
                return loaded
    if isinstance(profile, Mapping | list):
        return profile
    return {
        "status": "PASS",
        "selected_batch_size": 1,
        "candidate_batch_sizes": [1, 2, 4],
        "profiled": True,
        "source": "default-safe-generation-batch-one",
    }


def _selected_dev_artifacts_reusable(run_root: Path) -> bool:
    selection_path = run_root / "selection/best_checkpoint.json"
    marker_path = run_root / "selection/dev_artifacts.json"
    required = (
        run_root / "reasoning/dev_reasoning.jsonl",
        run_root / "predictions/dev_predictions.jsonl",
        run_root / "judge/dev_judge_responses.jsonl",
        run_root / "metrics/dev_reasoning_metrics.json",
        run_root / "reasoning/dev_chunks_manifest.json",
    )
    if not all(path.exists() for path in (selection_path, marker_path, *required)):
        return False
    try:
        selection = _load_mapping(selection_path)
        marker = _load_mapping(marker_path)
        best_pointer = read_generation_checkpoint_pointer(run_root, "best", allow_legacy=True)
        best_checkpoint = run_root / best_pointer["path"]
        selected = selection.get("dev_artifacts", {})
        return (
            marker.get("status") == "PASS"
            and selected.get("epoch") == marker.get("epoch")
            and selection.get("best_epoch") == marker.get("epoch")
            and str(selection.get("sha256") or selection.get("checkpoint_sha256")) == marker.get("checkpoint_sha256")
            and str(best_pointer.get("checkpoint_sha256", "")).upper() == str(marker.get("checkpoint_sha256", "")).upper()
            and best_checkpoint.is_file()
            and sha256_file(best_checkpoint) == marker.get("checkpoint_sha256")
            and sha256_file(required[0]) == marker.get("reasoning_sha256")
            and sha256_file(required[1]) == marker.get("predictions_sha256")
            and sha256_file(required[2]) == marker.get("judge_sha256")
            and sha256_file(required[3]) == marker.get("metrics_sha256")
            and sha256_file(required[4]) == marker.get("chunks_manifest_sha256")
        )
    except (GenerationCheckpointError, OSError, TypeError, ValueError, KeyError):
        return False


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
    dev_sample_ids = [f"fixture_{entry.run_id}_dev_{index}" for index in range(6)]
    test_sample_ids = [f"fixture_{entry.run_id}_test_{index}" for index in range(8)]
    manifest = run_component_bundle(
        context.run_root,
        executor_kind=spec.executor_kind,
        dev_sample_ids=dev_sample_ids,
        test_sample_ids=test_sample_ids,
        seed=int(entry.seed or 20260521),
        config_hash=sha256_json({"entry": entry.run_id, "variant": spec.variant_id}),
        data_hash="fixture-data",
        model_hash="fixture-model",
        allow_synthetic=True,
    )
    atomic_write_json(Path(context.run_root) / "training/resource_usage.json", {"fixture": True, "successful_gpu_hours": manifest["cost_gpu_hours"], "failed_or_retried_gpu_hours": 0.0, "component_cost_is_measured_sum": True})
    return StageOutcome.passed(
        summary={"executor_kind": spec.executor_kind, "component_count": len(manifest["component_names"]), "cost_gpu_hours": manifest["cost_gpu_hours"], "synthetic_results": True},
        expected_files=("components/state.json", "components/events.jsonl", "components/component_manifest.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json", "training/resource_usage.json"),
    )


def _execute_components(context: RunContext, entry: RunEntry) -> StageOutcome:
    if context.fixture:
        return _fixture_component_bundle(context, entry)
    spec = _execution_spec(context.root, entry)
    family = spec.model_family
    cache = read_family_status(context.root, family, "cache")
    snapshot = resolve_local_snapshot(context.root, cache.get("local_path"))
    if cache.get("status") != "PASS" or not snapshot:
        return StageOutcome.blocked(f"Phase 15 local snapshot is unavailable for component family {family}")
    try:
        bundle = load_vipragsent(context.root / "data/processed/vipragsent")
    except Exception as exc:
        return StageOutcome.blocked(f"component bundle dataset is unavailable: {exc}")
    if len(bundle.dev) != 1999 or len(bundle.test) != 2000:
        return StageOutcome.blocked(f"component bundle requires frozen dev/test counts 1999/2000, got {len(bundle.dev)}/{len(bundle.test)}")
    from .executors.component_production import ProductionComponentRunner

    production_runner = ProductionComponentRunner(context.root, entry=entry, bundle=bundle)
    try:
        manifest = run_component_bundle(
            context.run_root,
            executor_kind=spec.executor_kind,
            dev_sample_ids=[example.sample_id for example in bundle.dev],
            test_sample_ids=[example.sample_id for example in bundle.test],
            seed=int(entry.seed),
            config_hash=sha256_json(dict(entry.raw)),
            data_hash=bundle.fingerprint,
            model_hash=sha256_json({"family": family, "revision": cache.get("revision") or entry.model_revision}),
            resume=bool(context.metadata.get("resume", False)),
            model_loader=production_runner._load_runtime,
            component_runner=production_runner,
            allow_synthetic=False,
        )
    except Exception as exc:
        return StageOutcome.failed(f"production component bundle failed: {type(exc).__name__}: {exc}")
    atomic_write_json(Path(context.run_root) / "training/resource_usage.json", {"fixture": False, "successful_gpu_hours": manifest.get("total_measured_gpu_hours", 0.0), "failed_or_retried_gpu_hours": 0.0, "component_cost_is_measured_sum": True, "component_count": manifest.get("component_count")})
    return StageOutcome.passed(summary=manifest, expected_files=("components/state.json", "components/events.jsonl", "components/component_manifest.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json", "training/resource_usage.json"))


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
    selected_device, device_blocker = _resolve_production_device(root)
    if device_blocker:
        return StageOutcome.blocked(device_blocker)
    cache = read_family_status(root, family, "cache")
    snapshot = resolve_local_snapshot(root, cache.get("local_path"))
    if not snapshot:
        return StageOutcome.blocked(f"Phase 15 local snapshot is unavailable for {family}")
    model, spec = build_production_model(family, spec_entry.variant_id, local_snapshot=snapshot, execution_mode="production", selected_device=selected_device)
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
    preprocessor = _build_production_preprocessor(
        family,
        preprocessing_name=entry.preprocessing_name or "vncorenlp_rdrsegmenter",
        preprocessing_version=entry.preprocessing_version or "locked-v1",
        tokenizer_revision=spec.tokenizer_revision,
        model_revision=spec.revision,
    )
    collator = BatchCollator(tokenizer, preprocessor, q3_masks=q3_masks, budget=str(entry.budget) if entry.research_question == "Q3" else None, mask_hash=q3_mask_hash, class_weights=weights.as_dict(), rationale_records=rationale_records, rationale_target_max_length=resolved.rationale_target_max_length)
    evaluation_collator = BatchCollator(tokenizer, preprocessor, class_weights=weights.as_dict(), rationale_records=rationale_records, rationale_target_max_length=resolved.rationale_target_max_length)
    batch_size = resolved.physical_batch_size
    def batches(examples: list[DatasetExample]) -> list[dict[str, Any]]:
        return [collator(examples[index:index + batch_size]) for index in range(0, len(examples), batch_size)]
    train_batches = batches(bundle.train)
    dev_batches = [evaluation_collator(bundle.dev[index:index + batch_size]) for index in range(0, len(bundle.dev), batch_size)]
    test_batches = [evaluation_collator(bundle.test[index:index + batch_size]) for index in range(0, len(bundle.test), batch_size)]
    config = TrainingConfig.from_resolved(resolved)
    engine = TrainingEngine(model, config, run_id="model", checkpoint_root=Path(context.run_root) / "_engine_checkpoints", class_weights=weights, resolved_config=resolved.as_dict(), selected_device=selected_device)
    state = engine.train(
        train_batches,
        seed=int(entry.seed),
        dev_batches=dev_batches,
        test_batches=test_batches,
        resume=bool(context.metadata.get("resume", False)),
        output_root=Path(context.run_root) / "_engine_output",
        run_metadata={"mode": "full", "model_revision": spec.revision, "tokenizer_revision": spec.tokenizer_revision, "model_repository": spec.repo_id},
    )
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


def _validate_production_source_reference(context: RunContext, source_path: Path, source_run_id: object | None) -> str | None:
    if context.fixture:
        return None
    runs_root = (context.root / "results/runs").resolve()
    if source_run_id:
        source_run_root = runs_root / str(source_run_id)
    else:
        try:
            relative = source_path.resolve().relative_to(runs_root)
        except ValueError:
            return "production source dependencies must be inside an approved results/runs directory"
        if not relative.parts:
            return "production source dependency does not identify an approved run"
        source_run_root = runs_root / relative.parts[0]
    try:
        source_path.resolve().relative_to(source_run_root.resolve())
    except ValueError:
        return "explicit source dependency does not belong to its declared approved run"
    approval_errors = validate_approval_record(source_run_root, expected_run_id=source_run_root.name)
    if approval_errors:
        return "production source approval is incomplete: " + "; ".join(approval_errors)
    return None


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
    source_error = _validate_production_source_reference(context, source_path, entry.raw.get("source_run_id"))
    if source_error:
        return StageOutcome.blocked(source_error)
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
        if state.get("run_status") == "APPROVED" and approval.get("status") == "APPROVED" and not validate_approval_record(run_root, expected_run_id=run_id):
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


def _fixture_reasoning_judge(context: RunContext) -> ReasoningJudge:
    def transport(**_: Any) -> dict[str, Any]:
        return {"output": json.dumps({label: 0 for label in PRAGMATIC_LABELS}), "usage": {"input_tokens": 1, "output_tokens": 1}, "id": "fixture-judge"}

    return ReasoningJudge(context.root, transport=transport, cache_root=Path(context.run_root) / "judge/cache", sleep_fn=lambda _: None)


def _fixture_reasoning_rows(entry: RunEntry, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(8):
        gold = {label: int((index + offset) % 2) for offset, label in enumerate(PRAGMATIC_LABELS)}
        rows.append({"sample_id": f"fixture_{entry.run_id}_{split}_{index}", "text": "câu tiếng Việt kiểm thử", "gold": gold})
    return rows


def _fixture_cot_reasoning_stage(context: RunContext, entry: RunEntry, stage: str) -> StageOutcome:
    run_root = Path(context.run_root)
    spec = _execution_spec(context.root, entry)
    if stage == "train_generation":
        try:
            resolved = resolve_training_config(entry, spec, root=context.root, runtime_status={"successful_batch": 1})
            persist_resolved_training_config(context.root, run_root, resolved)
        except Exception:
            atomic_write_text(run_root / "config_snapshot.yaml", "fixture: true\nexecutor_kind: generation_trainable\n")
        history = [{"epoch": 1, "train_loss": 0.5, "optimizer_steps": 1.0, "selection_metric": "full_split_macro_pragmatic_f1_all_zero_fallback_dev"}]
        atomic_write_json(run_root / "training/history.json", history)
        _csv_history(run_root / "training/history.csv", history)
        atomic_write_json(run_root / "training/optimizer_summary.json", {"executor_kind": "generation_trainable", "optimizer": "fixture_generation_optimizer", "classifier_heads": 0, "direct_label_target": False})
        atomic_write_json(run_root / "training/scheduler_summary.json", {"scheduler": "fixture_generation_scheduler", "additional_training": True})
        atomic_write_json(run_root / "training/resource_usage.json", {"fixture": True, "successful_gpu_hours": 0.0, "failed_or_retried_gpu_hours": 0.0, "peak_vram_gb": 0.0, "optimizer_steps": 1})
        variant_fingerprint = sha256_json({"system_id": entry.system_id, "variant": spec.variant_id, "fixture": True})
        fixture_model = torch.nn.Linear(1, 1)
        fixture_provenance = {
            "model": {"class": "torch.nn.modules.linear.Linear"},
            "model_artifact": {"identity": "fixture-generation-model@local"},
            "tokenizer_artifact": {"identity": "fixture-generation-tokenizer@local"},
            "dataset": {"identity": "fixture-generation", "hash": "fixture-data"},
            "data_hash": "fixture-data",
            "optimizer": {"class": "fixture", "param_group_count": 0},
            "scheduler": {"class": "fixture"},
            "rng": {"seed": entry.seed, "streams": ["python", "numpy", "torch"]},
            "data_order": {"sample_ids": []},
            "config": {"executor": "generation_trainable", "variant": spec.variant_id},
            "model_environment": {"device": "cpu", "dtype": "float32"},
        }
        epoch_path = run_root / "checkpoints/epoch_0001/model.pt"
        fixture_manifest = save_generation_checkpoint(
            epoch_path,
            fixture_model,
            None,
            None,
            {"epoch": 1, "selection_metric": None, "data_order": []},
            fixture_provenance,
            metadata={"executor_kind": "generation_trainable", "variant_id": spec.variant_id, "synthetic_results": True},
            fixture_mode=True,
            epoch=1,
            variant_fingerprint=variant_fingerprint,
            selection_metric_name=GENERATION_SELECTION_METRIC_NAME,
        )
        latest_pointer = write_generation_checkpoint_pointer(run_root, "latest", "checkpoints/epoch_0001/model.pt", variant_fingerprint=variant_fingerprint)
        best_pointer = write_generation_checkpoint_pointer(run_root, "best", "checkpoints/epoch_0001/model.pt", selection_metric_value=0.0, variant_fingerprint=variant_fingerprint)
        digest = str(best_pointer["checkpoint_sha256"])
        atomic_write_json(run_root / "checkpoints/checkpoint_manifest.json", {"status": "PASS", "executor_kind": "generation_trainable", "synthetic_results": True, "checkpoint_sha256": digest, "best": best_pointer["path"], "latest": latest_pointer["path"], "best_pointer": "checkpoints/best_checkpoint.json", "latest_pointer": "checkpoints/latest_checkpoint.json", "best_checkpoint_sha256": digest, "latest_checkpoint_sha256": latest_pointer["checkpoint_sha256"], "provenance_sha256": fixture_manifest.provenance_sha256, "variant_fingerprint": variant_fingerprint, "best_epoch": 1, "latest_epoch": 1, "prompt_hash": _load_mapping(context.root / "configs/experiments/generation_reasoning_protocol.yaml").get("generation_prompt_hash"), "rationale_source_hash": "fixture-source"})
        atomic_write_json(run_root / "selection/best_checkpoint.json", {"status": "PASS", "path": best_pointer["path"], "sha256": digest, "best_epoch": 1, "selection_metric": GENERATION_SELECTION_METRIC_NAME, "selection_metric_name": GENERATION_SELECTION_METRIC_NAME, "value": 0.0, "checkpoint_path": best_pointer["path"], "checkpoint_sha256": digest})
        atomic_write_json(run_root / "selection/selection_metric.json", {"name": "full_split_macro_pragmatic_f1_all_zero_fallback_dev", "value": 0.0})
        atomic_write_json(run_root / "selection/thresholds.json", {"status": "NOT_APPLICABLE", "reason": "reasoning judge emits strict binary labels"})
        return StageOutcome.passed(summary={"executor_kind": "generation_trainable", "generation_only": True, "synthetic_results": True}, expected_files=("training/history.json", "training/history.csv", "training/optimizer_summary.json", "training/scheduler_summary.json", "training/resource_usage.json", "checkpoints/checkpoint_manifest.json", "checkpoints/epoch_0001/model.pt", "checkpoints/epoch_0001/model.pt.manifest.json", "checkpoints/latest_checkpoint.json", "checkpoints/best_checkpoint.json", "selection/best_checkpoint.json"))
    if stage == "generate_dev_reasoning":
        rows = [{**row, "split": "dev", "generated_reasoning": "phân tích dấu hiệu ngôn ngữ trong câu", "generation_status": "PASS", "truncated": False, "failure_reason": None} for row in _fixture_reasoning_rows(entry, "dev")]
        atomic_write_text(run_root / "reasoning/dev_reasoning.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        return StageOutcome.passed(summary={"split": "dev", "generation_count": len(rows), "decoding": "locked_greedy", "synthetic_results": True}, expected_files=("reasoning/dev_reasoning.jsonl",))
    if stage == "judge_dev_reasoning":
        rows = _read_jsonl(run_root / "reasoning/dev_reasoning.jsonl") if (run_root / "reasoning/dev_reasoning.jsonl").exists() else []
        if not rows:
            return StageOutcome.blocked("dev reasoning must be generated before judging")
        judge = _fixture_reasoning_judge(context)
        decisions: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        for row in rows:
            decision = judge.judge(str(row.get("generated_reasoning", "")))
            decisions.append({"sample_id": row["sample_id"], **decision})
            predictions.append(build_reasoning_prediction_row(str(row["sample_id"]), row["gold"], str(row.get("generated_reasoning", "")), decision))
        judge.write_artifacts(run_root, "dev", predictions, decisions)
        atomic_write_text(run_root / "predictions/dev_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        return StageOutcome.passed(summary={"split": "dev", "judge_protocol_id": judge.judge_protocol_id, "synthetic_results": True}, expected_files=("judge/dev_judge_responses.jsonl", "judge/cache_manifest.json", "judge/usage.json", "judge/invalid_outputs.jsonl", "predictions/dev_predictions.jsonl"))
    if stage == "compute_dev_reasoning_metrics":
        predictions = _read_jsonl(run_root / "predictions/dev_predictions.jsonl") if (run_root / "predictions/dev_predictions.jsonl").exists() else []
        if not predictions:
            return StageOutcome.blocked("dev judge predictions are missing")
        metrics = compute_reasoning_metrics(predictions)
        metrics.update({"status": "PASS", "split": "dev", "generation_protocol_id": "reasoning_generation_shared_judge_v1", "judge_protocol_id": "reasoning_judge_gpt41mini_zeroshot_v1", "synthetic_results": True})
        atomic_write_json(run_root / "metrics/dev_reasoning_metrics.json", metrics)
        return StageOutcome.passed(summary=metrics, expected_files=("metrics/dev_reasoning_metrics.json",))
    if stage == "freeze_selection":
        metrics_path = run_root / "metrics/dev_reasoning_metrics.json"
        if not metrics_path.exists():
            return StageOutcome.blocked("cot-only selection requires checkpoint and judged dev metrics")
        metrics = _load_mapping(metrics_path)
        try:
            pointer = read_generation_checkpoint_pointer(run_root, "best", allow_legacy=False)
            pointer = write_generation_checkpoint_pointer(
                run_root,
                "best",
                pointer["path"],
                selection_metric_value=float(metrics.get("primary_macro_f1", 0.0)),
                variant_fingerprint=str(pointer["variant_fingerprint"]),
            )
        except GenerationCheckpointError as exc:
            return StageOutcome.blocked(str(exc))
        digest = str(pointer["checkpoint_sha256"])
        atomic_write_json(run_root / "selection/best_checkpoint.json", {"status": "PASS", "path": pointer["path"], "sha256": digest, "best_epoch": int(pointer["epoch"]), "selection_metric": GENERATION_SELECTION_METRIC_NAME, "selection_metric_name": GENERATION_SELECTION_METRIC_NAME, "value": metrics.get("primary_macro_f1", 0.0), "primary_metric": metrics.get("primary_macro_f1", 0.0), "checkpoint_path": pointer["path"], "checkpoint_sha256": digest})
        atomic_write_json(run_root / "selection/selection_metric.json", {"name": "full_split_macro_pragmatic_f1_all_zero_fallback_dev", "value": metrics.get("primary_macro_f1", 0.0), "test_access": False})
        atomic_write_json(run_root / "selection/thresholds.json", {"status": "NOT_APPLICABLE", "reason": "reasoning judge emits strict binary labels; no dev threshold tuning"})
        atomic_write_json(run_root / "selection/freeze_manifest.json", {"frozen": True, "checkpoint": {"path": pointer["path"], "sha256": digest, "checkpoint_sha256": digest, "epoch": pointer["epoch"]}, "checkpoint_sha256": digest, "dev_metric": metrics.get("primary_macro_f1", 0.0), "test_access": False})
        return StageOutcome.passed(summary={"frozen": True, "best_dev_metric": metrics.get("primary_macro_f1", 0.0), "test_access": False}, expected_files=("selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json", "selection/freeze_manifest.json"))
    if stage == "generate_test_reasoning":
        if not (run_root / "selection/freeze_manifest.json").exists():
            return StageOutcome.blocked("test reasoning is prohibited before freeze_selection")
        rows = [{**row, "split": "test", "generated_reasoning": "phân tích dấu hiệu ngôn ngữ trong câu", "generation_status": "PASS", "truncated": False, "failure_reason": None} for row in _fixture_reasoning_rows(entry, "test")]
        atomic_write_text(run_root / "reasoning/test_reasoning.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        return StageOutcome.passed(summary={"split": "test", "generation_count": len(rows), "selection_frozen": True, "synthetic_results": True}, expected_files=("reasoning/test_reasoning.jsonl",))
    if stage == "judge_test_reasoning":
        rows = _read_jsonl(run_root / "reasoning/test_reasoning.jsonl") if (run_root / "reasoning/test_reasoning.jsonl").exists() else []
        if not rows:
            return StageOutcome.blocked("test reasoning must be generated before judging")
        judge = _fixture_reasoning_judge(context)
        decisions: list[dict[str, Any]] = []
        predictions: list[dict[str, Any]] = []
        for row in rows:
            decision = judge.judge(str(row.get("generated_reasoning", "")))
            decisions.append({"sample_id": row["sample_id"], **decision})
            predictions.append(build_reasoning_prediction_row(str(row["sample_id"]), row["gold"], str(row.get("generated_reasoning", "")), decision))
        judge.write_artifacts(run_root, "test", predictions, decisions)
        atomic_write_text(run_root / "predictions/test_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        return StageOutcome.passed(summary={"split": "test", "judge_protocol_id": judge.judge_protocol_id, "synthetic_results": True}, expected_files=("judge/test_judge_responses.jsonl", "judge/cache_manifest.json", "judge/usage.json", "judge/invalid_outputs.jsonl", "predictions/test_predictions.jsonl"))
    if stage == "compute_test_reasoning_metrics":
        predictions = _read_jsonl(run_root / "predictions/test_predictions.jsonl") if (run_root / "predictions/test_predictions.jsonl").exists() else []
        if not predictions:
            return StageOutcome.blocked("test judge predictions are missing")
        metrics = compute_reasoning_metrics(predictions)
        metrics.update({"status": "PASS", "split": "test", "generation_protocol_id": "reasoning_generation_shared_judge_v1", "judge_protocol_id": "reasoning_judge_gpt41mini_zeroshot_v1", "synthetic_results": True})
        atomic_write_json(run_root / "metrics/test_reasoning_metrics.json", metrics)
        return StageOutcome.passed(summary=metrics, expected_files=("metrics/test_reasoning_metrics.json",))
    return _production_generation_stage(context, entry, stage)


def _production_reasoning_records(root: Path, tokenizer: Any, examples: list[DatasetExample], prompt_template: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in examples:
        input_ids, attention_mask = _encode_text(tokenizer, prompt_template.replace("{TEXT}", example.text))
        records.append({"sample_id": example.sample_id, "input_ids": input_ids, "attention_mask": attention_mask, "gold": {label: int(example.labels[label]) for label in PRAGMATIC_LABELS}, "text": example.text})
    return records


def _production_generation_artifact_stage(context: RunContext, entry: RunEntry, stage: str) -> StageOutcome:
    """Judge or score existing generation artifacts without loading a model."""
    run_root = Path(context.run_root)
    split = "dev" if "dev" in stage else "test"
    judge = ReasoningJudge(context.root, cache_root=run_root / "judge/cache", require_deployment_manifest=True)
    if stage.startswith("judge_"):
        reasoning_path = run_root / f"reasoning/{split}_reasoning.jsonl"
        if not reasoning_path.exists():
            return StageOutcome.blocked(f"{split} reasoning is missing")
        if split == "dev" and _selected_dev_artifacts_reusable(run_root):
            return StageOutcome.passed(summary={"split": split, "reused_best_epoch_artifacts": True}, expected_files=("judge/dev_judge_responses.jsonl", "predictions/dev_predictions.jsonl"))
        bundle = load_vipragsent(context.root / "data/processed/vipragsent")
        generated = _read_jsonl(reasoning_path)
        gold = {row.sample_id: {label: int(row.labels[label]) for label in PRAGMATIC_LABELS} for row in getattr(bundle, split)}
        predictions: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for row in generated:
            sample_id = str(row["sample_id"])
            decision = judge.judge(str(row.get("generated_reasoning", "")))
            decisions.append({"sample_id": sample_id, **dict(decision)})
            predictions.append(
                build_reasoning_prediction_row(
                    sample_id,
                    gold[sample_id],
                    str(row.get("generated_reasoning", "")),
                    decision,
                    truncated=bool(row.get("truncated")),
                )
            )
        judge.write_artifacts(run_root, split, predictions, decisions)
        atomic_write_text(run_root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        return StageOutcome.passed(summary={"split": split, "judge_protocol_id": judge.judge_protocol_id}, expected_files=(f"judge/{split}_judge_responses.jsonl", f"predictions/{split}_predictions.jsonl"))
    if stage.startswith("compute_"):
        predictions_path = run_root / f"predictions/{split}_predictions.jsonl"
        if not predictions_path.exists():
            return StageOutcome.blocked(f"{split} judge predictions are missing")
        if split == "dev" and _selected_dev_artifacts_reusable(run_root):
            metrics_path = run_root / "metrics/dev_reasoning_metrics.json"
            return StageOutcome.passed(summary=_load_mapping(metrics_path) if metrics_path.exists() else {}, expected_files=("metrics/dev_reasoning_metrics.json",))
        metrics = compute_reasoning_metrics(_read_jsonl(predictions_path), diagnostics=judge.diagnostics)
        metrics.update({"status": "PASS", "split": split, "judge_protocol_id": judge.judge_protocol_id, "judge_prompt_hash": judge.prompt_hash, "judge_schema_hash": judge.schema_hash, "generation_protocol_id": "reasoning_generation_shared_judge_v1"})
        atomic_write_json(run_root / f"metrics/{split}_reasoning_metrics.json", metrics)
        return StageOutcome.passed(summary=metrics, expected_files=(f"metrics/{split}_reasoning_metrics.json",))
    return StageOutcome.failed(f"unsupported production generation artifact stage: {stage}")


def _production_generation_stage(context: RunContext, entry: RunEntry, stage: str) -> StageOutcome:
    # Classify artifact-only work before touching device, snapshot, model, or
    # tokenizer resolution.  Existing reasoning can be judged/scored from
    # files alone.
    if stage.startswith("judge_") or stage.startswith("compute_"):
        return _production_generation_artifact_stage(context, entry, stage)
    data_hash = context.metadata.get("data_hash")
    if not is_real_dataset_hash(data_hash):
        return StageOutcome.blocked("production generation requires a real context.metadata data_hash")
    if not generation_targets_available(context.root):
        return StageOutcome.blocked("generation reasoning protocol files are incomplete")
    injected = context.metadata.get("reasoning_executor")
    if injected is not None:
        handler = getattr(injected, stage, None)
        if callable(handler):
            result = handler()
            return result if isinstance(result, StageOutcome) else StageOutcome.passed(summary=dict(result))
    spec = _execution_spec(context.root, entry)
    family = spec.model_family
    selected_device, device_blocker = _resolve_production_device(context.root)
    if device_blocker:
        return StageOutcome.blocked(device_blocker)
    cache = read_family_status(context.root, family, "cache")
    snapshot = resolve_local_snapshot(context.root, cache.get("local_path"))
    if not snapshot:
        return StageOutcome.blocked(f"Phase 15 local snapshot is unavailable for {family}")
    from ..data.tokenizers import create_tokenizer
    from ..models.factory import build_production_model

    model, runtime_spec = build_production_model(family, "cot_only_vistral", local_snapshot=snapshot, execution_mode="production", selected_device=selected_device)
    tokenizer = create_tokenizer(family, revision=runtime_spec.tokenizer_revision, local_path=snapshot, execution_mode="production")
    runtime_status = read_family_status(context.root, family, "batch")
    resolved = resolve_training_config(entry, spec, root=context.root, runtime_status=runtime_status)
    if stage == "train_generation":
        persist_resolved_training_config(context.root, context.run_root, resolved)
    judge = ReasoningJudge(context.root, cache_root=Path(context.run_root) / "judge/cache", require_deployment_manifest=True)
    executor = ReasoningGenerationExecutor(
        context.root,
        model=model,
        tokenizer=tokenizer,
        judge=judge,
        run_root=context.run_root,
        seed=entry.seed,
        config_hash=resolved.config_hash,
        data_hash=str(data_hash),
        dataset_identity=str(context.metadata.get("dataset_identity", data_hash)),
        production_provenance_required=True,
        fixture_mode=False,
        physical_batch_size=resolved.physical_batch_size,
        gradient_accumulation_steps=resolved.gradient_accumulation_steps,
        pad_token_id=getattr(tokenizer, "pad_token_id", None),
        generation_profile=_production_generation_profile(context),
        generation_batch_size=context.metadata.get("generation_batch_size"),
    )
    bundle = load_vipragsent(context.root / "data/processed/vipragsent")
    prompt = (context.root / str(executor.protocol["generation_prompt_path"])).read_text(encoding="utf-8")
    if stage == "train_generation":
        train_records, source_report = build_cot_training_records(context.root, [{"sample_id": row.sample_id, "text": row.text} for row in bundle.train], tokenizer=tokenizer)
        optimizer, optimizer_summary = build_optimizer(model, optimizer_name=resolved.optimizer, learning_rate=resolved.learning_rate, weight_decay=resolved.weight_decay)
        steps_per_epoch = generation_optimizer_steps_per_epoch(len(train_records), resolved.physical_batch_size, resolved.gradient_accumulation_steps)
        scheduler, scheduler_summary = build_scheduler(optimizer, scheduler_name=resolved.scheduler, warmup_ratio=resolved.warmup_ratio, total_steps=steps_per_epoch * resolved.maximum_epochs)
        dev_records = _production_reasoning_records(context.root, tokenizer, bundle.dev, prompt)
        best_metric = float("-inf")
        best_epoch = 0
        best_hash = ""
        history: list[dict[str, float]] = []
        latest_epoch = 0
        latest_metric: float | None = None
        for epoch in range(1, resolved.maximum_epochs + 1):
            history.extend(executor.train_generation(train_records, optimizer=optimizer, epochs=1, scheduler=scheduler, gradient_clipping=resolved.gradient_clipping, epoch_start=epoch))
            epoch_root = Path(context.run_root) / "epochs" / f"epoch_{epoch}"
            # The canonical post-training checkpoint is the generation
            # identity for this epoch.  Save it before DEV inference so the
            # generation contract never falls back to hashing live weights.
            epoch_path = f"checkpoints/epoch_{epoch:04d}/model.pt"
            epoch_checkpoint = executor.write_epoch_checkpoint(
                epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                selection_metric=None,
            )
            latest_pointer = executor.write_checkpoint_pointer(
                "latest",
                epoch_path,
                selection_metric_value=None,
            )
            if str(latest_pointer["checkpoint_sha256"]) != str(epoch_checkpoint):
                return StageOutcome.blocked("latest generation checkpoint pointer SHA does not match the epoch payload")
            generated_dev = executor.generate_reasoning_split("dev", dev_records, artifact_root=epoch_root)
            gold_dev = {str(row["sample_id"]): row["gold"] for row in dev_records}
            dev_predictions, _ = executor.judge_reasoning_split("dev", generated_dev, gold_dev, artifact_root=epoch_root)
            dev_metrics = executor.compute_split_metrics("dev", dev_predictions, artifact_root=epoch_root)
            latest_epoch = epoch
            latest_metric = float(dev_metrics["primary_macro_f1"])
            latest_pointer = executor.write_checkpoint_pointer(
                "latest",
                epoch_path,
                selection_metric_value=latest_metric,
            )
            history[-1]["dev_primary_macro_f1"] = float(dev_metrics["primary_macro_f1"])
            if float(dev_metrics["primary_macro_f1"]) > best_metric:
                best_metric = float(dev_metrics["primary_macro_f1"])
                best_epoch = epoch
                best_pointer = executor.write_checkpoint_pointer("best", epoch_path, selection_metric_value=best_metric)
                best_hash = str(best_pointer["checkpoint_sha256"])
            atomic_write_json(Path(context.run_root) / f"metrics/dev_reasoning_metrics_epoch_{epoch}.json", dev_metrics | {"checkpoint_sha256": epoch_checkpoint})
        if latest_epoch < 1:
            return StageOutcome.blocked("generation run completed without a latest checkpoint epoch")
        latest_hash = str(latest_pointer["checkpoint_sha256"])
        if not best_hash:
            best_pointer = executor.write_checkpoint_pointer("best", f"checkpoints/epoch_{best_epoch:04d}/model.pt", selection_metric_value=best_metric)
            best_hash = str(best_pointer["checkpoint_sha256"])
        best_checkpoint = resolve_generation_checkpoint_pointer(context.run_root, "best", allow_legacy=False)
        try:
            executor.load_checkpoint(best_checkpoint, expected_sha256=best_hash)
        except GenerationCheckpointError as exc:
            return StageOutcome.blocked(str(exc))
        atomic_write_json(Path(context.run_root) / "training/history.json", history)
        atomic_write_text(Path(context.run_root) / "training/history.csv", "epoch,train_loss,optimizer_steps,dev_primary_macro_f1\n" + "\n".join(f"{row.get('epoch')},{row.get('train_loss')},{row.get('optimizer_steps')},{row.get('dev_primary_macro_f1', '')}" for row in history) + "\n")
        manifest = executor.write_checkpoint_manifest(best_epoch=best_epoch, selection_metric=best_metric, latest_epoch=latest_epoch, latest_selection_metric=latest_metric, rationale_source_hash=str(source_report.get("source_sha256", "NOT_PROVIDED")))
        dev_artifacts = executor.publish_dev_artifacts(best_epoch)
        best_pointer = read_generation_checkpoint_pointer(context.run_root, "best", allow_legacy=False)
        atomic_write_json(Path(context.run_root) / "selection/best_checkpoint.json", {"status": "PASS", "path": best_pointer["path"], "sha256": best_hash, "best_epoch": best_epoch, "selection_metric": GENERATION_SELECTION_METRIC_NAME, "selection_metric_name": GENERATION_SELECTION_METRIC_NAME, "value": best_metric, "checkpoint_path": best_pointer["path"], "checkpoint_sha256": best_hash, "dev_artifacts": dev_artifacts})
        atomic_write_json(Path(context.run_root) / "selection/selection_metric.json", {"name": GENERATION_SELECTION_METRIC_NAME, "value": best_metric, "best_epoch": best_epoch})
        atomic_write_json(Path(context.run_root) / "selection/thresholds.json", {"status": "NOT_APPLICABLE", "reason": "reasoning judge emits strict binary labels; no dev threshold tuning"})
        atomic_write_json(Path(context.run_root) / "training/optimizer_summary.json", optimizer_summary | {"executor_kind": "generation_trainable", "classifier_heads": 0})
        atomic_write_json(Path(context.run_root) / "training/scheduler_summary.json", scheduler_summary | {"executor_kind": "generation_trainable"})
        atomic_write_json(Path(context.run_root) / "training/resource_usage.json", {"successful_gpu_hours": 0.0, "failed_or_retried_gpu_hours": 0.0, "peak_vram_gb": 0.0, "synthetic_results": False})
        return StageOutcome.passed(summary={"history": history, "rationale_source": source_report, "best_epoch": best_epoch, "best_dev_metric": best_metric, "latest_epoch": latest_epoch, "latest_checkpoint_sha256": latest_hash, "checkpoint_manifest": manifest}, expected_files=("training/history.json", "training/history.csv", "training/optimizer_summary.json", "training/scheduler_summary.json", "training/resource_usage.json", "checkpoints/checkpoint_manifest.json", "checkpoints/latest_checkpoint.json", "checkpoints/best_checkpoint.json", "checkpoints/load_report.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json"))
    if stage in {"generate_dev_reasoning", "generate_test_reasoning"}:
        split = "dev" if stage.startswith("generate_dev") else "test"
        if split == "dev" and _selected_dev_artifacts_reusable(Path(context.run_root)):
            return StageOutcome.passed(summary={"split": "dev", "reused_best_epoch_artifacts": True}, expected_files=("reasoning/dev_reasoning.jsonl", "selection/dev_artifacts.json"))
        try:
            if split == "test":
                executor.load_frozen_checkpoint()
            else:
                executor.load_selected_checkpoint()
        except GenerationCheckpointError as exc:
            return StageOutcome.blocked(str(exc))
        rows = _production_reasoning_records(context.root, tokenizer, getattr(bundle, split), prompt)
        generated = executor.generate_reasoning_split(split, rows)
        return StageOutcome.passed(summary={"split": split, "generation_count": len(generated), "decoding": executor.protocol["decoding"]}, expected_files=(f"reasoning/{split}_reasoning.jsonl",))
    split = "dev" if "dev" in stage else "test"
    reasoning_path = Path(context.run_root) / f"reasoning/{split}_reasoning.jsonl"
    if stage.startswith("judge_"):
        if not reasoning_path.exists():
            return StageOutcome.blocked(f"{split} reasoning is missing")
        if split == "dev" and _selected_dev_artifacts_reusable(Path(context.run_root)):
            return StageOutcome.passed(summary={"split": "dev", "reused_best_epoch_artifacts": True}, expected_files=("judge/dev_judge_responses.jsonl", "predictions/dev_predictions.jsonl"))
        generated = _read_jsonl(reasoning_path)
        gold = {row.sample_id: {label: int(row.labels[label]) for label in PRAGMATIC_LABELS} for row in getattr(bundle, split)}
        predictions, _ = executor.judge_reasoning_split(split, generated, gold)
        return StageOutcome.passed(summary={"split": split, "judge_protocol_id": judge.judge_protocol_id}, expected_files=(f"judge/{split}_judge_responses.jsonl", "predictions/{split}_predictions.jsonl"))
    if stage.startswith("compute_"):
        predictions_path = Path(context.run_root) / f"predictions/{split}_predictions.jsonl"
        if not predictions_path.exists():
            return StageOutcome.blocked(f"{split} judge predictions are missing")
        if split == "dev" and _selected_dev_artifacts_reusable(Path(context.run_root)):
            metrics_path = Path(context.run_root) / "metrics/dev_reasoning_metrics.json"
            return StageOutcome.passed(summary=_load_mapping(metrics_path) if metrics_path.exists() else {}, expected_files=("metrics/dev_reasoning_metrics.json",))
        metrics = executor.compute_split_metrics(split, _read_jsonl(predictions_path))
        return StageOutcome.passed(summary=metrics, expected_files=(f"metrics/{split}_reasoning_metrics.json",))
    return StageOutcome.failed(f"unsupported production generation stage: {stage}")


def _generation_stage(context: RunContext, entry: RunEntry, stage: str) -> StageOutcome:
    if context.fixture:
        return _fixture_cot_reasoning_stage(context, entry, stage)
    return _production_generation_stage(context, entry, stage)


def _fixture_explanation_stage(context: RunContext, entry: RunEntry, stage: str) -> StageOutcome:
    run_root = Path(context.run_root)
    if stage == "resolve_approved_full_vistral_source":
        atomic_write_json(run_root / "source/source_provenance.json", {"status": "PASS", "source_system_id": "vipragsent_full_vistral", "source_checkpoint_key": f"vipragsent_full_vistral:{entry.seed}", "source_run_id": "fixture_full_vistral", "source_checkpoint_sha256": "fixture-source-checkpoint", "source_approval_sha256": "fixture-source-approval", "same_seed_source": True, "same_seed": True, "additional_training": False, "direct_classification_outputs_used": False, "rationale_decoder_enabled_at_inference": True, "native_causal_lm_generation_used": False, "inference_output_source": "judge_of_rationale_decoder_output", "synthetic_results": True})
        return StageOutcome.passed(summary={"source_system_id": "vipragsent_full_vistral", "same_seed": True, "additional_training": False, "synthetic_results": True}, expected_files=("source/source_provenance.json",))
    if stage == "validate_source_checkpoint":
        provenance = _load_mapping(run_root / "source/source_provenance.json")
        if provenance.get("status") != "PASS" or provenance.get("same_seed") is not True:
            return StageOutcome.blocked("explanation-only requires an exact approved same-seed full Vistral source")
        return StageOutcome.passed(summary={"source_checkpoint_valid": True, "additional_training": False, "synthetic_results": True}, expected_files=("source/source_provenance.json",))
    if stage in {"generate_dev_reasoning_from_rationale_decoder", "generate_test_reasoning_from_rationale_decoder"}:
        split = "dev" if stage.startswith("generate_dev") else "test"
        if not (run_root / "source/source_provenance.json").exists():
            return StageOutcome.blocked("approved source must be resolved before rationale-decoder inference")
        rows = [{**row, "split": split, "generated_reasoning": "rationale decoder fixture output", "generation_status": "PASS", "truncated": False, "failure_reason": None, "inference_output_source": "judge_of_rationale_decoder_output"} for row in _fixture_reasoning_rows(entry, split)]
        atomic_write_text(run_root / f"reasoning/{split}_reasoning.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        return StageOutcome.passed(summary={"split": split, "rationale_decoder_only": True, "direct_classification_outputs_used": False, "additional_training": False, "synthetic_results": True}, expected_files=(f"reasoning/{split}_reasoning.jsonl",))
    if stage in {"judge_dev_reasoning", "judge_test_reasoning"}:
        split = "dev" if "dev" in stage else "test"
        rows = _read_jsonl(run_root / f"reasoning/{split}_reasoning.jsonl") if (run_root / f"reasoning/{split}_reasoning.jsonl").exists() else []
        if not rows:
            return StageOutcome.blocked(f"{split} rationale reasoning is missing")
        judge = _fixture_reasoning_judge(context)
        predictions: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for row in rows:
            decision = judge.judge(str(row.get("generated_reasoning", "")))
            decisions.append({"sample_id": row["sample_id"], **decision})
            predictions.append(build_reasoning_prediction_row(str(row["sample_id"]), row["gold"], str(row.get("generated_reasoning", "")), decision))
        judge.write_artifacts(run_root, split, predictions, decisions)
        atomic_write_text(run_root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        return StageOutcome.passed(summary={"split": split, "judge_protocol_id": judge.judge_protocol_id, "inference_output_source": "judge_of_rationale_decoder_output", "synthetic_results": True}, expected_files=(f"judge/{split}_judge_responses.jsonl", f"predictions/{split}_predictions.jsonl", "judge/cache_manifest.json", "judge/usage.json", "judge/invalid_outputs.jsonl"))
    if stage in {"compute_dev_reasoning_metrics", "compute_test_reasoning_metrics"}:
        split = "dev" if "dev" in stage else "test"
        path = run_root / f"predictions/{split}_predictions.jsonl"
        if not path.exists():
            return StageOutcome.blocked(f"{split} rationale judge predictions are missing")
        metrics = compute_reasoning_metrics(_read_jsonl(path)) | {"status": "PASS", "split": split, "inference_output_source": "judge_of_rationale_decoder_output", "source_run_id": "fixture_full_vistral", "additional_training": False, "direct_classification_outputs_used": False, "synthetic_results": True}
        atomic_write_json(run_root / f"metrics/{split}_reasoning_metrics.json", metrics)
        return StageOutcome.passed(summary=metrics, expected_files=(f"metrics/{split}_reasoning_metrics.json",))
    return StageOutcome.failed(f"unsupported explanation-only fixture stage: {stage}")


def _production_explanation_source_entry(entry: RunEntry) -> dict[str, Any]:
    """Bind an explanation run to its approved same-seed full-model source."""
    source_entry = dict(entry.raw)
    source_entry.update(
        {
            "seed": entry.seed,
            "source_checkpoint_id": f"vipragsent_full_vistral:{entry.seed}",
        }
    )
    if entry.model_revision:
        source_entry["model_revision"] = entry.model_revision
    if entry.tokenizer_revision:
        source_entry["tokenizer_revision"] = entry.tokenizer_revision
    return source_entry


def _resolve_production_explanation_source(context: RunContext, entry: RunEntry):
    return resolve_approved_full_vistral_source(
        context.root,
        _production_explanation_source_entry(entry),
    )


def _production_explanation_identity(
    context: RunContext,
    protocol: Mapping[str, Any],
) -> SharedInferenceIdentity:
    """Build the frozen identity shared by explanation inference and chunks."""
    protocol_hash = sha256_json(protocol)
    configured = context.metadata.get("explanation_inference_identity")
    if configured is None:
        configured = context.metadata.get("generation_identity")
    values = dict(configured) if isinstance(configured, Mapping) else {}
    configured_hash = values.get("protocol_hash") or context.metadata.get("generation_protocol_hash")
    if configured_hash not in (None, "", protocol_hash):
        raise ExplanationRuntimeError("explanation protocol hash does not match the locked generation protocol")
    configured_protocol_id = values.get("protocol_id")
    protocol_id = str(protocol.get("protocol_version", ""))
    if configured_protocol_id not in (None, "", protocol_id):
        raise ExplanationRuntimeError("explanation protocol identity does not match the locked generation protocol")

    environment_path = Path(context.run_root) / "environment.json"
    environment = _load_mapping(environment_path)
    environment_identity = (
        context.metadata.get("environment_identity")
        or environment.get("environment_identity")
        or "production"
    )
    environment_version = context.metadata.get("environment_version")
    if not environment_version and environment_path.exists():
        environment_version = sha256_file(environment_path)
    if not environment_version:
        environment_version = f"torch-{torch.__version__}"

    values.update(
        {
            "protocol_id": protocol_id,
            "protocol_hash": protocol_hash,
            "protocol_version": str(values.get("protocol_version") or "v1"),
            "environment_identity": str(environment_identity),
            "environment_version": str(environment_version),
        }
    )
    return SharedInferenceIdentity.from_mapping(values)


def _build_production_explanation_runtime(
    context: RunContext,
    entry: RunEntry,
    source: Any,
    model: torch.nn.Module,
    tokenizer: Any,
    judge: ReasoningJudge,
) -> ExplanationOnlyRuntime:
    # The approved-source resolver has already verified this exact checkpoint
    # path and digest.  Preserve that boundary so request construction does
    # not hash the large source checkpoint a second time.
    source_identity = ValidatedSourceCheckpointIdentity.from_approved_source(source)
    data_hash = context.metadata.get("data_hash") or source_identity.dataset_hash
    if not is_real_dataset_hash(data_hash):
        raise ExplanationRuntimeError("production explanation inference requires a canonical SHA-256 dataset hash")
    dataset_identity = str(context.metadata.get("dataset_identity") or source_identity.dataset_identity)
    if not dataset_identity:
        raise ExplanationRuntimeError("production explanation inference requires a dataset identity")
    protocol = judge.protocol
    config = ExplanationOnlyConfig(
        identity=_production_explanation_identity(context, protocol),
        decoder_max_tokens=int(protocol.get("decoding", {}).get("max_new_tokens", 160)),
        generation_profile=_production_generation_profile(context),
    )
    request = ExplanationOnlyRequest(
        seed=entry.seed,
        source_checkpoint=source_identity,
        config=config,
        data_hash=str(data_hash),
        dataset_identity=dataset_identity,
        batch_size=context.metadata.get("generation_batch_size"),
        artifact_root=Path(context.run_root),
        legacy_artifact_root=Path(context.run_root) / "legacy_explanation",
        fixture_mode=False,
    )
    return ExplanationOnlyRuntime(model, tokenizer, request, run_root=context.run_root)


def _production_explanation_records(tokenizer: Any, examples: list[DatasetExample]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for example in examples:
        input_ids, attention = _encode_text(tokenizer, example.text)
        records.append({"sample_id": example.sample_id, "input_ids": input_ids, "attention_mask": attention})
    return records


def _read_production_explanation_rows_without_model(
    run_root: Path,
    split: str,
    sample_ids: list[str],
    source: Any,
    engine_fingerprint: str,
) -> list[dict[str, Any]]:
    """Validate committed rationale artifacts without instantiating 7B state."""
    reasoning_path = run_root / f"reasoning/{split}_reasoning.jsonl"
    manifest_path = run_root / f"reasoning/{split}_chunks_manifest.json"
    if not reasoning_path.exists() or not manifest_path.exists():
        raise ExplanationRuntimeError(f"{split} committed rationale artifacts are missing")
    manifest = _load_mapping(manifest_path)
    contract = manifest.get("generation_contract")
    if not isinstance(contract, Mapping):
        raise ExplanationRuntimeError(f"{split} rationale chunks lack a canonical generation contract")
    checkpoint_identity = contract.get("checkpoint_identity")
    if not isinstance(checkpoint_identity, Mapping) or str(checkpoint_identity.get("checkpoint_sha256", "")).upper() != str(source.checkpoint_sha256).upper():
        raise ExplanationRuntimeError(f"{split} rationale chunks are bound to a different source checkpoint")
    store = GenerationChunkStore(run_root, split, sample_ids, generation_contract=contract)
    rows = store.committed_rows()
    observed_ids = [str(row.get("sample_id", "")) for row in rows]
    if manifest.get("complete") is not True or observed_ids != sample_ids:
        raise ExplanationRuntimeError(f"{split} committed rationale chunks are incomplete or out of order")
    for row in rows:
        if str(row.get("engine_fingerprint", "")) != engine_fingerprint:
            raise ExplanationRuntimeError(f"{split} chunk engine identity mismatch")
        if str(row.get("source_checkpoint_sha256", "")).upper() != str(source.checkpoint_sha256).upper():
            raise ExplanationRuntimeError(f"{split} chunk source checkpoint binding mismatch")
        if str(row.get("source_checkpoint_key", "")) != f"vipragsent_full_vistral:{source.seed}":
            raise ExplanationRuntimeError(f"{split} chunk source checkpoint key mismatch")
    return rows


def _production_explanation_artifact_stage(context: RunContext, entry: RunEntry, stage: str, source: Any) -> StageOutcome:
    """Judge or score existing rationale artifacts without loading the source model."""
    run_root = Path(context.run_root)
    judge = ReasoningJudge(context.root, cache_root=run_root / "judge/cache", require_deployment_manifest=True)
    identity = _production_explanation_identity(context, judge.protocol)
    if stage.startswith("judge_"):
        split = "dev" if "dev" in stage else "test"
        bundle = load_vipragsent(context.root / "data/processed/vipragsent")
        sample_ids = [str(example.sample_id) for example in getattr(bundle, split)]
        try:
            reasoning = _read_production_explanation_rows_without_model(run_root, split, sample_ids, source, identity.fingerprint)
        except (ExplanationRuntimeError, OSError, RuntimeError, ValueError) as exc:
            return StageOutcome.blocked(str(exc))
        gold = {example.sample_id: {label: int(example.labels[label]) for label in PRAGMATIC_LABELS} for example in getattr(bundle, split)}
        predictions: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for row in reasoning:
            sample_id = str(row["sample_id"])
            decision = judge.judge(str(row.get("generated_reasoning", "")))
            decisions.append({"sample_id": sample_id, **dict(decision)})
            predictions.append(build_reasoning_prediction_row(sample_id, gold[sample_id], str(row.get("generated_reasoning", "")), decision, truncated=bool(row.get("truncated"))))
        judge.write_artifacts(run_root, split, predictions, decisions)
        atomic_write_text(run_root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        return StageOutcome.passed(summary={"split": split, "source_run_id": source.run_id, "judge_protocol_id": judge.judge_protocol_id}, expected_files=(f"judge/{split}_judge_responses.jsonl", f"predictions/{split}_predictions.jsonl"))
    if stage.startswith("compute_"):
        split = "dev" if "dev" in stage else "test"
        path = run_root / f"predictions/{split}_predictions.jsonl"
        if not path.exists():
            return StageOutcome.blocked(f"{split} rationale judge predictions are missing")
        metrics = compute_reasoning_metrics(_read_jsonl(path), diagnostics=judge.diagnostics) | {"status": "PASS", "split": split, "inference_output_source": "judge_of_rationale_decoder_output"}
        atomic_write_json(run_root / f"metrics/{split}_reasoning_metrics.json", metrics)
        return StageOutcome.passed(summary=metrics, expected_files=(f"metrics/{split}_reasoning_metrics.json",))
    return StageOutcome.failed(f"unsupported production explanation artifact stage: {stage}")


def _production_explanation_stage(context: RunContext, entry: RunEntry, stage: str) -> StageOutcome:
    run_root = Path(context.run_root)
    if stage == "resolve_approved_full_vistral_source":
        try:
            source = _resolve_production_explanation_source(context, entry)
        except Exception as exc:
            return StageOutcome.blocked(str(exc))
        atomic_write_json(run_root / "source/source_provenance.json", {"status": "PASS", "source": source.as_dict(context.root), "source_system_id": "vipragsent_full_vistral", "same_seed_source": True, "additional_training": False, "direct_classification_outputs_used": False, "rationale_decoder_enabled_at_inference": True, "native_causal_lm_generation_used": False, "inference_output_source": "judge_of_rationale_decoder_output"})
        return StageOutcome.passed(summary=source.as_dict(context.root), expected_files=("source/source_provenance.json",))
    source_payload = _load_mapping(run_root / "source/source_provenance.json")
    if not source_payload:
        return StageOutcome.blocked("approved full Vistral source has not been resolved")
    try:
        source = _resolve_production_explanation_source(context, entry)
    except Exception as exc:
        return StageOutcome.blocked(str(exc))
    if stage == "validate_source_checkpoint":
        report = validate_source_checkpoint(context.root, source)
        return StageOutcome.passed(summary=report, expected_files=("source/source_provenance.json",)) if report["status"] == "PASS" else StageOutcome.blocked(*report["errors"])
    if stage.startswith("judge_") or stage.startswith("compute_"):
        return _production_explanation_artifact_stage(context, entry, stage, source)
    spec = _execution_spec(context.root, entry)
    selected_device, device_blocker = _resolve_production_device(context.root)
    if device_blocker:
        return StageOutcome.blocked(device_blocker)
    cache = read_family_status(context.root, spec.model_family, "cache")
    snapshot = resolve_local_snapshot(context.root, cache.get("local_path"))
    if not snapshot:
        return StageOutcome.blocked(f"Phase 15 local snapshot is unavailable for {spec.model_family}")
    from ..data.tokenizers import create_tokenizer
    from ..models.factory import build_production_model

    model, runtime_spec = build_production_model(spec.model_family, "explanation_only_vistral", local_snapshot=snapshot, execution_mode="production", selected_device=selected_device)
    load_checkpoint(
        source.checkpoint_path,
        model,
        allow_legacy_fixture=context.fixture,
        required_head_prefixes=infer_required_head_prefixes(model),
        report_path=run_root / "checkpoints/source_load_report.json",
    )
    device = resolve_model_input_device(model)
    write_device_report(
        run_root / "training/device_report.json",
        assert_runtime_device_contract(model, device, model_family=str(spec.model_family)),
    )
    tokenizer = create_tokenizer(spec.model_family, revision=runtime_spec.tokenizer_revision, local_path=snapshot, execution_mode="production")
    judge = ReasoningJudge(context.root, cache_root=run_root / "judge/cache", require_deployment_manifest=True)
    bundle = load_vipragsent(context.root / "data/processed/vipragsent")
    try:
        runtime = _build_production_explanation_runtime(context, entry, source, model, tokenizer, judge)
    except (ExplanationRuntimeError, OSError, RuntimeError, ValueError) as exc:
        return StageOutcome.blocked(str(exc))
    if stage in {"generate_dev_reasoning_from_rationale_decoder", "generate_test_reasoning_from_rationale_decoder"}:
        split = "dev" if stage.startswith("generate_dev") else "test"
        records = _production_explanation_records(tokenizer, getattr(bundle, split))
        try:
            runtime.generate_reasoning_split(split, records)
        except (ExplanationRuntimeError, OSError, RuntimeError, ValueError) as exc:
            return StageOutcome.blocked(str(exc))
        return StageOutcome.passed(summary={"split": split, "source_run_id": source.run_id, "additional_training": False, "direct_classification_outputs_used": False, "inference_output_source": "judge_of_rationale_decoder_output"}, expected_files=(f"reasoning/{split}_reasoning.jsonl",))
    if stage in {"judge_dev_reasoning", "judge_test_reasoning"}:
        split = "dev" if "dev" in stage else "test"
        records = _production_explanation_records(tokenizer, getattr(bundle, split))
        sample_ids = [str(record["sample_id"]) for record in records]
        try:
            reasoning = runtime.committed_rows_for_downstream(split, sample_ids, records)
        except (ExplanationRuntimeError, OSError, RuntimeError, ValueError) as exc:
            return StageOutcome.blocked(str(exc))
        if not reasoning:
            return StageOutcome.blocked(f"{split} committed rationale chunks are missing")
        gold = {example.sample_id: {label: int(example.labels[label]) for label in PRAGMATIC_LABELS} for example in getattr(bundle, split)}
        predictions: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for row in reasoning:
            decision = judge.judge(str(row.get("generated_reasoning", "")))
            decisions.append({"sample_id": row["sample_id"], **decision})
            predictions.append(
                build_reasoning_prediction_row(
                    str(row["sample_id"]),
                    gold[str(row["sample_id"])],
                    str(row.get("generated_reasoning", "")),
                    decision,
                    truncated=bool(row.get("truncated")),
                )
            )
        judge.write_artifacts(run_root, split, predictions, decisions)
        atomic_write_text(run_root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in predictions))
        metrics = compute_reasoning_metrics(predictions, diagnostics=judge.diagnostics) | {"status": "PASS", "split": split, "inference_output_source": "judge_of_rationale_decoder_output"}
        atomic_write_json(run_root / f"metrics/{split}_reasoning_metrics.json", metrics)
        return StageOutcome.passed(summary=metrics, expected_files=(f"judge/{split}_judge_responses.jsonl", f"predictions/{split}_predictions.jsonl"))
    if stage in {"compute_dev_reasoning_metrics", "compute_test_reasoning_metrics"}:
        split = "dev" if "dev" in stage else "test"
        path = run_root / f"predictions/{split}_predictions.jsonl"
        if not path.exists():
            return StageOutcome.blocked(f"{split} rationale judge predictions are missing")
        metrics = compute_reasoning_metrics(_read_jsonl(path), diagnostics=judge.diagnostics) | {"status": "PASS", "inference_output_source": "judge_of_rationale_decoder_output"}
        atomic_write_json(run_root / f"metrics/{split}_reasoning_metrics.json", metrics)
        return StageOutcome.passed(summary=metrics, expected_files=(f"metrics/{split}_reasoning_metrics.json",))
    return StageOutcome.failed(f"unsupported explanation-only production stage: {stage}")


def _explanation_stage(context: RunContext, entry: RunEntry, stage: str) -> StageOutcome:
    return _fixture_explanation_stage(context, entry, stage) if context.fixture else _production_explanation_stage(context, entry, stage)


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
        partial = bool(result.get("partial"))
        applicable = set(result.get("applicable_external_datasets", []))
        prediction_files = {"vsfc": "predictions/uit_vsfc_test_predictions.jsonl", "vsmec": "predictions/uit_vsmec_test_predictions.jsonl", "aivivn": "predictions/aivivn_test_predictions.jsonl"}
        expected = tuple(prediction_files[key] for key in sorted(applicable)) + (("metrics/partial_external_retention_metrics.json",) if partial else ("metrics/external_retention_metrics.json",)) + ("metrics/test_metrics.json", "external/external_evaluation_manifest.json")
        return StageOutcome.passed(summary=result, expected_files=expected)
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
    if entry.system_id == "cot_only_vistral":
        try:
            pointer = read_generation_checkpoint_pointer(run_root, "best", allow_legacy=True)
            selected_path = run_root / str(payload.get("path") or payload.get("checkpoint_path") or "")
            resolved_path = run_root / str(pointer["path"])
            if selected_path.resolve() != resolved_path.resolve() or str(pointer.get("checkpoint_sha256", "")).upper() != str(payload.get("sha256", "")).upper():
                return StageOutcome.failed("selection freeze checkpoint does not match the verified best pointer")
        except (GenerationCheckpointError, OSError, ValueError) as exc:
            return StageOutcome.failed(f"selection freeze checkpoint pointer is invalid: {exc}")
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
    if entry.research_question == "Q1a" and not entry.is_azure and entry.system_id not in {"cot_only_vistral", "explanation_only_vistral"}:
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
        **expected_inference_provenance(entry.system_id, execution_kind=entry.execution_kind),
    })
    checkpoint_manifest = _load_mapping(run_root / "checkpoints/checkpoint_manifest.json")
    generation_pointer_data: dict[str, Any] = {}
    if entry.system_id == "cot_only_vistral":
        try:
            best_pointer = read_generation_checkpoint_pointer(run_root, "best", allow_legacy=True)
            latest_pointer = read_generation_checkpoint_pointer(run_root, "latest", allow_legacy=True)
            generation_pointer_data = {
                "best_checkpoint_pointer": "checkpoints/best_checkpoint.json" if not best_pointer.get("legacy") else "NOT_APPLICABLE_LEGACY",
                "latest_checkpoint_pointer": "checkpoints/latest_checkpoint.json" if not latest_pointer.get("legacy") else "NOT_APPLICABLE_LEGACY",
                "best_checkpoint_path": best_pointer["path"],
                "latest_checkpoint_path": latest_pointer["path"],
                "best_checkpoint_sha256": best_pointer["checkpoint_sha256"],
                "latest_checkpoint_sha256": latest_pointer["checkpoint_sha256"],
                "best_checkpoint_provenance_sha256": best_pointer.get("provenance_sha256", "NOT_PROVIDED"),
                "latest_checkpoint_provenance_sha256": latest_pointer.get("provenance_sha256", "NOT_PROVIDED"),
                "variant_fingerprint": best_pointer.get("variant_fingerprint", "NOT_PROVIDED"),
            }
        except GenerationCheckpointError as exc:
            return StageOutcome.blocked(f"generation checkpoint pointer validation failed: {exc}")
    resolved_config = _load_mapping(run_root / "training/resolved_training_config.json")
    class_weight_path = run_root / "training/class_weights.json"
    manifest.update({
        "resolved_training_config_hash": resolved_config.get("config_hash", "NOT_APPLICABLE"),
        "class_weights_path": _relative(class_weight_path, context.root) if class_weight_path.exists() else "NOT_APPLICABLE",
        "class_weights_sha256": sha256_file(class_weight_path) if class_weight_path.exists() else "NOT_APPLICABLE",
        "q3_mask_hash": checkpoint_manifest.get("q3_mask_hash", "NOT_APPLICABLE"),
        "q3_budget": entry.budget if entry.research_question == "Q3" else "NOT_APPLICABLE",
        "generation_protocol_id": _load_mapping(context.root / "configs/experiments/generation_reasoning_protocol.yaml").get("protocol_version", "NOT_APPLICABLE") if entry.system_id in {"cot_only_vistral", "explanation_only_vistral"} else "NOT_APPLICABLE",
        "provenance_contract_version": 1,
        **generation_pointer_data,
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
        "dev": _load_mapping(run_root / ("metrics/dev_reasoning_metrics.json" if entry.system_id in {"cot_only_vistral", "explanation_only_vistral"} else "metrics/dev_metrics.json")),
        "test": _load_mapping(run_root / ("metrics/test_reasoning_metrics.json" if entry.system_id in {"cot_only_vistral", "explanation_only_vistral"} else "metrics/test_metrics.json")),
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
    elif entry.system_id == "cot_only_vistral":
        required = ["training/history.csv", "training/history.json", "training/optimizer_summary.json", "training/scheduler_summary.json", "training/resource_usage.json", "checkpoints/checkpoint_manifest.json", "checkpoints/latest_checkpoint.json", "checkpoints/best_checkpoint.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json", "selection/freeze_manifest.json", "reasoning/dev_reasoning.jsonl", "reasoning/test_reasoning.jsonl", "judge/dev_judge_responses.jsonl", "judge/test_judge_responses.jsonl", "judge/cache_manifest.json", "judge/usage.json", "judge/invalid_outputs.jsonl", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_reasoning_metrics.json", "metrics/test_reasoning_metrics.json"]
    elif entry.system_id == "explanation_only_vistral":
        required = ["source/source_provenance.json", "reasoning/dev_reasoning.jsonl", "reasoning/test_reasoning.jsonl", "judge/dev_judge_responses.jsonl", "judge/test_judge_responses.jsonl", "judge/cache_manifest.json", "judge/usage.json", "judge/invalid_outputs.jsonl", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_reasoning_metrics.json", "metrics/test_reasoning_metrics.json"]
    elif entry.research_question == "Q1b":
        if entry.system_id == "phobert_pol_single":
            required = ["predictions/uit_vsfc_test_predictions.jsonl", "predictions/aivivn_test_predictions.jsonl", "metrics/partial_external_retention_metrics.json", "metrics/test_metrics.json", "external/external_evaluation_manifest.json"]
        elif entry.system_id == "phobert_emo_single":
            required = ["predictions/uit_vsmec_test_predictions.jsonl", "metrics/partial_external_retention_metrics.json", "metrics/test_metrics.json", "external/external_evaluation_manifest.json"]
        else:
            required = ["predictions/uit_vsfc_test_predictions.jsonl", "predictions/uit_vsmec_test_predictions.jsonl", "predictions/aivivn_test_predictions.jsonl", "metrics/external_retention_metrics.json", "metrics/test_metrics.json", "external/external_evaluation_manifest.json"]
    elif entry.research_question == "Q4":
        required = ["source/source_provenance.json", "paper_artifacts/q4_pragmatic_calibration_per_seed.json", "figure_backing/q4_pragmatic_reliability_bins.json", "figure_backing/q4_learning_curves.json"]
    elif entry.execution_kind in {ExecutionKind.EVALUATION_ONLY.value, ExecutionKind.CHECKPOINT_REUSE.value, ExecutionKind.ARTIFACT_EXTRACTION.value}:
        required = ["checkpoint_reference.json", "predictions/test_predictions.jsonl", "metrics/test_metrics.json"]
    else:
        required = ["training/history.csv", "training/history.json", "training/optimizer_summary.json", "training/scheduler_summary.json", "training/resource_usage.json", "checkpoints/checkpoint_manifest.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json"]
    if entry.research_question == "Q1a" and not entry.is_azure and entry.system_id not in {"cot_only_vistral", "explanation_only_vistral"}:
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
    missing.extend(validate_inference_provenance(manifest, source="run_manifest", allow_fixture_parser=context.fixture))
    if entry.system_id == "explanation_only_vistral":
        source = _load_mapping(run_root / "source/source_provenance.json")
        source_payload = {"system_id": entry.system_id, **source}
        missing.extend(validate_inference_provenance(source_payload, source="source_provenance", allow_fixture_parser=context.fixture))
    if entry.system_id == "cot_only_vistral":
        for kind in GENERATION_CHECKPOINT_POINTER_KINDS:
            try:
                read_generation_checkpoint_pointer(run_root, kind, allow_legacy=True)
            except GenerationCheckpointError as exc:
                missing.append(f"{kind} generation checkpoint pointer: {exc}")
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
        lines.extend([f"## {key}", json.dumps(summary.get(key), ensure_ascii=False, sort_keys=True) if isinstance(summary.get(key), dict | list) else str(summary.get(key)), ""])
    atomic_write_text(run_root / "review_summary.md", "\n".join(lines))
    return StageOutcome.passed(summary={"validation_status": "PASS", "review_summary_sha256": sha256_file(run_root / "review_summary.json")}, expected_files=("review_summary.json", "review_summary.md", "approval_status.json"))


def _azure_required_files(entry: RunEntry) -> tuple[str, ...]:
    files = ["azure/request_manifest.json", "azure/response_manifest.json", "azure/usage.json", "azure/usage_records.jsonl", "azure/cost_ledger.json", "azure/invalid_outputs.jsonl", "azure/cache_manifest.json"]
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


def _load_azure_usage_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"Azure usage record {path}:{line_number} is not an object")
        records.append(dict(payload))
    return records


def _build_azure_cost_ledger(records: list[Mapping[str, Any]], *, synthetic: bool) -> dict[str, Any]:
    uncached = [record for record in records if not bool(record.get("cache_hit"))]
    priced = [record for record in uncached if bool(record.get("cost_counted")) and record.get("request_cost_usd") is not None]
    usage_unavailable = [record for record in uncached if record.get("cost_status") == "USAGE_UNAVAILABLE"]
    total_cost = sum(float(record["request_cost_usd"]) for record in priced)
    non_cached_input_cost = sum(float(record.get("non_cached_input_cost_usd", 0.0) or 0.0) for record in priced)
    cached_input_cost = sum(float(record.get("cached_input_cost_usd", 0.0) or 0.0) for record in priced)
    output_cost = sum(float(record.get("output_cost_usd", 0.0) or 0.0) for record in priced)
    return {
        "schema_version": 1,
        "currency": "USD",
        "unit": "per_1_million_tokens",
        "input_usd_per_1m": AZURE_USER_SUPPLIED_RATES_USD_PER_1M["input"],
        "cached_input_usd_per_1m": AZURE_USER_SUPPLIED_RATES_USD_PER_1M["cached_input"],
        "output_usd_per_1m": AZURE_USER_SUPPLIED_RATES_USD_PER_1M["output"],
        "cost_accounting_method": AZURE_COST_ACCOUNTING_METHOD,
        "cost_verification_status": AZURE_COST_VERIFICATION_STATUS,
        "successful_response_records": len(records),
        "successful_uncached_logical_requests": len(uncached),
        "successful_uncached_priced_requests": len(priced),
        "cached_reuses": sum(1 for record in records if bool(record.get("cache_hit"))),
        "usage_unavailable_successes": len(usage_unavailable),
        "input_tokens": sum(int(record.get("input_tokens", 0) or 0) for record in priced),
        "cached_input_tokens": sum(int(record.get("cached_input_tokens", 0) or 0) for record in priced),
        "non_cached_input_tokens": sum(int(record.get("non_cached_input_tokens", 0) or 0) for record in priced),
        "output_tokens": sum(int(record.get("output_tokens", 0) or 0) for record in priced),
        "non_cached_input_cost_usd": round(non_cached_input_cost, 12),
        "cached_input_cost_usd": round(cached_input_cost, 12),
        "output_cost_usd": round(output_cost, 12),
        "total_azure_cost_usd": round(total_cost, 12),
        "failed_attempts_cost_usd": 0.0,
        "retry_attempts_are_not_costed_separately": True,
        "synthetic_results": synthetic,
        "status": "NOT_APPLICABLE_FIXTURE" if synthetic else ("NO_SUCCESSFUL_RESPONSES" if not records else ("INCOMPLETE_USAGE" if usage_unavailable else "PASS")),
    }


def _persist_azure_usage_record(run_root: Path, *, sample_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Persist one successful response and its local cost exactly once."""

    usage_path = run_root / "azure/usage_records.jsonl"
    ledger_path = run_root / "azure/cost_ledger.json"
    cache_hit = bool(result.get("cache_hit"))
    logical_key = str(result.get("cache_key") or f"sample_id:{sample_id}")
    usage_cost = azure_successful_usage_cost(result.get("usage") if isinstance(result.get("usage"), Mapping) else None)
    record = {
        "sample_id": str(sample_id),
        "logical_key": logical_key,
        "request_id": result.get("request_id"),
        "response_id": result.get("response_id"),
        "model": result.get("observed_model") or result.get("expected_model_family"),
        "model_version": result.get("observed_model_version") or result.get("expected_model_version"),
        "deployment": result.get("deployment"),
        "input_tokens": usage_cost["input_tokens"],
        "cached_input_tokens": usage_cost["cached_input_tokens"],
        "non_cached_input_tokens": usage_cost["non_cached_input_tokens"],
        "output_tokens": usage_cost["output_tokens"],
        "non_cached_input_cost_usd": usage_cost.get("non_cached_input_cost_usd"),
        "cached_input_cost_usd": usage_cost.get("cached_input_cost_usd"),
        "output_cost_usd": usage_cost.get("output_cost_usd"),
        "retry_count": int(result.get("retry_count", 0) or 0),
        "cache_hit": cache_hit,
        "response_status": "CACHE_REUSED_SUCCESSFUL_RESPONSE" if cache_hit else "SUCCESSFUL_AND_VALIDATED_RESPONSE",
        "cost_status": "CACHE_REUSE_NO_NEW_COST" if cache_hit else usage_cost["cost_status"],
        "cost_counted": not cache_hit and usage_cost["cost_status"] == "USAGE_AVAILABLE",
        "request_cost_usd": 0.0 if cache_hit else usage_cost["request_cost_usd"],
        "cost_accounting_method": AZURE_COST_ACCOUNTING_METHOD,
        "cost_verification_status": AZURE_COST_VERIFICATION_STATUS,
        "request_timestamp": result.get("request_timestamp"),
    }
    with exclusive_lock(usage_path.with_suffix(".lock")):
        records = _load_azure_usage_records(usage_path)
        existing_index = {str(item.get("logical_key")): index for index, item in enumerate(records) if item.get("logical_key")}
        existing = records[existing_index[logical_key]] if logical_key in existing_index else None
        if existing is None or (not existing.get("cost_counted") and record["cost_counted"]):
            if existing is None:
                records.append(record)
            else:
                records[existing_index[logical_key]] = record
            atomic_write_text(usage_path, "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records))
            prior_ledger = _load_mapping(ledger_path)
            ledger = _build_azure_cost_ledger(records, synthetic=False)
            ledger["generated_at"] = prior_ledger.get("generated_at") or utc_now()
            atomic_write_json(ledger_path, ledger)
            return record
        return existing


def _ensure_azure_cost_artifacts(run_root: Path, *, synthetic: bool, usage_records: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    usage_path = run_root / "azure/usage_records.jsonl"
    records = _load_azure_usage_records(usage_path) if usage_path.exists() else [dict(item) for item in (usage_records or [])]
    if not usage_path.exists():
        atomic_write_text(usage_path, "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records))
    ledger_path = run_root / "azure/cost_ledger.json"
    prior_ledger = _load_mapping(ledger_path)
    ledger = _build_azure_cost_ledger(records, synthetic=synthetic)
    ledger["generated_at"] = prior_ledger.get("generated_at") or utc_now()
    atomic_write_json(ledger_path, ledger)
    return ledger


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
    ledger = _ensure_azure_cost_artifacts(run_root, synthetic=synthetic, usage_records=usage_records)
    persisted_usage = _load_azure_usage_records(run_root / "azure/usage_records.jsonl")
    retry_count = sum(int(record.get("retry_count", 0) or 0) for record in persisted_usage)
    input_tokens = sum(int(record.get("input_tokens", 0) or 0) for record in persisted_usage)
    output_tokens = sum(int(record.get("output_tokens", 0) or 0) for record in persisted_usage)
    cache_hits = sum(1 for record in persisted_usage if record.get("cache_hit") is True)
    atomic_write_json(run_root / "azure/request_manifest.json", {"job_id": entry.run_id, "job_type": entry.variant, "deployment": settings.deployment, "batch_deployment": settings.batch_deployment, "temperature": 0, "strict_schema": True, "requested": requested, "synthetic_results": synthetic})
    atomic_write_json(run_root / "azure/response_manifest.json", {"requested": requested, "successful": successful, "invalid": invalid, "missing": 0, "failed": len(failures) - invalid, "filtered": 0, "retried": retry_count, "synthetic_results": synthetic})
    atomic_write_json(run_root / "azure/usage.json", {"request_count": requested, "input_tokens": input_tokens, "output_tokens": output_tokens, "cached_input_tokens": ledger["cached_input_tokens"], "non_cached_input_tokens": ledger["non_cached_input_tokens"], "cache_hits": cache_hits, "cache_misses": max(0, successful - cache_hits), "failed_requests": len(failures) - invalid, "retried_requests": retry_count, "invalid_output_rate": invalid / requested if requested else 0.0, "non_cached_input_cost_usd": ledger["non_cached_input_cost_usd"], "cached_input_cost_usd": ledger["cached_input_cost_usd"], "output_cost_usd": ledger["output_cost_usd"], "total_azure_cost_usd": ledger["total_azure_cost_usd"], "cost_accounting_method": AZURE_COST_ACCOUNTING_METHOD, "cost_verification_status": AZURE_COST_VERIFICATION_STATUS, "usage_records_path": "azure/usage_records.jsonl", "cost_ledger_path": "azure/cost_ledger.json", "synthetic_results": synthetic})
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
        _ensure_azure_cost_artifacts(run_root, synthetic=True, usage_records=[])
        atomic_write_text(run_root / "azure/invalid_outputs.jsonl", "")
        atomic_write_json(run_root / "azure/cache_manifest.json", {"cache_entries": 4, "request_hashes": [sha256_json({"job_id": entry.run_id, "index": i}) for i in range(4)], "synthetic_results": True})
        if entry.variant == "rationale_generation":
            atomic_write_text(run_root / "azure/rationale.jsonl", "")
            atomic_write_json(run_root / "azure/rationale_failures.json", [])
        else:
            rows = [{"sample_id": f"fixture_{entry.run_id}_{index}", "split": "test", "system_id": entry.system_id, "seed": entry.seed, "gold": {}, "probabilities": {}, "predictions": {}, "invalid_status": False, "failure_reason": None} for index in range(4)]
            atomic_write_text(run_root / "predictions/test_predictions.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        return StageOutcome.passed(summary={"azure_request_count": 4, "azure_input_tokens": 0, "azure_output_tokens": 0, "azure_cache_hits": 0, "azure_cache_misses": 4}, expected_files=_azure_required_files(entry))
    from ..azure.client import (
        AzureCache,
        AzureResponsesClient,
        AzureSafetyBudgetError,
        AzureSafetyCeilings,
        AzureSafetyLedger,
        AzureSettings,
    )
    from ..azure.prompts import validate_task_demo_manifest
    from ..data.loaders import read_csv

    try:
        settings = AzureSettings.from_env()
    except ValueError as exc:
        return StageOutcome.blocked(str(exc))
    transport = context.metadata.get("azure_transport")
    if transport is not None and not callable(transport):
        return StageOutcome.failed("injected azure_transport must be callable")
    safety_payload = context.metadata.get("azure_safety_ceilings")
    if safety_payload is not None and not isinstance(safety_payload, Mapping):
        return StageOutcome.failed("azure_safety_ceilings must be a mapping")
    safety_values = dict(safety_payload or {})
    # Production artifact validation requires locally verified spend.  A
    # caller may opt into an explicit UNKNOWN result only for non-production
    # diagnostics; this boundary remains fail-closed by default.
    safety_values.setdefault("allow_unknown_spend", False)
    try:
        safety = AzureSafetyCeilings.from_mapping(safety_values)
    except (TypeError, ValueError) as exc:
        return StageOutcome.blocked(f"invalid Azure safety ceilings: {exc}")
    client = AzureResponsesClient(settings, transport=transport, cache=AzureCache(run_root / "azure/cache"), safety=safety, safety_ledger=AzureSafetyLedger(safety))
    task = str(entry.task or "pragmatic")
    if entry.variant == "rationale_generation":
        input_path = context.root / "data/processed/rationales/azure_rationale_input_train.jsonl"
        if not input_path.exists():
            return StageOutcome.blocked("rationale input manifest is missing")
        inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        try:
            client.preflight_logical_requests(len(inputs))
        except AzureSafetyBudgetError as exc:
            return StageOutcome.blocked(str(exc))
        schema = {"strict": True, "schema": __import__("vipragsent.azure.schemas", fromlist=["strict_rationale_schema"]).strict_rationale_schema()}
        records, failures, usage = [], [], []
        for index, item in enumerate(inputs):
            try:
                result = client.create_structured(prompt=f"Generate a rationale for this Vietnamese comment:\n{item['comment']}", task="rationale", schema=schema, max_output_tokens=256, sample_id=str(item["sample_id"]), input_payload=item)
                _persist_azure_usage_record(run_root, sample_id=str(item["sample_id"]), result=result)
                records.append({"sample_id": item["sample_id"], "rationale_target": result["labels"]["rationale"], **{key: result.get(key) for key in ("prompt_hash", "schema_hash", "response_id", "deployment", "observed_model", "observed_model_version", "usage")}})
                usage.append({**dict(result.get("usage", {})), "retry_count": result.get("retry_count", 0), "cache_hit": result.get("cache_hit", False)})
            except AzureSafetyBudgetError as exc:
                failures.append({"sample_id": item.get("sample_id"), "status": "SAFETY_BUDGET_EXCEEDED", "error": str(exc)})
                failures.extend({"sample_id": remainder.get("sample_id"), "status": "NOT_ATTEMPTED_AFTER_SAFETY_STOP", "error": str(exc)} for remainder in inputs[index + 1 :])
                break
            except Exception as exc:
                failures.append({"sample_id": item.get("sample_id"), "status": "FAILED", "error": str(exc)})
        atomic_write_text(run_root / "azure/rationale.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records))
        atomic_write_json(run_root / "azure/rationale_failures.json", failures)
        _write_azure_manifests(run_root, entry, len(inputs), len(records), len(failures), usage, settings, synthetic=False, failures=failures)
        if failures:
            return StageOutcome.failed(f"Azure execution rejected {len(failures)} of {len(inputs)} requests")
        return StageOutcome.passed(summary={"azure_request_count": len(inputs), "azure_invalid_output_rate": 0.0}, expected_files=tuple(_azure_required_files(entry)))
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
    try:
        client.preflight_logical_requests(len(rows))
    except AzureSafetyBudgetError as exc:
        return StageOutcome.blocked(str(exc))
    for index, row in enumerate(rows):
        prompt = _render_azure_prompt(prompt_task, str(row.get("text", "")), demonstrations, schema)
        try:
            result = client.create_structured(prompt=prompt, task=prompt_task, schema=schema, max_output_tokens=128 if prompt_task == "pragmatic" else 32, sample_id=str(row["sample_id"]), input_payload=row)
            _persist_azure_usage_record(run_root, sample_id=str(row["sample_id"]), result=result)
            labels = result["labels"]
            records.append({"sample_id": row["sample_id"], "split": "test", "system_id": entry.system_id, "seed": entry.seed, "gold": {key: row[key] for key in labels if key in row}, "predictions": labels, "probabilities": {}, "invalid_status": False, "failure_reason": None})
            usage.append({**dict(result.get("usage", {})), "retry_count": result.get("retry_count", 0), "cache_hit": result.get("cache_hit", False)})
        except AzureSafetyBudgetError as exc:
            failures.append({"sample_id": row.get("sample_id"), "status": "SAFETY_BUDGET_EXCEEDED", "error": str(exc)})
            failures.extend({"sample_id": remainder.get("sample_id"), "status": "NOT_ATTEMPTED_AFTER_SAFETY_STOP", "error": str(exc)} for remainder in rows[index + 1 :])
            break
        except Exception as exc:
            failures.append({"sample_id": row.get("sample_id"), "status": "INVALID", "error": str(exc)})
    atomic_write_text(run_root / "predictions/test_predictions.jsonl", "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records))
    _write_azure_manifests(run_root, entry, len(rows), len(records), len(failures), usage, settings, synthetic=False, failures=failures)
    if failures:
        return StageOutcome.failed(f"Azure execution rejected {len(failures)} of {len(rows)} requests")
    return StageOutcome.passed(summary={"azure_request_count": len(rows), "azure_invalid_output_rate": 0.0}, expected_files=tuple(_azure_required_files(entry)))


def _azure_validate(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    required = [run_root / name for name in _azure_required_files(entry)]
    missing = [str(path.relative_to(run_root)) for path in required if not path.exists()]
    if missing:
        return StageOutcome.failed("Azure response validation missing: " + "; ".join(missing))
    response = json.loads((run_root / "azure/response_manifest.json").read_text(encoding="utf-8"))
    if int(response.get("requested", 0)) != int(response.get("successful", 0)) + int(response.get("invalid", 0)) + int(response.get("missing", 0)) + int(response.get("failed", 0)):
        return StageOutcome.failed("Azure response accounting does not close over requested requests")
    if not context.fixture:
        ledger = _load_mapping(run_root / "azure/cost_ledger.json")
        if ledger.get("cost_accounting_method") != AZURE_COST_ACCOUNTING_METHOD or ledger.get("cost_verification_status") != AZURE_COST_VERIFICATION_STATUS:
            return StageOutcome.failed("Azure cost ledger does not use the authoritative local-usage accounting policy")
        if ledger.get("status") != "PASS":
            return StageOutcome.failed(f"Azure cost ledger is not complete: {ledger.get('status', 'UNKNOWN')}")
        if int(response.get("invalid", 0) or 0) or int(response.get("missing", 0) or 0) or int(response.get("failed", 0) or 0):
            return StageOutcome.failed("Azure response validation found invalid, missing, or failed responses")
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
        "generate_dev_reasoning": lambda: _generation_stage(context, entry, "generate_dev_reasoning") if entry.system_id == "cot_only_vistral" else _explanation_stage(context, entry, "generate_dev_reasoning_from_rationale_decoder"),
        "judge_dev_reasoning": lambda: _generation_stage(context, entry, "judge_dev_reasoning") if entry.system_id == "cot_only_vistral" else _explanation_stage(context, entry, "judge_dev_reasoning"),
        "compute_dev_reasoning_metrics": lambda: _generation_stage(context, entry, "compute_dev_reasoning_metrics") if entry.system_id == "cot_only_vistral" else _explanation_stage(context, entry, "compute_dev_reasoning_metrics"),
        "generate_test_reasoning": lambda: _generation_stage(context, entry, "generate_test_reasoning") if entry.system_id == "cot_only_vistral" else _explanation_stage(context, entry, "generate_test_reasoning_from_rationale_decoder"),
        "judge_test_reasoning": lambda: _generation_stage(context, entry, "judge_test_reasoning") if entry.system_id == "cot_only_vistral" else _explanation_stage(context, entry, "judge_test_reasoning"),
        "compute_test_reasoning_metrics": lambda: _generation_stage(context, entry, "compute_test_reasoning_metrics") if entry.system_id == "cot_only_vistral" else _explanation_stage(context, entry, "compute_test_reasoning_metrics"),
        "resolve_approved_full_vistral_source": lambda: _explanation_stage(context, entry, "resolve_approved_full_vistral_source"),
        "validate_source_checkpoint": lambda: _explanation_stage(context, entry, "validate_source_checkpoint"),
        "generate_dev_reasoning_from_rationale_decoder": lambda: _explanation_stage(context, entry, "generate_dev_reasoning_from_rationale_decoder"),
        "generate_test_reasoning_from_rationale_decoder": lambda: _explanation_stage(context, entry, "generate_test_reasoning_from_rationale_decoder"),
        "generate_dev": lambda: _generation_stage(context, entry, "generate_dev_reasoning"),
        "parse_dev": lambda: _generation_stage(context, entry, "judge_dev_reasoning"),
        "generate_test": lambda: _generation_stage(context, entry, "generate_test_reasoning"),
        "parse_test": lambda: _generation_stage(context, entry, "judge_test_reasoning"),
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
