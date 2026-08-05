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
from ..training.engine import TrainingConfig, TrainingEngine
from .contracts import (
    ExecutionKind,
    RunContext,
    RunEntry,
    StageOutcome,
)
from .preflight_single import run_single_preflight
from .run_store import RunStore, artifact_hashes, git_commit, utc_now

StageHandler = Callable[[], StageOutcome]


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    atomic_write_json(path, dict(payload))
    return path.as_posix()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _copy(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _entry_variant(entry: RunEntry) -> str:
    if entry.system_id == "vipragsent_no_auxiliary_vistral":
        return entry.system_id
    if "pol_single" in entry.system_id:
        return "phobert_pol_single"
    if "emo_single" in entry.system_id:
        return "phobert_emo_single"
    if "no_multitask" in entry.system_id:
        return "no_multitask"
    if "no_emotion" in entry.system_id:
        return "no_emotion_auxiliary"
    if "no_polarity" in entry.system_id:
        return "no_polarity_auxiliary"
    if "no_rationale" in entry.system_id:
        return "no_rationale"
    if entry.system_id in {"vipragsent_full_vistral", "vipragsent_full_phobert"}:
        return entry.system_id
    if "vistral" in entry.system_id:
        return "vistral_pragmatic_sft"
    if "sailor" in entry.system_id:
        return "sailor_pragmatic_sft"
    if "pragmatic" in entry.system_id:
        return "phobert_pragmatic_finetune"
    return entry.system_id


def _active_tasks(entry: RunEntry) -> set[str]:
    name = _entry_variant(entry)
    if name in {"phobert_pol_single"}:
        return {"polarity"}
    if name in {"phobert_emo_single"}:
        return {"emotion"}
    if name == "vipragsent_no_auxiliary_vistral":
        return {"pragmatic"}
    if name == "no_emotion_auxiliary":
        return {"pragmatic", "polarity"}
    if name == "no_polarity_auxiliary":
        return {"pragmatic", "emotion"}
    if "pragmatic" in name or "sft" in name or "pragmatic" in entry.task:
        return {"pragmatic"}
    return {"pragmatic", "polarity", "emotion"}


def _metric_name(entry: RunEntry) -> str:
    selection = str(entry.raw.get("selection_metric") or entry.raw.get("primary_dev_selection_metric") or "")
    if "polarity" in selection or "polarity" in entry.task:
        return "dev_polarity_macro_f1"
    if "emotion" in selection or ("emotion" in entry.task and "pragmatic" not in entry.task):
        return "dev_emotion_macro_f1"
    if entry.research_question == "Q3" or "sarcasm" in selection:
        return "dev_sarcasm_macro_f1"
    return "dev_macro_pragmatic_f1"


def _fixture_batches(entry: RunEntry, split: str, *, batch_size: int = 4) -> list[dict[str, Any]]:
    tasks = _active_tasks(entry)
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
            batch["pragmatic_pos_weight"] = {key: torch.tensor(1.0) for key in PRAGMATIC_LABELS}
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
    active = [key for key in PRAGMATIC_LABELS if true[key]]
    if active:
        output["per_label_f1"] = {key: binary_macro_f1(true[key], pred[key]) for key in active}
        output["macro_pragmatic_f1"] = float(np.mean(list(output["per_label_f1"].values())))
        output["raw_positive_probabilities"] = probabilities
        output["gold_pragmatic"] = true
    else:
        output["per_label_f1"] = {}
        output["macro_pragmatic_f1"] = "NOT_APPLICABLE"
    return output


def _fixture_train(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    family = "causal" if entry.backbone in {"sailor_7b", "vistral_7b"} else "encoder"
    config = VariantConfig(name=_entry_variant(entry), backbone_family=family, hidden_size=12, vocab_size=32, rationale_enabled_for_training=False)
    model = build_dummy_model(config)
    output_root = run_root / "_engine_output"
    engine = TrainingEngine(
        model,
        TrainingConfig(
            learning_rate=0.05,
            weight_decay=0.0,
            max_epochs=2,
            physical_batch_size=4,
            effective_batch_size=4,
            gradient_accumulation_steps=1,
            precision="fp32",
            primary_metric=_metric_name(entry),
            patience=10,
            use_uncertainty_weighting=config.has_uncertainty_weighting,
        ),
        run_id="model",
        checkpoint_root=run_root / "_engine_checkpoints",
    )
    train_batches = _fixture_batches(entry, "train")
    dev_batches = _fixture_batches(entry, "dev")
    test_batches = _fixture_batches(entry, "test")
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
    atomic_write_json(run_root / "training/optimizer_summary.json", {"optimizer": "AdamW", "learning_rate": 0.05, "weight_decay": 0.0, "fixture": True})
    atomic_write_json(run_root / "training/scheduler_summary.json", {"scheduler": "linear", "warmup_ratio": 0.1, "fixture": True})
    atomic_write_json(run_root / "training/resource_usage.json", {"fixture": True, "successful_gpu_hours": 0.0, "failed_or_retried_gpu_hours": 0.0, "peak_vram_gb": 0.0})
    engine_checkpoint = run_root / "_engine_checkpoints/model/best.pt"
    best_checkpoint = run_root / "checkpoints/best/model.pt"
    latest_checkpoint = run_root / "checkpoints/latest/model.pt"
    _copy(engine_checkpoint, best_checkpoint)
    _copy(run_root / "_engine_checkpoints/model/epoch_002.pt", latest_checkpoint)
    checkpoint_hash = sha256_file(best_checkpoint)
    atomic_write_json(run_root / "checkpoints/checkpoint_manifest.json", {"status": "PASS", "best": _relative(best_checkpoint, run_root), "latest": _relative(latest_checkpoint, run_root), "checkpoint_sha256": checkpoint_hash, "model_revision": "fixture", "variant_fingerprint": sha256_json({"variant": _entry_variant(entry), "tasks": sorted(_active_tasks(entry))})})
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
    family = entry.backbone
    cache = read_family_status(root, family, "cache")
    snapshot = cache.get("local_path")
    if not snapshot:
        return StageOutcome.blocked(f"Phase 15 local snapshot is unavailable for {family}")
    model, spec = build_production_model(family, _entry_variant(entry), local_snapshot=snapshot, execution_mode="production")
    tokenizer = create_tokenizer(family, revision=spec.tokenizer_revision, local_path=snapshot, execution_mode="production")
    bundle = load_vipragsent(root / "data/processed/vipragsent")
    preprocessor = TextPreprocessor(PreprocessingSpec(family, entry.preprocessing_name or "vncorenlp_rdrsegmenter", entry.preprocessing_version or "locked-v1", tokenizer_revision=spec.tokenizer_revision, model_revision=spec.revision, execution_mode="production"))
    collator = BatchCollator(tokenizer, preprocessor)
    batch_size = int(read_family_status(root, family, "batch").get("successful_batch") or 1)
    def batches(examples: list[DatasetExample]) -> list[dict[str, Any]]:
        return [collator(examples[index:index + batch_size]) for index in range(0, len(examples), batch_size)]
    train_batches, dev_batches, test_batches = batches(bundle.train), batches(bundle.dev), batches(bundle.test)
    config = TrainingConfig(primary_metric=_metric_name(entry), physical_batch_size=batch_size, effective_batch_size=int(entry.raw.get("effective_batch_size") or 32), precision=str(entry.raw.get("precision") or "bf16"))
    engine = TrainingEngine(model, config, run_id="model", checkpoint_root=Path(context.run_root) / "_engine_checkpoints")
    state = engine.train(train_batches, seed=int(entry.seed), dev_batches=dev_batches, test_batches=test_batches, output_root=Path(context.run_root) / "_engine_output", run_metadata={"mode": "full", "model_revision": spec.revision, "tokenizer_revision": spec.tokenizer_revision, "model_repository": spec.repo_id})
    # Production uses the same engine outputs and checkpoint contract as the fixture adapter.
    return _materialize_engine_outputs(context, entry, state, model_revision=spec.revision, tokenizer_revision=spec.tokenizer_revision)


def _materialize_engine_outputs(context: RunContext, entry: RunEntry, state: Any, *, model_revision: str, tokenizer_revision: str) -> StageOutcome:
    run_root = Path(context.run_root)
    output_root = run_root / "_engine_output"
    _copy(output_root / "dev_predictions.jsonl", run_root / "predictions/dev_predictions.jsonl")
    _copy(output_root / "test_predictions.jsonl", run_root / "predictions/test_predictions.jsonl")
    atomic_write_json(run_root / "training/history.json", state.history)
    _csv_history(run_root / "training/history.csv", state.history)
    atomic_write_json(run_root / "training/optimizer_summary.json", {"optimizer": "AdamW", "learning_rate": "locked_config", "weight_decay": "locked_config"})
    atomic_write_json(run_root / "training/scheduler_summary.json", {"scheduler": "locked_config", "warmup_ratio": "locked_config"})
    atomic_write_json(run_root / "training/resource_usage.json", {"fixture": False, "successful_gpu_hours": "measured", "failed_or_retried_gpu_hours": "measured"})
    checkpoint = run_root / "_engine_checkpoints/model/best.pt"
    latest = sorted((run_root / "_engine_checkpoints/model").glob("epoch_*.pt"))[-1]
    best = run_root / "checkpoints/best/model.pt"
    latest_target = run_root / "checkpoints/latest/model.pt"
    _copy(checkpoint, best)
    _copy(latest, latest_target)
    digest = sha256_file(best)
    atomic_write_json(run_root / "checkpoints/checkpoint_manifest.json", {"status": "PASS", "best": _relative(best, run_root), "latest": _relative(latest_target, run_root), "checkpoint_sha256": digest, "model_revision": model_revision, "tokenizer_revision": tokenizer_revision})
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
    """Create source-shaped Q4 outputs without invoking training."""
    run_root = Path(context.run_root)
    rows: list[dict[str, Any]] = []
    for split in ("dev", "test"):
        for index in range(8):
            gold = {label: int((index + offset) % 3 == 0) for offset, label in enumerate(PRAGMATIC_LABELS)}
            probabilities = {label: round(0.2 + 0.1 * ((index + offset) % 6), 4) for offset, label in enumerate(PRAGMATIC_LABELS)}
            predictions = {label: int(probabilities[label] >= 0.5) for label in PRAGMATIC_LABELS}
            rows.append({"sample_id": f"fixture_{entry.run_id}_{split}_{index}", "split": split, "gold": gold, "probabilities": probabilities, "predictions": predictions})
        split_rows = [row for row in rows if row["split"] == split]
        atomic_write_text(run_root / f"predictions/{split}_predictions.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in split_rows))
        atomic_write_json(run_root / f"metrics/{split}_metrics.json", _metrics_from_rows(run_root / f"predictions/{split}_predictions.jsonl"))
    source_id = str(entry.source_checkpoint_id or entry.system_id)
    source_hash = sha256_json({"source_checkpoint_id": source_id, "fixture": True})
    atomic_write_json(run_root / "checkpoint_reference.json", {"source": source_id, "source_sha256": source_hash, "source_approval_required": True, "source_status": "FIXTURE_SOURCE_ONLY", "training_applicability": "NOT_APPLICABLE", "not_applicable_reason": "Q4 extracts raw pragmatic probabilities from an approved upstream checkpoint; it does not train."})
    atomic_write_json(run_root / "selection/best_checkpoint.json", {"path": "checkpoint_reference.json", "sha256": source_hash, "best_epoch": "NOT_APPLICABLE"})
    atomic_write_json(run_root / "selection/selection_metric.json", {"name": "dev_macro_pragmatic_f1", "value": _metrics_from_rows(run_root / "predictions/dev_predictions.jsonl").get("macro_pragmatic_f1", 0.0), "best_epoch": "NOT_APPLICABLE"})
    atomic_write_json(run_root / "selection/thresholds.json", {label: 0.5 for label in PRAGMATIC_LABELS})
    atomic_write_json(run_root / "training/history.json", [{"epoch": "NOT_APPLICABLE", "dev_macro_pragmatic_f1": _metrics_from_rows(run_root / "predictions/dev_predictions.jsonl").get("macro_pragmatic_f1", 0.0), "dev_loss": "NOT_APPLICABLE", "seconds": 0.0, "source_checkpoint_reuse": True}])
    return StageOutcome.passed(summary={"training_applicability": "NOT_APPLICABLE", "not_applicable_reason": "Q4 raw-probability extraction from an approved source; no training was run."}, expected_files=("checkpoint_reference.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json"))


def _preflight(context: RunContext, entry: RunEntry) -> StageOutcome:
    report = run_single_preflight(context.root, entry, kind="azure" if entry.is_azure else "experiment", run_id=entry.run_id, fixture=context.fixture, dry_run=context.dry_run)
    atomic_write_json(Path(context.run_root) / "preflight.json", report)
    if report["passed"]:
        return StageOutcome.passed(summary=report, expected_files=("preflight.json",))
    return StageOutcome.blocked(*report["blockers"])


def _evaluate_dev(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    path = run_root / "predictions/dev_predictions.jsonl"
    if not path.exists():
        return StageOutcome.blocked("dev predictions are missing")
    metrics = _metrics_from_rows(path)
    atomic_write_json(run_root / "metrics/dev_metrics.json", metrics)
    return StageOutcome.passed(summary=metrics, expected_files=("predictions/dev_predictions.jsonl", "metrics/dev_metrics.json"))


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
    metrics = _metrics_from_rows(prediction)
    metrics["thresholds_source"] = "selection/freeze_manifest.json"
    metrics["test_threshold_tuning"] = False
    atomic_write_json(run_root / "metrics/test_metrics.json", metrics)
    return StageOutcome.passed(summary=metrics, expected_files=("predictions/test_predictions.jsonl", "metrics/test_metrics.json"))


def _q4_sidecars(context: RunContext, entry: RunEntry) -> None:
    run_root = Path(context.run_root)
    prediction_path = run_root / "predictions/test_predictions.jsonl"
    if not prediction_path.exists():
        return
    rows = _read_jsonl(prediction_path)
    true = {key: [] for key in PRAGMATIC_LABELS}
    probabilities = {key: [] for key in PRAGMATIC_LABELS}
    for row in rows:
        for key in PRAGMATIC_LABELS:
            if key in row.get("gold", {}) and key in row.get("probabilities", {}):
                true[key].append(int(row["gold"][key]))
                value = row["probabilities"][key]
                probabilities[key].append(float(value[-1] if isinstance(value, list) else value))
    if not all(true[key] for key in PRAGMATIC_LABELS):
        return
    from ..evaluation.metrics import pragmatic_ece

    ece_by_label, macro_ece, bins = pragmatic_ece(true, probabilities, bins=10)
    checkpoint_id = str(entry.source_checkpoint_id or entry.system_id)
    q4_row = {
        "system": entry.system_id,
        "display_name": entry.display_name,
        "checkpoint_id": checkpoint_id,
        "seed": entry.seed,
        "split": "vipragsent_test",
        "per_label_pragmatic_ece": ece_by_label,
        "macro_pragmatic_ece": macro_ece,
        "bin_count": 10,
        "temperature_scaling": False,
        "probability_aggregation": "none",
        "prediction_file": _relative(prediction_path, context.root),
        "prediction_file_sha256": sha256_file(prediction_path),
        "config_hash": sha256_file(run_root / "config_snapshot.yaml"),
        "code_commit": git_commit(context.root),
    }
    atomic_write_json(run_root / "paper_artifacts/q4_pragmatic_calibration_per_seed.json", q4_row)
    reliability_rows = [{"system": entry.system_id, "seed": entry.seed, "label": label, **row} for label, label_rows in bins.items() for row in label_rows]
    atomic_write_json(run_root / "figure_backing/q4_pragmatic_reliability_bins.json", reliability_rows)
    history_path = run_root / "training/history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
    curves = [{"system": entry.system_id, "seed": entry.seed, "epoch": row.get("epoch"), "dev_macro_pragmatic_f1": row.get("dev_macro_pragmatic_f1"), "dev_loss": row.get("dev_loss"), "wall_seconds": row.get("seconds")} for row in history]
    atomic_write_json(run_root / "figure_backing/q4_learning_curves.json", curves)


def _export_artifacts(context: RunContext, entry: RunEntry) -> StageOutcome:
    run_root = Path(context.run_root)
    if entry.research_question == "Q4":
        _q4_sidecars(context, entry)
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
    elif entry.execution_kind in {ExecutionKind.EVALUATION_ONLY.value, ExecutionKind.CHECKPOINT_REUSE.value, ExecutionKind.ARTIFACT_EXTRACTION.value}:
        required = ["checkpoint_reference.json", "predictions/test_predictions.jsonl", "metrics/test_metrics.json"]
    else:
        required = ["training/history.csv", "training/history.json", "training/optimizer_summary.json", "training/scheduler_summary.json", "training/resource_usage.json", "checkpoints/checkpoint_manifest.json", "selection/best_checkpoint.json", "selection/selection_metric.json", "selection/thresholds.json", "predictions/dev_predictions.jsonl", "predictions/test_predictions.jsonl", "metrics/dev_metrics.json", "metrics/test_metrics.json"]
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
        "train_or_reuse": lambda: _train_or_reuse(context, entry),
        "evaluate_dev": lambda: _evaluate_dev(context, entry),
        "freeze_selection": lambda: _freeze_selection(context, entry),
        "evaluate_test": lambda: _evaluate_test(context, entry),
        "export_artifacts": lambda: _export_artifacts(context, entry),
        "validate_artifacts": lambda: _validate_artifacts(context, entry),
        "generate_review_summary": lambda: StageOutcome.passed(summary={"deferred": True}),
    }


def _train_or_reuse(context: RunContext, entry: RunEntry) -> StageOutcome:
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
