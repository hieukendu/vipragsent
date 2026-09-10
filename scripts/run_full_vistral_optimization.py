"""Run and evaluate isolated warm-start optimization trials for full Vistral.

Trials are deliberately kept outside the canonical Q1a run directories.  The
training command never reads the test split; the evaluate command can read it
only after the dev-selected checkpoint and thresholds have been frozen.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.constants import PRAGMATIC_LABELS
from vipragsent.data.collation import BatchCollator
from vipragsent.data.loaders import DatasetExample, load_vipragsent
from vipragsent.data.tokenizers import create_tokenizer
from vipragsent.evaluation.metrics import binary_macro_f1, macro_pragmatic_f1
from vipragsent.hashing import sha256_file, sha256_json
from vipragsent.models.factory import build_production_model
from vipragsent.models.variants import VariantConfig
from vipragsent.orchestration.stage_registry import _build_production_preprocessor
from vipragsent.runtime.model_assets import read_family_status, resolve_local_snapshot
from vipragsent.training.checkpoints import infer_required_head_prefixes, load_checkpoint
from vipragsent.training.class_weights import compute_train_only_class_weights
from vipragsent.training.engine import RunState, TrainingConfig, TrainingEngine
from vipragsent.training.seeding import seed_everything

FAMILY = "vistral_7b"
VARIANT = "vipragsent_full_vistral"
MODEL_REVISION = "d331b64e61b935cc43c2b3010ae9fb4fde599b45"
TOKENIZER_REVISION = MODEL_REVISION
SEEDS = (20260521, 20260522, 20260523)
SOURCE_RUNS = {seed: f"q1a_vipragsent_full_vistral_{seed}" for seed in SEEDS}
SOURCE_REPOSITORIES = {
    20260521: "Thundergod2007/vipragsent-experiment-artifacts-overflow-017",
    20260522: "Thundergod2007/vipragsent-experiment-artifacts-overflow-017",
    20260523: "Thundergod2007/vipragsent-experiment-artifacts-overflow-016",
}
SOURCE_REMOTE_PATHS = {
    seed: (
        "campaigns/vipragsent-v7-6693e7e9eef08ef1/SHARD_GPU40_7B_INTEGRATION/"
        f"q1a_vipragsent_full_vistral_{seed}/checkpoints/best/model.pt"
    )
    for seed in SEEDS
}
EXPECTED_SOURCE_HASHES = {
    seed: json.loads(
        (ROOT / "results/runs" / SOURCE_RUNS[seed] / "checkpoints/checkpoint_manifest.json").read_text(encoding="utf-8")
    )["checkpoint_sha256"].lower()
    for seed in SEEDS
}


def _json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default
    except (OSError, json.JSONDecodeError):
        return default


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _source_checkpoint(seed: int) -> Path:
    repository = SOURCE_REPOSITORIES[seed].replace("/", "--")
    pattern = (
        ROOT
        / "hub"
        / f"models--{repository}"
        / "snapshots"
    )
    matches = sorted(
        pattern.glob(
            "*/campaigns/vipragsent-v7-6693e7e9eef08ef1/SHARD_GPU40_7B_INTEGRATION/"
            f"q1a_vipragsent_full_vistral_{seed}/checkpoints/best/model.pt"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one cached source checkpoint for seed {seed}, found {len(matches)}"
        )
    path = matches[0]
    actual = sha256_file(path).lower()
    expected = EXPECTED_SOURCE_HASHES[seed]
    if actual != expected:
        raise RuntimeError(f"source checkpoint hash mismatch for seed {seed}: {actual} != {expected}")
    return path


def _snapshot() -> Path:
    status = read_family_status(ROOT, FAMILY, "cache")
    snapshot = resolve_local_snapshot(ROOT, status.get("local_path"))
    if snapshot is None or not snapshot.exists():
        raise RuntimeError("the pinned local Vistral snapshot is unavailable")
    return snapshot


def _load_rationale_records() -> None:
    # The first optimization trial intentionally omits rationale tensors so
    # the additional update is focused on the six reported classification heads.
    return None


def _batches(
    split_rows: list[DatasetExample],
    *,
    tokenizer: Any,
    preprocessor: Any,
    weights: Any,
    physical_batch_size: int,
    rationale_records: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    collator = BatchCollator(
        tokenizer,
        preprocessor,
        class_weights=weights.as_dict(),
        rationale_records=rationale_records,
        rationale_target_max_length=160,
    )
    return [
        collator(split_rows[index : index + physical_batch_size])
        for index in range(0, len(split_rows), physical_batch_size)
    ]


def _data_context(physical_batch_size: int, *, include_train: bool = True) -> tuple[Any, Any, Any, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    bundle = load_vipragsent(ROOT / "data/processed/vipragsent")
    snapshot = _snapshot()
    tokenizer = create_tokenizer(
        FAMILY,
        revision=TOKENIZER_REVISION,
        local_path=snapshot,
        execution_mode="production",
    )
    preprocessor = _build_production_preprocessor(
        FAMILY,
        preprocessing_name="vncorenlp_rdrsegmenter",
        preprocessing_version="locked-v1",
        tokenizer_revision=TOKENIZER_REVISION,
        model_revision=MODEL_REVISION,
    )
    weights = compute_train_only_class_weights(
        bundle.train,
        dataset_hash=bundle.fingerprint,
        code_commit=_git_commit(),
    )
    rationale_records = _load_rationale_records()
    train_batches = (
        _batches(
            bundle.train,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            weights=weights,
            physical_batch_size=physical_batch_size,
            rationale_records=rationale_records,
        )
        if include_train
        else []
    )
    evaluation_records = None
    dev_batches = _batches(
        bundle.dev,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=physical_batch_size,
        rationale_records=evaluation_records,
    )
    test_batches = _batches(
        bundle.test,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=physical_batch_size,
        rationale_records=evaluation_records,
    )
    return bundle, weights, snapshot, train_batches, dev_batches, test_batches


def _trial_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()
    if not normalized:
        raise ValueError("trial name cannot be empty")
    return normalized


def _config(args: argparse.Namespace, *, qlora: dict[str, Any]) -> TrainingConfig:
    physical = int(args.physical_batch_size)
    effective = int(args.effective_batch_size)
    if effective % physical:
        raise ValueError("effective batch size must be divisible by physical batch size")
    return TrainingConfig(
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        max_epochs=int(args.max_epochs),
        effective_batch_size=effective,
        physical_batch_size=physical,
        max_grad_norm=1.0,
        patience=int(args.patience),
        min_delta=0.0001,
        precision="bf16",
        gradient_accumulation_steps=effective // physical,
        primary_metric="dev_macro_pragmatic_f1",
        scheduler=str(args.scheduler),
        warmup_ratio=float(args.warmup_ratio),
        use_uncertainty_weighting=True,
        rationale_beta=0.0,
        rationale_target_max_length=160,
        optimizer="paged_adamw_8bit",
        gradient_checkpointing=True,
        deterministic_algorithms="warn_only",
        qlora=qlora,
    )


def _engine(
    *,
    output_root: Path,
    config: TrainingConfig,
    weights: Any,
    snapshot: Path,
    trial: str,
    selected_device: int,
    seed: int | None = None,
) -> tuple[TrainingEngine, Any]:
    if seed is not None:
        # Keep fresh runs reproducible before heads and LoRA adapters exist.
        seed_everything(int(seed))
    model, spec = build_production_model(
        FAMILY,
        VARIANT,
        local_snapshot=snapshot,
        execution_mode="production",
        selected_device=selected_device,
    )
    resolved = {
        "system_id": f"{VARIANT}_optimization_{trial}",
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
        "rationale_training": False,
        "rationale_beta": config.rationale_beta,
        "rationale_target_max_length": config.rationale_target_max_length,
        "rationale_inference": False,
        "qlora": config.qlora,
        "selection_metric": config.primary_metric,
        "optimization_trial": trial,
    }
    engine = TrainingEngine(
        model,
        config,
        run_id="model",
        checkpoint_root=output_root / "_engine_checkpoints",
        class_weights=weights,
        resolved_config=resolved,
        selected_device=selected_device,
    )
    return engine, spec


def _set_focus(model: Any, focus: str) -> None:
    if focus == "full":
        return
    if focus != "pragmatic":
        raise ValueError(f"unsupported focus: {focus}")
    current = model.config
    model.config = VariantConfig(
        "vipragsent_no_auxiliary_vistral",
        backbone_family=current.backbone_family,
        hidden_size=current.hidden_size,
        vocab_size=current.vocab_size,
        rationale_vocab_size=current.rationale_vocab_size,
        rationale_enabled_for_training=current.has_rationale_decoder,
        use_uncertainty_weighting=True,
    )


def _pragmatic_metrics(selection: Any) -> dict[str, Any]:
    per_label = {
        label: binary_macro_f1(selection.true[label], selection.predictions[label])
        for label in PRAGMATIC_LABELS
    }
    return {
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


def _write_prediction(path: Path, engine: TrainingEngine, selection: Any, batches: list[dict[str, Any]]) -> None:
    ids = [sample_id for batch in batches for sample_id in batch.get("sample_ids", [])]
    engine._write_prediction_jsonl(path, engine._prediction_rows(selection, sample_ids=ids))


def train(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    if seed not in SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    trial = _trial_name(args.trial)
    output_root = ROOT / "results/runs" / f"q1a_vipragsent_full_vistral_optimization_{trial}_{seed}"
    if output_root.exists() and not args.force:
        raise FileExistsError(f"candidate output already exists; use --force only for an explicit rerun: {output_root}")
    if args.force and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source_checkpoint = _source_checkpoint(seed) if args.init == "source" else None
    bundle, weights, snapshot, train_batches, dev_batches, _ = _data_context(
        int(args.physical_batch_size), include_train=True
    )
    config = _config(args, qlora={"rank": 16, "alpha": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"]})
    engine, spec = _engine(
        output_root=output_root,
        config=config,
        weights=weights,
        snapshot=snapshot,
        trial=trial,
        selected_device=int(args.device),
        seed=seed,
    )
    source_load = None
    if source_checkpoint is not None:
        source_load = load_checkpoint(
            source_checkpoint,
            engine.model,
            loss_aggregator=engine.loss_aggregator,
            required_head_prefixes=infer_required_head_prefixes(engine.model),
            report_path=output_root / "source_checkpoint_load_report.json",
        )
    initial_model_hash = sha256_json(sorted(engine.model.state_dict()))
    _set_focus(engine.model, args.focus)
    initial_selection = engine._evaluate_dev(dev_batches)
    updates_per_epoch = (
        len(train_batches) + config.gradient_accumulation_steps - 1
    ) // config.gradient_accumulation_steps
    engine._ensure_scheduler(updates_per_epoch * config.max_epochs)
    initial_state = RunState(
        epoch=0,
        best_metric=initial_selection.metric,
        best_loss=initial_selection.total_loss,
        best_epoch=0,
        status="RUNNING",
        thresholds=dict(initial_selection.thresholds),
    )
    # Persist the measured starting point so a bad update cannot replace it.
    engine.checkpoints.save("best", engine.model, engine.optimizer, engine.scheduler, engine.loss_aggregator, initial_state)
    engine.checkpoints.save("epoch_000", engine.model, engine.optimizer, engine.scheduler, engine.loss_aggregator, initial_state)

    manifest = {
        "schema_version": 1,
        "status": "RUNNING",
        "run_id": output_root.name,
        "trial": trial,
        "seed": seed,
        "focus": args.focus,
        "source_run_id": SOURCE_RUNS[seed],
        "initialization": args.init,
        "source_repository": SOURCE_REPOSITORIES[seed] if source_checkpoint is not None else None,
        "source_remote_path": SOURCE_REMOTE_PATHS[seed] if source_checkpoint is not None else None,
        "source_checkpoint": str(source_checkpoint.relative_to(ROOT)) if source_checkpoint is not None else None,
        "source_checkpoint_sha256": sha256_file(source_checkpoint) if source_checkpoint is not None else None,
        "source_manifest_sha256": EXPECTED_SOURCE_HASHES[seed] if source_checkpoint is not None else None,
        "source_checkpoint_load_status": source_load.report.status if source_load is not None else "NOT_APPLICABLE",
        "source_checkpoint_matched_ratio": source_load.report.matched_ratio if source_load is not None else None,
        "source_checkpoint_missing_keys": list(source_load.report.missing_keys) if source_load is not None else [],
        "source_checkpoint_unexpected_keys": list(source_load.report.unexpected_keys) if source_load is not None else [],
        "source_model_state_key_fingerprint": initial_model_hash,
        "base_model_repository": spec.repo_id,
        "base_model_revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "dataset_fingerprint": bundle.fingerprint,
        "dataset_split_sizes": {key: len(value) for key, value in bundle.splits.items()},
        "code_commit": _git_commit(),
        "training_config": asdict(config),
        "selection_protocol": "dev_only_frozen_threshold_v1",
        "test_evaluated_during_training": False,
        "initial_dev": _pragmatic_metrics(initial_selection),
    }
    atomic_write_json(output_root / "optimization_manifest.json", manifest)
    atomic_write_json(output_root / "training/class_weights.json", weights.as_dict())
    atomic_write_json(output_root / "training/resolved_training_config.json", asdict(config))
    atomic_write_json(output_root / "selection/initial_dev.json", _pragmatic_metrics(initial_selection))
    state = engine.train(
        train_batches,
        seed=seed,
        dev_batches=dev_batches,
        test_batches=None,
        resume=True,
        output_root=output_root / "_engine_output",
        run_metadata={
            "mode": "warm_start_optimization" if source_checkpoint is not None else "fresh_base_optimization",
            "trial": trial,
            "focus": args.focus,
            "source_run_id": SOURCE_RUNS[seed] if source_checkpoint is not None else None,
            "source_checkpoint_sha256": sha256_file(source_checkpoint) if source_checkpoint is not None else None,
            "test_evaluated_during_training": False,
        },
    )
    final_dev = engine._evaluate_dev(dev_batches, thresholds_override=state.thresholds)
    atomic_write_json(output_root / "metrics/dev_metrics.json", _pragmatic_metrics(final_dev))
    _write_prediction(output_root / "predictions/dev_predictions.jsonl", engine, final_dev, dev_batches)
    atomic_write_json(
        output_root / "selection/best_checkpoint.json",
        {
            "status": "PASS",
            "path": "_engine_checkpoints/model/best.pt",
            "best_epoch": state.best_epoch,
            "best_dev_metric": state.best_metric,
            "thresholds": state.thresholds,
        },
    )
    manifest.update(
        {
            "status": state.status,
            "best_epoch": state.best_epoch,
            "best_dev": _pragmatic_metrics(final_dev),
            "selected_thresholds": dict(state.thresholds),
            "history": state.history,
            "test_evaluated_during_training": False,
        }
    )
    atomic_write_json(output_root / "optimization_manifest.json", manifest)
    print(json.dumps({"status": state.status, "run_root": str(output_root), "best_dev": manifest["best_dev"]}, indent=2))
    return 0


def evaluate(args: argparse.Namespace) -> int:
    output_root = Path(args.run_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    manifest = _json(output_root / "optimization_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("status") != "PASS":
        raise RuntimeError(f"candidate training manifest is not complete: {output_root}")
    seed = int(manifest["seed"])
    config_data = dict(manifest["training_config"])
    config_data["qlora"] = dict(config_data.get("qlora") or {})
    config = TrainingConfig(**config_data)
    bundle, weights, snapshot, _, dev_batches, test_batches = _data_context(
        config.physical_batch_size, include_train=False
    )
    engine, spec = _engine(
        output_root=output_root,
        config=config,
        weights=weights,
        snapshot=snapshot,
        trial=str(manifest["trial"]),
        selected_device=int(args.device),
        seed=seed,
    )
    best_checkpoint = output_root / "_engine_checkpoints/model/best.pt"
    load_result = load_checkpoint(
        best_checkpoint,
        engine.model,
        loss_aggregator=engine.loss_aggregator,
        required_head_prefixes=infer_required_head_prefixes(engine.model),
        report_path=output_root / "evaluation_checkpoint_load_report.json",
    )
    _set_focus(engine.model, str(manifest.get("focus", "full")))
    thresholds = dict(manifest.get("selected_thresholds") or _json(output_root / "_engine_output/thresholds.json", {}))
    if set(thresholds) != set(PRAGMATIC_LABELS):
        raise RuntimeError("candidate does not contain a complete frozen pragmatic threshold set")
    engine.gate.freeze_checkpoint()
    test_selection = engine._evaluate_dev(test_batches, thresholds_override=thresholds)
    metrics = _pragmatic_metrics(test_selection)
    atomic_write_json(output_root / "metrics/test_metrics.json", metrics)
    _write_prediction(output_root / "predictions/test_predictions.jsonl", engine, test_selection, test_batches)
    atomic_write_json(
        output_root / "evaluation_manifest.json",
        {
            "status": "PASS",
            "run_id": output_root.name,
            "seed": seed,
            "source_run_id": manifest["source_run_id"],
            "checkpoint": "_engine_checkpoints/model/best.pt",
            "checkpoint_sha256": sha256_file(best_checkpoint),
            "checkpoint_load_status": load_result.report.status,
            "checkpoint_matched_ratio": load_result.report.matched_ratio,
            "thresholds": thresholds,
            "selection_protocol": "dev_only_frozen_threshold_v1",
            "test_evaluated_after_dev_freeze": True,
            "test_selection_used_for_any_decision": False,
            "dataset_fingerprint": bundle.fingerprint,
            "base_model_repository": spec.repo_id,
            "base_model_revision": spec.revision,
            "tokenizer_revision": spec.tokenizer_revision,
            "metrics": metrics,
        },
    )
    print(json.dumps({"status": "PASS", "run_root": str(output_root), "test": metrics}, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    train_parser.add_argument("--trial", required=True)
    train_parser.add_argument("--focus", choices=("full", "pragmatic"), default="full")
    train_parser.add_argument("--init", choices=("source", "base"), default="source")
    train_parser.add_argument("--learning-rate", type=float, default=2e-5)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--scheduler", choices=("linear", "cosine"), default="cosine")
    train_parser.add_argument("--warmup-ratio", type=float, default=0.05)
    train_parser.add_argument("--physical-batch-size", type=int, choices=(2, 4, 8), default=4)
    train_parser.add_argument("--effective-batch-size", type=int, default=16)
    train_parser.add_argument("--max-epochs", type=int, default=1)
    train_parser.add_argument("--patience", type=int, default=1)
    train_parser.add_argument("--device", type=int, default=0)
    train_parser.add_argument("--force", action="store_true")
    train_parser.set_defaults(handler=train)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--run-root", required=True)
    evaluate_parser.add_argument("--device", type=int, default=0)
    evaluate_parser.set_defaults(handler=evaluate)
    return parser


if __name__ == "__main__":
    parsed = _parser().parse_args()
    raise SystemExit(parsed.handler(parsed))
