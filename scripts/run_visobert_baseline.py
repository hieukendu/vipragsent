"""Run the locked ViSoBERT no-rationale baseline for the reviewer gaps.

The baseline uses the same frozen ViPragSent split, class-weight recipe,
dev-only threshold selection, optimizer schedule, and three seeds as the
audited XLM-R protocol.  ViSoBERT's rationale decoder is intentionally not
constructed; all six pragmatic heads plus polarity and emotion remain active.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.constants import EXPECTED_SPLIT_COUNTS
from vipragsent.data.loaders import load_vipragsent
from vipragsent.data.preprocessing import PreprocessingSpec, TextPreprocessor
from vipragsent.data.tokenizers import create_tokenizer
from vipragsent.hashing import sha256_file
from vipragsent.models.factory import build_production_model
from vipragsent.runtime.model_assets import read_family_status, resolve_local_snapshot
from vipragsent.training.class_weights import compute_train_only_class_weights
from vipragsent.training.engine import TrainingEngine
from vipragsent.training.seeding import seed_everything

from run_priority12_xlmr import (
    PROTOCOL_PATH,
    RESULT_ROOT,
    SEEDS,
    _batches,
    _copy_report_bundle,
    _git_commit,
    _loss_multipliers,
    _metrics,
    _resource_snapshot,
    _validate_run,
    _write_predictions,
    _training_config,
)

FAMILY = "visobert"
MODEL_REPOSITORY = "uitnlp/visobert"
MODEL_REVISION = "196a62afad9cbe4f52a54aabad828b13f0eec59a"
MODEL_VARIANT = "no_rationale"
REPORT_ROOT = ROOT / "reports/reviewer_gaps/visobert_baseline"
CAMPAIGN_ID = os.environ.get("VIPRAGSENT_HF_CAMPAIGN", "reviewer-gaps-20261005")
HF_QUEUE = ROOT / os.environ.get("VIPRAGSENT_HF_QUEUE", "runtime/reviewer_gaps_hf_upload_queue.jsonl")


def _protocol() -> dict[str, Any]:
    payload = yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8")) or {}
    expected = {str(key): int(value) for key, value in payload["data"]["expected_split_counts"].items()}
    if expected != {key: int(value) for key, value in EXPECTED_SPLIT_COUNTS.items()}:
        raise RuntimeError(f"ViPragSent split contract differs from the frozen dataset: {expected}")
    protocol = copy.deepcopy(payload)
    protocol["experiment_name"] = "reviewer_gap_visobert_baseline"
    protocol["model"] = {
        **protocol["model"],
        "family": FAMILY,
        "repository": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
    }
    protocol["training"] = dict(protocol["training"])
    return protocol


def _snapshot() -> Path:
    status = read_family_status(ROOT, FAMILY, "cache")
    if status.get("status") != "PASS":
        raise RuntimeError(f"Pinned {FAMILY} cache is not PASS: {status.get('status')}")
    snapshot = resolve_local_snapshot(ROOT, status.get("local_path"))
    if snapshot is None or not snapshot.exists():
        raise RuntimeError("Pinned local ViSoBERT snapshot is unavailable")
    required = [snapshot / "config.json", snapshot / "sentencepiece.bpe.model"]
    model_files = (snapshot / "pytorch_model.bin", snapshot / "model.safetensors")
    if any(not path.exists() for path in required) or not any(path.exists() for path in model_files):
        raise RuntimeError("ViSoBERT snapshot is missing config, SentencePiece, or model weights")
    return snapshot


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


def _run_id(seed: int) -> str:
    return f"reviewer_visobert_baseline_{seed}"


def run_one(*, seed: int, device: int, force: bool) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"unsupported seed: {seed}")
    protocol = _protocol()
    run_id = _run_id(seed)
    run_root = RESULT_ROOT / run_id
    report_root = REPORT_ROOT / f"seed_{seed}"
    if run_root.exists() or report_root.exists():
        if not force:
            raise FileExistsError(f"ViSoBERT output exists; use --force for an explicit rerun: {run_root}")
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

    batch_probe = read_family_status(ROOT, FAMILY, "batch")
    if batch_probe.get("status") != "PASS" or not batch_probe.get("frozen"):
        raise RuntimeError(f"Pinned ViSoBERT batch probe is not frozen PASS: {batch_probe.get('status')}")
    physical_batch = int(batch_probe["successful_batch"])
    effective_batch = int(protocol["training"]["effective_batch_size"])
    if effective_batch % physical_batch:
        raise RuntimeError(f"Effective batch {effective_batch} is not divisible by ViSoBERT physical batch {physical_batch}")
    protocol["training"]["physical_batch_size"] = physical_batch
    protocol["training"]["gradient_accumulation_steps"] = effective_batch // physical_batch

    snapshot = _snapshot()
    tokenizer = create_tokenizer(
        FAMILY,
        revision=MODEL_REVISION,
        local_path=snapshot,
        execution_mode="production",
        use_fast=False,
    )
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
    config = _training_config(
        protocol,
        primary_metric="dev_macro_pragmatic_f1",
        use_uncertainty=True,
        rationale_beta_override=0.0,
    )
    train_batches = _batches(
        bundle.train,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=config.physical_batch_size,
        rationale_records=None,
        rationale_target_max_length=config.rationale_target_max_length,
    )
    dev_batches = _batches(
        bundle.dev,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=config.physical_batch_size,
        rationale_records=None,
        rationale_target_max_length=config.rationale_target_max_length,
    )

    seed_everything(seed)
    model, spec = build_production_model(
        FAMILY,
        MODEL_VARIANT,
        local_snapshot=snapshot,
        execution_mode="production",
        selected_device=device,
    )
    loss_multipliers = _loss_multipliers(protocol)
    resolved = {
        "experiment_name": protocol["experiment_name"],
        "experiment": "reviewer_gap_visobert_baseline",
        "run_id": run_id,
        "seed": seed,
        "variant_key": "visobert_baseline",
        "variant_id": MODEL_VARIANT,
        "model_family": FAMILY,
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
        "rationale": {"enabled": False, "beta": 0.0, "artifact": None},
        "batch_probe": batch_probe,
        "gradient_strategy": protocol["training"]["gradient_strategy"],
        "batch_order": protocol["training"]["batch_order"],
        "dataset": {"root": protocol["data"]["dataset_root"], "fingerprint": bundle.fingerprint, "split_sizes": split_sizes, "sample_id_order": "frozen CSV order"},
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
        "experiment": "reviewer_gap_visobert_baseline",
        "variant_key": "visobert_baseline",
        "variant_id": MODEL_VARIANT,
        "seed": seed,
        "model_family": FAMILY,
        "backbone_architecture": "ViSoBERT XLM-R encoder",
        "base_model_repository": spec.repo_id,
        "base_model_revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision,
        "local_snapshot": str(snapshot.relative_to(ROOT)),
        "dataset_fingerprint": bundle.fingerprint,
        "dataset_split_sizes": split_sizes,
        "rationale_artifact": None,
        "rationale_artifact_sha256": None,
        "rationale_record_count": 0,
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
        "uncertainty_log_variances_learned": True,
        "rationale": {"enabled": False, "beta": 0.0, "decoder": False},
        "selection_metric": "dev_macro_pragmatic_f1",
        "selection_split": "dev",
        "threshold_source_split": "dev",
        "thresholds_frozen_before_test": True,
        "test_evaluated_during_training": False,
        "test_evaluated_after_checkpoint_and_threshold_freeze": False,
        "hardware_before": _resource_snapshot(device),
        "batch_probe_reference": batch_probe,
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
        run_metadata={"experiment": "reviewer_gap_visobert_baseline", "run_id": run_id, "test_evaluated_during_training": False},
    )
    final_dev = engine._evaluate_dev(dev_batches, thresholds_override=state.thresholds)
    final_train = engine._evaluate_dev(train_batches, thresholds_override=state.thresholds)
    _write_predictions(run_root / "predictions/train_predictions.jsonl", engine, final_train, train_batches)
    _write_predictions(run_root / "predictions/dev_predictions.jsonl", engine, final_dev, dev_batches)
    atomic_write_json(run_root / "metrics/train_metrics.json", _metrics(final_train))
    atomic_write_json(run_root / "metrics/dev_metrics.json", _metrics(final_dev))
    atomic_write_json(run_root / "selection/thresholds.json", {"source_split": "dev", "selection_metric": "dev_macro_pragmatic_f1", "thresholds": dict(state.thresholds), "best_epoch": state.best_epoch})

    engine.assert_test_access()
    test_batches = _batches(
        bundle.test,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        weights=weights,
        physical_batch_size=config.physical_batch_size,
        rationale_records=None,
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
    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(best_engine_checkpoint, best_checkpoint)
    except OSError:
        shutil.copy2(best_engine_checkpoint, best_checkpoint)
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
    atomic_write_json(run_root / "selection/selection_metric.json", {"name": "dev_macro_pragmatic_f1", "value": state.best_metric, "best_epoch": state.best_epoch})
    atomic_write_json(run_root / "training/history.json", state.history)
    if not (run_root / "training/device_report.json").exists():
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
    validation = _validate_run(run_root, manifest, bundle, active_auxiliaries={"polarity", "emotion"})
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
    return {
        "status": "PASS",
        "run_id": run_id,
        "run_root": str(run_root),
        "report_root": str(report_root),
        "validation": validation,
        "test": _metrics(final_test),
        "best_epoch": state.best_epoch,
        "peak_vram_gb": peak_memory,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the ViSoBERT reviewer-gap baseline")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = SEEDS if args.run_all else (args.seed,) if args.seed is not None else ()
    if not seeds:
        parser.error("--seed is required unless --run-all is used")
    for seed in seeds:
        result = run_one(seed=seed, device=args.device, force=args.force)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
