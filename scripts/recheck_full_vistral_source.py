#!/usr/bin/env python3
"""Re-evaluate an audited full-Vistral source checkpoint without changing it.

The checkpoint and thresholds are selected artifacts.  This command recomputes
dev first and then test with the frozen dev thresholds, writing a new audit
directory instead of touching the canonical run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from run_full_vistral_optimization import (
    SOURCE_RUNS,
    _data_context,
    _engine,
    _pragmatic_metrics,
    _source_checkpoint,
    _write_prediction,
)
from vipragsent.atomic import atomic_write_json
from vipragsent.constants import PRAGMATIC_LABELS
from vipragsent.hashing import sha256_file
from vipragsent.training.checkpoints import infer_required_head_prefixes, load_checkpoint
from vipragsent.training.engine import TrainingConfig
from vipragsent.training.seeding import seed_everything


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(source_root: Path) -> TrainingConfig:
    resolved = _json(source_root / "training/resolved_training_config.json")
    return TrainingConfig(
        learning_rate=float(resolved["learning_rate"]),
        weight_decay=float(resolved["weight_decay"]),
        max_epochs=int(resolved["maximum_epochs"]),
        effective_batch_size=int(resolved["effective_batch_size"]),
        physical_batch_size=int(resolved["physical_batch_size"]),
        max_grad_norm=float(resolved["gradient_clipping"]),
        patience=int(resolved["patience"]),
        min_delta=float(resolved["minimum_delta"]),
        precision=str(resolved["precision"]),
        gradient_accumulation_steps=int(resolved["gradient_accumulation_steps"]),
        primary_metric=str(resolved["selection_metric"]),
        scheduler=str(resolved["scheduler"]),
        warmup_ratio=float(resolved["warmup_ratio"]),
        use_uncertainty_weighting=bool(resolved["uncertainty_weighting_enabled"]),
        rationale_beta=float(resolved["rationale_beta"]),
        rationale_target_max_length=int(resolved["rationale_target_max_length"]),
        optimizer=str(resolved["optimizer"]),
        gradient_checkpointing=bool(resolved["gradient_checkpointing"]),
        deterministic_algorithms=str(resolved["deterministic_algorithms"]),
        qlora=dict(resolved["qlora"]),
    )


def run(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    source_root = ROOT / "results/runs" / SOURCE_RUNS[seed]
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"audit output is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    source_checkpoint = _source_checkpoint(seed)
    source_selection = _json(source_root / "selection/thresholds.json")
    if set(source_selection) != set(PRAGMATIC_LABELS):
        raise RuntimeError("canonical source does not contain all six frozen pragmatic thresholds")

    # Ensure model construction itself is deterministic for this audit.
    seed_everything(seed)
    config = _config(source_root)
    bundle, weights, snapshot, _, dev_batches, test_batches = _data_context(
        config.physical_batch_size,
        include_train=False,
    )
    engine, spec = _engine(
        output_root=output_root,
        config=config,
        weights=weights,
        snapshot=snapshot,
        trial=f"source_recheck_{seed}",
        selected_device=int(args.device),
    )
    load_result = load_checkpoint(
        source_checkpoint,
        engine.model,
        loss_aggregator=engine.loss_aggregator,
        required_head_prefixes=infer_required_head_prefixes(engine.model),
        report_path=output_root / "checkpoint_load_report.json",
    )
    if load_result.report.status != "PASS" or load_result.report.matched_ratio != 1.0:
        raise RuntimeError(f"source checkpoint load was not exact: {load_result.report}")

    dev_selection = engine._evaluate_dev(dev_batches, thresholds_override=source_selection)
    engine.gate.freeze_checkpoint()
    test_selection = engine._evaluate_dev(test_batches, thresholds_override=source_selection)
    dev_metrics = _pragmatic_metrics(dev_selection)
    test_metrics = _pragmatic_metrics(test_selection)
    atomic_write_json(output_root / "metrics/dev_metrics.json", dev_metrics)
    atomic_write_json(output_root / "metrics/test_metrics.json", test_metrics)
    _write_prediction(output_root / "predictions/dev_predictions.jsonl", engine, dev_selection, dev_batches)
    _write_prediction(output_root / "predictions/test_predictions.jsonl", engine, test_selection, test_batches)
    atomic_write_json(
        output_root / "recheck_manifest.json",
        {
            "status": "PASS",
            "seed": seed,
            "source_run_id": SOURCE_RUNS[seed],
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "checkpoint_load_status": load_result.report.status,
            "checkpoint_matched_ratio": load_result.report.matched_ratio,
            "missing_keys": list(load_result.report.missing_keys),
            "unexpected_keys": list(load_result.report.unexpected_keys),
            "dataset_fingerprint": bundle.fingerprint,
            "base_model_repository": spec.repo_id,
            "base_model_revision": spec.revision,
            "tokenizer_revision": spec.tokenizer_revision,
            "thresholds": source_selection,
            "selection_protocol": "canonical_dev_thresholds_frozen_before_test_v1",
            "test_selection_used_for_any_decision": False,
            "dev_metrics": dev_metrics,
            "test_metrics": test_metrics,
        },
    )
    print(json.dumps({"status": "PASS", "output_root": str(output_root), "dev": dev_metrics, "test": test_metrics}, indent=2))
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, choices=tuple(SOURCE_RUNS), required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output-root", required=True)
    return p


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
