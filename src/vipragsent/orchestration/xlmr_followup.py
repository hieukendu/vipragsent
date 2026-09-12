"""Production helpers for the isolated XLM-R-large follow-up lane.

The canonical Q1b/Q4 executors deliberately enforce the original inventory's
approved-source semantics.  The requested XLM-R rerun has different lineage:
Q1b trains a new full XLM-R source before external retention evaluation, while
Q4 reads the already completed V37 full XLM-R source.  These helpers keep that
lineage explicit without weakening the canonical executors.
"""

from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..atomic import atomic_write_json, atomic_write_text
from ..constants import EMOTION_LABELS, POLARITY_LABELS, PRAGMATIC_LABELS
from ..evaluation.external_retention import evaluate_external_retention
from ..evaluation.production import evaluate_q4_seed
from ..hashing import sha256_file, sha256_json
from ..orchestration.executors.external_retention import DATASET_KEYS, MANIFEST_KEYS, _load_csv
from ..orchestration.status import RuntimeBlocked
from ..runtime.device import (
    assert_runtime_device_contract,
    move_batch_to_model_device,
    resolve_model_input_device,
    write_device_report,
)
from ..runtime.hardware import validate_hardware
from ..runtime.model_assets import read_family_status, resolve_local_snapshot
from ..training.checkpoints import infer_required_head_prefixes, load_checkpoint
from .contracts import RunEntry
from .executors.generation import _encode_text

XLMR_FAMILY = "xlmr_large"
XLMR_VARIANT = "vipragsent_full_xlmr_large"
XLMR_SOURCE_PREFIX = "q1a_vipragsent_full_XLM_R_large_optimization_pragmatic_warmup020_irony105_implicit101_sarcasm101_code104_v37_"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeBlocked(f"invalid JSON artifact: {path}") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeBlocked(f"JSON artifact is not an object: {path}")
    return dict(payload)


def _load_external_datasets(root: Path) -> tuple[dict[str, list[Any]], dict[str, str], str]:
    manifest_path = root / "data/manifests/external_datasets.json"
    if not manifest_path.exists():
        raise RuntimeBlocked("official external dataset manifest is missing")
    manifest = _load_json(manifest_path)
    datasets: dict[str, list[Any]] = {}
    source_hashes: dict[str, str] = {}
    for dataset in DATASET_KEYS:
        item = manifest.get("datasets", {}).get(MANIFEST_KEYS[dataset], {})
        path_value = item.get("normalized_path")
        # The checked-in external manifest was generated on Windows and may
        # contain backslash separators.  Normalize before resolving against
        # the POSIX repository root so the bundled AIVIVN test is found on
        # Linux runners as well.
        normalized_path = str(path_value).replace("\\", "/") if path_value else ""
        path = root / normalized_path if normalized_path else None
        if path is None or not path.exists() or item.get("status") != "PASS":
            raise RuntimeBlocked(f"official normalized external test is unavailable for {dataset}")
        expected = str(item.get("checksum") or "")
        actual = sha256_file(path)
        if expected and expected != actual:
            raise RuntimeBlocked(f"official external test hash mismatch for {dataset}")
        datasets[dataset] = _load_csv(path, dataset)
        source_hashes[dataset] = actual
    return datasets, source_hashes, sha256_file(manifest_path)


def _load_xlmr_source_model(root: Path, run_root: Path, checkpoint: Path) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    hardware = validate_hardware(root)
    if hardware.get("status") != "PASS":
        raise RuntimeBlocked("validated CUDA runtime is unavailable for the XLM-R follow-up evaluator")
    selected_device = hardware.get("selected_device_index")
    cache = read_family_status(root, XLMR_FAMILY, "cache")
    snapshot = resolve_local_snapshot(root, cache.get("local_path"))
    if not snapshot:
        raise RuntimeBlocked("Phase 15 local snapshot is unavailable for xlmr_large")
    from ..data.tokenizers import create_tokenizer
    from ..models.factory import build_production_model

    model, runtime_spec = build_production_model(
        XLMR_FAMILY,
        XLMR_VARIANT,
        local_snapshot=snapshot,
        execution_mode="production",
        selected_device=selected_device,
    )
    load_result = load_checkpoint(
        checkpoint,
        model,
        allow_legacy_fixture=False,
        required_head_prefixes=infer_required_head_prefixes(model),
        report_path=run_root / "checkpoints/xlmr_q1b_source_load_report.json",
    )
    tokenizer = create_tokenizer(
        XLMR_FAMILY,
        revision=runtime_spec.tokenizer_revision,
        local_path=snapshot,
        execution_mode="production",
    )
    device = resolve_model_input_device(model)
    report = assert_runtime_device_contract(model, device, model_family=XLMR_FAMILY)
    write_device_report(run_root / "training/xlmr_q1b_device_report.json", report)
    model.eval()
    return model, tokenizer, {
        "device": str(device),
        "model_revision": runtime_spec.revision,
        "tokenizer_revision": runtime_spec.tokenizer_revision,
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_load_report": load_result.report.as_dict(),
    }


def _predict_external(model: torch.nn.Module, tokenizer: Any, dataset: str, example: Any) -> str:
    input_ids, attention = _encode_text(tokenizer, str(example.text))
    batch = move_batch_to_model_device(
        {"input_ids": input_ids, "attention_mask": attention},
        model,
        device=resolve_model_input_device(model),
    )
    with torch.no_grad():
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = output.get("logits", {}) if isinstance(output, Mapping) else getattr(output, "logits", {})
    task = "emotion" if dataset == "vsmec" else "polarity"
    task_logits = logits.get(task) if isinstance(logits, Mapping) else None
    if task_logits is None:
        raise RuntimeBlocked(f"XLM-R source did not expose the routed {task} head")
    values = task_logits[0] if getattr(task_logits, "ndim", 0) > 1 else task_logits
    index = int(torch.argmax(values).item())
    return (EMOTION_LABELS if task == "emotion" else POLARITY_LABELS)[index]


def evaluate_xlmr_q1b_from_current_checkpoint(
    root: str | Path,
    entry: RunEntry,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Evaluate official Q1b tests from the newly trained full XLM-R source."""

    if entry.research_question != "Q1b" or entry.backbone != XLMR_FAMILY:
        raise RuntimeBlocked("XLM-R follow-up Q1b evaluator received a non-XLM-R entry")
    root = Path(root)
    output_root = Path(output_root)
    checkpoint = output_root / "checkpoints/best/model.pt"
    if not checkpoint.exists():
        raise RuntimeBlocked("Q1b XLM-R evaluation requires the frozen best checkpoint")
    datasets, source_hashes, external_manifest_hash = _load_external_datasets(root)
    model = tokenizer = None
    try:
        model, tokenizer, source = _load_xlmr_source_model(root, output_root, checkpoint)
        predictions = {
            dataset: {
                example.sample_id: _predict_external(model, tokenizer, dataset, example)
                for example in examples
            }
            for dataset, examples in datasets.items()
        }
        result = evaluate_external_retention(
            datasets,
            predictions,
            source_checkpoint_id=f"xlmr_followup_full:{entry.seed}",
            source_seed=entry.seed,
            external_manifest_hash=external_manifest_hash,
            output_root=output_root,
        )
        source_summary = {
            "run_id": entry.run_id,
            "system_id": entry.system_id,
            "seed": entry.seed,
            "backbone": entry.backbone,
            "variant_id": XLMR_VARIANT,
            "checkpoint_path": "checkpoints/best/model.pt",
            "checkpoint_sha256": source["checkpoint_sha256"],
            "model_revision": source["model_revision"],
            "tokenizer_revision": source["tokenizer_revision"],
        }
        atomic_write_json(
            output_root / "external/external_evaluation_manifest.json",
            {
                "status": "PASS",
                "research_question": "Q1b",
                "followup_lane": "xlmr_q1b_train_and_external",
                "source_run_id": entry.run_id,
                "source_summary": source_summary,
                "source_summary_sha256": sha256_json(source_summary),
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "source_variant_fingerprint": sha256_json({"backbone": XLMR_FAMILY, "variant": XLMR_VARIANT}),
                "external_dataset_manifest_sha256": external_manifest_hash,
                "normalized_test_hashes": source_hashes,
                "external_finetuning": False,
                "optimizer_steps": 0,
                "backward_calls": 0,
                "train_loader_created": False,
                "training_applicability": "NEW_SOURCE_TRAINED;_EXTERNAL_EVALUATION_ONLY",
                "applicable_external_datasets": list(DATASET_KEYS),
                "partial": False,
                "predictor_factory": "xlmr_followup_current_checkpoint",
                "checkpoint_key": f"xlmr_followup_full:{entry.seed}",
                "source_seed": entry.seed,
                "producer_id": entry.run_id,
                "producer_run_id": entry.run_id,
                "producer_kind": "xlmr_followup_trainable_checkpoint",
            },
        )
        return result | {
            "followup_lane": "xlmr_q1b_train_and_external",
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_variant_fingerprint": sha256_json({"backbone": XLMR_FAMILY, "variant": XLMR_VARIANT}),
            "training_applicability": "NEW_SOURCE_TRAINED;_EXTERNAL_EVALUATION_ONLY",
            "producer_id": entry.run_id,
            "producer_run_id": entry.run_id,
            "producer_kind": "xlmr_followup_trainable_checkpoint",
            "checkpoint_key": f"xlmr_followup_full:{entry.seed}",
            "source_seed": entry.seed,
        }
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _copy_with_hash(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return sha256_file(target)


def _resolve_v37_source(root: Path, entry: RunEntry) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_id = str(entry.raw.get("source_run_id") or "")
    if not source_id or not source_id.startswith(XLMR_SOURCE_PREFIX):
        raise RuntimeBlocked("XLM-R Q4 must declare an explicit V37 full XLM-R source run")
    source_root = (root / "results/runs" / source_id).resolve()
    try:
        source_root.relative_to((root / "results/runs").resolve())
    except ValueError as exc:
        raise RuntimeBlocked("XLM-R Q4 source must remain inside results/runs") from exc
    if not source_root.is_dir():
        raise RuntimeBlocked(f"XLM-R Q4 source run is missing: {source_id}")
    optimization = _load_json(source_root / "optimization_manifest.json")
    checkpoint_manifest = _load_json(source_root / "checkpoints/checkpoint_manifest.json")
    config_snapshot = source_root / "config_snapshot.yaml"
    history_path = source_root / "training/history.json"
    if not history_path.exists():
        history_path = source_root / "_engine_output/training_history.json"
    required = (
        source_root / "predictions/test_predictions.jsonl",
        history_path,
        source_root / "checkpoints/best/model.pt",
        source_root / "checkpoints/checkpoint_manifest.json",
        config_snapshot,
    )
    missing = [str(path.relative_to(source_root)) for path in required if not path.exists()]
    if missing:
        raise RuntimeBlocked("XLM-R Q4 source artifacts are missing: " + ", ".join(missing))
    if optimization.get("status") != "PASS":
        raise RuntimeBlocked("XLM-R Q4 source optimization manifest is not PASS")
    if optimization.get("model_family") != XLMR_FAMILY or optimization.get("variant_id") != XLMR_VARIANT:
        raise RuntimeBlocked("XLM-R Q4 source is not the full XLM-R-large ViPragSent variant")
    if str(optimization.get("seed")) != str(entry.seed):
        raise RuntimeBlocked("XLM-R Q4 source seed does not match the Q4 entry")
    checkpoint_sha = sha256_file(source_root / "checkpoints/best/model.pt")
    if checkpoint_manifest.get("status") != "PASS" or checkpoint_manifest.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeBlocked("XLM-R Q4 source checkpoint manifest does not verify best/model.pt")
    if optimization.get("checkpoint_sha256") not in (None, "", checkpoint_sha):
        raise RuntimeBlocked("XLM-R Q4 source optimization manifest checkpoint hash disagrees")
    return source_root, optimization, checkpoint_manifest, {"checkpoint_sha256": checkpoint_sha, "history_path": history_path}


def extract_xlmr_q4_source(
    root: str | Path,
    entry: RunEntry,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Extract calibration/history artifacts from the approved V37 XLM-R run."""

    root = Path(root)
    output_root = Path(output_root)
    source_root, optimization, checkpoint_manifest, source = _resolve_v37_source(root, entry)
    prediction_path = source_root / "predictions/test_predictions.jsonl"
    predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not predictions:
        raise RuntimeBlocked("XLM-R Q4 source predictions are empty")
    true: dict[str, list[int]] = {label: [] for label in PRAGMATIC_LABELS}
    probabilities: dict[str, list[float]] = {label: [] for label in PRAGMATIC_LABELS}
    for row in predictions:
        gold = row.get("gold", {})
        probs = row.get("probabilities", {})
        if any(label not in gold or label not in probs for label in PRAGMATIC_LABELS):
            raise RuntimeBlocked("XLM-R Q4 source predictions do not contain six pragmatic gold/probability values")
        for label in PRAGMATIC_LABELS:
            true[label].append(int(gold[label]))
            probabilities[label].append(float(probs[label]))
    history = json.loads(Path(source["history_path"]).read_text(encoding="utf-8"))
    if not isinstance(history, list) or not history or any(not isinstance(row, Mapping) for row in history):
        raise RuntimeBlocked("XLM-R Q4 source learning history is empty or malformed")
    copied = {
        "test_predictions.jsonl": _copy_with_hash(prediction_path, output_root / "source/test_predictions.jsonl"),
        "training_history.json": _copy_with_hash(Path(source["history_path"]), output_root / "source/training_history.json"),
        "checkpoint_manifest.json": _copy_with_hash(source_root / "checkpoints/checkpoint_manifest.json", output_root / "source/checkpoint_manifest.json"),
        "config_snapshot.yaml": _copy_with_hash(source_root / "config_snapshot.yaml", output_root / "source/config_snapshot.yaml"),
        "best_model.pt": _copy_with_hash(source_root / "checkpoints/best/model.pt", output_root / "source/best_model.pt"),
    }
    q4 = evaluate_q4_seed(probabilities, true, seed=int(entry.seed))
    source_integrity = {
        "source_run_id": source_root.name,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_optimization_manifest_sha256": sha256_file(source_root / "optimization_manifest.json"),
        "source_checkpoint_manifest_sha256": sha256_file(source_root / "checkpoints/checkpoint_manifest.json"),
        "source_prediction_sha256": sha256_file(prediction_path),
        "source_history_sha256": sha256_file(Path(source["history_path"])),
    }
    source_integrity_hash = sha256_json(source_integrity)
    q4_payload = {
        "status": "PASS",
        "followup_lane": "xlmr_q4_from_v37_full_checkpoint",
        "checkpoint_id": f"vipragsent_full_xlmr_large:{entry.seed}",
        "source_run_id": source_root.name,
        "seed": int(entry.seed),
        "split": "vipragsent_test",
        "per_label_pragmatic_ece": q4["ece_by_label"],
        "macro_pragmatic_ece": q4["macro_pragmatic_ece"],
        "reliability_bins": q4["reliability_bins"],
        "prediction_file": "source/test_predictions.jsonl",
        "prediction_file_sha256": copied["test_predictions.jsonl"],
        "config_hash": copied["config_snapshot.yaml"],
        "checkpoint_manifest_sha256": copied["checkpoint_manifest.json"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "code_commit": optimization.get("code_commit", "NOT_APPLICABLE"),
        "source_integrity_sha256": source_integrity_hash,
        "temperature_scaling": False,
        "bin_count": 10,
        "probability_aggregation": "none",
    }
    atomic_write_json(output_root / "paper_artifacts/q4_pragmatic_calibration_per_seed.json", q4_payload)
    reliability_rows = [
        {"system": "xlmr_followup_q4", "seed": entry.seed, "label": label, **row}
        for label, rows in q4["reliability_bins"].items()
        for row in rows
    ]
    curves = [{"system": "xlmr_followup_q4", "seed": entry.seed, **dict(row)} for row in history]
    atomic_write_json(output_root / "figure_backing/q4_pragmatic_reliability_bins.json", reliability_rows)
    atomic_write_json(output_root / "figure_backing/q4_learning_curves.json", curves)
    for label in PRAGMATIC_LABELS:
        atomic_write_text(
            output_root / f"figures/q4_{label}_reliability.svg",
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"320\" height=\"180\"><title>XLM-R Q4 reliability source-backed figure</title></svg>\n",
        )
    provenance = {
        "status": "PASS",
        "followup_lane": "xlmr_q4_from_v37_full_checkpoint",
        "source_run_id": source_root.name,
        "source_hashes": copied,
        "source_integrity": source_integrity,
        "source_integrity_sha256": source_integrity_hash,
        "prediction_hash": copied["test_predictions.jsonl"],
        "history_hash": copied["training_history.json"],
        "config_hash": copied["config_snapshot.yaml"],
        "checkpoint_sha256": source["checkpoint_sha256"],
        "checkpoint_manifest_status": checkpoint_manifest.get("status"),
        "source_model_family": optimization.get("model_family"),
        "source_variant_id": optimization.get("variant_id"),
        "source_approval_basis": "V37 optimization_manifest PASS plus checkpoint/prediction/history hash verification",
        "code_commit": optimization.get("code_commit", "NOT_APPLICABLE"),
        "synthetic_history": False,
        "training_applicability": "NOT_APPLICABLE",
    }
    atomic_write_json(output_root / "source/source_provenance.json", provenance)
    return {"status": "PASS", "q4": q4_payload, "provenance": provenance, "figure_count": len(PRAGMATIC_LABELS)}
