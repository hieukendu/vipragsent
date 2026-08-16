from __future__ import annotations

import csv
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...atomic import atomic_write_json, atomic_write_text
from ...constants import EMOTION_LABELS, POLARITY_LABELS
from ...evaluation.external_retention import NormalizedExternalExample, evaluate_external_retention
from ...evaluation.metrics import multiclass_macro_f1
from ...hashing import sha256_file, sha256_json
from ...orchestration.status import RuntimeBlocked

DATASET_KEYS = ("vsfc", "vsmec", "aivivn")
MANIFEST_KEYS = {"vsfc": "uit_vsfc", "vsmec": "uit_vsmec", "aivivn": "aivivn_human_derived_3way"}


def _load_csv(path: Path, dataset: str) -> list[NormalizedExternalExample]:
    if path.name.casefold() != "test.csv" or "train" in path.as_posix().casefold():
        raise RuntimeBlocked(f"Q1b may load only official normalized test files: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    label_column = "polarity" if dataset in {"vsfc", "aivivn"} else "emotion"
    examples = [NormalizedExternalExample(str(row["sample_id"] if "sample_id" in row else row["id"]), str(row.get("text", row.get("comment", ""))), str(row[label_column] if label_column in row else row["label"])) for row in rows]
    ids = [row.sample_id for row in examples]
    if len(ids) != len(set(ids)):
        raise RuntimeBlocked(f"Q1b normalized {dataset} test IDs are not unique")
    return examples


def _approved_source_fixture_compatibility(root: Path, entry: Mapping[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Legacy injected-predictor path retained only for existing CPU fixtures."""
    requested_system = str(entry.get("system_id", ""))
    requested_seed = entry.get("seed")
    requested_source = str(entry.get("source_checkpoint_id") or entry.get("checkpoint_role") or "")
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for summary_path in sorted((root / "results/runs").glob("*/review_summary.json")):
        run_root = summary_path.parent
        approval_path = run_root / "approval_status.json"
        if not approval_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        if approval.get("status") != "APPROVED":
            continue
        system_match = str(summary.get("system_id", "")) in {requested_system, requested_source} if requested_source else str(summary.get("system_id", "")) == requested_system
        seed_match = requested_seed in (None, "", "NOT_APPLICABLE") or str(summary.get("seed")) == str(requested_seed)
        if system_match and seed_match:
            candidates.append((run_root, summary, approval))
    if len(candidates) != 1:
        raise RuntimeBlocked(f"Q1b requires exactly one approved upstream source for {requested_system}; found {len(candidates)}")
    source_root, summary, approval = candidates[0]
    if str(summary.get("USER_REVIEW_STATUS")) != "PENDING" or str(summary.get("NEXT_RUN_ALLOWED")) != "NO":
        raise RuntimeBlocked("approved Q1b source has invalid review-gate fields")
    return source_root, summary, approval


def evaluate_external_retention_from_disk(
    root: str | Path,
    entry: Mapping[str, Any],
    *,
    output_root: str | Path,
    predictor: Callable[[str, NormalizedExternalExample], str] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    output_root = Path(output_root)
    if str(entry.get("research_question")) != "Q1b":
        raise RuntimeBlocked("disk external-retention executor accepts only Q1b entries")
    if bool(entry.get("external_finetuning", False)):
        raise RuntimeBlocked("Q1b external_finetuning must be false")
    manifest_path = root / "data/manifests/external_datasets.json"
    if not manifest_path.exists():
        raise RuntimeBlocked("official external dataset manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets: dict[str, list[NormalizedExternalExample]] = {}
    source_files: dict[str, str] = {}
    for dataset in DATASET_KEYS:
        item = manifest.get("datasets", {}).get(MANIFEST_KEYS[dataset], {})
        path_value = item.get("normalized_path")
        path = root / str(path_value) if path_value else None
        if path is None or not path.exists() or item.get("status") != "PASS":
            raise RuntimeBlocked(f"official normalized external test is unavailable for {dataset}")
        if item.get("checksum") and sha256_file(path) != item["checksum"]:
            raise RuntimeBlocked(f"official external test hash mismatch for {dataset}")
        datasets[dataset] = _load_csv(path, dataset)
        source_files[dataset] = sha256_file(path)
    factory = None
    if predictor is None:
        from ..q1b_predictor import DiskBackedQ1BPredictor

        factory = DiskBackedQ1BPredictor(root, entry)
        source_root = factory.source.run_root
        source_summary = {"checkpoint_path": factory.source.checkpoint_key, "seed": factory.source.seed, "system_id": factory.source.system_id}
        source_approval = {"sha256": factory.source.approval_sha256}
        source_checkpoint_sha256 = factory.source.checkpoint_sha256
        source_variant_fingerprint = factory.source.variant_fingerprint
        applicable = set(factory.applicable_datasets)
        predictions = {dataset: {example.sample_id: str(factory.predict(dataset, example)) for example in examples} for dataset, examples in datasets.items() if dataset in applicable}
    else:
        source_root, source_summary, source_approval = _approved_source_fixture_compatibility(root, entry)
        source_checkpoint_sha256 = str(source_summary.get("checkpoint_sha256", ""))
        source_variant_fingerprint = str(source_summary.get("variant_fingerprint", ""))
        applicable = set(DATASET_KEYS)
        predictions = {dataset: {example.sample_id: str(predictor(dataset, example)) for example in examples} for dataset, examples in datasets.items()}
    predictor_provenance = factory.provenance() if factory is not None else {}
    producer_fields = {
        "producer_id": predictor_provenance.get("producer_id"),
        "producer_run_id": predictor_provenance.get("producer_run_id"),
        "producer_kind": predictor_provenance.get("producer_kind"),
        "checkpoint_key": predictor_provenance.get("checkpoint_key") or source_summary.get("checkpoint_path"),
        "source_seed": predictor_provenance.get("source_seed", source_summary.get("seed")),
        "dependency_graph_sha256": predictor_provenance.get("dependency_graph_sha256"),
        "dependency_source_sha256": predictor_provenance.get("dependency_source_sha256"),
    }
    if applicable == set(DATASET_KEYS):
        result = evaluate_external_retention(
            datasets,
            predictions,
            source_checkpoint_id=str(source_summary.get("checkpoint_path") or entry.get("source_checkpoint_id") or source_summary.get("system_id")),
            source_seed=source_summary.get("seed"),
            external_manifest_hash=sha256_file(manifest_path),
            output_root=output_root,
        ) | {"source_checkpoint_sha256": source_checkpoint_sha256, "source_variant_fingerprint": source_variant_fingerprint, **producer_fields}
    else:
        partial_scores: dict[str, float] = {}
        for dataset in sorted(applicable):
            task = "emotion" if dataset == "vsmec" else "polarity"
            labels = EMOTION_LABELS if task == "emotion" else POLARITY_LABELS
            gold = [row.label for row in datasets[dataset]]
            pred = [predictions[dataset][row.sample_id] for row in datasets[dataset]]
            partial_scores[f"{dataset}_macro_f1"] = multiclass_macro_f1(gold, pred, labels)
            filename = {"vsfc": "uit_vsfc_test_predictions.jsonl", "vsmec": "uit_vsmec_test_predictions.jsonl", "aivivn": "aivivn_test_predictions.jsonl"}[dataset]
            atomic_write_text(output_root / "predictions" / filename, "".join(json.dumps({"sample_id": row.sample_id, "text": row.text, "gold": row.label, "prediction": predictions[dataset][row.sample_id]}, ensure_ascii=False, sort_keys=True) + "\n" for row in datasets[dataset]))
        result = {**partial_scores, "ord_f1": "NOT_APPLICABLE", "partial": True, "applicable_external_datasets": sorted(applicable), "source_checkpoint_id": str(source_summary.get("checkpoint_path") or entry.get("source_checkpoint_id") or source_summary.get("system_id")), "source_seed": source_summary.get("seed"), "source_checkpoint_sha256": source_checkpoint_sha256, "source_variant_fingerprint": source_variant_fingerprint, "external_finetuning": False, "external_manifest_hash": sha256_file(manifest_path), "optimizer_steps": 0, "train_loader_created": False, **producer_fields}
        atomic_write_json(output_root / "metrics/partial_external_retention_metrics.json", result)
    external_manifest = {
        "status": "PASS",
        "research_question": "Q1b",
        "source_run_id": source_root.name,
        "source_summary_sha256": sha256_file(source_root / "review_summary.json"),
        "source_approval_sha256": sha256_json(source_approval),
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_variant_fingerprint": source_variant_fingerprint,
        "external_dataset_manifest_sha256": sha256_file(manifest_path),
        "normalized_test_hashes": source_files,
        "external_finetuning": False,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "label_routing": {"vsfc": "polarity", "vsmec": "emotion", "aivivn": "polarity"},
        "applicable_external_datasets": sorted(applicable),
        "partial": applicable != set(DATASET_KEYS),
        "predictor_factory": predictor_provenance if factory is not None else "fixture_injected_predictor_compatibility",
        **producer_fields,
    }
    atomic_write_json(output_root / "external/external_evaluation_manifest.json", external_manifest)
    return result | {"external_evaluation_manifest": external_manifest}
