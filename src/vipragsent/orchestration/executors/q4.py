from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...atomic import atomic_write_json, atomic_write_text
from ...constants import PRAGMATIC_LABELS
from ...evaluation.production import evaluate_q4_seed
from ...hashing import sha256_file, sha256_json
from ...orchestration.status import RuntimeBlocked


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_source(root: Path, entry: Mapping[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    requested_id = str(entry.get("source_run_id") or entry.get("approved_source_run_id") or "")
    requested_system = str(entry.get("system_id", ""))
    requested_seed = str(entry.get("seed"))
    candidates: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    for summary_path in sorted((root / "results/runs").glob("*/review_summary.json")):
        run_root = summary_path.parent
        if requested_id and run_root.name != requested_id:
            continue
        approval_path = run_root / "approval_status.json"
        if not approval_path.exists():
            continue
        summary = _load(summary_path)
        approval = _load(approval_path)
        if approval.get("status") != "APPROVED":
            continue
        if str(summary.get("system_id")) != requested_system or str(summary.get("seed")) != requested_seed:
            continue
        candidates.append((run_root, summary, approval))
    if len(candidates) != 1:
        raise RuntimeBlocked(f"Q4 requires exactly one approved source for {requested_system}/{requested_seed}; found {len(candidates)}")
    return candidates[0]


def _copy_with_hash(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return sha256_file(target)


def resolve_and_extract_q4_source(root: str | Path, entry: Mapping[str, Any], *, output_root: str | Path) -> dict[str, Any]:
    root = Path(root)
    output_root = Path(output_root)
    source_root, summary, approval = _resolve_source(root, entry)
    required = {
        "predictions/test_predictions.jsonl": source_root / "predictions/test_predictions.jsonl",
        "training/history.json": source_root / "training/history.json",
        "checkpoints/checkpoint_manifest.json": source_root / "checkpoints/checkpoint_manifest.json",
        "config_snapshot.yaml": source_root / "config_snapshot.yaml",
        "provenance.json": source_root / "provenance.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise RuntimeBlocked("Q4 approved source artifacts are missing: " + ", ".join(missing))
    predictions = [json.loads(line) for line in required["predictions/test_predictions.jsonl"].read_text(encoding="utf-8").splitlines() if line.strip()]
    if not predictions:
        raise RuntimeBlocked("Q4 approved source predictions are empty")
    true: dict[str, list[int]] = {label: [] for label in PRAGMATIC_LABELS}
    probabilities: dict[str, list[float]] = {label: [] for label in PRAGMATIC_LABELS}
    for row in predictions:
        gold = row.get("gold", {})
        probs = row.get("probabilities", {})
        if any(label not in gold or label not in probs for label in PRAGMATIC_LABELS):
            raise RuntimeBlocked("Q4 source predictions do not contain six pragmatic gold/probability values")
        for label in PRAGMATIC_LABELS:
            true[label].append(int(gold[label]))
            probabilities[label].append(float(probs[label]))
    history = _load(required["training/history.json"])
    if not isinstance(history, list) or not history or any(not isinstance(row, Mapping) for row in history):
        raise RuntimeBlocked("Q4 approved source learning history is empty or malformed")
    source = output_root / "source"
    copied_hashes = {name: _copy_with_hash(path, source / Path(name).name) for name, path in required.items()}
    q4 = evaluate_q4_seed(probabilities, true, seed=int(entry["seed"]))
    q4_payload = {
        "status": "PASS",
        "checkpoint_id": str(entry.get("source_checkpoint_id") or summary.get("checkpoint_path")),
        "source_run_id": source_root.name,
        "seed": int(entry["seed"]),
        "split": "vipragsent_test",
        "per_label_pragmatic_ece": q4["ece_by_label"],
        "macro_pragmatic_ece": q4["macro_pragmatic_ece"],
        "reliability_bins": q4["reliability_bins"],
        "prediction_file": "source/test_predictions.jsonl",
        "prediction_file_sha256": copied_hashes["predictions/test_predictions.jsonl"],
        "config_hash": sha256_file(required["config_snapshot.yaml"]),
        "checkpoint_manifest_sha256": copied_hashes["checkpoints/checkpoint_manifest.json"],
        "code_commit": summary.get("code_commit", "NOT_APPLICABLE"),
        "approval_record_sha256": sha256_json(approval),
        "temperature_scaling": False,
        "bin_count": 10,
        "probability_aggregation": "none",
    }
    atomic_write_json(output_root / "paper_artifacts/q4_pragmatic_calibration_per_seed.json", q4_payload)
    reliability_rows = [{"system": entry.get("system_id"), "seed": entry.get("seed"), "label": label, **row} for label, rows in q4["reliability_bins"].items() for row in rows]
    curves = [{"system": entry.get("system_id"), "seed": entry.get("seed"), **dict(row)} for row in history]
    atomic_write_json(output_root / "figure_backing/q4_pragmatic_reliability_bins.json", reliability_rows)
    atomic_write_json(output_root / "figure_backing/q4_learning_curves.json", curves)
    for label in PRAGMATIC_LABELS:
        atomic_write_text(output_root / f"figures/q4_{label}_reliability.svg", "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"320\" height=\"180\"><title>Q4 reliability source-backed figure</title></svg>\n")
    provenance = {
        "status": "PASS",
        "source_run_id": source_root.name,
        "source_hashes": copied_hashes,
        "approval_record_sha256": sha256_json(approval),
        "prediction_hash": copied_hashes["predictions/test_predictions.jsonl"],
        "history_hash": copied_hashes["training/history.json"],
        "config_hash": copied_hashes["config_snapshot.yaml"],
        "code_commit": summary.get("code_commit", "NOT_APPLICABLE"),
        "synthetic_history": False,
        "training_applicability": "NOT_APPLICABLE",
    }
    atomic_write_json(output_root / "source/source_provenance.json", provenance)
    return {"status": "PASS", "q4": q4_payload, "provenance": provenance, "figure_count": len(PRAGMATIC_LABELS)}
