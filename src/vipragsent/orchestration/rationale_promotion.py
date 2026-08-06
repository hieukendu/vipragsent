from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, atomic_write_text
from ..hashing import sha256_file, sha256_json
from .review import validate_review_summary
from .run_store import RunStore

CANONICAL_RATIONALE_PATH = Path("data/processed/rationales/approved_generated_rationales_train.jsonl")
CANONICAL_MANIFEST_PATH = Path("data/manifests/approved_generated_rationales_train.json")
REQUIRED_CANONICAL_FIELDS = (
    "sample_id",
    "rationale",
    "source_run_id",
    "source_response_id",
    "source_prompt_hash",
    "source_schema_hash",
    "source_deployment",
    "source_model_version",
    "source_record_hash",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _train_ids(root: Path, explicit: Iterable[str] | None = None) -> list[str]:
    if explicit is not None:
        result = [str(value) for value in explicit]
    else:
        path = root / "data/processed/vipragsent/train.csv"
        if not path.exists():
            raise ValueError("frozen ViPragSent train split is missing")
        import csv

        with path.open(encoding="utf-8", newline="") as handle:
            result = [str(row["sample_id"]) for row in csv.DictReader(handle)]
    if len(result) != len(set(result)):
        raise ValueError("frozen train sample IDs are not unique")
    return result


def _source_artifact_hash(source_root: Path) -> str:
    names = (
        "azure/request_manifest.json",
        "azure/response_manifest.json",
        "azure/usage.json",
        "azure/invalid_outputs.jsonl",
        "azure/cache_manifest.json",
        "azure/rationale.jsonl",
        "azure/rationale_failures.json",
    )
    return sha256_json({name: sha256_file(source_root / name) for name in names})


def _validate_source(root: Path, source_run_id: str, train_ids: list[str]) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = root / "results/runs" / source_run_id
    required = [
        "state.json",
        "review_summary.json",
        "approval_status.json",
        "checksums.sha256",
        "azure/request_manifest.json",
        "azure/response_manifest.json",
        "azure/usage.json",
        "azure/invalid_outputs.jsonl",
        "azure/cache_manifest.json",
        "azure/rationale.jsonl",
        "azure/rationale_failures.json",
    ]
    missing = [name for name in required if not (source_root / name).exists()]
    if missing:
        raise ValueError("rationale source artifacts are missing: " + ", ".join(missing))
    state = _load(source_root / "state.json")
    if source_run_id != "azure_rationale_generation":
        raise ValueError("source run is not the locked Azure rationale-generation job")
    if state.get("run_status") != "COMPLETED_PENDING_APPROVAL":
        raise ValueError("rationale source run is not completed pending approval")
    summary = _load(source_root / "review_summary.json")
    summary_errors = validate_review_summary(summary, completed=True)
    if summary_errors:
        raise ValueError("rationale review summary is not PASS: " + "; ".join(summary_errors))
    approval = _load(source_root / "approval_status.json")
    if approval.get("status") != "APPROVED" or not isinstance(approval.get("record"), Mapping):
        raise ValueError("rationale source approval is not APPROVED")
    record = dict(approval["record"])
    if record.get("review_summary_sha256") != sha256_file(source_root / "review_summary.json"):
        raise ValueError("rationale approval does not bind the current review summary")
    if record.get("artifact_checksum_file_sha256") != sha256_file(source_root / "checksums.sha256"):
        raise ValueError("rationale approval does not bind the current checksum file")
    context = type("PromotionContext", (), {"root": root, "entry": type("Entry", (), {"run_id": source_run_id, "is_azure": True})(), "fixture": False, "run_root": source_root})()
    checksum_errors = RunStore(context).validate_checksums()  # type: ignore[arg-type]
    if checksum_errors:
        raise ValueError("rationale source checksums do not validate: " + "; ".join(checksum_errors))
    request = _load(source_root / "azure/request_manifest.json")
    response = _load(source_root / "azure/response_manifest.json")
    usage = _load(source_root / "azure/usage.json")
    if request.get("deployment") in (None, "") or request.get("model") in (None, ""):
        raise ValueError("rationale request provenance is incomplete")
    requested = int(response.get("requested", 0))
    successful = int(response.get("successful", 0))
    failed = int(response.get("failed", 0))
    invalid = int(response.get("invalid", 0))
    missing_count = int(response.get("missing", 0))
    if requested != successful + failed + invalid + missing_count:
        raise ValueError("rationale request accounting does not close")
    if int(usage.get("request_count", requested)) != requested:
        raise ValueError("rationale usage accounting does not match requested count")
    records = _jsonl(source_root / "azure/rationale.jsonl")
    failures = _load(source_root / "azure/rationale_failures.json")
    if not isinstance(failures, list):
        raise ValueError("rationale failures artifact must be a list")
    return source_root, request, response, record, records, [dict(item) for item in failures]


def promote_approved_rationales(
    root: str | Path = ".",
    *,
    source_run_id: str = "azure_rationale_generation",
    train_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = Path(root)
    frozen_ids = _train_ids(root, train_ids)
    source_root, request, response, approval_record, records, failures = _validate_source(root, source_run_id, frozen_ids)
    expected_ids = set(frozen_ids)
    successful: dict[str, dict[str, Any]] = {}
    for raw in records:
        sample_id = str(raw.get("sample_id", ""))
        if not sample_id or sample_id in successful:
            raise ValueError("rationale successful records contain duplicate or empty sample IDs")
        if sample_id not in expected_ids:
            raise ValueError(f"rationale successful record is outside the frozen train split: {sample_id}")
        rationale = str(raw.get("rationale_target", "")).strip()
        if not rationale:
            raise ValueError(f"successful rationale record has no rationale: {sample_id}")
        required_provenance = ("response_id", "prompt_hash", "schema_hash", "deployment")
        if any(raw.get(field) in (None, "") for field in required_provenance):
            raise ValueError(f"rationale provenance is incomplete for {sample_id}")
        model_version = raw.get("observed_model_version") or raw.get("observed_model") or request.get("model")
        canonical = {
            "sample_id": sample_id,
            "rationale": rationale,
            "source_run_id": source_run_id,
            "source_response_id": str(raw["response_id"]),
            "source_prompt_hash": str(raw["prompt_hash"]),
            "source_schema_hash": str(raw["schema_hash"]),
            "source_deployment": str(raw["deployment"]),
            "source_model_version": str(model_version),
            "source_record_hash": sha256_json(raw),
        }
        successful[sample_id] = canonical
    failed_ids: list[str] = []
    for item in failures:
        sample_id = str(item.get("sample_id", ""))
        if not sample_id or sample_id in failed_ids:
            raise ValueError("rationale failure records contain duplicate or empty sample IDs")
        if sample_id not in expected_ids:
            raise ValueError(f"rationale failure record is outside the frozen train split: {sample_id}")
        failed_ids.append(sample_id)
    failed_set = set(failed_ids)
    if set(successful) & failed_set:
        raise ValueError("successful and failed rationale IDs overlap")
    missing_ids = expected_ids - set(successful) - failed_set
    requested = int(response.get("requested", 0))
    if len(records) != int(response.get("successful", 0)):
        raise ValueError("rationale successful record count does not match the response manifest")
    if len(failed_set) != int(response.get("invalid", 0)) + int(response.get("failed", 0)):
        raise ValueError("rationale failure record count does not match the response manifest")
    if requested != len(successful) + len(failed_set) + int(response.get("missing", 0)):
        raise ValueError("rationale source counts do not match the promoted ID accounting")
    ordered_rows = [successful[sample_id] for sample_id in frozen_ids if sample_id in successful]
    canonical_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in ordered_rows)
    target = root / CANONICAL_RATIONALE_PATH
    manifest_path = root / CANONICAL_MANIFEST_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = target.with_name(f".{target.name}.{os.getpid()}.candidate")
    candidate_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.candidate")
    try:
        atomic_write_text(candidate, canonical_text)
        candidate_hash = sha256_file(candidate)
        manifest = {
            "schema_version": 1,
            "status": "PASS",
            "train_count": len(frozen_ids),
            "requested_count": requested,
            "successful_count": len(successful),
            "failed_count": len(failed_set),
            "missing_count": len(missing_ids),
            "coverage_rate": len(successful) / len(frozen_ids) if frozen_ids else 0.0,
            "successful_id_sha256": sha256_json(sorted(successful)),
            "failed_id_sha256": sha256_json(sorted(failed_set)),
            "missing_id_sha256": sha256_json(sorted(missing_ids)),
            "canonical_file_sha256": candidate_hash,
            "source_artifact_sha256": _source_artifact_hash(source_root),
            "approval_record_sha256": sha256_json(approval_record),
            "source_run_id": source_run_id,
            "source_request_manifest_sha256": sha256_file(source_root / "azure/request_manifest.json"),
            "source_response_manifest_sha256": sha256_file(source_root / "azure/response_manifest.json"),
            "source_deployment": request.get("deployment"),
            "source_model_version": request.get("model"),
            "rationale_target_max_length": 160,
        }
        atomic_write_json(candidate_manifest, manifest)
        os.replace(candidate, target)
        os.replace(candidate_manifest, manifest_path)
    finally:
        for path in (candidate, candidate_manifest):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    loaded = _load(manifest_path)
    if loaded.get("canonical_file_sha256") != sha256_file(target):
        raise ValueError("promoted rationale manifest hash does not match the canonical file")
    return {
        "status": "PASS",
        "source_run_id": source_run_id,
        "canonical_path": CANONICAL_RATIONALE_PATH.as_posix(),
        "manifest_path": CANONICAL_MANIFEST_PATH.as_posix(),
        "manifest": loaded,
        "canonical_file_sha256": sha256_file(target),
        "provenance_retained": True,
    }


def load_approved_rationales(root: str | Path = ".") -> dict[str, dict[str, Any]]:
    root = Path(root)
    path = root / CANONICAL_RATIONALE_PATH
    manifest_path = root / CANONICAL_MANIFEST_PATH
    if not path.exists() or not manifest_path.exists():
        raise ValueError("approved canonical rationale file and manifest are required")
    manifest = _load(manifest_path)
    if manifest.get("canonical_file_sha256") != sha256_file(path):
        raise ValueError("approved canonical rationale file hash does not match its manifest")
    rows = _jsonl(path)
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(REQUIRED_CANONICAL_FIELDS) - set(row) or not row.get("rationale"):
            raise ValueError("approved canonical rationale row is incomplete")
        sample_id = str(row["sample_id"])
        if sample_id in records:
            raise ValueError("approved canonical rationale IDs are not unique")
        records[sample_id] = row
    if int(manifest.get("successful_count", len(rows))) != len(rows):
        raise ValueError("approved canonical rationale count does not match its manifest")
    return records
