from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json
from ..hashing import sha256_file

MODEL_FAMILY_STATES = {
    "cache": Path("data/model_cache_status"),
    "smoke": Path("data/model_smoke_status"),
    "batch": Path("data/batch_probe_status"),
}


def family_status_path(root: str | Path, family: str, category: str) -> Path:
    if category not in MODEL_FAMILY_STATES:
        raise ValueError(f"Unknown model-family status category: {category}")
    return Path(root) / MODEL_FAMILY_STATES[category] / f"{family}.json"


def read_family_status(root: str | Path, family: str, category: str) -> dict[str, Any]:
    path = family_status_path(root, family, category)
    if not path.exists():
        return {"model_family": family, "category": category, "status": "PENDING_NOT_REQUESTED"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"model_family": family, "category": category, "status": "FAIL", "error": str(exc)}
    return dict(payload)


def write_family_status(root: str | Path, family: str, category: str, payload: Mapping[str, Any]) -> Path:
    path = family_status_path(root, family, category)
    value = {"model_family": family, "category": category, **dict(payload)}
    atomic_write_json(path, value)
    return path


def merge_family_manifest(root: str | Path, registry: Mapping[str, Mapping[str, Any]], *, requested_family: str | None = None) -> dict[str, Any]:
    """Merge one family result without turning unrelated pending families into a blocker."""
    records: list[dict[str, Any]] = []
    for family, spec in registry.items():
        cache = read_family_status(root, family, "cache")
        smoke = read_family_status(root, family, "smoke")
        batch = read_family_status(root, family, "batch")
        status = cache.get("status", "PENDING_NOT_REQUESTED")
        if (
            status == "PASS"
            and smoke.get("status") == "PASS"
            and smoke.get("actual_local_loads") is True
            and batch.get("status") == "PASS"
            and batch.get("frozen") is True
            and batch.get("fixture_probe") is not True
        ):
            status = "PASS"
        elif requested_family == family and any(item.get("status") in {"FAIL", "BLOCKED"} for item in (cache, smoke, batch)):
            status = "BLOCKED" if any(item.get("status") == "BLOCKED" for item in (cache, smoke, batch)) else "FAIL"
        elif requested_family != family:
            status = "PENDING_NOT_REQUESTED"
        else:
            status = "PENDING"
        records.append({
            "name": family,
            "repo_id": spec.get("repo_id"),
            "revision": spec.get("revision"),
            "tokenizer_revision": spec.get("tokenizer_revision"),
            "architecture": spec.get("architecture"),
            "quantization": spec.get("quantization", "none"),
            "status": status,
            "cache_status": cache.get("status", "PENDING_NOT_REQUESTED"),
            "smoke_status": smoke.get("status", "PENDING_NOT_REQUESTED"),
            "batch_probe_status": batch.get("status", "PENDING_NOT_REQUESTED"),
            "local_path": cache.get("local_path"),
            "verification_hash": smoke.get("verification_hash"),
        })
    selected = next((item for item in records if item["name"] == requested_family), None) if requested_family else None
    return {
        "schema_version": 2,
        "models": records,
        "requested_model_family": requested_family,
        "family_status": selected["status"] if selected else None,
        "weights_downloaded": bool(records) and all(item["status"] == "PASS" for item in records),
        "global_status": "PASS" if records and all(item["status"] == "PASS" for item in records) else "PENDING",
        "blockers": [f"{item['name']}: {item['status']}" for item in records if item["status"] in {"FAIL", "BLOCKED"}],
    }


def cache_record_from_snapshot(family: str, spec: Mapping[str, Any], local_path: str | Path, *, status: str = "PASS", error: str | None = None) -> dict[str, Any]:
    path = Path(local_path)
    record: dict[str, Any] = {
        "model_family": family,
        "status": status,
        "repo_id": spec.get("repo_id"),
        "revision": spec.get("revision"),
        "tokenizer_revision": spec.get("tokenizer_revision"),
        "local_path": str(path),
        "snapshot_files": sorted(item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()) if path.exists() else [],
    }
    if error:
        record["error"] = error
    if path.exists():
        record["manifest_hash"] = sha256_file(path / "config.json") if (path / "config.json").exists() else None
    return record
