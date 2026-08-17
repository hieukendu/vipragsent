"""Atomic, resumable persistence for generation chunks.

Chunks are deliberately independent of model checkpoints.  A chunk is a
completed generation boundary: it is written atomically and registered in the
manifest before any downstream judge is allowed to consume it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, atomic_write_text, exclusive_lock
from ..hashing import sha256_json


class GenerationPersistenceError(RuntimeError):
    """Generation artifacts cannot be safely resumed or committed."""


class GenerationChunkStore:
    """Persist ordered, idempotent generation chunks for one split."""

    SCHEMA_VERSION = 1
    GENERATION_CONTRACT_VERSION = 1
    _CONTRACT_FIELDS = (
        "source_identity",
        "code_identity",
        "model_identity",
        "tokenizer_identity",
        "checkpoint_identity",
        "config_identity",
        "dataset_identity",
        "split",
        "data_hash",
        "input_record_digest",
        "record_order_digest",
        "seed",
        "system_identity",
        "budget",
    )

    def __init__(
        self,
        root: str | Path,
        split: str,
        sample_ids: Sequence[str],
        *,
        generation_contract: Mapping[str, Any] | None = None,
        fixture_mode: bool = False,
        legacy_mode: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = str(split)
        self.sample_ids = [str(value) for value in sample_ids]
        if len(self.sample_ids) != len(set(self.sample_ids)):
            raise GenerationPersistenceError(f"duplicate sample IDs in {self.split} generation input")
        self.fixture_mode = bool(fixture_mode)
        self.legacy_mode = bool(legacy_mode)
        self.generation_contract = self._normalize_contract(generation_contract)
        self.reasoning_root = self.root / "reasoning"
        self.chunk_root = self.reasoning_root / f"{self.split}_chunks"
        self.manifest_path = self.reasoning_root / f"{self.split}_chunks_manifest.json"
        self.chunk_root.mkdir(parents=True, exist_ok=True)
        self._validate_production_contract()
        self._manifest = self._load_or_create_manifest()

    @classmethod
    def _normalize_contract(cls, value: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise GenerationPersistenceError("generation contract must be a JSON object")
        contract = dict(value)
        try:
            version = int(contract.get("contract_version", -1))
        except (TypeError, ValueError) as exc:
            raise GenerationPersistenceError("generation contract version is invalid") from exc
        if version != cls.GENERATION_CONTRACT_VERSION:
            raise GenerationPersistenceError("unsupported generation contract version")
        missing = [field for field in cls._CONTRACT_FIELDS if field not in contract]
        if missing:
            raise GenerationPersistenceError(f"generation contract is missing identity fields: {missing}")
        try:
            encoded = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise GenerationPersistenceError(f"generation contract is not JSON serializable: {exc}") from exc

    def _validate_manifest_contract(self, manifest: Mapping[str, Any]) -> None:
        persisted = manifest.get("generation_contract")
        if persisted is None:
            if not (self.fixture_mode or self.legacy_mode):
                raise GenerationPersistenceError(
                    "generation manifest lacks a canonical contract; production resume is blocked"
                )
            return
        normalized = self._normalize_contract(persisted)
        if str(manifest.get("generation_contract_sha256", "")) != sha256_json(normalized):
            raise GenerationPersistenceError("generation manifest generation contract hash mismatch")
        if self.generation_contract is None:
            if not (self.fixture_mode or self.legacy_mode):
                raise GenerationPersistenceError("production generation contract is missing")
            return
        if normalized != self.generation_contract:
            raise GenerationPersistenceError("generation contract identity mismatch")

    @staticmethod
    def _production_contract_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip().upper() in {"", "NONE", "NULL", "UNKNOWN", "NOT_PROVIDED", "NOT PROVIDED"}
        if isinstance(value, Mapping):
            if not value:
                return True
            if set(value) == {"identity"} and str(value["identity"]).strip().lower().endswith("@local"):
                return True
            return any(
                GenerationChunkStore._production_contract_missing(item)
                for item in value.values()
                if not isinstance(item, Mapping)
            )
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return any(GenerationChunkStore._production_contract_missing(item) for item in value)
        return False

    def _validate_production_contract(self) -> None:
        if self.generation_contract is None or self.fixture_mode or self.legacy_mode:
            return
        for field in self._CONTRACT_FIELDS:
            if field == "budget" and self.generation_contract[field] == "NOT_APPLICABLE":
                continue
            if self._production_contract_missing(self.generation_contract[field]):
                raise GenerationPersistenceError(f"production generation contract identity is missing: {field}")
        data_hash = str(self.generation_contract["data_hash"]).strip().upper()
        if not re.fullmatch(r"[0-9A-F]{64}", data_hash):
            raise GenerationPersistenceError("production generation contract data_hash must be a canonical SHA-256 digest")

    def _load_or_create_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            if self.generation_contract is None and not (self.fixture_mode or self.legacy_mode):
                raise GenerationPersistenceError("production generation requires a canonical generation contract")
            manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "split": self.split,
                "sample_ids": list(self.sample_ids),
                "chunks": [],
                "complete": False,
            }
            if self.generation_contract is not None:
                manifest["generation_contract"] = self.generation_contract
                manifest["generation_contract_sha256"] = sha256_json(self.generation_contract)
            else:
                manifest["legacy_compatibility"] = True
            atomic_write_json(self.manifest_path, manifest)
            return manifest
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GenerationPersistenceError(f"invalid generation chunk manifest: {self.manifest_path}") from exc
        if not isinstance(manifest, Mapping) or int(manifest.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise GenerationPersistenceError(f"unsupported generation chunk manifest: {self.manifest_path}")
        if str(manifest.get("split")) != self.split or [str(value) for value in manifest.get("sample_ids", [])] != self.sample_ids:
            raise GenerationPersistenceError("generation chunk manifest input identity mismatch")
        self._validate_manifest_contract(manifest)
        chunks = manifest.get("chunks", [])
        if not isinstance(chunks, list):
            raise GenerationPersistenceError("generation chunk manifest has invalid chunks")
        for item in chunks:
            if not isinstance(item, Mapping) or not str(item.get("path", "")) or not str(item.get("sha256", "")):
                raise GenerationPersistenceError("generation chunk manifest contains an invalid chunk entry")
            path = self.root / str(item["path"])
            if not path.exists() or sha256_json(self._read_rows(path)) != str(item["sha256"]):
                raise GenerationPersistenceError(f"generation chunk is missing or corrupt: {path}")
        return dict(manifest)

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GenerationPersistenceError(f"invalid generation chunk: {path}") from exc

    def _write_manifest(self) -> None:
        atomic_write_json(self.manifest_path, self._manifest)

    def committed_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        expected = set(self.sample_ids)
        for item in sorted(self._manifest.get("chunks", []), key=lambda value: int(value["index"])):
            manifest_contract_sha = self._manifest.get("generation_contract_sha256")
            if manifest_contract_sha is not None and str(item.get("generation_contract_sha256", "")) != str(manifest_contract_sha):
                raise GenerationPersistenceError("generation chunk contract identity is invalid")
            chunk_rows = self._read_rows(self.root / str(item["path"]))
            expected_chunk_ids = [str(value) for value in item.get("sample_ids", [])]
            observed_ids = [str(row.get("sample_id", "")) for row in chunk_rows]
            if observed_ids != expected_chunk_ids or not set(observed_ids) <= expected or seen.intersection(observed_ids):
                raise GenerationPersistenceError("generation chunk ordering or sample identity is invalid")
            seen.update(observed_ids)
            rows.extend(chunk_rows)
        return rows

    def committed_sample_ids(self) -> set[str]:
        return {str(row["sample_id"]) for row in self.committed_rows()}

    def next_index(self) -> int:
        chunks = self._manifest.get("chunks", [])
        return max((int(item["index"]) for item in chunks), default=-1) + 1

    def commit(self, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        materialized = [dict(row) for row in rows]
        if not materialized:
            raise GenerationPersistenceError("cannot commit an empty generation chunk")
        ids = [str(row.get("sample_id", "")) for row in materialized]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise GenerationPersistenceError("generation chunk has missing or duplicate sample IDs")
        if not set(ids) <= set(self.sample_ids):
            raise GenerationPersistenceError("generation chunk contains an unexpected sample ID")
        digest = sha256_json(materialized)
        with exclusive_lock(self.manifest_path.with_suffix(self.manifest_path.suffix + ".lock")):
            self._manifest = self._load_or_create_manifest()
            for item in self._manifest.get("chunks", []):
                if [str(value) for value in item.get("sample_ids", [])] == ids:
                    if str(item.get("sha256")) != digest:
                        raise GenerationPersistenceError("attempted to rewrite a committed generation chunk")
                    return dict(item)
            if self.committed_sample_ids().intersection(ids):
                raise GenerationPersistenceError("generation chunk would duplicate committed sample work")
            index = self.next_index()
            relative_path = (Path("reasoning") / f"{self.split}_chunks" / f"chunk_{index:06d}.jsonl").as_posix()
            path = self.root / relative_path
            atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized))
            entry = {
                "index": index,
                "path": relative_path,
                "sample_ids": ids,
                "sha256": digest,
                "row_count": len(materialized),
                "generation_contract_sha256": self._manifest.get("generation_contract_sha256"),
            }
            self._manifest.setdefault("chunks", []).append(entry)
            self._manifest["complete"] = len(self.committed_sample_ids()) == len(self.sample_ids)
            self._write_manifest()
            return dict(entry)

    def mark_complete(self) -> None:
        if self.committed_sample_ids() != set(self.sample_ids):
            raise GenerationPersistenceError("cannot mark generation chunks complete before every sample is committed")
        self._manifest["complete"] = True
        self._write_manifest()
