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
        self._manifest_revision = self._manifest_revision_for(self._manifest)
        self._manifest_signature = self._file_signature(self.manifest_path)
        self._committed_rows_cache: list[dict[str, Any]] = []
        self._committed_sample_ids_cache: set[str] = set()
        self._chunk_entries_by_ids: dict[tuple[str, ...], dict[str, Any]] = {}
        self._next_chunk_index = 0
        self._initialize_committed_state()

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
    def _production_contract_missing(
        value: Any,
        *,
        allow_optional_config_values: bool = False,
        _root: bool = True,
    ) -> bool:
        """Find missing sentinels anywhere in a production identity value.

        Identity mappings are recursive.  ``config_identity`` also contains
        optional configuration fields such as an absent batch profile, so
        those nested ``None``/empty mappings remain valid while its required
        top-level value is still checked.
        """
        if value is None:
            return not (allow_optional_config_values and not _root)
        if isinstance(value, str):
            return value.strip().upper() in {"", "NONE", "NULL", "UNKNOWN", "NOT_PROVIDED", "NOT PROVIDED"}
        if isinstance(value, Mapping):
            if not value:
                return _root or not allow_optional_config_values
            if "identity" in value and str(value["identity"]).strip().lower().endswith("@local"):
                return True
            return any(
                GenerationChunkStore._production_contract_missing(
                    item,
                    allow_optional_config_values=allow_optional_config_values,
                    _root=False,
                )
                for item in value.values()
            )
        if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            return any(
                GenerationChunkStore._production_contract_missing(
                    item,
                    allow_optional_config_values=allow_optional_config_values,
                    _root=False,
                )
                for item in value
            )
        return False

    def _validate_production_contract(self) -> None:
        if self.generation_contract is None or self.fixture_mode or self.legacy_mode:
            return
        for field in self._CONTRACT_FIELDS:
            if field == "budget" and self.generation_contract[field] == "NOT_APPLICABLE":
                continue
            if self._production_contract_missing(
                self.generation_contract[field],
                allow_optional_config_values=field == "config_identity",
            ):
                raise GenerationPersistenceError(f"production generation contract identity is missing: {field}")
        data_hash = str(self.generation_contract["data_hash"]).strip().upper()
        if not re.fullmatch(r"[0-9A-F]{64}", data_hash):
            raise GenerationPersistenceError("production generation contract data_hash must be a canonical SHA-256 digest")

    @staticmethod
    def _manifest_revision_for(manifest: Mapping[str, Any]) -> str:
        return sha256_json(dict(manifest))

    def _validate_manifest_shape(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        try:
            schema_version = int(manifest.get("schema_version", -1))
        except (TypeError, ValueError) as exc:
            raise GenerationPersistenceError(f"unsupported generation chunk manifest: {self.manifest_path}") from exc
        if schema_version != self.SCHEMA_VERSION:
            raise GenerationPersistenceError(f"unsupported generation chunk manifest: {self.manifest_path}")
        if str(manifest.get("split")) != self.split or [str(value) for value in manifest.get("sample_ids", [])] != self.sample_ids:
            raise GenerationPersistenceError("generation chunk manifest input identity mismatch")
        self._validate_manifest_contract(manifest)
        chunks = manifest.get("chunks", [])
        if not isinstance(chunks, list):
            raise GenerationPersistenceError("generation chunk manifest has invalid chunks")
        indexes: set[int] = set()
        ordered_indexes: list[int] = []
        for item in chunks:
            if not isinstance(item, Mapping) or not str(item.get("path", "")) or not str(item.get("sha256", "")):
                raise GenerationPersistenceError("generation chunk manifest contains an invalid chunk entry")
            try:
                index = int(item.get("index", -1))
            except (TypeError, ValueError) as exc:
                raise GenerationPersistenceError("generation chunk manifest contains an invalid chunk index") from exc
            if index < 0 or index in indexes:
                raise GenerationPersistenceError("generation chunk manifest contains duplicate or invalid chunk indexes")
            indexes.add(index)
            ordered_indexes.append(index)
            if not isinstance(item.get("sample_ids", []), list):
                raise GenerationPersistenceError("generation chunk manifest contains invalid sample IDs")
        if ordered_indexes != list(range(len(indexes))):
            raise GenerationPersistenceError("generation chunk manifest indexes must be contiguous and ordered")
        return dict(manifest)

    def _read_manifest(self) -> dict[str, Any]:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GenerationPersistenceError(f"invalid generation chunk manifest: {self.manifest_path}") from exc
        if not isinstance(manifest, Mapping):
            raise GenerationPersistenceError("generation chunk manifest must be a JSON object")
        return self._validate_manifest_shape(manifest)

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
        return self._read_manifest()

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        try:
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GenerationPersistenceError(f"invalid generation chunk: {path}") from exc

    def _write_manifest(self) -> None:
        atomic_write_json(self.manifest_path, self._manifest)
        self._manifest_signature = self._file_signature(self.manifest_path)

    @staticmethod
    def _file_signature(path: Path) -> tuple[int, int, int, int]:
        try:
            stat = path.stat()
        except OSError as exc:
            raise GenerationPersistenceError(f"generation manifest is missing: {path}") from exc
        return (int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))

    def _validate_chunk_entry(
        self,
        item: Mapping[str, Any],
        *,
        seen: set[str],
    ) -> list[dict[str, Any]]:
        manifest_contract_sha = self._manifest.get("generation_contract_sha256")
        if manifest_contract_sha is not None and str(item.get("generation_contract_sha256", "")) != str(manifest_contract_sha):
            raise GenerationPersistenceError("generation chunk contract identity is invalid")
        path = self.root / str(item["path"])
        chunk_rows = self._read_rows(path)
        if any(not isinstance(row, Mapping) for row in chunk_rows):
            raise GenerationPersistenceError(f"generation chunk contains a non-object row: {path}")
        if sha256_json(chunk_rows) != str(item["sha256"]):
            raise GenerationPersistenceError(f"generation chunk is missing or corrupt: {path}")
        expected_chunk_ids = [str(value) for value in item.get("sample_ids", [])]
        observed_ids = [str(row.get("sample_id", "")) for row in chunk_rows]
        try:
            expected_row_count = int(item.get("row_count", len(chunk_rows)))
        except (TypeError, ValueError) as exc:
            raise GenerationPersistenceError("generation chunk row count is invalid") from exc
        if (
            observed_ids != expected_chunk_ids
            or len(chunk_rows) != expected_row_count
            or len(observed_ids) != len(set(observed_ids))
            or not set(observed_ids) <= set(self.sample_ids)
            or seen.intersection(observed_ids)
        ):
            raise GenerationPersistenceError("generation chunk ordering or sample identity is invalid")
        seen.update(observed_ids)
        return chunk_rows

    def _register_chunk(self, item: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
        entry = dict(item)
        ids = tuple(str(value) for value in entry.get("sample_ids", []))
        if ids in self._chunk_entries_by_ids:
            raise GenerationPersistenceError("generation manifest contains a duplicate chunk")
        self._chunk_entries_by_ids[ids] = entry
        self._committed_rows_cache.extend(dict(row) for row in rows)
        self._committed_sample_ids_cache.update(ids)
        self._next_chunk_index = max(self._next_chunk_index, int(entry["index"]) + 1)

    def _validate_next_sample_order(self, ids: Sequence[str], *, offset: int | None = None) -> None:
        if offset is None:
            offset = len(self._committed_rows_cache)
        expected = self.sample_ids[offset:offset + len(ids)]
        if list(ids) != expected:
            raise GenerationPersistenceError("generation chunks must preserve exact sample record ordering")

    def _initialize_committed_state(self) -> None:
        seen: set[str] = set()
        for item in sorted(self._manifest.get("chunks", []), key=lambda value: int(value["index"])):
            rows = self._validate_chunk_entry(item, seen=seen)
            self._validate_next_sample_order([str(value) for value in item.get("sample_ids", [])])
            self._register_chunk(item, rows)

    def _reconcile_manifest_under_lock(self) -> None:
        current_signature = self._file_signature(self.manifest_path)
        if current_signature == self._manifest_signature:
            return
        manifest = self._read_manifest()
        manifest_revision = self._manifest_revision_for(manifest)
        if self._manifest_revision == manifest_revision:
            self._manifest_signature = current_signature
            return
        current_chunks = list(self._manifest.get("chunks", []))
        observed_chunks = list(manifest.get("chunks", []))
        if len(observed_chunks) < len(current_chunks) or observed_chunks[:len(current_chunks)] != current_chunks:
            raise GenerationPersistenceError("generation manifest changed outside the append-only commit boundary")
        self._manifest = manifest
        seen = set(self._committed_sample_ids_cache)
        for item in observed_chunks[len(current_chunks):]:
            if int(item["index"]) != self._next_chunk_index:
                raise GenerationPersistenceError("generation manifest appended a non-monotonic chunk index")
            rows = self._validate_chunk_entry(item, seen=seen)
            self._validate_next_sample_order([str(value) for value in item.get("sample_ids", [])])
            self._register_chunk(item, rows)
        self._manifest_revision = manifest_revision
        self._manifest_signature = current_signature

    def _validate_all_cached_chunks(self) -> None:
        seen: set[str] = set()
        offset = 0
        for item in sorted(self._manifest.get("chunks", []), key=lambda value: int(value["index"])):
            self._validate_chunk_entry(item, seen=seen)
            ids = [str(value) for value in item.get("sample_ids", [])]
            self._validate_next_sample_order(ids, offset=offset)
            offset += len(ids)

    def committed_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._committed_rows_cache]

    def committed_sample_ids(self) -> set[str]:
        return set(self._committed_sample_ids_cache)

    def next_index(self) -> int:
        return self._next_chunk_index

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
            self._reconcile_manifest_under_lock()
            key = tuple(ids)
            existing = self._chunk_entries_by_ids.get(key)
            if existing is not None:
                if str(existing.get("sha256")) != digest:
                    raise GenerationPersistenceError("attempted to rewrite a committed generation chunk")
                return dict(existing)
            if self._committed_sample_ids_cache.intersection(ids):
                raise GenerationPersistenceError("generation chunk would duplicate committed sample work")
            self._validate_next_sample_order(ids)
            index = self.next_index()
            relative_path = (Path("reasoning") / f"{self.split}_chunks" / f"chunk_{index:06d}.jsonl").as_posix()
            path = self.root / relative_path
            if path.exists():
                raise GenerationPersistenceError(f"generation chunk path already exists: {path}")
            atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in materialized))
            entry = {
                "index": index,
                "path": relative_path,
                "sample_ids": ids,
                "sha256": digest,
                "row_count": len(materialized),
                "generation_contract_sha256": self._manifest.get("generation_contract_sha256"),
            }
            candidate_manifest = dict(self._manifest)
            candidate_manifest["chunks"] = [dict(item) for item in self._manifest.get("chunks", [])] + [entry]
            candidate_manifest["complete"] = len(self._committed_sample_ids_cache | set(ids)) == len(self.sample_ids)
            self._manifest = candidate_manifest
            self._write_manifest()
            self._register_chunk(entry, materialized)
            self._manifest_revision = self._manifest_revision_for(self._manifest)
            return dict(entry)

    def mark_complete(self) -> None:
        with exclusive_lock(self.manifest_path.with_suffix(self.manifest_path.suffix + ".lock")):
            self._reconcile_manifest_under_lock()
            if self.committed_sample_ids() != set(self.sample_ids):
                raise GenerationPersistenceError("cannot mark generation chunks complete before every sample is committed")
            self._validate_all_cached_chunks()
            self._manifest["complete"] = True
            self._write_manifest()
            self._manifest_revision = self._manifest_revision_for(self._manifest)
