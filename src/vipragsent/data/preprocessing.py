from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..constants import MAX_SEQUENCE_LENGTH
from ..atomic import atomic_write_text
from ..hashing import sha256_file, sha256_json
from ..orchestration.status import RuntimeBlocked


class Segmenter(Protocol):
    version: str
    resource_checksum: str

    def segment(self, text: str) -> str: ...


@dataclass(frozen=True)
class DeterministicSegmenter:
    """Fixture-only whitespace segmenter."""

    version: str = "fixture-whitespace"
    resource_checksum: str = "fixture"

    def segment(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())


class VnCoreNLPSegmenter:
    """Adapter for an already-installed VnCoreNLP RDRSegmenter resource tree."""

    def __init__(self, resource_dir: str | Path, *, client: Any | None = None, version: str | None = None) -> None:
        self.resource_dir = Path(resource_dir).expanduser().resolve()
        if not self.resource_dir.exists():
            raise RuntimeBlocked(f"VnCoreNLP resources are missing: {self.resource_dir}")
        self._check_java_17()
        self.resource_checksum = self._resource_checksum()
        self.version = version or self._read_version()
        self.client = client or self._build_client()

    @classmethod
    def from_env(cls, resource_dir: str | Path | None = None) -> "VnCoreNLPSegmenter":
        configured = resource_dir or os.getenv("VNCORENLP_HOME")
        if not configured:
            raise RuntimeBlocked("VNCORENLP_HOME is required for production PhoBERT preprocessing")
        return cls(configured)

    def _check_java_17(self) -> None:
        try:
            result = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeBlocked("Java 17 LTS is required for VnCoreNLP") from exc
        match = re.search(r'version "(\d+)', result.stderr + result.stdout)
        if not match or match.group(1) != "17":
            raise RuntimeBlocked("Java 17 LTS is required for VnCoreNLP")

    def _resource_checksum(self) -> str:
        files = sorted(path for path in self.resource_dir.rglob("*") if path.is_file())
        if not files:
            raise RuntimeBlocked(f"VnCoreNLP resource directory is empty: {self.resource_dir}")
        return sha256_json([{"path": path.relative_to(self.resource_dir).as_posix(), "sha256": sha256_file(path)} for path in files])

    def _read_version(self) -> str:
        for name in ("VERSION", "version.txt", "vncorenlp.version"):
            path = self.resource_dir / name
            if path.exists():
                return path.read_text(encoding="utf-8").strip()
        return "vncorenlp-resource-tree"

    def _build_client(self) -> Any:
        try:
            from vncorenlp import VnCoreNLP
        except ImportError as exc:
            raise RuntimeBlocked("vncorenlp Python adapter is not installed") from exc
        try:
            return VnCoreNLP(str(self.resource_dir), annotators="wseg", quiet=True)
        except TypeError:
            return VnCoreNLP(str(self.resource_dir), annotators="wseg")
        except Exception as exc:
            raise RuntimeBlocked(f"VnCoreNLP failed to open configured resources: {exc}") from exc

    def segment(self, text: str) -> str:
        if hasattr(self.client, "word_segment"):
            value = self.client.word_segment(text)
        elif hasattr(self.client, "segment"):
            value = self.client.segment(text)
        elif callable(self.client):
            value = self.client(text)
        else:
            raise RuntimeBlocked("Configured VnCoreNLP client has no word-segmentation method")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeBlocked("VnCoreNLP returned an empty segmentation")
        return value


@dataclass(frozen=True)
class PreprocessingSpec:
    backbone: str
    preprocessing_name: str
    preprocessing_version: str
    max_length: int = MAX_SEQUENCE_LENGTH
    tokenizer_revision: str | None = None
    model_revision: str | None = None
    execution_mode: str = "production"


class TextPreprocessor:
    def __init__(self, spec: PreprocessingSpec, segmenter: Segmenter | None = None) -> None:
        self.spec = spec
        if spec.execution_mode not in {"fixture", "production"}:
            raise ValueError("execution_mode must be fixture or production")
        if spec.execution_mode == "fixture" and spec.backbone == "phobert_base":
            self.segmenter = segmenter or DeterministicSegmenter()
        else:
            self.segmenter = segmenter

    def prepare_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFC", text)
        if self.spec.backbone == "phobert_base":
            if isinstance(self.segmenter, DeterministicSegmenter) and self.spec.execution_mode != "fixture":
                raise RuntimeBlocked("Whitespace segmentation is fixture-only")
            if self.segmenter is None:
                raise RuntimeBlocked("A VnCoreNLPSegmenter is required for production PhoBERT preprocessing")
            return self.segmenter.segment(normalized)
        return normalized

    def cache_key(self, sample_id: str, text: str) -> str:
        if not sample_id:
            raise ValueError("sample_id is required for a preprocessing cache key")
        return sha256_json({
            "sample_id": sample_id,
            "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest().upper(),
            "model_revision": self.spec.model_revision,
            "tokenizer_revision": self.spec.tokenizer_revision,
            "preprocessing_name": self.spec.preprocessing_name,
            "preprocessing_version": self.spec.preprocessing_version,
            "max_length": self.spec.max_length,
            "vncorenlp_version": getattr(self.segmenter, "version", None),
            "vncorenlp_resource_checksum": getattr(self.segmenter, "resource_checksum", None),
        })

    def write_cache(self, examples: list[dict[str, Any]], output: str | Path) -> dict[str, Any]:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        truncations = 0
        records: list[dict[str, Any]] = []
        for example in examples:
            prepared = self.prepare_text(example["text"])
            tokens = prepared.split()
            truncated = len(tokens) > self.spec.max_length
            truncations += int(truncated)
            records.append({
                "sample_id": example["sample_id"],
                "raw_text_sha256": hashlib.sha256(example["text"].encode("utf-8")).hexdigest().upper(),
                "prepared_text": prepared,
                "cache_key": self.cache_key(example["sample_id"], example["text"]),
                "truncated": truncated,
                "token_count_before_truncation": len(tokens),
                "token_count_after_truncation": min(len(tokens), self.spec.max_length),
                "preprocessing_name": self.spec.preprocessing_name,
                "preprocessing_version": self.spec.preprocessing_version,
                "segmenter_version": getattr(self.segmenter, "version", None),
                "segmenter_resource_checksum": getattr(self.segmenter, "resource_checksum", None),
            })
        atomic_write_text(
            output,
            "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        )
        return {"rows": len(records), "truncated_rows": truncations, "truncation_rate": truncations / len(records) if records else 0.0}


class DummyTokenizer:
    """Small tokenizer explicitly reserved for fixture mode."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    revision = "fixture"

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str, max_length: int = MAX_SEQUENCE_LENGTH) -> list[int]:
        ids = [3 + (sum(ord(char) for char in token) % (self.vocab_size - 3)) for token in text.split()]
        return [self.bos_token_id, *ids[: max_length - 2], self.eos_token_id]

    def batch_encode(self, texts: list[str], max_length: int = MAX_SEQUENCE_LENGTH) -> dict[str, list[list[int]]]:
        encoded = [self.encode(text, max_length=max_length) for text in texts]
        width = max((len(item) for item in encoded), default=0)
        input_ids = [item + [self.pad_token_id] * (width - len(item)) for item in encoded]
        attention_mask = [[int(token != self.pad_token_id) for token in item] for item in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}
