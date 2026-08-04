from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from ..constants import MAX_SEQUENCE_LENGTH
from ..hashing import sha256_json


class Segmenter(Protocol):
    version: str
    resource_checksum: str

    def segment(self, text: str) -> str: ...


@dataclass(frozen=True)
class DeterministicSegmenter:
    version: str = "vncorenlp-rdrsegmenter-pending"
    resource_checksum: str = "UNAVAILABLE"

    def segment(self, text: str) -> str:
        # The real runtime injects VnCoreNLP RDRSegmenter. This deterministic fallback is fixture-only.
        return re.sub(r"\s+", " ", text.strip())


@dataclass(frozen=True)
class PreprocessingSpec:
    backbone: str
    preprocessing_name: str
    preprocessing_version: str
    max_length: int = MAX_SEQUENCE_LENGTH
    tokenizer_revision: str | None = None


class TextPreprocessor:
    def __init__(self, spec: PreprocessingSpec, segmenter: Segmenter | None = None) -> None:
        self.spec = spec
        self.segmenter = segmenter or DeterministicSegmenter()

    def prepare_text(self, text: str) -> str:
        normalized = unicodedata.normalize("NFC", text)
        if self.spec.backbone == "phobert_base":
            return self.segmenter.segment(normalized)
        return normalized

    def cache_key(self, sample_id: str, text: str) -> str:
        return sha256_json({
            "sample_id": sample_id,
            "raw_text": text,
            "model_revision": self.spec.preprocessing_version,
            "tokenizer_revision": self.spec.tokenizer_revision,
            "preprocessing_version": self.spec.preprocessing_version,
            "max_length": self.spec.max_length,
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
                "raw_text": example["text"],
                "prepared_text": prepared,
                "cache_key": self.cache_key(example["sample_id"], example["text"]),
                "truncated": truncated,
                "token_count_before_truncation": len(tokens),
                "token_count_after_truncation": min(len(tokens), self.spec.max_length),
                "preprocessing_name": self.spec.preprocessing_name,
                "preprocessing_version": self.spec.preprocessing_version,
                "segmenter_version": self.segmenter.version,
                "segmenter_resource_checksum": self.segmenter.resource_checksum,
            })
        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return {"rows": len(records), "truncated_rows": truncations, "truncation_rate": truncations / len(records) if records else 0.0}


class DummyTokenizer:
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2

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
