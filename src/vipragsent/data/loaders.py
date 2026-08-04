from __future__ import annotations

import csv
import json
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..constants import ALL_LABEL_KEYS, DATASET_SPLITS, EXPECTED_SPLIT_COUNTS, PRAGMATIC_LABELS, SPLIT_SEED
from ..hashing import fingerprint_files, sha256_file
from .labels import validate_label_dict


class DatasetValidationError(ValueError):
    """Raised when frozen project data fails a contract check."""


@dataclass(frozen=True)
class DatasetExample:
    sample_id: str
    text: str
    labels: dict[str, Any]
    split: str
    source_dataset: str = "SEACrowd/ViSoBERT"

    @classmethod
    def from_row(cls, row: dict[str, str], *, split: str | None = None) -> "DatasetExample":
        actual_split = split or row.get("split", "")
        if actual_split not in DATASET_SPLITS:
            raise DatasetValidationError(f"Invalid split for {row.get('sample_id')}: {actual_split!r}")
        if not row.get("sample_id"):
            raise DatasetValidationError("Missing sample_id")
        if "text" not in row:
            raise DatasetValidationError(f"Missing text column for {row['sample_id']}")
        labels = {key: row.get(key) for key in ALL_LABEL_KEYS}
        return cls(
            sample_id=row["sample_id"],
            text=row["text"],
            labels=validate_label_dict(labels),
            split=actual_split,
            source_dataset=row.get("source_dataset", "SEACrowd/ViSoBERT"),
        )


@dataclass
class DatasetBundle:
    splits: dict[str, list[DatasetExample]]
    fingerprint: str
    manifest: dict[str, Any]

    @property
    def train(self) -> list[DatasetExample]:
        return self.splits["train"]

    @property
    def dev(self) -> list[DatasetExample]:
        return self.splits["dev"]

    @property
    def test(self) -> list[DatasetExample]:
        return self.splits["test"]


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _validate_split_manifest(path: Path, examples: dict[str, DatasetExample]) -> None:
    rows = read_csv(path)
    expected = {"sample_id", "source_row", "split", "split_seed"}
    if set(rows[0]) != expected if rows else True:
        raise DatasetValidationError("split_manifest.csv has an unexpected schema")
    seen: set[str] = set()
    for row in rows:
        sample_id = row["sample_id"]
        if sample_id in seen or sample_id not in examples:
            raise DatasetValidationError(f"Invalid or duplicate split-manifest ID: {sample_id}")
        seen.add(sample_id)
        if row["split"] != examples[sample_id].split or int(row["split_seed"]) != SPLIT_SEED:
            raise DatasetValidationError(f"Split manifest mismatch for {sample_id}")
    if seen != set(examples):
        raise DatasetValidationError("Split manifest does not cover every sample")


def _validate_q3_mask_files(q3_dir: Path, train: list[DatasetExample]) -> dict[str, Any]:
    from .masks import validate_q3_masks

    result = validate_q3_masks(q3_dir, {example.sample_id: example for example in train})
    return result


def validate_vipragsent(package_dir: str | Path) -> dict[str, Any]:
    root = Path(package_dir)
    data_dir = root / "02_vipragsent"
    full_rows = read_csv(data_dir / "vipragsent_full.csv")
    if len(full_rows) != sum(EXPECTED_SPLIT_COUNTS.values()):
        raise DatasetValidationError(f"Expected 11997 rows, got {len(full_rows)}")
    examples: dict[str, DatasetExample] = {}
    for row in full_rows:
        example = DatasetExample.from_row(row)
        if example.sample_id in examples:
            raise DatasetValidationError(f"Duplicate sample_id: {example.sample_id}")
        if not example.text.strip():
            raise DatasetValidationError(f"Missing text: {example.sample_id}")
        examples[example.sample_id] = example
    _validate_split_manifest(data_dir / "split_manifest.csv", examples)
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        rows = read_csv(data_dir / f"{split}.csv")
        ids = [row["sample_id"] for row in rows]
        if len(rows) != expected or len(set(ids)) != expected:
            raise DatasetValidationError(f"Unexpected {split} split size or uniqueness")
        if set(ids) != {sample_id for sample_id, item in examples.items() if item.split == split}:
            raise DatasetValidationError(f"{split} does not match frozen full-data split")
    q3_report = _validate_q3_mask_files(root / "04_q3_low_resource_sarcasm", [e for e in examples.values() if e.split == "train"])
    sarcasm_positives = sum(example.labels["sarcasm"] for example in examples.values() if example.split == "train")
    if sarcasm_positives != 545:
        raise DatasetValidationError(f"Expected 545 train sarcasm positives, got {sarcasm_positives}")
    return {
        "rows": len(examples),
        "split_sizes": dict(EXPECTED_SPLIT_COUNTS),
        "unique_sample_ids": True,
        "missing_text_rows": 0,
        "train_sarcasm_positives": sarcasm_positives,
        "q3": q3_report,
        "label_keys": list(ALL_LABEL_KEYS),
        "deduplication_performed": False,
    }


def load_vipragsent(processed_dir: str | Path = "data/processed/vipragsent") -> DatasetBundle:
    root = Path(processed_dir)
    splits: dict[str, list[DatasetExample]] = {}
    files: list[Path] = []
    all_ids: set[str] = set()
    for split in DATASET_SPLITS:
        path = root / f"{split}.csv"
        files.append(path)
        rows = read_csv(path)
        examples = [DatasetExample.from_row(row, split=split) for row in rows]
        if all_ids.intersection(example.sample_id for example in examples):
            raise DatasetValidationError("Sample ID appears in more than one split")
        all_ids.update(example.sample_id for example in examples)
        splits[split] = examples
    manifest_path = root.parent.parent / "manifests" / "dataset_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return DatasetBundle(splits=splits, fingerprint=fingerprint_files(files), manifest=manifest)


def calculate_loss_weights(train: Iterable[DatasetExample]) -> dict[str, Any]:
    rows = list(train)
    if not rows:
        raise ValueError("Cannot calculate weights from an empty train split")
    pragmatic: dict[str, float] = {}
    for key in PRAGMATIC_LABELS:
        positives = sum(int(row.labels[key]) for row in rows)
        negatives = len(rows) - positives
        pragmatic[key] = float(negatives / positives) if positives else math.inf
    result: dict[str, Any] = {"pragmatic_pos_weight": pragmatic, "class_weight": {}}
    for field in ("polarity", "emotion"):
        values = sorted({row.labels[field] for row in rows})
        counts = {value: sum(row.labels[field] == value for row in rows) for value in values}
        result["class_weight"][field] = {
            value: float(len(rows) / (len(values) * count)) if count else math.inf
            for value, count in counts.items()
        }
    result["source_split"] = "train"
    return result


def ingest_zip(
    zip_path: str | Path,
    *,
    raw_root: str | Path = "data/raw/vipragsent_package",
    processed_root: str | Path = "data/processed",
    manifest_root: str | Path = "data/manifests",
) -> dict[str, Any]:
    """Extract and freeze the supplied package without changing its source files."""
    zip_path = Path(zip_path)
    raw_root = Path(raw_root)
    processed_root = Path(processed_root)
    manifest_root = Path(manifest_root)
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    if raw_root.exists() and any(raw_root.iterdir()):
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (raw_root / member.filename).resolve()
            if not str(target).startswith(str(raw_root.resolve())):
                raise DatasetValidationError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(raw_root)
    package_candidates = [path for path in raw_root.iterdir() if path.is_dir()]
    if len(package_candidates) != 1:
        raise DatasetValidationError("Expected one top-level dataset package directory")
    package_dir = package_candidates[0]
    report = validate_vipragsent(package_dir)
    vipragsent_target = processed_root / "vipragsent"
    if vipragsent_target.exists():
        shutil.rmtree(vipragsent_target)
    vipragsent_target.mkdir(parents=True, exist_ok=True)
    for filename in ("train.csv", "dev.csv", "test.csv", "vipragsent_full.csv", "label_schema.json"):
        shutil.copy2(package_dir / "02_vipragsent" / filename, vipragsent_target / filename)
    q3_target = processed_root / "q3_low_resource_sarcasm"
    q3_target.mkdir(parents=True, exist_ok=True)
    for source in (package_dir / "04_q3_low_resource_sarcasm").glob("*.csv"):
        shutil.copy2(source, q3_target / source.name)
    aivivn_target = processed_root / "external" / "aivivn_human_derived_3way"
    aivivn_target.mkdir(parents=True, exist_ok=True)
    for filename in ("train.csv", "dev.csv", "test.csv", "manifest.json"):
        shutil.copy2(package_dir / "03_aivivn_human_derived_3way" / filename, aivivn_target / filename)
    rationale_target = processed_root / "rationales"
    rationale_target.mkdir(parents=True, exist_ok=True)
    source_rationale = package_dir / "05_rationale_generation" / "rationale_generation_input_train.jsonl"
    active_rationale = rationale_target / "azure_rationale_input_train.jsonl"
    sanitized: list[dict[str, Any]] = []
    with source_rationale.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            sanitized.append({
                "sample_id": item["sample_id"],
                "comment": item["comment"],
                "gold_labels": validate_label_dict(item["gold_labels"]),
            })
    with active_rationale.open("w", encoding="utf-8") as handle:
        for item in sanitized:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    files = [path for path in vipragsent_target.glob("*.csv")]
    dataset_manifest = {
        "package": "ViPragSent_Experiment_Dataset_FINAL_V8",
        "source_zip": str(zip_path),
        "source_zip_sha256": sha256_file(zip_path),
        "validation": report,
        "processed_files": {path.name: sha256_file(path) for path in sorted(files)},
        "processed_fingerprint": fingerprint_files(files),
        "rationale_source_sha256": sha256_file(source_rationale),
        "active_rationale_sha256": sha256_file(active_rationale),
        "rationale_rows": len(sanitized),
        "duplicate_policy": "retain_all_rows_and_near_overlaps",
        "raw_text_preserved": True,
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "dataset_manifest.json").write_text(json.dumps(dataset_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (manifest_root / "loss_weights_train.json").write_text(json.dumps(calculate_loss_weights(load_vipragsent(vipragsent_target).train), indent=2) + "\n", encoding="utf-8")
    return dataset_manifest
