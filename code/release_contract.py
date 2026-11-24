"""Verify the manifest-defined release without deleting unrelated recovery files."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_release(release: Path, work: Path) -> dict:
    release = release.resolve()
    manifest = json.loads((release / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
    listed = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in listed:
            raise ValueError("Unsafe or duplicate release-manifest path")
        source = (release / relative).resolve()
        if not source.is_relative_to(release):
            raise ValueError("Release path escapes via symlink")
        if not source.is_file() or source.stat().st_size != entry["bytes"] or digest(source) != entry["sha256"].lower():
            raise ValueError(f"Release checksum mismatch: {relative}")
        listed.add(relative.as_posix())
    checksums = {}
    for line in (release / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            value, name = line.split(maxsplit=1)
            checksums[name.lstrip("*")] = value.lower()
    if checksums != {e["path"]: e["sha256"].lower() for e in manifest["files"]}:
        raise ValueError("CHECKSUMS.sha256 and FILE_MANIFEST.json disagree")
    # Run the supplied verifier on an exact temporary view of listed files.
    # The source tree stays read-only; report extra recovery files separately.
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="release-", dir=work) as temporary:
        view = Path(temporary)
        for name in listed | {"FILE_MANIFEST.json", "CHECKSUMS.sha256"}:
            destination = view / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(release / name, destination)
        result = subprocess.run([sys.executable, "-B", str(view / "verify_release.py")],
                                capture_output=True, text=True, check=True)
        verified = json.loads(result.stdout)
        aggregate = subprocess.run([sys.executable, "-B", str(view / "validate_saved_results.py")],
                                   capture_output=True, text=True, check=True)
        aggregate_verified = json.loads(aggregate.stdout)
        if aggregate_verified.get("status") != "PASS_AGGREGATE_VALIDATION":
            raise ValueError("Aggregate result validation failed")
    if verified.get("status") != "PASS":
        raise ValueError("Release content verification failed")
    dataset = json.loads((release / "dataset_manifest.json").read_text(encoding="utf-8"))
    masks = json.loads((release / "q3_masks/manifest.json").read_text(encoding="utf-8"))
    hashes = [("rationales/azure_rationale_input_train.jsonl", dataset["active_rationale_sha256"])]
    hashes.extend((f"q3_masks/budget_{b}_masks.csv", h) for b, h in masks["mask_hashes"].items())
    differences = [{"path": name, "historical_sha256": historical.lower(),
                    "release_sha256": digest(release / name)}
                   for name, historical in hashes if digest(release / name) != historical.lower()]
    if differences:
        raise ValueError("Current input hash locks disagree with released files")
    extras = sorted(p.relative_to(release).as_posix() for p in release.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                    and p.relative_to(release).parts[0] != "submission"
                    and p.relative_to(release).as_posix() not in listed | {"FILE_MANIFEST.json", "CHECKSUMS.sha256"})
    return {"status": "PASS", "verification_scope": "manifest_defined_release_files_only",
            "release_content_verification": verified, "extra_files_outside_release": extras,
            "aggregate_result_verification": {
                "status": aggregate_verified["status"],
                "aggregate_run_splits": aggregate_verified["aggregate_run_splits"],
                "Q1b_all_six_rows_F1_from_confusion_counts": aggregate_verified["Q1b_all_six_rows_F1_from_confusion_counts"],
                "Q3_Vistral32_independent_dev_threshold_reselection_verified": aggregate_verified["Q3_Vistral32_independent_dev_threshold_reselection_verified"],
                "training_or_API_rerun": False,
            },
            "current_input_hash_locks_verified": True,
            "historical_provenance_verified": False,
            "released_gold_labels_modified": False,
            "q3_masks_restored_to_historical_paper_cohort": True,
            "historical_Q3_mask_hashes_verified": True}
