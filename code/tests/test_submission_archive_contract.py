from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from build_submission import source_paths, write_archive
from release_paths import default_release
from verify_submission import sha, verify_archives

ROOT = Path(__file__).resolve().parents[1]


def test_default_resolution_does_not_silently_use_obsolete_data_tree(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    obsolete = tmp_path / "data/ARR_clean"
    obsolete.mkdir(parents=True)
    (obsolete / "FILE_MANIFEST.json").write_text("{}")
    assert default_release(code) == tmp_path / "paper/ARR_clean"
    current = tmp_path / "ARR_clean"
    current.mkdir()
    (current / "FILE_MANIFEST.json").write_text("{}")
    assert default_release(code) == current


def test_archive_replacement_cannot_reuse_a_pass_receipt(tmp_path):
    code, release = tmp_path / "code", tmp_path / "ARR_clean"
    (code / "submission").mkdir(parents=True)
    (release / "submission").mkdir(parents=True)
    software, data = code / "submission/software_anonymous.zip", release / "submission/ARR_clean_anonymous.zip"
    software.write_bytes(b"original software")
    data.write_bytes(b"original data")
    receipt = {"software_sha256": sha(software), "data_sha256": sha(data)}
    data.write_bytes(b"replacement data")
    with pytest.raises(ValueError, match="Actual ZIP hash differs"):
        verify_archives(code, release, expected_receipt=receipt)


@pytest.fixture
def frozen_package(tmp_path):
    code, release = tmp_path / "code", tmp_path / "ARR_clean"
    canonical = default_release(ROOT)
    for source in source_paths(ROOT):
        destination = code / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    manifest = json.loads((canonical / "FILE_MANIFEST.json").read_text())
    names = [e["path"] for e in manifest["files"]] + ["FILE_MANIFEST.json", "CHECKSUMS.sha256"]
    for name in names:
        destination = release / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(canonical / name, destination)
    write_archive(code / "submission/software_anonymous.zip",
                  [(p, "code/" + p.relative_to(code).as_posix()) for p in source_paths(code)])
    return code, release


def data_archive(release):
    manifest = json.loads((release / "FILE_MANIFEST.json").read_text())
    names = [e["path"] for e in manifest["files"]] + ["FILE_MANIFEST.json", "CHECKSUMS.sha256"]
    write_archive(release / "submission/ARR_clean_anonymous.zip", [(release / n, "ARR_clean/" + n) for n in names])


def test_complete_archive_validates_actual_masks_availability_and_primary_cohort(frozen_package):
    code, release = frozen_package
    data_archive(release)
    report = verify_archives(code, release)
    assert report["availability_report_in_actual_data_zip"]
    assert round(report["q2_full_macro_mean_percent"], 2) == 93.67
    assert report["obsolete_Q2_followup_runs"] == 0
    assert all(v["rationale_active_positives"] == 545 and v["rationale_active_negatives"] == 0
               for v in report["q3_mask_alignment"].values())


@pytest.mark.parametrize("problem", ["wrong_masks", "missing_availability", "mismatched_pdf_reference"])
def test_fresh_checksums_cannot_certify_invalid_data_payload(frozen_package, problem):
    code, release = frozen_package
    if problem == "missing_availability":
        (release / "AVAILABILITY.json").unlink()
        expected = "Availability report missing"
    elif problem == "mismatched_pdf_reference":
        path = release / "AVAILABILITY.json"
        value = json.loads(path.read_text())
        value["paper_pdf_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
        expected = "Availability and table bindings refer to different PDFs"
    else:
        mask = release / "q3_masks/budget_32_masks.csv"
        rows = list(csv.DictReader(io.StringIO(mask.read_text())))
        for row in rows:
            row["rationale_loss_mask"] = "1" if row["is_sarcasm_positive"] == "0" or row["positive_selected_for_budget"] == "1" else "0"
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        mask.write_bytes(stream.getvalue().encode())
        checksum = hashlib.sha256(mask.read_bytes()).hexdigest()
        path = release / "q3_masks/manifest.json"
        value = json.loads(path.read_text())
        value["mask_hashes"]["32"] = checksum.upper()
        path.write_text(json.dumps(value))
        path = release / "SOURCE_RECEIPT.json"
        value = json.loads(path.read_text())
        value["delivered_file_sha256"]["q3_masks/budget_32_masks.csv"] = checksum
        path.write_text(json.dumps(value))
        expected = "Paper Q3 rationale mask"
    subprocess.run([sys.executable, "-B", str(release / "refresh_manifest.py")], check=True, capture_output=True)
    data_archive(release)
    receipt = {"software_sha256": sha(code / "submission/software_anonymous.zip"),
               "data_sha256": sha(release / "submission/ARR_clean_anonymous.zip")}
    with pytest.raises(ValueError, match=expected):
        verify_archives(code, release, expected_receipt=receipt)
