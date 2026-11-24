"""Validate the actual anonymous ZIPs against their receipt and frozen data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import statistics
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from build_submission import audit_bytes, source_paths
from release_paths import default_release


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_archives(root: Path, release: Path, *, expected_receipt: dict | None = None) -> dict:
    root, release = root.resolve(), release.resolve()
    software = root / "submission/software_anonymous.zip"
    data = release / "submission/ARR_clean_anonymous.zip"
    hashes = {"software_sha256": sha(software), "data_sha256": sha(data)}
    if expected_receipt is not None:
        require(all(expected_receipt.get(k) == v for k, v in hashes.items()),
                "Actual ZIP hash differs from the verification receipt")
    work = root / ".work/package_verification"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zip-", dir=work) as temporary:
        view = Path(temporary)
        members = {}
        for path, prefix in ((software, "code"), (data, "ARR_clean")):
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                require(len(names) == len(set(names)), "Duplicate ZIP member")
                require(archive.testzip() is None, "ZIP CRC failure")
                for name in names:
                    relative = PurePosixPath(name)
                    require(relative.parts and relative.parts[0] == prefix
                            and ".." not in relative.parts and "\\" not in name
                            and not any(":" in part for part in relative.parts)
                            and not relative.is_absolute() and not name.endswith("/"), "Unsafe ZIP member")
                    destination = (view / name).resolve()
                    require(destination.is_relative_to(view), "ZIP member escapes temporary view")
                    payload = archive.read(name)
                    audit_bytes(name, payload)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                members[prefix] = set(names)
        current_source = {"code/" + p.relative_to(root).as_posix(): p for p in source_paths(root)}
        require(members["code"] == set(current_source), "Software ZIP member inventory differs from current source")
        require(all((view / name).read_bytes() == path.read_bytes() for name, path in current_source.items()),
                "Software ZIP bytes differ from current source")
        extracted = view / "ARR_clean"
        manifest = json.loads((extracted / "FILE_MANIFEST.json").read_text(encoding="utf-8"))
        expected_names = {"ARR_clean/" + e["path"] for e in manifest["files"]}
        expected_names.update({"ARR_clean/FILE_MANIFEST.json", "ARR_clean/CHECKSUMS.sha256"})
        require(members["ARR_clean"] == expected_names, "Data ZIP member inventory differs from its manifest")
        require("ARR_clean/AVAILABILITY.json" in members["ARR_clean"], "Availability report missing from actual data ZIP")
        for name in members["ARR_clean"]:
            require((view / name).read_bytes() == (release / name.removeprefix("ARR_clean/")).read_bytes(),
                    "Data ZIP bytes differ from canonical ARR_clean")
        outputs = {}
        for script in ("verify_release.py", "validate_saved_results.py"):
            result = subprocess.run([sys.executable, "-B", str(extracted / script)],
                                    capture_output=True, text=True, encoding="utf-8")
            require(result.returncode == 0, f"Extracted {script} failed: {result.stderr.strip()}")
            outputs[script] = json.loads(result.stdout)
        dataset, aggregate = outputs["verify_release.py"], outputs["validate_saved_results.py"]
        require(dataset["status"] == "PASS" and aggregate["status"] == "PASS_AGGREGATE_VALIDATION",
                "Extracted data or aggregate validation did not pass")
        mask_report = {}
        for budget in ("32", "64", "128", "256", "512", "full"):
            name = f"q3_masks/budget_{budget}_masks.csv"
            payload = (extracted / name).read_bytes()
            rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
            positive = [r for r in rows if r["is_sarcasm_positive"] == "1"]
            negative = [r for r in rows if r["is_sarcasm_positive"] == "0"]
            require(len(positive) == 545 and len(negative) == 7453, "Q3 positive/negative cohort mismatch")
            require(all(r["rationale_loss_mask"] == "1" for r in positive)
                    and all(r["rationale_loss_mask"] == "0" for r in negative), "Q3 rationale policy mismatch")
            runtime = root / "data/processed/q3_low_resource_sarcasm" / Path(name).name
            if runtime.exists():
                require(runtime.read_bytes() == payload, "Runtime Q3 mask differs from canonical data ZIP")
            mask_report[budget] = {
                "rationale_active_positives": 545, "rationale_active_negatives": 0,
                "out_of_budget_positive_rationales_active": sum(r["positive_selected_for_budget"] == "0" for r in positive),
                "runtime_mask_bytes_match": True if runtime.exists() else None,
                "mask_sha256": hashlib.sha256(payload).hexdigest(),
            }
        payload = json.loads((extracted / "replay/AGGREGATE_RESULTS.json").read_text())
        bindings = json.loads((extracted / "replay/TABLE_BINDINGS.json").read_text())
        full = next(g for g in bindings["groups"] if g["group"] == "Q2_full")
        scores = [100 * r["macro_f1"] for r in payload["runs"] if r["target"] in full["targets"] and r["split"] == "test"]
        require(len(scores) == 3 and round(statistics.mean(scores), 2) == 93.67, "Q2 Full is not the primary paper cohort")
        require(not any(r["target"].startswith("Q2_followup/") for r in payload["runs"]), "Obsolete Q2 cohort in data ZIP")
        availability = json.loads((extracted / "AVAILABILITY.json").read_text())
        require(availability["paper_pdf_sha256"] == bindings["paper_pdf_sha256"]
                == aggregate["paper_pdf_sha256"], "Packaged PDF reference metadata differs")
    # Detect a concurrent replacement while the ZIPs were being checked.
    require(hashes == {"software_sha256": sha(software), "data_sha256": sha(data)}, "ZIP changed during verification")
    return {
        "status": "PASS_FINAL_VERIFICATION", "verification_scope": "actual_software_and_data_zip_bytes",
        **hashes, "receipt_hashes_match_actual_archives": True,
        "source_matches_packaged_checksums": True, "canonical_data_matches_actual_zip": True,
        "source_and_archive_anonymity_check": "PASS", "availability_report_in_actual_data_zip": True,
        "paper_pdf_sha256": bindings["paper_pdf_sha256"],
        "paper_pdf_version": bindings["paper_pdf_version"],
        "availability_metadata_matches_table_bindings": True,
        "historical_posthoc_annotation_notes_available": availability["historical_posthoc_annotation_notes_available"],
        "paper_wording_revision_applied": availability["paper_wording_revision_applied"],
        "source_file_count": len(members["code"]), "canonical_release_file_count": len(members["ARR_clean"]),
        "dataset": dataset, "aggregate": aggregate, "q3_mask_alignment": mask_report,
        "q2_full_macro_mean_percent": statistics.mean(scores),
        "q2_full_macro_sample_sd_percent": statistics.stdev(scores), "obsolete_Q2_followup_runs": 0,
        "historical_notes_recovery_waived_by_user": availability["historical_notes_recovery_waived_by_user"],
        "training_or_API_rerun": False, "bootstrap_reexecuted": False,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=default_release(root))
    args = parser.parse_args()
    receipt = json.loads((root / "submission/FINAL_VERIFICATION.json").read_text(encoding="utf-8"))
    result = verify_archives(root, args.release, expected_receipt=receipt)
    print(json.dumps({k: result[k] for k in ("status", "receipt_hashes_match_actual_archives",
                     "availability_report_in_actual_data_zip", "q2_full_macro_mean_percent",
                     "obsolete_Q2_followup_runs", "software_sha256", "data_sha256")}, indent=2))


if __name__ == "__main__":
    main()
