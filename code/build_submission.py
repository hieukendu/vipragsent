"""Build separate anonymous software and data packages in their respective trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from release_contract import validate_release
from release_paths import default_release

ROOT_FILES = (
    "LICENSE", "README.md", "README_REPRODUCE.md",
    "pyproject.toml", "requirements-paper.txt", "Makefile", ".gitignore", ".env.example",
    "prepare_inputs.py", "release_contract.py", "verify_predictions.py",
    "verify_current_q2.py", "build_submission.py", "reproduction_inputs.py", "release_paths.py",
    "verify_submission.py",
)
SOURCE_DIRECTORIES = ("src", "scripts", "configs", "schemas", "prompts", "tests", "docs")
OMIT_COMPONENTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache",
                   ".restore_private", ".work", "_tmp_fixture_artifacts", ".git", ".venv"}
PENDING_MATERIALS = ()


def source_paths(root: Path) -> list[Path]:
    paths = [root / name for name in ROOT_FILES]
    for directory in SOURCE_DIRECTORIES:
        paths.extend(path for path in (root / directory).rglob("*")
                     if path.is_file() and not set(path.relative_to(root).parts) & OMIT_COMPONENTS
                     and path.suffix not in {".pyc", ".pyo"})
    for path in paths:
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Missing or unsafe source path: {path.name}")
    return sorted(paths)


def audit_bytes(relative: str, payload: bytes) -> None:
    if any(part in OMIT_COMPONENTS for part in Path(relative).parts) or relative.endswith((".pyc", ".pyo")):
        raise ValueError(f"Private/cache path in submission: {relative}")
    text = payload.decode("utf-8-sig").replace("\r\n", "\n")
    # Detect private destinations without embedding any submitting author's identity.
    if re.search(r"\b[\w.-]+/vipragsent-[\w.-]*(?:checkpoints|experiment-artifacts)\b", text, re.I):
        raise ValueError(f"Personal experiment repository in submission: {relative}")
    if re.search(r"(?m)^\s*(?:checkpoint_repository|artifact_repository_family)\s*:\s*(?!null\b|false\b|#|$)\S+", text):
        raise ValueError(f"Non-anonymous upload destination in submission: {relative}")
    if Path(relative).name == "LICENSE" and "MIT License" in text:
        if not re.search(r"^Copyright \(c\) \d{4} Anonymous contributors$", text, re.M):
            raise ValueError("Non-anonymous code license attribution")
    host_path_pattern = r"(?:[A-Z]:[\\/](?:Users|vipragsent)|/" + "home" + r"/[^/\s]+|/" + "workspace" + "/)"
    if re.search(host_path_pattern, text, re.I):
        raise ValueError(f"Workstation path in submission: {relative}")
    if re.search(r"(?:ghp_|github_pat_|hf_)[A-Za-z0-9]{20,}", text):
        raise ValueError(f"Credential-like token in submission: {relative}")


def write_archive(destination: Path, pairs: list[tuple[Path, str]]) -> list[dict]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries=[]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in pairs:
            payload=path.read_bytes()
            audit_bytes(relative,payload)
            entries.append({"path":relative,"bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest()})
            info=zipfile.ZipInfo(relative,date_time=(2026,10,10,0,0,0))
            info.create_system=0
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,payload)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist())!={e["path"] for e in entries}:
            raise ValueError("ZIP integrity or member mismatch")
        for entry in entries:
            payload=archive.read(entry["path"])
            audit_bytes(entry["path"],payload)
            if hashlib.sha256(payload).hexdigest()!=entry["sha256"]:
                raise ValueError("ZIP member checksum mismatch")
    return entries


def main() -> None:
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description="Build separate anonymous ARR software and data ZIPs")
    parser.add_argument("--release",type=Path,default=default_release(root))
    args=parser.parse_args()
    release=args.release.resolve()
    report=validate_release(release,root/".work/release_validation")
    source=source_paths(root)
    manifest=json.loads((release/"FILE_MANIFEST.json").read_text(encoding="utf-8"))
    names=sorted([e["path"] for e in manifest["files"]]+["FILE_MANIFEST.json","CHECKSUMS.sha256"])
    availability=json.loads((release/"AVAILABILITY.json").read_text(encoding="utf-8"))
    if "AVAILABILITY.json" not in names:
        raise ValueError("Availability report must be included in the release manifest")
    software=root/"submission/software_anonymous.zip"
    data=release/"submission/ARR_clean_anonymous.zip"
    source_entries=write_archive(software,[(p,"code/"+p.relative_to(root).as_posix()) for p in source])
    data_entries=write_archive(data,[(release/name,"ARR_clean/"+name) for name in names])
    package={"anonymity_and_package_integrity":"PASS","submission_readiness":availability["full_submission_readiness"],
        "source_file_count":len(source_entries),"canonical_release_file_count":len(data_entries),
        "software_sha256":hashlib.sha256(software.read_bytes()).hexdigest(),"data_sha256":hashlib.sha256(data.read_bytes()).hexdigest(),
        "excluded_private_backups_caches_weights_and_downloaded_records":True,"historical_provenance_verified":False,
        "full_training_reproduction":False,"current_input_hash_locks_verified":True,"historical_Q3_mask_hashes_verified":True,
        "aggregate_result_verification":report["aggregate_result_verification"],
        "pending_independent_checks":[],
        "historical_notes_recovery_waived_by_user":availability["historical_notes_recovery_waived_by_user"],
        "files":source_entries+data_entries}
    from verify_submission import verify_archives
    final=verify_archives(root,release,expected_receipt=package)
    (root/"submission/PACKAGE_REPORT.json").write_text(json.dumps(package,indent=2)+"\n",encoding="utf-8")
    (root/"submission/FINAL_VERIFICATION.json").write_text(json.dumps(final,indent=2)+"\n",encoding="utf-8")
    extracted={**final,"status":"PASS_EXTRACTED_PACKAGE_VALIDATION"}
    (root/"submission/EXTRACTED_PACKAGE_VALIDATION.json").write_text(json.dumps(extracted,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"software_archive":str(software),"data_archive":str(data),
        "anonymity_and_package_integrity":"PASS","submission_readiness":package["submission_readiness"],
        "source_files":len(source_entries),"data_files":len(data_entries)},indent=2))


if __name__ == "__main__":
    main()
