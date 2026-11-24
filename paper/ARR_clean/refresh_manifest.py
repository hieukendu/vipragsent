"""Regenerate current release byte locks after an intentional release edit."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    entries = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(ROOT)
        if name.parts[0] == "submission" or name.as_posix() in ("FILE_MANIFEST.json", "CHECKSUMS.sha256"):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Unsafe release file")
        if any(p in ("__pycache__", ".pytest_cache", ".git", ".work") for p in name.parts) or path.suffix in (".pyc", ".pyo"):
            raise ValueError("Remove cache/private files before refreshing the release")
        entries.append({"path": name.as_posix(), "bytes": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"release": "ARR_SUPPLEMENT_20261010", "algorithm": "SHA-256",
                "scope": "paper_aligned_dataset_and_aggregate_experiment_results",
                "excluded_working_directories": ["submission"],
                "full_paper_reproduction": False, "files": entries}
    (ROOT/"FILE_MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8", newline="\n")
    (ROOT/"CHECKSUMS.sha256").write_text("".join(e["sha256"]+"  "+e["path"]+"\n" for e in entries), encoding="utf-8", newline="\n")
    print(json.dumps({"current_release_files": len(entries), "manifest_refresh_modifies_payloads": False}))


if __name__ == "__main__":
    main()
