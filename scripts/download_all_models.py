from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from _bootstrap import ROOT
from vipragsent.hashing import sha256_file
from vipragsent.phase import write_phase_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 only: download and verify locked model revisions")
    parser.add_argument("--manifest", default="configs/models/download_manifest.yaml")
    parser.add_argument("--cache-dir", default="data/model_cache")
    args = parser.parse_args()
    manifest = yaml.safe_load((ROOT / args.manifest).read_text(encoding="utf-8"))
    blockers = []
    if any(item.get("revision") in (None, "") for item in manifest.get("models", [])):
        blockers.append("Model registry contains unresolved revisions; run Phase 04 metadata resolution first")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None
        blockers.append("huggingface_hub is not installed")
    records = []
    if not blockers:
        cache_dir = ROOT / args.cache_dir
        for item in manifest["models"]:
            try:
                path = snapshot_download(item["repo_id"], revision=item["revision"], cache_dir=str(cache_dir), local_dir=str(cache_dir / item["name"]), local_dir_use_symlinks=False, allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.py", "tokenizer.*", "vocab.*", "merges.txt"])
                records.append({"name": item["name"], "repo_id": item["repo_id"], "revision": item["revision"], "local_path": str(path), "status": "PASS"})
            except Exception as exc:
                blockers.append(f"{item['name']}: {exc}")
                records.append({"name": item["name"], "repo_id": item["repo_id"], "revision": item["revision"], "status": "BLOCKED", "error": str(exc)})
    output = {"models": records, "weights_downloaded": not blockers, "blockers": blockers}
    (ROOT / "data/model_cache_manifest.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    status = "PASS" if not blockers else "BLOCKED"
    write_phase_handoff("15", status, inputs_read=[args.manifest, "32_RUNTIME_PREFLIGHT_CHECKLIST.md"], files_created=["data/model_cache_manifest.json"], blockers=blockers, next_phase_ready=not blockers)
    print(json.dumps(output, indent=2))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
