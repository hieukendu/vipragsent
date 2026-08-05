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
    parser.add_argument("--model-family", help="Download exactly one locked model family; omit to process the complete manifest")
    args = parser.parse_args()
    manifest = yaml.safe_load((ROOT / args.manifest).read_text(encoding="utf-8"))
    all_models = list(manifest.get("models", []))
    selected_models = all_models
    if args.model_family:
        selected_models = [item for item in all_models if item.get("name") == args.model_family]
    blockers = []
    if not selected_models:
        blockers.append(f"Unknown model family: {args.model_family}")
    if any(item.get("revision") in (None, "") for item in selected_models):
        blockers.append("Model registry contains unresolved revisions; run Phase 04 metadata resolution first")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        snapshot_download = None
        blockers.append("huggingface_hub is not installed")
    records_by_name = {}
    existing_manifest_path = ROOT / "data/model_cache_manifest.json"
    if args.model_family and existing_manifest_path.exists():
        try:
            existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
            records_by_name = {str(item.get("name")): dict(item) for item in existing.get("models", []) if item.get("name")}
        except (OSError, json.JSONDecodeError):
            records_by_name = {}
    if not blockers:
        cache_dir = ROOT / args.cache_dir
        for item in selected_models:
            try:
                path = snapshot_download(item["repo_id"], revision=item["revision"], cache_dir=str(cache_dir), local_dir=str(cache_dir / item["name"]), local_dir_use_symlinks=False, allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.py", "tokenizer.*", "vocab.*", "merges.txt"])
                records_by_name[item["name"]] = {"name": item["name"], "repo_id": item["repo_id"], "revision": item["revision"], "local_path": str(path), "status": "PASS"}
            except Exception as exc:
                blockers.append(f"{item['name']}: {exc}")
                records_by_name[item["name"]] = {"name": item["name"], "repo_id": item["repo_id"], "revision": item["revision"], "status": "BLOCKED", "error": str(exc)}
    records = [records_by_name[name] for name in (item["name"] for item in all_models) if name in records_by_name]
    missing = [item["name"] for item in all_models if not any(record.get("name") == item["name"] and record.get("status") == "PASS" for record in records)]
    if args.model_family and not blockers and missing:
        blockers.append("Remaining model families require separate Phase 15 prompts: " + ", ".join(missing))
    output = {"models": records, "requested_model_family": args.model_family, "weights_downloaded": not blockers and len(records) == len(all_models), "blockers": blockers}
    (ROOT / "data/model_cache_manifest.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    status = "PASS" if not blockers else "BLOCKED"
    write_phase_handoff("15", status, inputs_read=[args.manifest, "32_RUNTIME_PREFLIGHT_CHECKLIST.md"], files_created=["data/model_cache_manifest.json"], blockers=blockers, next_phase_ready=not blockers)
    print(json.dumps(output, indent=2))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
