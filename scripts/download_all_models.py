from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.phase import write_phase_handoff
from vipragsent.runtime.model_assets import (
    cache_record_from_snapshot,
    merge_family_manifest,
    read_family_status,
    write_family_status,
)


def _load_models(path: Path) -> dict[str, dict[str, object]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(item["name"]): dict(item) for item in payload.get("models", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 only: prepare exactly one locked model family")
    parser.add_argument("--manifest", default="configs/models/download_manifest.yaml")
    parser.add_argument("--cache-dir", default="data/model_cache")
    parser.add_argument("--model-family", help="Prepare exactly one locked model family; omit only for an explicitly approved complete Phase 15 operation")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    hf_token = os.getenv("HF_TOKEN")
    models = _load_models(ROOT / args.manifest)
    selected_names = [args.model_family] if args.model_family else list(models)
    blockers: list[str] = []
    if not selected_names or any(name not in models for name in selected_names):
        blockers.append(f"Unknown model family: {args.model_family}")
    records: dict[str, dict[str, object]] = {}
    for family, spec in models.items():
        previous = read_family_status(ROOT, family, "cache")
        if family not in selected_names:
            records[family] = {"name": family, **spec, "status": previous.get("status", "PENDING_NOT_REQUESTED"), "family_request_status": "PENDING_NOT_REQUESTED"}
    if not blockers:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            snapshot_download = None
            blockers.append(f"huggingface_hub is not installed: {exc}")
        for family in selected_names:
            spec = models[family]
            if not spec.get("revision") or not spec.get("tokenizer_revision"):
                error = "model registry contains an unresolved revision"
                blockers.append(f"{family}: {error}")
                write_family_status(ROOT, family, "cache", {"status": "BLOCKED", "error": error, "revision": spec.get("revision")})
                continue
            if snapshot_download is None:
                error = "huggingface_hub is unavailable"
                write_family_status(ROOT, family, "cache", {"status": "BLOCKED", "error": error, "revision": spec.get("revision")})
                continue
            try:
                cache_dir = ROOT / args.cache_dir
                local_path = snapshot_download(
                    str(spec["repo_id"]),
                    revision=str(spec["revision"]),
                    cache_dir=str(cache_dir),
                    local_dir=str(cache_dir / family),
                    token=hf_token or None,
                    allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.py", "tokenizer.*", "vocab.*", "merges.txt", "*.codes" ],
                )
                record = cache_record_from_snapshot(family, spec, local_path)
                write_family_status(ROOT, family, "cache", record)
                records[family] = {"name": family, **spec, **record, "family_request_status": "PASS"}
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                blockers.append(f"{family}: {error}")
                write_family_status(ROOT, family, "cache", {"status": "BLOCKED", "error": error, "revision": spec.get("revision")})
                records[family] = {"name": family, **spec, "status": "BLOCKED", "error": error, "family_request_status": "BLOCKED"}
    manifest = merge_family_manifest(ROOT, models, requested_family=args.model_family)
    # Download success is independently reportable; global weights_downloaded remains false until all families pass smoke and batch gates.
    selected_cache_status = read_family_status(ROOT, args.model_family, "cache").get("status") if args.model_family else manifest.get("global_status")
    manifest.update({"requested_model_family": args.model_family, "selected_family_status": selected_cache_status, "download_blockers": blockers})
    atomic_write_json(ROOT / "data/model_cache_manifest.json", manifest)
    download_status = "PASS" if args.model_family and selected_cache_status == "PASS" and not blockers else "BLOCKED" if blockers else "PASS" if manifest.get("weights_downloaded") else "BLOCKED"
    handoff = write_phase_handoff(
        "15",
        download_status,
        inputs_read=[args.manifest, "32_RUNTIME_PREFLIGHT_CHECKLIST.md"],
        files_created=["data/model_cache_manifest.json", *[f"data/model_cache_status/{family}.json" for family in selected_names]],
        blockers=blockers,
        next_phase_ready=False,
        model_family=args.model_family,
    )
    output = {
        "family_status": download_status,
        "phase15_handoff_status": handoff.status,
        "selected_model_family": args.model_family,
        "selected_family_download_status": selected_cache_status,
        "global_weights_downloaded": manifest.get("weights_downloaded", False),
        "models": manifest.get("models", []),
        "blockers": blockers,
        "other_families": {name: records.get(name, {}).get("status", "PENDING_NOT_REQUESTED") for name in models if name != args.model_family},
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0 if download_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
