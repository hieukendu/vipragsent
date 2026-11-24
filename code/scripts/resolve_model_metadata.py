from __future__ import annotations

import argparse
import json

import yaml

from _bootstrap import ROOT
from vipragsent.phase import write_phase_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve immutable Hugging Face model metadata without downloading weights")
    parser.add_argument("--registry", default="configs/models/model_registry.yaml")
    args = parser.parse_args()
    registry_path = ROOT / args.registry
    config = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    report = {"models": {}, "blockers": []}
    try:
        from huggingface_hub import HfApi
    except ImportError:
        report["blockers"].append("huggingface_hub is not installed")
        (ROOT / "data/manifests/model_resolution_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_phase_handoff("04", "BLOCKED", inputs_read=[args.registry, "31_IMPLEMENTATION_DECISIONS.md"], files_created=["data/manifests/model_resolution_report.json"], blockers=report["blockers"], next_phase_ready=False)
        return 2
    api = HfApi()
    for name, item in config["models"].items():
        try:
            info = api.model_info(item["repo_id"], revision="main")
            card = info.cardData or {}
            resolved = {"repo_id": item["repo_id"], "revision": info.sha, "tokenizer_revision": info.sha, "license": card.get("license", "unknown"), "architecture": item["architecture"], "gated": bool(getattr(info, "gated", False)), "vocab_size": None}
            report["models"][name] = resolved
            item.update({"revision": info.sha, "tokenizer_revision": info.sha, "license": resolved["license"], "gated": resolved["gated"]})
        except Exception as exc:
            report["blockers"].append(f"{name}: {exc}")
    (ROOT / "data/manifests/model_resolution_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["blockers"]:
        write_phase_handoff("04", "BLOCKED", inputs_read=[args.registry, "31_IMPLEMENTATION_DECISIONS.md"], files_created=["data/manifests/model_resolution_report.json"], blockers=report["blockers"], next_phase_ready=False)
        return 2
    registry_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    download_manifest = ROOT / "configs/models/download_manifest.yaml"
    download = yaml.safe_load(download_manifest.read_text(encoding="utf-8"))
    for item in download["models"]:
        resolved = report["models"][item["name"]]
        item.update({"revision": resolved["revision"], "tokenizer_revision": resolved["tokenizer_revision"]})
    download_manifest.write_text(yaml.safe_dump(download, sort_keys=False), encoding="utf-8")
    write_phase_handoff("04", "PASS", inputs_read=[args.registry, "31_IMPLEMENTATION_DECISIONS.md"], files_created=[args.registry, "configs/models/download_manifest.yaml", "data/manifests/model_resolution_report.json"], tests_run=["Hugging Face immutable metadata resolution"], tests_passed=True, next_phase_ready=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
