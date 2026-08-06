from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.phase import write_phase_handoff
from vipragsent.runtime.model_smoke import verify_model_family


def _registry(root: Path, manifest_path: str) -> dict[str, dict[str, object]]:
    payload = yaml.safe_load((root / manifest_path).read_text(encoding="utf-8")) or {}
    raw = payload.get("models", {})
    if isinstance(raw, list):
        return {str(item["name"]): dict(item) for item in raw}
    return {str(name): dict(spec) for name, spec in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 offline tokenizer/model smoke for exactly one family")
    parser.add_argument("--manifest", default="data/model_cache_manifest.json")
    parser.add_argument("--model-family", required=True)
    parser.add_argument("--registry", default="configs/models/model_registry.yaml")
    parser.add_argument("--fake", action="store_true", help="Use the tiny fake loader contract for CPU tests only")
    args = parser.parse_args()
    registry = _registry(ROOT, args.registry)
    report = verify_model_family(ROOT, args.model_family, registry=registry, fake=args.fake)
    atomic_write_json(ROOT / "data/model_smoke_report.json", report | {"selected_model_family": args.model_family, "manifest": args.manifest, "actual_local_loads": not args.fake})
    status = "PASS" if report.get("status") == "PASS" else "BLOCKED"
    handoff = write_phase_handoff(
        "15",
        status,
        inputs_read=[args.manifest, args.registry, "32_RUNTIME_PREFLIGHT_CHECKLIST.md"],
        files_created=["data/model_smoke_report.json", f"data/model_smoke_status/{args.model_family}.json"],
        tests_run=["offline tokenizer load", "offline model load", "forward", "backward", "finite loss", "gradient checks"],
        tests_passed=status == "PASS",
        blockers=list(report.get("blockers", [])),
        next_phase_ready=False,
        model_family=args.model_family,
    )
    report["phase15_handoff_status"] = handoff.status
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
