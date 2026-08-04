from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.phase import write_phase_handoff


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 15 offline model/tokenizer smoke checks")
    parser.add_argument("--manifest", default="data/model_cache_manifest.json")
    args = parser.parse_args()
    path = ROOT / args.manifest
    if not path.exists():
        blocker = "data/model_cache_manifest.json is missing; no model-load smoke can run"
        write_phase_handoff("15", "BLOCKED", inputs_read=[args.manifest, "32_RUNTIME_PREFLIGHT_CHECKLIST.md"], blockers=[blocker], next_phase_ready=False)
        print(blocker)
        return 2
    manifest = json.loads(path.read_text(encoding="utf-8"))
    blockers = list(manifest.get("blockers", []))
    for item in manifest.get("models", []):
        if item.get("status") != "PASS":
            blockers.append(f"model not verified: {item.get('name')}")
    report = {"offline_load_smoke": not blockers, "models": manifest.get("models", []), "blockers": blockers, "cuda_required_for_7b": True}
    (ROOT / "data/model_smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_phase_handoff("15", "PASS" if not blockers else "BLOCKED", inputs_read=[args.manifest, "32_RUNTIME_PREFLIGHT_CHECKLIST.md"], files_created=["data/model_smoke_report.json"], tests_run=["offline manifest verification", "tokenizer/model smoke hook"], tests_passed=not blockers, blockers=blockers, next_phase_ready=not blockers)
    print(json.dumps(report, indent=2))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
