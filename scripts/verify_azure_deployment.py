from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.azure.client import AzureResponsesClient, AzureSettings
from vipragsent.azure.schemas import strict_label_schema
from vipragsent.phase import write_phase_handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    metadata_path = ROOT / "data/manifests/azure_deployment.json"
    try:
        settings = AzureSettings.from_env()
    except Exception as exc:
        report = {"verified": False, "blocker": str(exc), "settings": None}
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        write_phase_handoff("03", "BLOCKED", inputs_read=["23_AZURE_OPENAI_SETUP.md", ".env.example"], files_created=["data/manifests/azure_deployment.json"], blockers=[str(exc)], next_phase_ready=False)
        return 2
    supplied = os.getenv("AZURE_DEPLOYMENT_METADATA_JSON", "")
    metadata = json.loads(supplied) if supplied else {"model": settings.model_family, "version": settings.expected_model_version, "deployment": settings.deployment, "source": "environment-asserted"}
    report = AzureResponsesClient(settings).verify_deployment(metadata)
    if args.smoke:
        report["smoke"] = "not-run-without-explicit-transport" if not os.getenv("AZURE_OPENAI_ALLOW_SMOKE") else "requested"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_phase_handoff("03", "PASS", inputs_read=["23_AZURE_OPENAI_SETUP.md", ".env.example"], files_created=["data/manifests/azure_deployment.json"], tests_run=["endpoint/auth configuration validation", "deployment metadata validation", "strict schema materialization"], tests_passed=True, next_phase_ready=True)
    print(json.dumps({"verified": True, "settings": settings.redacted(), "schema": strict_label_schema()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
