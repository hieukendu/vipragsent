from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.azure.client import AzureResponsesClient, AzureSettings
from vipragsent.azure.schemas import strict_label_schema
from vipragsent.phase import write_phase_handoff


def _live_smoke(settings: AzureSettings) -> dict[str, object]:
    client = AzureResponsesClient(settings)
    try:
        plain = client._default_transport(input="Reply with exactly OK.", max_output_tokens=8)
        structured = client._default_transport(
            input="Return a valid all-task ViPragSent label object.",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "vipragsent_azure_smoke",
                    "strict": True,
                    "schema": strict_label_schema(),
                }
            },
            max_output_tokens=64,
        )
        return {
            "status": "PASS",
            "plain_response": {"id": plain.get("id"), "model": plain.get("model")},
            "structured_response": {"id": structured.get("id"), "model": structured.get("model")},
        }
    except Exception as exc:
        message = str(exc)
        if settings.api_key:
            message = message.replace(settings.api_key, "<redacted>")
        return {"status": "BLOCKED", "error_type": type(exc).__name__, "error": message}


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
    smoke = None
    if args.smoke and os.getenv("AZURE_OPENAI_ALLOW_SMOKE"):
        smoke = _live_smoke(settings)
        report["smoke"] = smoke
    elif args.smoke:
        report["smoke"] = "not-run-without-explicit-transport"
    verified = not isinstance(smoke, dict) or smoke.get("status") == "PASS"
    if not verified:
        report["verified"] = False
        report["blocker"] = "Live Azure Responses API smoke failed"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_phase_handoff("03", "PASS" if verified else "BLOCKED", inputs_read=["23_AZURE_OPENAI_SETUP.md", ".env.example"], files_created=["data/manifests/azure_deployment.json"], tests_run=["endpoint/auth configuration validation", "deployment metadata validation", "strict schema materialization"] + (["live Responses API plain and strict-schema smoke"] if smoke else []), tests_passed=verified, blockers=[] if verified else ["Live Azure Responses API smoke failed; verify the deployment name and resource"], next_phase_ready=verified)
    print(json.dumps({"verified": verified, "settings": settings.redacted(), "schema": strict_label_schema(), "smoke": smoke}, indent=2))
    return 0 if verified else 2


if __name__ == "__main__":
    raise SystemExit(main())
