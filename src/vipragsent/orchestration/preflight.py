from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..artifacts.schemas import validate_artifact_tree
from ..data.loaders import load_vipragsent
from ..protocol import validate_protocol_resolution
from ..runtime.hardware import validate_hardware
from .status import RunExitCode


@dataclass
class PreflightResult:
    passed: bool
    blockers: list[str]
    warnings: list[str]
    checks: dict[str, bool]
    scientific_protocol_conflicts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["exit_code"] = self.exit_code
        return payload

    @property
    def exit_code(self) -> int:
        if self.blockers:
            return RunExitCode.BLOCKED
        if self.scientific_protocol_conflicts:
            return RunExitCode.PROTOCOL_FAILURE
        return RunExitCode.SUCCESS


def run_preflight(root: str | Path = ".", *, mode: str = "full") -> PreflightResult:
    root = Path(root)
    blockers: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    scientific_protocol_conflicts = validate_protocol_resolution(root)["scientific_protocol_conflicts"]
    processed = root / "data" / "processed" / "vipragsent"
    checks["vipragsent_processed"] = processed.exists() and all((processed / f"{split}.csv").exists() for split in ("train", "dev", "test"))
    if not checks["vipragsent_processed"]:
        blockers.append("Processed ViPragSent splits are missing")
    else:
        try:
            load_vipragsent(processed)
        except Exception as exc:
            blockers.append(f"ViPragSent validation failed: {exc}")
    external_manifest = root / "data" / "manifests" / "external_datasets.json"
    checks["external_manifest"] = external_manifest.exists()
    if not checks["external_manifest"]:
        blockers.append("External dataset manifest is missing")
    external_files = [root / "data" / "processed" / "external" / "uit_vsfc" / "test.csv", root / "data" / "processed" / "external" / "uit_vsmec" / "test.csv"]
    bundled_aivivn = root / "data/processed/external/aivivn_human_derived_3way/test.csv"
    checks["external_official_tests"] = all(path.exists() for path in external_files) and bundled_aivivn.exists()
    if mode == "full" and not checks["external_official_tests"]:
        blockers.append("UIT-VSFC and/or UIT-VSMEC official test files are missing; use the manual-drop fallback")
    if external_manifest.exists():
        external = json.loads(external_manifest.read_text(encoding="utf-8"))
        checks["external_provenance_complete"] = all(item.get("status") == "PASS" for item in external.get("datasets", {}).values())
        if mode == "full" and not checks["external_provenance_complete"]:
            blockers.append("External dataset provenance is incomplete; official/manual-drop checks must pass")
    settings_present = bool(os.getenv("AZURE_OPENAI_ENDPOINT")) and bool(os.getenv("AZURE_OPENAI_DEPLOYMENT"))
    checks["azure_credentials"] = settings_present
    if mode == "full" and not settings_present:
        blockers.append("Azure credentials/deployment are not configured")
    azure_manifest = root / "data" / "manifests" / "azure_deployment.json"
    azure_report: dict[str, Any] = {}
    if azure_manifest.exists():
        try:
            azure_report = json.loads(azure_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blockers.append("Azure deployment manifest is invalid JSON")
    checks["azure_deployment_verified"] = azure_report.get("verified") is True
    smoke = azure_report.get("smoke")
    checks["azure_live_smoke"] = isinstance(smoke, dict) and smoke.get("status") == "PASS"
    if mode == "full" and not checks["azure_deployment_verified"]:
        blockers.append("Azure deployment live verification failed or is missing")
    if mode == "full" and not checks["azure_live_smoke"]:
        blockers.append("Azure Responses API plain/strict Structured Output smoke has not passed")
    registry = root / "configs" / "models" / "model_registry.yaml"
    checks["model_revisions_pinned"] = registry.exists() and "revision: null" not in registry.read_text(encoding="utf-8")
    if mode == "full" and not checks["model_revisions_pinned"]:
        blockers.append("Immutable Hugging Face model revisions are not pinned")
    weights_manifest = root / "data" / "model_cache_manifest.json"
    if weights_manifest.exists():
        try:
            json.loads(weights_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            blockers.append("Model cache manifest is invalid JSON")
    registry_payload = yaml.safe_load((root / "configs/models/model_registry.yaml").read_text(encoding="utf-8")) if (root / "configs/models/model_registry.yaml").exists() else {}
    model_families = [str(name) for name in (registry_payload or {}).get("models", {})]
    from ..runtime.model_assets import read_family_status

    family_records = {
        family: {
            "cache": read_family_status(root, family, "cache"),
            "smoke": read_family_status(root, family, "smoke"),
            "batch": read_family_status(root, family, "batch"),
        }
        for family in model_families
    }
    family_weights_verified = bool(family_records) and all(
        record["cache"].get("status") == "PASS"
        and record["smoke"].get("status") == "PASS"
        and record["smoke"].get("actual_local_loads") is True
        and record["batch"].get("status") == "PASS"
        and record["batch"].get("frozen") is True
        for record in family_records.values()
    )
    checks["model_family_statuses"] = family_weights_verified
    checks["model_weights_verified"] = family_weights_verified
    if mode == "full" and not checks["model_weights_verified"]:
        blockers.append("Every model family must pass cache, actual offline smoke, and frozen physical-batch verification")
    hardware = validate_hardware(root)
    checks["cuda_a100_or_mig"] = hardware.get("checks", {}).get("a100_or_approved_mig", False)
    checks["hardware_runtime_exact"] = hardware.get("status") == "PASS"
    if mode == "full" and not checks["cuda_a100_or_mig"]:
        blockers.append("A100 20 GB or an A100 MIG profile is not available")
    java_output = ""
    try:
        java_result = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10, check=False)
        java_output = java_result.stderr + java_result.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    java_match = re.search(r'version "(\d+)', java_output)
    requires_vncorenlp = (root / "configs/runtime/vncorenlp.yaml").exists()
    checks["java_17"] = bool(java_match and java_match.group(1) == "17") if requires_vncorenlp else True
    if mode == "full" and requires_vncorenlp and not checks["java_17"]:
        blockers.append("Java 17 LTS is required for VnCoreNLP")
    vncorenlp_path = Path(os.getenv("VNCORENLP_HOME", str(root / "data/model_cache/vncorenlp")))
    checks["vncorenlp_resources"] = vncorenlp_path.exists() if requires_vncorenlp else True
    if mode == "full" and requires_vncorenlp and not checks["vncorenlp_resources"]:
        blockers.append("Pinned VnCoreNLP RDRSegmenter resources are missing")
    checks["peft_installed"] = bool(importlib.util.find_spec("peft"))
    checks["bitsandbytes_installed"] = bool(importlib.util.find_spec("bitsandbytes"))
    if mode == "full" and not checks["peft_installed"]:
        blockers.append("PEFT is not installed for QLoRA")
    if mode == "full" and not checks["bitsandbytes_installed"]:
        blockers.append("bitsandbytes is not installed for NF4 QLoRA")
    prompt_root = root / "data/manifests/prompts"
    required_prompts = [prompt_root / name for name in ("pragmatic_v1.json", "polarity_v1.json", "emotion_v1.json", "q3_budget_32_v1.json", "q3_budget_64_v1.json", "q3_budget_128_v1.json", "q3_budget_256_v1.json", "q3_budget_512_v1.json", "q3_budget_full_v1.json")]
    checks["prompt_manifests"] = all(path.exists() for path in required_prompts)
    if mode == "full" and not checks["prompt_manifests"]:
        blockers.append("Frozen task-specific Azure prompt manifests are incomplete")
    checks["model_smoke_report"] = bool(family_records) and all(
        record["smoke"].get("status") == "PASS" and record["smoke"].get("actual_local_loads") is True
        for record in family_records.values()
    )
    if mode == "full" and not checks["model_smoke_report"]:
        blockers.append("Phase 15 actual per-family model/tokenizer smoke reports are missing or incomplete")
    checks["azure_direct_endpoint_absent"] = "api.openai.com" not in " ".join(path.read_text(encoding="utf-8", errors="ignore") for path in (root / "configs").rglob("*.yaml"))
    artifact_root = root / "experiment_artifacts"
    artifact_errors = validate_artifact_tree(artifact_root) if artifact_root.exists() and any(path.is_file() for path in artifact_root.rglob("*")) else []
    checks["artifact_schema"] = not artifact_errors
    if mode == "full" and artifact_errors:
        blockers.extend(f"Artifact schema error: {error}" for error in artifact_errors)
    rationale_path = root / "data/processed/rationales/azure_rationale_input_train.jsonl"
    rationale_text = rationale_path.read_text(encoding="utf-8", errors="ignore") if rationale_path.exists() else ""
    checks["active_rationale_manifest_sanitized"] = "TO_BE_FILLED_WITH_EXACT_GPT_4O_MINI_SNAPSHOT" not in rationale_text
    if mode == "full" and not checks["active_rationale_manifest_sanitized"]:
        blockers.append("Active rationale manifest contains a legacy generator placeholder")
    checks["expected_run_inventory"] = (root / "reports/expected_experiment_runs.json").exists()
    if mode == "full" and not checks["expected_run_inventory"]:
        blockers.append("Expected experiment run inventory is missing")
    if mode == "fixture":
        blockers = []
        scientific_protocol_conflicts = []
    return PreflightResult(not blockers and not scientific_protocol_conflicts, blockers, warnings, checks, scientific_protocol_conflicts)
