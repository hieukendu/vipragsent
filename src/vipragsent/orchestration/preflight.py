from __future__ import annotations

import json
import os
import re
import subprocess
import importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ..artifacts.schemas import validate_artifact_tree
from ..data.loaders import load_vipragsent


@dataclass
class PreflightResult:
    passed: bool
    blockers: list[str]
    warnings: list[str]
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_preflight(root: str | Path = ".", *, mode: str = "full") -> PreflightResult:
    root = Path(root)
    blockers: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
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
    checks["external_official_tests"] = all(path.exists() for path in external_files)
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
    registry = root / "configs" / "models" / "model_registry.yaml"
    checks["model_revisions_pinned"] = registry.exists() and "revision: null" not in registry.read_text(encoding="utf-8")
    if mode == "full" and not checks["model_revisions_pinned"]:
        blockers.append("Immutable Hugging Face model revisions are not pinned")
    weights_manifest = root / "data" / "model_cache_manifest.json"
    checks["model_weights_verified"] = weights_manifest.exists()
    if mode == "full" and not checks["model_weights_verified"]:
        blockers.append("Model weights have not passed Phase 15 offline verification")
    try:
        import torch

        device_names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    except ImportError:
        device_names = []
    checks["cuda_a100_or_mig"] = any("A100" in name or "MIG" in name for name in device_names)
    if mode == "full" and not checks["cuda_a100_or_mig"]:
        blockers.append("A100 20 GB or an A100 MIG profile is not available")
    java_output = ""
    try:
        java_result = subprocess.run(["java", "-version"], capture_output=True, text=True, timeout=10, check=False)
        java_output = java_result.stderr + java_result.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    java_match = re.search(r'version "(\d+)', java_output)
    checks["java_17"] = bool(java_match and java_match.group(1) == "17")
    if mode == "full" and not checks["java_17"]:
        blockers.append("Java 17 LTS is required for VnCoreNLP")
    vncorenlp_path = Path(os.getenv("VNCORENLP_HOME", str(root / "data/model_cache/vncorenlp")))
    checks["vncorenlp_resources"] = vncorenlp_path.exists()
    if mode == "full" and not checks["vncorenlp_resources"]:
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
    checks["model_smoke_report"] = (root / "data/model_smoke_report.json").exists()
    if mode == "full" and not checks["model_smoke_report"]:
        blockers.append("Phase 15 model/tokenizer smoke report is missing")
    checks["azure_direct_endpoint_absent"] = "api.openai.com" not in " ".join(path.read_text(encoding="utf-8", errors="ignore") for path in (root / "configs").rglob("*.yaml"))
    checks["artifact_schema"] = not validate_artifact_tree(root / "experiment_artifacts") if mode == "fixture" else True
    if mode == "fixture":
        blockers = []
    return PreflightResult(not blockers, blockers, warnings, checks)
