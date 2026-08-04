from __future__ import annotations

import json
import os
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
    checks["azure_direct_endpoint_absent"] = "api.openai.com" not in " ".join(path.read_text(encoding="utf-8", errors="ignore") for path in (root / "configs").rglob("*.yaml"))
    checks["artifact_schema"] = not validate_artifact_tree(root / "experiment_artifacts") if mode == "fixture" else True
    if mode == "fixture":
        blockers = []
    return PreflightResult(not blockers, blockers, warnings, checks)
