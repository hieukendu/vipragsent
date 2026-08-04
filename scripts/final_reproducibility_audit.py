from __future__ import annotations

import json
import re
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.artifacts.schemas import validate_artifact_tree
from vipragsent.orchestration.preflight import run_preflight


def main() -> int:
    blockers: list[str] = []
    artifact_errors = validate_artifact_tree(ROOT / "experiment_artifacts")
    if artifact_errors:
        blockers.extend(artifact_errors)
    if (ROOT / "experiment_artifacts" / "figures" / "Figure 5.svg").exists():
        blockers.append("Prohibited Figure 5 artifact exists")
    configs = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (ROOT / "configs").rglob("*") if path.is_file())
    if "explanation_at_inference" in configs:
        blockers.append("Prohibited explanation-at-inference config is present")
    if "api.openai.com" in configs:
        blockers.append("Direct OpenAI endpoint appears in active configuration")
    secret_pattern = re.compile(r"sk-[A-Za-z0-9_-]{16,}")
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts and ".codex_input" not in path.parts and path.suffix not in {".zip", ".bin", ".safetensors"}:
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
                assigned_secret = any(
                    line.split("=", 1)[1].strip()
                    for line in content.splitlines()
                    if line.strip().startswith("AZURE_OPENAI_API_KEY=") and line.split("=", 1)[1].strip()
                )
                if secret_pattern.search(content) or assigned_secret:
                    blockers.append(f"Possible secret in {path.relative_to(ROOT)}")
            except OSError:
                pass
    preflight = run_preflight(ROOT, mode="full")
    blockers.extend(preflight.blockers)
    passed = not blockers
    report = {"EXPERIMENT_REPOSITORY_READY": passed, "blockers": blockers, "fixture_artifacts_valid": not artifact_errors, "full_preflight": preflight.as_dict()}
    (ROOT / "REPRODUCIBILITY_REPORT.md").write_text("# Reproducibility report\n\nEXPERIMENT_REPOSITORY_READY=" + str(passed).lower() + "\n\n" + "\n".join(f"- {item}" for item in blockers or ["None"]) + "\n", encoding="utf-8")
    (ROOT / "RELEASE_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ROOT / "EXPERIMENT_MODEL_REGISTRY.md").write_text("# Experiment model registry\n\nSee `configs/models/model_registry.yaml`. Immutable revisions remain a preflight requirement.\n", encoding="utf-8")
    (ROOT / "DATASET_CARD.md").write_text("# Dataset card\n\nViPragSent V8 is validated from the supplied archive. Restricted external datasets remain manual-drop inputs.\n", encoding="utf-8")
    (ROOT / "KNOWN_LIMITATIONS.md").write_text("# Known limitations\n\nHuman error analysis and qualitative approval remain manual by design. Full execution also requires external data, Azure access, and Phase 15 model verification.\n", encoding="utf-8")
    (ROOT / "FINAL_CHECKSUMS.sha256").write_text("", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
