from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.artifacts.exporter import export_fixture_artifacts
from vipragsent.artifacts.schemas import validate_artifact_tree
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.preflight import run_preflight
from vipragsent.protocol import compare_frozen_hashes, validate_protocol_resolution


DEFERRED_RUNTIME_REQUIREMENTS = [
    "A100 or A100 MIG runtime",
    "Java 17 and VnCoreNLP resources",
    "PEFT",
    "bitsandbytes",
    "model downloads",
    "real Phase 15 model/tokenizer/QLoRA smoke",
]


def _tracked_files() -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [line for line in result.stdout.splitlines() if line]


def _stable_checksum_files() -> list[Path]:
    roots = ("src", "scripts", "configs", "schemas", "docs", "tests", "prompts", "data/manifests", "data/processed/vipragsent", "data/processed/q3_low_resource_sarcasm")
    root_files = ("README.md", "LICENSE", "Makefile", ".gitignore", ".env.example", "pyproject.toml", "reports/phase_14_5_frozen_hash_baseline.json")
    paths: list[Path] = []
    for relative in roots:
        directory = ROOT / relative
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    paths.extend(ROOT / relative for relative in root_files if (ROOT / relative).exists())
    excluded_fragments = ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "tokenization_cache", "progress", "handoff", "status", ".pyc")
    excluded_names = {"FINAL_CHECKSUMS.sha256", "SETUP_CHECKSUMS.sha256"}
    return sorted({path.resolve() for path in paths if path.name not in excluded_names and not any(fragment in path.as_posix() for fragment in excluded_fragments)}, key=lambda path: path.relative_to(ROOT).as_posix())


def _write_final_checksums() -> dict[str, object]:
    paths = _stable_checksum_files()
    lines = [f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}" for path in paths]
    checksum_path = ROOT / "FINAL_CHECKSUMS.sha256"
    atomic_write_text(checksum_path, "\n".join(lines) + "\n")
    return {"path": checksum_path.relative_to(ROOT).as_posix(), "file_count": len(paths), "sha256": sha256_file(checksum_path), "self_excluded": all("FINAL_CHECKSUMS.sha256" not in line for line in lines), "stable_posix_paths": all("\\" not in line.split("  ", 1)[-1] for line in lines)}


def _secret_findings(tracked: list[str]) -> list[str]:
    findings: list[str] = []
    pattern = re.compile(r"(?:AZURE_OPENAI_API_KEY|KAGGLE_KEY)[ \t]*=[ \t]*[^\s#]+|sk-[A-Za-z0-9_-]{16,}")
    for relative in tracked:
        if relative == ".env" or relative.endswith(".zip"):
            continue
        path = ROOT / relative
        if path.suffix in {".bin", ".pt", ".pth", ".safetensors"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            findings.append(relative)
    return findings


def _regeneration_check() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="vipragsent-audit-") as temporary:
        output = Path(temporary)
        manifest = export_fixture_artifacts(repo_root=ROOT, output_root=output)
        errors = validate_artifact_tree(output / "artifacts")
        return {"fixture_table_regeneration": not errors, "fixture_figure_regeneration": (output / "artifacts/figures/per_phenomenon_f1.svg").exists(), "artifact_schema_errors": errors, "synthetic_only": manifest["synthetic_results"] is True}


def main() -> int:
    tracked = _tracked_files()
    protocol = validate_protocol_resolution(ROOT)
    preflight = run_preflight(ROOT, mode="full")
    inventory = build_expected_runs(ROOT)
    frozen_hashes = compare_frozen_hashes(ROOT)
    regeneration = _regeneration_check()
    fixture_root = ROOT / "runs/fixture"
    fixture_manifest = fixture_root / "FIXTURE_VALIDATION_MANIFEST.json"
    fixture_valid = fixture_manifest.exists() and json.loads(fixture_manifest.read_text(encoding="utf-8")).get("core_experiments_ready") is False
    production_root = ROOT / "experiment_artifacts"
    production_files = [path for path in production_root.rglob("*") if path.is_file()] if production_root.exists() else []
    production_placeholder_files = [path.relative_to(ROOT).as_posix() for path in production_files if "fixture" in path.read_text(encoding="utf-8", errors="ignore").casefold()]
    full_manifest = ROOT / "FINAL_EXPERIMENT_MANIFEST.json"
    secret_findings = _secret_findings(tracked)
    errors: list[str] = []
    if not frozen_hashes["unchanged"]:
        errors.append("frozen data/provenance hash baseline changed: " + ", ".join(frozen_hashes["changed"]))
    if not fixture_valid:
        errors.append("fixture validation manifest is missing or claims core completion")
    if production_placeholder_files:
        errors.append("fixture markers found in production artifact root: " + ", ".join(production_placeholder_files))
    if not regeneration["fixture_table_regeneration"] or not regeneration["fixture_figure_regeneration"]:
        errors.append("temporary deterministic table/figure regeneration failed")
    if secret_findings:
        errors.append("possible secret in tracked files: " + ", ".join(secret_findings))
    if "api.openai.com" in "\n".join((ROOT / "configs").rglob("*") and [path.read_text(encoding="utf-8", errors="ignore") for path in (ROOT / "configs").rglob("*") if path.is_file()] or []):
        errors.append("direct OpenAI endpoint appears in active configuration")
    if any(path.name == "Figure 5.svg" for path in ROOT.rglob("*")):
        errors.append("prohibited Figure 5 artifact exists")
    if any(relative.endswith(".pyc") or "__pycache__" in relative for relative in tracked):
        errors.append("tracked Python bytecode exists")
    if not full_manifest.exists() or json.loads(full_manifest.read_text(encoding="utf-8")).get("mode") != "full":
        full_run_blocker = "complete Phase 16 production manifest is not present"
    else:
        full_run_blocker = None
    blockers = list(preflight.blockers)
    if full_run_blocker:
        blockers.append(full_run_blocker)
    if protocol["scientific_protocol_conflicts"]:
        blockers.append("unresolved scientific protocol conflict")
    passed = not errors and not preflight.passed and bool(blockers)
    checksum = _write_final_checksums()
    report = {
        "status": "BLOCKED" if blockers or not passed else "PASS",
        "EXPERIMENT_REPOSITORY_READY": False,
        "implementation_checks_passed": not errors,
        "blockers": blockers,
        "errors": errors,
        "scientific_protocol_conflicts": protocol["scientific_protocol_conflicts"],
        "protocol_resolution_status": protocol["resolution_status"],
        "full_preflight": preflight.as_dict(),
        "fixture_artifacts_valid": fixture_valid,
        "production_placeholder_files": production_placeholder_files,
        "expected_run_count": inventory["derived_run_count"],
        "expected_run_counts_by_question": inventory["counts_by_question"],
        "frozen_hash_comparison": frozen_hashes,
        "deterministic_regeneration": regeneration,
        "final_checksums": checksum,
        "deferred_server_requirements": DEFERRED_RUNTIME_REQUIREMENTS,
    }
    atomic_write_text(ROOT / "REPRODUCIBILITY_REPORT.md", "# Reproducibility report\n\n" + "\n".join([
        "EXPERIMENT_REPOSITORY_READY=false",
        f"Status: {report['status']}",
        "",
        "## Blockers",
        *[f"- {item}" for item in blockers or ["None"]],
        "",
        "## Scientific protocol conflicts",
        *[f"- {item}" for item in protocol["scientific_protocol_conflicts"] or ["None"]],
        "",
        "The report distinguishes implementation checks from deferred runtime and protocol readiness.",
    ]) + "\n")
    atomic_write_json(ROOT / "RELEASE_MANIFEST.json", report)
    atomic_write_text(ROOT / "EXPERIMENT_MODEL_REGISTRY.md", "# Experiment model registry\n\nThe locked model and tokenizer revisions are recorded in `configs/models/model_registry.yaml`. Weight download and offline smoke verification remain Phase 15 operations.\n")
    atomic_write_text(ROOT / "DATASET_CARD.md", "# Dataset card\n\nThe frozen ViPragSent V8 processed splits and Q3 masks are validated from the supplied archive. Official external test inputs remain access-controlled manual-drop data and are not redistributed by this repository.\n")
    atomic_write_text(ROOT / "KNOWN_LIMITATIONS.md", "# Known limitations\n\nThis audit is blocked until the Phase 15 runtime prerequisites are supplied and sequential runs receive explicit user approval. Human error analysis and qualitative approval remain manual by design.\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
