from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.config_validation import validate_config_tree
from vipragsent.constants import RUNTIME_PREFLIGHT_CHECKLIST
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.preflight import run_preflight
from vipragsent.protocol import compare_frozen_hashes, validate_protocol_resolution
from vipragsent.runtime.phase15_state import reconcile_phase15_state

DEFERRED_RUNTIME_REQUIREMENTS = [
    "A100 or A100 MIG runtime",
    "Java 17 and VnCoreNLP resources",
    "PEFT",
    "bitsandbytes",
    "model downloads",
    "real Phase 15 model/tokenizer/QLoRA smoke",
]


def _git_clean() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    return not result.stdout.strip()


def _setup_checksum_files() -> list[Path]:
    roots = ("src", "scripts", "configs", "schemas", "docs", "tests", "prompts", "data/manifests", "data/processed/vipragsent", "data/processed/q3_low_resource_sarcasm")
    root_files = ("pyproject.toml", "README.md", "LICENSE", "Makefile", ".gitignore", ".env.example", "reports/phase_14_5_frozen_hash_baseline.json")
    paths: list[Path] = []
    for relative in roots:
        directory = ROOT / relative
        if directory.exists():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    paths.extend(ROOT / relative for relative in root_files if (ROOT / relative).exists())
    excluded = ("__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "tokenized_text", "runs", "results", "experiment_artifacts", "checkpoints", "predictions")
    return sorted({path.resolve() for path in paths if not any(part in path.as_posix() for part in excluded) and path.suffix not in {".pyc", ".pt", ".pth", ".bin", ".safetensors"}}, key=lambda path: path.relative_to(ROOT).as_posix())


def _write_setup_checksums() -> dict[str, object]:
    output = ROOT / "SETUP_CHECKSUMS.sha256"
    lines = [f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}" for path in _setup_checksum_files() if path.resolve() != output.resolve()]
    atomic_write_text(output, "\n".join(lines) + "\n")
    return {"path": output.name, "file_count": len(lines), "self_excluded": all("SETUP_CHECKSUMS.sha256" not in line for line in lines), "stable_posix_paths": all("\\" not in line.split("  ", 1)[-1] for line in lines)}


def main() -> int:
    audit_path = ROOT / "reports/production_implementation_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {"implementation_passed": False, "errors": ["production implementation audit is missing"], "scientific_protocol_conflicts": []}
    config = validate_config_tree(ROOT)
    semantic_path = ROOT / "reports/semantic_config_audit.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8")) if semantic_path.exists() else {"passed": False, "errors": ["semantic audit missing"]}
    fixture_state_path = ROOT / "runs/fixture/dag_state.json"
    fixture_manifest_path = ROOT / "runs/fixture/FIXTURE_VALIDATION_MANIFEST.json"
    fixture_passed = fixture_state_path.exists() and fixture_manifest_path.exists() and json.loads(fixture_state_path.read_text(encoding="utf-8")).get("status") == "PASS" and json.loads(fixture_manifest_path.read_text(encoding="utf-8")).get("core_experiments_ready") is False
    protocol = validate_protocol_resolution(ROOT)
    frozen_hashes = compare_frozen_hashes(ROOT)
    preflight = run_preflight(ROOT, mode="full")
    implementation_errors = list(audit.get("errors", [])) + list(config.get("errors", [])) + list(semantic.get("errors", []))
    if not fixture_passed:
        implementation_errors.append("fixture DAG/manifest validation did not pass")
    if not frozen_hashes["unchanged"]:
        implementation_errors.append("frozen data/provenance hash baseline changed")
    conflicts = protocol["scientific_protocol_conflicts"]
    implementation_ready = bool(audit.get("implementation_passed")) and not implementation_errors
    phase14_ready = implementation_ready and not conflicts
    if implementation_errors:
        status = "FAIL"
        current_phase = "14.5"
        blockers = implementation_errors
    elif conflicts:
        status = "BLOCKED"
        current_phase = "14.5"
        blockers = ["unresolved scientific protocol conflict"]
    else:
        status = "PASS"
        current_phase = "15"
        blockers = []
    setup_manifest = {
        "project": "ViPragSent",
        "setup_implementation_ready": implementation_ready,
        "setup_frozen": phase14_ready,
        "runtime_dependencies_pending": bool(preflight.blockers),
        "setup_ready": phase14_ready,
        "weights_downloaded": False,
        "full_run_started": False,
        "status": status,
        "scientific_protocol_conflicts": conflicts,
        "blockers": blockers,
        "deferred_runtime_requirements": DEFERRED_RUNTIME_REQUIREMENTS,
        "checks": {
            "config_validation": config["passed"],
            "semantic_config_audit": bool(semantic.get("passed")),
            "production_implementation_audit": bool(audit.get("implementation_passed")),
            "fixture_dag": fixture_passed,
            "frozen_data_hashes": frozen_hashes,
            "full_runtime_preflight": preflight.as_dict(),
            "git_worktree_clean_at_freeze": _git_clean(),
            "python": platform.python_version(),
            "azure_env_present": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
        },
        "source_zip_sha256": json.loads((ROOT / "data/manifests/input_checksums.json").read_text(encoding="utf-8"))["ViPragSent_Experiment_Dataset_FINAL_V8.zip"]["sha256"],
        "azure_deployment_manifest": "data/manifests/azure_deployment.json",
    }
    atomic_write_json(ROOT / "SETUP_FREEZE_MANIFEST.json", setup_manifest)
    ready_lines = [
        "# Setup readiness",
        "",
        f"SETUP_IMPLEMENTATION_READY={str(implementation_ready).lower()}",
        f"SETUP_FROZEN={str(phase14_ready).lower()}",
        f"RUNTIME_DEPENDENCIES_PENDING={str(bool(preflight.blockers)).lower()}",
        "",
        "Phase 15 model download and runtime smoke are intentionally deferred.",
        "",
        "## Scientific protocol conflicts",
        *[f"- `{item}`" for item in conflicts or ["None"]],
        "",
        "## Implementation blockers",
        *[f"- {item}" for item in implementation_errors or ["None"]],
        "",
        "## Deferred runtime requirements",
        *[f"- {item}" for item in DEFERRED_RUNTIME_REQUIREMENTS],
    ]
    atomic_write_text(ROOT / "SETUP_READY.md", "\n".join(ready_lines) + "\n")
    checksum = _write_setup_checksums()
    state_path = ROOT / "PROJECT_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"project": "ViPragSent"}
    state.update({
        "current_phase": current_phase,
        "weights_downloaded": False,
        "full_run_started": False,
        "setup_implementation_ready": implementation_ready,
        "setup_frozen": phase14_ready,
        "runtime_dependencies_pending": bool(preflight.blockers),
        "runtime_environment_ready": False,
        "core_experiments_ready": False,
        "manual_paper_analysis_pending": True,
        "scientific_protocol_conflicts": conflicts,
        "blockers": blockers,
    })
    atomic_write_json(state_path, state)
    # Setup refreshes are derivative writers. Preserve a finalized Phase 15
    # handoff when one is already present instead of resetting runtime state.
    reconcile_phase15_state(ROOT, require_local_snapshot=False)
    phase14 = {
        "phase": "14",
        "status": status,
        "inputs_read": ["30_SPEC_COMPLETENESS_AUDIT.md", "31_IMPLEMENTATION_DECISIONS.md", RUNTIME_PREFLIGHT_CHECKLIST, "reports/production_implementation_audit.json"],
        "files_created": ["SETUP_FREEZE_MANIFEST.json", "SETUP_CHECKSUMS.sha256", "SETUP_READY.md", "PROJECT_STATE.json"],
        "tests_run": ["configuration validation", "semantic configuration audit", "production implementation audit", "fixture DAG and manifest validation", "frozen data hash comparison", "full runtime preflight"],
        "tests_passed": implementation_ready,
        "production_implementation_audit_passed": bool(audit.get("implementation_passed")),
        "scientific_protocol_conflicts": conflicts,
        "blockers": blockers,
        "deferred_runtime_requirements": DEFERRED_RUNTIME_REQUIREMENTS,
        "next_phase": "15" if phase14_ready else None,
        "next_phase_ready": phase14_ready,
        "setup_checksum": checksum,
    }
    atomic_write_json(ROOT / "reports/phases/phase_14_handoff.json", phase14)
    atomic_write_text(ROOT / "reports/phases/phase_14_status.md", "# Phase 14 status\n\n- Status: `" + status + "`\n- Tests passed: `" + str(implementation_ready).lower() + "`\n- Setup implementation ready: `" + str(implementation_ready).lower() + "`\n- Setup frozen: `" + str(phase14_ready).lower() + "`\n- Next phase ready: `" + str(phase14_ready).lower() + "`\n\n## Blockers\n\n" + "\n".join(f"- {item}" for item in blockers or ["None"]) + "\n")
    print(json.dumps(setup_manifest, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 2 if status == "BLOCKED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
