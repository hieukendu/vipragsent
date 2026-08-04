from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

from _bootstrap import ROOT
from vipragsent.config_validation import validate_config_tree
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.preflight import run_preflight
from vipragsent.phase import write_phase_handoff


def _git_clean() -> bool:
    result = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False)
    return not result.stdout.strip()


def _setup_files() -> list[Path]:
    files: list[Path] = []
    for relative_root in ("configs", "src", "scripts", "tests", "data/processed", "data/manifests"):
        base = ROOT / relative_root
        if base.exists():
            files.extend(path for path in base.rglob("*") if path.is_file())
    files.extend(ROOT / name for name in ("pyproject.toml", "README.md", ".env.example", "PROJECT_STATE.json"))
    return sorted(path for path in files if path.exists())


def main() -> int:
    preflight = run_preflight(ROOT, mode="full")
    config = validate_config_tree(ROOT)
    semantic_path = ROOT / "reports/semantic_config_audit.json"
    semantic = json.loads(semantic_path.read_text(encoding="utf-8")) if semantic_path.exists() else {"passed": False, "errors": ["semantic audit missing"]}
    fixture_state_path = ROOT / "runs/fixture/dag_state.json"
    fixture_passed = fixture_state_path.exists() and json.loads(fixture_state_path.read_text(encoding="utf-8")).get("status") == "PASS"
    blockers = list(preflight.blockers) + list(config["errors"]) + list(semantic.get("errors", []))
    if not fixture_passed:
        blockers.append("fixture DAG state is not PASS")
    setup_manifest = {
        "project": "ViPragSent",
        "setup_freeze_attempted": True,
        "setup_ready": not blockers,
        "weights_downloaded": False,
        "full_run_started": False,
        "checks": {
            "config_validation": config["passed"],
            "semantic_config_audit": bool(semantic.get("passed")),
            "fixture_dag": fixture_passed,
            "full_preflight": preflight.as_dict(),
            "git_worktree_clean_at_freeze": _git_clean(),
            "python": platform.python_version(),
            "azure_env_present": bool(os.getenv("AZURE_OPENAI_ENDPOINT")),
        },
        "blockers": blockers,
        "source_zip_sha256": json.loads((ROOT / "data/manifests/input_checksums.json").read_text(encoding="utf-8"))["ViPragSent_Experiment_Dataset_FINAL_V8.zip"]["sha256"],
    }
    (ROOT / "SETUP_FREEZE_MANIFEST.json").write_text(json.dumps(setup_manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["# Setup readiness", "", f"SETUP_READY={str(not blockers).lower()}", "", "The complete setup is intentionally not marked ready until all runtime preflight prerequisites pass.", "", "## Blockers"]
    lines.extend(f"- {item}" for item in blockers or ["None"])
    (ROOT / "SETUP_READY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    checksum_lines = []
    for path in _setup_files():
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(ROOT).as_posix()}")
    (ROOT / "SETUP_CHECKSUMS.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    state_path = ROOT / "PROJECT_STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"project": "ViPragSent"}
    state.update({"current_phase": "15", "weights_downloaded": False, "full_run_started": False, "setup_frozen": False, "core_experiments_ready": False, "blockers": blockers})
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    write_phase_handoff(
        "14",
        "PASS" if not blockers else "BLOCKED",
        inputs_read=["30_SPEC_COMPLETENESS_AUDIT.md", "31_IMPLEMENTATION_DECISIONS.md", "32_RUNTIME_PREFLIGHT_CHECKLIST.md"],
        files_created=["SETUP_FREEZE_MANIFEST.json", "SETUP_CHECKSUMS.sha256", "SETUP_READY.md", "reports/semantic_config_audit.json"],
        tests_run=["configuration validation", "semantic configuration audit", "fixture DAG state check", "full runtime preflight"],
        tests_passed=config["passed"] and bool(semantic.get("passed")) and fixture_passed,
        blockers=blockers,
        next_phase_ready=not blockers,
    )
    print(json.dumps(setup_manifest, indent=2))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
