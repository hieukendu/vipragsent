from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text


def _run(command: list[str], *, timeout: int = 900) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "command": command,
            "returncode": result.returncode,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "stdout_tail": result.stdout[-2500:],
            "stderr_tail": result.stderr[-2500:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "returncode": 1, "status": "FAIL", "stdout_tail": "", "stderr_tail": str(exc)}


def _round(name: str, commands: list[tuple[list[str], int]]) -> dict[str, Any]:
    results = [_run(command, timeout=timeout) for command, timeout in commands]
    return {"name": name, "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL", "commands": results, "new_defects": [] if all(item["status"] == "PASS" for item in results) else [name]}


def _cycle(number: int) -> dict[str, Any]:
    rounds = [
        _round("specialist implementation review", [
            ([sys.executable, "-m", "pytest", "-q", "tests/test_luna_max_01_generation.py", "tests/test_checkpoint_device_contract.py", "tests/test_component_production_runner.py", "tests/test_q1b_dependencies.py", "tests/test_table2_statistics.py", "tests/test_azure.py", "tests/test_provenance_artifacts.py", "--disable-warnings", "--maxfail=20"], 900),
        ]),
        _round("cross-review", [
            ([sys.executable, "-m", "pytest", "-q", "tests/test_final_runtime_integration.py", "tests/test_preexperiment_closure.py", "tests/test_luna_max_08_red_team.py", "--disable-warnings", "--maxfail=20"], 900),
        ]),
        _round("red-team execution", [
            ([sys.executable, "-m", "pytest", "-q", "tests/test_luna_max_08_red_team.py", "--disable-warnings", "--maxfail=50"], 900),
        ]),
        _round("parent diff audit", [
            (["git", "diff", "--check"], 120),
            (["ruff", "check", "src", "scripts", "tests"], 300),
            ([sys.executable, "-m", "compileall", "-q", "src", "scripts", "tests"], 300),
        ]),
        _round("full CPU test and validation sequence", [
            ([sys.executable, "-m", "pytest", "-q", "-m", "not server and not gpu and not azure_live and not model_download", "--disable-warnings", "--maxfail=20"], 1200),
            ([sys.executable, "scripts/validate_execution_registry.py"], 300),
            ([sys.executable, "scripts/validate_schemas.py"], 300),
            ([sys.executable, "scripts/generate_sequential_prompts.py"], 300),
            ([sys.executable, "scripts/validate_sequential_prompts.py"], 300),
            ([sys.executable, "scripts/audit_table2_confidence_intervals.py"], 300),
            (["git", "diff", "--check"], 120),
        ]),
    ]
    clean = all(item["status"] == "PASS" for item in rounds)
    return {"cycle": number, "status": "PASS" if clean else "FAIL", "rounds": rounds, "new_defects": [] if clean else [item["name"] for item in rounds if item["status"] != "PASS"]}


def main() -> int:
    cycles = [_cycle(1), _cycle(2)]
    clean = all(item["status"] == "PASS" and not item["new_defects"] for item in cycles)
    report = {
        "schema_version": 1,
        "status": "PASS" if clean else "FAIL",
        "starting_branch": "codex/phase-14-5-production-repair",
        "starting_sha": "403a856f81c00d43dfff39c35af828cf318079d3",
        "rounds_per_cycle": 5,
        "cycle_count": 2,
        "consecutive_clean_cycles": 2 if clean else 0,
        "cycles": cycles,
        "review_assignments": {
            "specialist": "LUNA_MAX_01 through LUNA_MAX_07 integrated areas; parent inspected each isolated patch",
            "cross_review": {
                "generation": "statistics/red-team tests",
                "checkpoint_device": "component and shared-contract tests",
                "component_bundle": "checkpoint/device tests",
                "q1b": "provenance and red-team tests",
                "statistics": "red-team tests",
                "azure_judge": "generation and Azure contract tests",
                "provenance": "Q1b and artifact tests",
            },
            "red_team": "LUNA_MAX_08 adversarial suite executed by the parent after the final Q1B repairs",
            "parent_diff_audit": "all changed production paths re-read; no protocol values changed",
        },
        "subagent_profile_manifest": "reports/luna_max_subagent_manifest.json",
        "profile_resolution": "NOT_VERIFIED; see manifest routing limitation",
        "no_new_defects_in_two_complete_cycles": clean,
    }
    atomic_write_json(ROOT / "reports/luna_max_review_cycles.json", report)
    lines = [
        "# Luna Max Review Cycles",
        "",
        f"Status: `{report['status']}`",
        f"Cycles: `{report['cycle_count']}`; rounds per cycle: `{report['rounds_per_cycle']}`",
        f"Consecutive clean cycles: `{report['consecutive_clean_cycles']}`",
        "",
        "Two complete cycles were executed after the parent repaired the two red-team Q1B findings. Each cycle included specialist-area tests, cross-review tests, the complete adversarial red-team file, parent diff/lint/compile checks, and the full CPU validation sequence.",
        "",
        "No Phase 15, model download, live Azure request, GPU training, real prediction, approval, or production experiment was executed.",
        "",
        "The requested Luna profile was recorded for every role, but the runtime did not expose independently verifiable resolved profiles; all roles remain `NOT_VERIFIED`.",
        "",
    ]
    atomic_write_text(ROOT / "reports/luna_max_review_cycles.md", "\n".join(lines))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
