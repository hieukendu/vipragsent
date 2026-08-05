from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json, atomic_write_text
from vipragsent.hashing import sha256_file, sha256_json

TESTS = (
    "tests/test_provenance_artifacts.py::test_explanation_manifest_truthful_rationale_inference",
    "tests/test_provenance_artifacts.py::test_explanation_validator_accepts_truthful_provenance",
    "tests/test_provenance_artifacts.py::test_cot_manifest_marks_native_causal_generation",
    "tests/test_provenance_artifacts.py::test_generation_provenance_system_specific",
)


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def _run_test(test_name: str, source_hash: str) -> dict[str, Any]:
    command = [sys.executable, "-m", "pytest", "-q", test_name]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180, check=False)
        stdout = result.stdout[-4000:]
        stderr = result.stderr[-4000:]
        return {
            "test_name": test_name,
            "command": command,
            "fixture_input_sha256": sha256_json({"test_name": test_name, "test_source_sha256": source_hash}),
            "observed_output_sha256": sha256_json({"returncode": result.returncode, "stdout": stdout, "stderr": stderr}),
            "returncode": result.returncode,
            "status": "PASS" if result.returncode == 0 else "FAIL",
            "stdout_tail": stdout,
            "stderr_tail": stderr,
            "synthetic": True,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "test_name": test_name,
            "command": command,
            "fixture_input_sha256": sha256_json({"test_name": test_name, "test_source_sha256": source_hash}),
            "observed_output_sha256": sha256_json({"error": str(exc)}),
            "returncode": 1,
            "status": "FAIL",
            "stdout_tail": "",
            "stderr_tail": str(exc),
            "synthetic": True,
        }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Local production correctness closure",
        "",
        f"Status: `{report['status']}`",
        f"Audited code SHA: `{report['code_commit_at_audit']}`",
        "",
        "This is production-shaped synthetic evidence only. It is not a real production run, approval, or claim that Phase 15 has passed.",
        "",
        "## Evidence",
        "",
        "| Defect | Test | Input hash | Output hash | Status |",
        "|---|---|---|---|---|",
    ]
    for item in report["evidence"]:
        lines.append(
            f"| {item['defect']} | `{item['test_name']}` | `{item['fixture_input_sha256']}` | `{item['observed_output_sha256']}` | `{item['status']}` |"
        )
    lines.extend(
        [
            "",
            "## Safety boundary",
            "",
            "Phase 15, model downloads, live Azure requests, GPU training, real predictions, approvals, and experiments were not executed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    source_hash = sha256_file(ROOT / "tests/test_provenance_artifacts.py")
    results = [_run_test(test_name, source_hash) for test_name in TESTS]
    by_name = {item["test_name"]: item for item in results}
    evidence = [
        {"defect": "Defect 9", **by_name[TESTS[0]]},
        {"defect": "Defect 9", **by_name[TESTS[1]]},
        {"defect": "Defect 9", **by_name[TESTS[2]]},
        {"defect": "Defect 10", **by_name[TESTS[3]]},
    ]
    status = "PASS" if all(item["status"] == "PASS" for item in evidence) else "FAIL"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "code_commit_at_audit": _git_sha(),
        "evidence_scope": "role-07 production-shaped synthetic CPU tests",
        "production_proof": False,
        "defects_covered": ["Defect 9", "Defect 10"],
        "test_source_sha256": source_hash,
        "evidence": evidence,
        "execution_safety": {
            "phase15_executed": False,
            "model_downloaded": False,
            "azure_request_made": False,
            "gpu_training_executed": False,
            "real_predictions_generated": False,
            "approval_recorded": False,
            "experiment_started": False,
        },
    }
    atomic_write_json(ROOT / "reports/local_production_correctness_closure.json", report)
    atomic_write_text(ROOT / "reports/local_production_correctness_closure.md", _markdown(report))
    atomic_write_json(
        ROOT / "reports/provenance_truthfulness_audit.json",
        {
            "schema_version": 1,
            "status": status,
            "code_commit_at_audit": report["code_commit_at_audit"],
            "scope": "Defect 9 system-specific provenance and Defect 10 executable evidence",
            "production_proof": False,
            "evidence": evidence,
            "execution_safety": report["execution_safety"],
        },
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
