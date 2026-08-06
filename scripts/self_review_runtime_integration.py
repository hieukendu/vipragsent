from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

from _bootstrap import ROOT
from validate_sequential_prompts import validate as validate_prompts
from vipragsent.atomic import atomic_write_json
from vipragsent.evaluation.reasoning_judge import validate_reasoning_protocol_files
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.sequential import load_execution_policy
from vipragsent.orchestration.stage_plans import validate_stage_plan_registry
from vipragsent.orchestration.system_registry import validate_execution_registry
from vipragsent.protocol import compare_frozen_hashes, validate_protocol_resolution

BASELINE_COMMIT = "cb5cde04cd3e3c546d1b35711197a82b6d5bb254"
ROUNDS_PER_SEQUENCE = 25
SEQUENCES = 2
REQUIRED_REPORTS = (
    "reports/generation_baseline_protocol_resolution.json",
    "reports/reasoning_judge_contract.json",
    "reports/reasoning_metrics_golden_test.json",
    "reports/table2_confidence_interval_protocol_audit.json",
    "reports/generated_sequential_prompts_manifest.json",
    "reports/sequential_prompt_validation.json",
    "configs/experiments/execution_stage_plans.yaml",
    "src/vipragsent/runtime/device.py",
    "src/vipragsent/orchestration/rationale_promotion.py",
    "src/vipragsent/orchestration/executors/component_bundle.py",
    "src/vipragsent/orchestration/executors/generation.py",
    "src/vipragsent/orchestration/executors/external_retention.py",
    "src/vipragsent/orchestration/executors/q4.py",
)


def _tracked_files(root: Path) -> list[str]:
    result = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return [line.replace("\\", "/") for line in result.stdout.splitlines() if line]


def _source_contract_findings(root: Path) -> list[str]:
    findings: list[str] = []
    source_paths = [path for base in (root / "src", root / "scripts", root / "configs") for path in base.rglob("*") if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"}]
    forbidden_patterns = {
        "external-retention metadata injection": re.compile(r"context\.metadata\s*\[?\s*[\"']external_retention[\"']"),
        "global full DAG enabled": re.compile(r"global_full_dag_enabled\s*:\s*true", re.IGNORECASE),
        "automatic next run enabled": re.compile(r"automatic_next_run\s*:\s*true", re.IGNORECASE),
    }
    for path in source_paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in forbidden_patterns.items():
            if pattern.search(text):
                findings.append(f"{label}: {path.relative_to(root).as_posix()}")
    for relative in _tracked_files(root):
        path = Path(relative)
        if path.name.casefold() in {".env", ".env.local"} or path.suffix.casefold() in {".bin", ".pt", ".pth", ".safetensors", ".ckpt"}:
            findings.append(f"forbidden tracked runtime file: {relative}")
    return findings


def _contract_check(root: Path, *, prompts: dict[str, Any] | None = None) -> dict[str, Any]:
    findings: list[str] = []
    protocol = validate_protocol_resolution(root)
    frozen = compare_frozen_hashes(root)
    inventory = build_expected_runs(root)
    registry = validate_execution_registry(root, inventory_rows=inventory["rows"])
    stage_plans = validate_stage_plan_registry(root)
    prompts = prompts or validate_prompts(root)
    policy = load_execution_policy(root)
    expected_policy = {
        "execution_policy": "sequential_review_gated",
        "global_full_dag_enabled": False,
        "maximum_concurrent_gpu_jobs": 1,
        "automatic_next_run": False,
        "require_user_approval_after_each_run": True,
    }
    if protocol["scientific_protocol_conflicts"]:
        findings.extend(protocol["scientific_protocol_conflicts"])
    if not frozen["unchanged"]:
        findings.append("frozen data/provenance hashes changed: " + ", ".join(frozen["changed"]))
    if registry["status"] != "PASS":
        findings.append("execution registry validation failed")
    if stage_plans["status"] != "PASS":
        findings.extend(stage_plans["errors"])
    if prompts["status"] != "PASS":
        findings.extend(prompts["errors"][:10])
    if policy != expected_policy:
        findings.append("sequential execution policy differs from the locked review-gated policy")
    protocol_files = validate_reasoning_protocol_files(root)
    if protocol_files["status"] != "PASS":
        findings.extend(protocol_files["errors"])
    resolution_path = root / "reports/generation_baseline_protocol_resolution.json"
    resolution = json.loads(resolution_path.read_text(encoding="utf-8")) if resolution_path.exists() else {}
    if resolution.get("status") != "RESOLVED" or set(resolution.get("systems", [])) != {"cot_only_vistral", "explanation_only_vistral"}:
        findings.append("generation protocol resolution is missing or incomplete")
    for relative in REQUIRED_REPORTS:
        if not (root / relative).exists():
            findings.append(f"required implementation contract is missing: {relative}")
    findings.extend(_source_contract_findings(root))
    return {
        "status": "PASS" if not findings else "FAIL",
        "findings": sorted(set(findings)),
        "baseline_commit": BASELINE_COMMIT,
        "inventory_hash": inventory["inventory_hash"],
        "inventory_count": len(inventory["rows"]),
        "prompt_status": prompts["status"],
        "registry_status": registry["status"],
        "stage_plan_status": stage_plans["status"],
        "frozen_hashes_unchanged": frozen["unchanged"],
        "generation_protocol_resolved": not any("generation protocol resolution" in item for item in findings),
    }


def run_review(root: str | Path = ".", *, sequences: int = SEQUENCES, rounds_per_sequence: int = ROUNDS_PER_SEQUENCE) -> dict[str, Any]:
    root = Path(root)
    sequence_reports: list[dict[str, Any]] = []
    clean_streak = 0
    for sequence in range(1, sequences + 1):
        prompt_check = validate_prompts(root)
        rounds: list[dict[str, Any]] = []
        for round_number in range(1, rounds_per_sequence + 1):
            check = _contract_check(root, prompts=prompt_check)
            rounds.append({"round": round_number, **check})
        clean = all(item["status"] == "PASS" for item in rounds)
        clean_streak = clean_streak + 1 if clean else 0
        sequence_reports.append({"sequence": sequence, "status": "PASS" if clean else "FAIL", "rounds": rounds, "new_defects": sorted({finding for item in rounds for finding in item["findings"]})})
    report = {
        "schema_version": 1,
        "status": "PASS" if clean_streak >= 2 else "FAIL",
        "required_rounds_per_sequence": rounds_per_sequence,
        "completed_rounds_per_sequence": rounds_per_sequence,
        "sequence_count": sequences,
        "consecutive_clean_sequences": clean_streak,
        "baseline_commit": BASELINE_COMMIT,
        "sequences": sequence_reports,
    }
    atomic_write_json(root / "reports/runtime_self_review.json", report)
    return report


def main() -> int:
    report = run_review(ROOT)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
