from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from _bootstrap import ROOT
from vipragsent.atomic import atomic_write_json
from vipragsent.hashing import sha256_file
from vipragsent.orchestration.inventory import build_expected_runs
from vipragsent.orchestration.sequential import (
    AZURE_JOB_TYPES,
    build_azure_job_inventory,
    load_execution_policy,
)


def _expected_entries(root: Path, inventory: dict[str, Any]) -> dict[str, Any]:
    expected = {f"experiment:{row['experiment_id']}": row for row in inventory["rows"]}
    expected |= {f"azure_job:{job['job_id']}": job for job in build_azure_job_inventory()}
    registry = yaml.safe_load((root / "configs/models/model_registry.yaml").read_text(encoding="utf-8"))
    expected |= {f"phase15:{name}": dict(model) | {"name": name} for name, model in registry["models"].items()}
    expected |= {f"aggregation:{question}": {"research_question": question} for question in ("Q1a", "Q1b", "Q2", "Q3", "Q4")}
    expected["final_aggregation:all"] = {"research_question": "all"}
    return expected


def _validate_prompt_text(path: Path, item: dict[str, Any], expected: dict[str, Any], all_experiment_ids: set[str], all_model_families: set[str], errors: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    kind = item.get("kind")
    identifier = str(item.get("id"))
    if sha256_file(path) != item.get("sha256"):
        errors.append(f"prompt hash mismatch: {path}")
    if kind == "experiment":
        mentioned = {value for value in all_experiment_ids if value in text}
        if mentioned != {identifier}:
            errors.append(f"{path}: prompt names experiment IDs {sorted(mentioned)}, expected only {identifier}")
        required = (identifier, "--stage preflight", "--stage all", "--resume", "print_run_review_summary.py", "PENDING_USER_APPROVAL", "stop")
    elif kind == "azure_job":
        required = (identifier, "--stage preflight", "--stage all", "--resume", "PENDING_USER_APPROVAL", "stop")
    elif kind == "phase15":
        mentioned = {value for value in all_model_families if value in text}
        if mentioned != {identifier}:
            errors.append(f"{path}: prompt names model families {sorted(mentioned)}, expected only {identifier}")
        required = (identifier, "download_all_models.py", "--model-family", "verify_model_smoke.py", "--model-family", "PENDING_USER_APPROVAL", "stop")
    else:
        required = ("aggregate_approved_runs.py", "APPROVED", "PENDING_USER_APPROVAL", "stop")
    for fragment in required:
        if fragment.casefold() not in text.casefold():
            errors.append(f"{path}: missing required instruction {fragment!r}")
    if kind in {"experiment", "azure_job", "aggregation", "final_aggregation"} and ("run_all_experiments.py" in text or "--mode full" in text):
        errors.append(f"{path}: global full-DAG execution appears in a sequential prompt")
    if kind == "experiment" and ("download_all_models.py" in text or "verify_model_smoke.py" in text):
        errors.append(f"{path}: Phase 15 execution appears in an experiment prompt")
    if kind == "azure_job" and expected.get("job_type") not in AZURE_JOB_TYPES:
        errors.append(f"{path}: unsupported Azure job type")


def validate(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    errors: list[str] = []
    try:
        policy = load_execution_policy(root)
    except Exception as exc:
        policy = {}
        errors.append(str(exc))
    manifest_path = root / "reports/sequential_prompt_manifest.json"
    alias_path = root / "reports/generated_sequential_prompts_manifest.json"
    if not manifest_path.exists():
        return {"status": "FAIL", "errors": ["sequential prompt manifest is missing"], "prompt_count": 0}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not alias_path.exists() or json.loads(alias_path.read_text(encoding="utf-8")) != manifest:
        errors.append("generated sequential prompt manifest alias is missing or stale")
    inventory = build_expected_runs(root)
    expected = _expected_entries(root, inventory)
    if manifest.get("execution_policy") != policy:
        errors.append("prompt manifest execution policy does not match the locked policy")
    if manifest.get("inventory_hash") != inventory["inventory_hash"]:
        errors.append("prompt manifest inventory hash is stale")
    seen: set[str] = set()
    all_experiment_ids = {str(row["experiment_id"]) for row in inventory["rows"]}
    registry = yaml.safe_load((root / "configs/models/model_registry.yaml").read_text(encoding="utf-8"))
    all_model_families = set(registry["models"])
    prompts = manifest.get("prompts", [])
    for item in prompts:
        key = f"{item.get('kind')}:{item.get('id')}"
        if key in seen:
            errors.append(f"duplicate prompt entry: {key}")
        seen.add(key)
        if key not in expected:
            errors.append(f"prompt has no current inventory entry: {key}")
            continue
        path = root / str(item.get("path", ""))
        if not path.exists():
            errors.append(f"prompt file is missing: {path}")
            continue
        _validate_prompt_text(path, item, expected[key], all_experiment_ids, all_model_families, errors)
    missing = sorted(set(expected) - seen)
    errors.extend(f"missing generated prompt: {key}" for key in missing)
    counts = {
        "experiment_count": len(inventory["rows"]),
        "azure_job_count": len(build_azure_job_inventory()),
        "phase15_model_count": len(all_model_families),
        "aggregation_count": 5,
    }
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        **counts,
        "prompt_count": len(prompts),
        "expected_prompt_count": len(expected),
        "inventory_hash": inventory["inventory_hash"],
        "execution_policy": policy,
        "approval_contract": manifest.get("approval_contract"),
    }
    atomic_write_json(root / "reports/sequential_prompt_validation.json", report)
    return report


def main() -> int:
    report = validate(ROOT)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
