from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ..atomic import atomic_write_json, atomic_write_text
from ..hashing import sha256_file, sha256_json
from ..protocol import validate_protocol_resolution


SEQUENTIAL_STAGES = (
    "preflight",
    "train_or_run",
    "evaluate_dev",
    "freeze_selection",
    "evaluate_test",
    "export_artifacts",
    "validate_artifacts",
    "generate_review_summary",
)

AZURE_JOB_TYPES = (
    "rationale_generation",
    "pragmatic_zero_shot",
    "pragmatic_8_shot",
    "polarity_dedicated_prompt",
    "emotion_dedicated_prompt",
    "q3_budget_specific_pragmatic_8_shot",
)

REVIEW_FIELDS = (
    "experiment_id", "research_question", "system_id", "display_name", "variant", "backbone", "seed", "budget",
    "execution_mode", "run_status", "user_review_status", "dataset_fingerprint", "split_hashes", "q3_mask_hash",
    "model_repository", "model_revision", "tokenizer_revision", "preprocessing_name", "preprocessing_version",
    "configuration_hash", "code_commit", "optimizer", "learning_rate", "weight_decay", "scheduler", "warmup_ratio",
    "precision", "physical_batch_size", "gradient_accumulation_steps", "effective_batch_size", "maximum_epochs",
    "actual_epochs", "best_epoch", "early_stopping_reason", "primary_dev_selection_metric", "best_dev_metric",
    "best_dev_loss", "frozen_thresholds", "per_label_dev_metrics", "per_label_test_metrics", "macro_pragmatic_f1",
    "polarity_macro_f1", "emotion_macro_f1", "pragmatic_ece", "invalid_output_rate", "wall_clock_time_seconds",
    "successful_gpu_hours", "failed_retried_gpu_hours", "peak_vram_gb", "azure_request_token_usage",
    "checkpoint_path", "artifact_paths", "artifact_sha256", "warnings", "blockers", "validation_status",
    "RUN_STATUS", "USER_REVIEW_STATUS", "NEXT_RUN_ALLOWED",
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def load_execution_policy(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    config = _load_yaml(root / "configs/master_run.yaml")
    policy_path = root / "configs/execution_policy.yaml"
    policy_config = _load_yaml(policy_path) if policy_path.exists() else config
    policy = {
        "execution_policy": policy_config.get("execution_policy"),
        "global_full_dag_enabled": policy_config.get("global_full_dag_enabled"),
        "maximum_concurrent_gpu_jobs": policy_config.get("maximum_concurrent_gpu_jobs"),
        "automatic_next_run": policy_config.get("automatic_next_run"),
        "require_user_approval_after_each_run": policy_config.get("require_user_approval_after_each_run"),
    }
    expected = {
        "execution_policy": "sequential_review_gated",
        "global_full_dag_enabled": False,
        "maximum_concurrent_gpu_jobs": 1,
        "automatic_next_run": False,
        "require_user_approval_after_each_run": True,
    }
    errors = [f"{key}={policy.get(key)!r}, expected {value!r}" for key, value in expected.items() if policy.get(key) != value]
    for key, value in expected.items():
        if config.get(key) != value:
            errors.append(f"master_run.yaml {key}={config.get(key)!r}, expected {value!r}")
    if errors:
        raise ValueError("Sequential execution policy is invalid: " + "; ".join(errors))
    return policy


def load_inventory(root: str | Path = ".") -> list[dict[str, Any]]:
    path = Path(root) / "reports/expected_experiment_runs.json"
    if not path.exists():
        from .inventory import build_expected_runs

        return list(build_expected_runs(root)["rows"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Expected experiment inventory rows are not a list")
    return [dict(row) for row in rows]


def find_experiment(root: str | Path, experiment_id: str) -> dict[str, Any]:
    matches = [row for row in load_inventory(root) if str(row.get("experiment_id", row.get("run_id"))) == experiment_id or str(row.get("run_id")) == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one inventory entry for {experiment_id!r}; found {len(matches)}")
    row = matches[0]
    row.setdefault("experiment_id", row.get("run_id"))
    row.setdefault("system_id", row.get("system"))
    return row


def build_azure_job_inventory() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = [
        {"job_id": "azure_rationale_generation", "job_type": "rationale_generation", "research_question": "setup", "task": "rationale", "budget": ""},
        {"job_id": "azure_pragmatic_zero_shot", "job_type": "pragmatic_zero_shot", "research_question": "Q1a", "task": "pragmatic", "budget": ""},
        {"job_id": "azure_pragmatic_8_shot", "job_type": "pragmatic_8_shot", "research_question": "Q1a", "task": "pragmatic", "budget": ""},
        {"job_id": "azure_polarity_dedicated", "job_type": "polarity_dedicated_prompt", "research_question": "Q1b", "task": "polarity", "budget": ""},
        {"job_id": "azure_emotion_dedicated", "job_type": "emotion_dedicated_prompt", "research_question": "Q1b", "task": "emotion", "budget": ""},
    ]
    for budget in ("32", "64", "128", "256", "512", "full"):
        jobs.append({"job_id": f"azure_q3_pragmatic_8_shot_{budget}", "job_type": "q3_budget_specific_pragmatic_8_shot", "research_question": "Q3", "task": "sarcasm", "budget": budget})
    for job in jobs:
        job.update({
            "display_name": job["job_id"].replace("_", " "),
            "execution_policy": "sequential_review_gated",
            "required_phase15_assets": "azure_deployment;prompt_manifest",
            "dependencies": "preflight_validation",
            "split": "vipragsent_train" if job["job_type"] == "rationale_generation" else "vipragsent_test",
            "model": "gpt-4.1-mini" if job["job_type"] != "rationale_generation" else "gpt-4.1-mini",
        })
    return jobs


def find_azure_job(job_id: str) -> dict[str, Any]:
    matches = [job for job in build_azure_job_inventory() if job["job_id"] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown Azure job ID: {job_id}")
    return matches[0]


def _git_commit(root: Path) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def preflight_entry(root: str | Path, entry: Mapping[str, Any], *, kind: str, dry_run: bool = False) -> dict[str, Any]:
    root = Path(root)
    load_execution_policy(root)
    protocol = validate_protocol_resolution(root)
    blockers = list(protocol["scientific_protocol_conflicts"])
    warnings: list[str] = []
    required = ("experiment_id", "system_id", "research_question", "variant", "backbone", "task", "split", "required_phase15_assets") if kind == "experiment" else ("job_id", "job_type", "task", "required_phase15_assets")
    missing = [key for key in required if entry.get(key) in (None, "")]
    if missing:
        blockers.append("single-run inventory entry is missing: " + ", ".join(missing))
    if kind == "azure" and entry.get("job_type") not in AZURE_JOB_TYPES:
        blockers.append(f"unsupported Azure job type: {entry.get('job_type')}")
    if not dry_run:
        if kind == "experiment" and entry.get("backbone") != "azure" and not (root / "data/model_cache_manifest.json").exists():
            blockers.append("Phase 15 verified model cache manifest is unavailable")
        if kind == "azure" and not (root / "data/manifests/azure_deployment.json").exists():
            blockers.append("Azure deployment manifest is unavailable")
    else:
        warnings.append("dry-run: no model, GPU, Azure, or external runtime was accessed")
    return {
        "kind": kind,
        "entry_id": str(entry.get("experiment_id", entry.get("job_id"))),
        "policy": load_execution_policy(root),
        "protocol_resolution": protocol["resolution_status"],
        "blockers": blockers,
        "warnings": warnings,
        "passed": not blockers,
        "code_commit": _git_commit(root),
    }


def _review_payload(entry: Mapping[str, Any], *, run_id: str, run_status: str, validation_status: str, blockers: Sequence[str], warnings: Sequence[str], root: Path) -> dict[str, Any]:
    manifest = root / "data/manifests/dataset_manifest.json"
    data_fingerprint = sha256_file(manifest) if manifest.exists() else None
    payload: dict[str, Any] = {field: None for field in REVIEW_FIELDS}
    payload.update({
        "experiment_id": entry.get("experiment_id", entry.get("job_id")),
        "research_question": entry.get("research_question"),
        "system_id": entry.get("system_id", entry.get("job_id")),
        "display_name": entry.get("display_name"),
        "variant": entry.get("variant", entry.get("job_type")),
        "backbone": entry.get("backbone", "azure" if "job_id" in entry else None),
        "seed": entry.get("seed"),
        "budget": entry.get("budget", ""),
        "execution_mode": "single_run_sequential_review_gated",
        "run_status": run_status,
        "user_review_status": "PENDING",
        "dataset_fingerprint": data_fingerprint,
        "split_hashes": {},
        "q3_mask_hash": None,
        "model_repository": entry.get("model_repository"),
        "model_revision": entry.get("model_revision"),
        "tokenizer_revision": entry.get("tokenizer_revision"),
        "preprocessing_name": entry.get("preprocessing_name"),
        "preprocessing_version": entry.get("preprocessing_version"),
        "configuration_hash": sha256_json(dict(entry)),
        "code_commit": _git_commit(root),
        "frozen_thresholds": {},
        "per_label_dev_metrics": {},
        "per_label_test_metrics": {},
        "artifact_paths": [],
        "artifact_sha256": {},
        "warnings": list(warnings),
        "blockers": list(blockers),
        "validation_status": validation_status,
        "RUN_STATUS": run_status,
        "USER_REVIEW_STATUS": "PENDING",
        "NEXT_RUN_ALLOWED": "NO",
        "run_id": run_id,
    })
    return payload


def render_review_summary(payload: Mapping[str, Any]) -> str:
    lines = ["# Sequential Run Review Summary", ""]
    for field in REVIEW_FIELDS:
        title = field.replace("_", " ")
        value = payload.get(field)
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = "null" if value is None else str(value)
        lines.append(f"## {title}")
        lines.append(rendered)
        lines.append("")
    return "\n".join(lines)


def write_review_artifacts(root: str | Path, entry: Mapping[str, Any], *, run_id: str, run_status: str, validation_status: str, blockers: Sequence[str] = (), warnings: Sequence[str] = ()) -> dict[str, str]:
    root = Path(root)
    output = root / "results/runs" / run_id
    payload = _review_payload(entry, run_id=run_id, run_status=run_status, validation_status=validation_status, blockers=blockers, warnings=warnings, root=root)
    atomic_write_json(output / "review_summary.json", payload)
    atomic_write_text(output / "review_summary.md", render_review_summary(payload))
    atomic_write_json(output / "approval_status.json", {"run_id": run_id, "status": "PENDING_USER_APPROVAL", "approved_by": None, "approved_at": None})
    return {
        "review_summary_json": (output / "review_summary.json").relative_to(root).as_posix(),
        "review_summary_md": (output / "review_summary.md").relative_to(root).as_posix(),
        "approval_status": (output / "approval_status.json").relative_to(root).as_posix(),
    }


StageHandler = Callable[[Mapping[str, Any], Path], Mapping[str, Any] | None]


def execute_sequential_run(
    root: str | Path,
    entry: Mapping[str, Any],
    *,
    kind: str,
    stage: str,
    run_id: str,
    resume: bool = False,
    dry_run: bool = False,
    fixture: bool = False,
    stage_handlers: Mapping[str, StageHandler] | None = None,
) -> tuple[dict[str, Any], int]:
    root = Path(root)
    if stage not in {"preflight", "all", *SEQUENTIAL_STAGES}:
        raise ValueError(f"Unknown sequential stage: {stage}")
    state_path = root / "runs/sequential" / f"{run_id}.json"
    state: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8")) if resume and state_path.exists() else {"run_id": run_id, "kind": kind, "entry_id": entry.get("experiment_id", entry.get("job_id")), "stages": {}, "status": "PENDING"}
    target_stages = SEQUENTIAL_STAGES if stage == "all" else (stage,)
    if dry_run:
        report = {"run_id": run_id, "kind": kind, "entry": dict(entry), "execution_policy": "sequential_review_gated", "stages": list(target_stages), "dry_run": True, "passed": True, "message": "No execution was performed; stop and await explicit user approval before a real run."}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return report, 0
    preflight = preflight_entry(root, entry, kind=kind, dry_run=fixture)
    state["preflight"] = preflight
    if not preflight["passed"]:
        state["status"] = "BLOCKED"
        atomic_write_json(state_path, state)
        artifacts = write_review_artifacts(root, entry, run_id=run_id, run_status="BLOCKED", validation_status="BLOCKED", blockers=preflight["blockers"], warnings=preflight["warnings"])
        state["review_artifacts"] = artifacts
        atomic_write_json(state_path, state)
        return state, 2
    handlers = dict(stage_handlers or {})
    for current in target_stages:
        if resume and state["stages"].get(current, {}).get("status") == "PASS":
            continue
        if current == "preflight":
            result: Mapping[str, Any] = {"status": "PASS", "summary": preflight}
        elif fixture:
            marker = root / "runs/sequential" / run_id / f"{current}.json"
            atomic_write_json(marker, {"run_id": run_id, "stage": current, "synthetic_results": True})
            result = {"status": "PASS", "summary": {"synthetic_results": True, "path": marker.relative_to(root).as_posix()}}
        elif current in handlers:
            result = handlers[current](entry, root) or {"status": "PASS"}
        else:
            result = {"status": "BLOCKED", "error": f"Live backend for stage {current!r} is not enabled in this setup-only environment"}
        state["stages"][current] = dict(result)
        if result.get("status") != "PASS":
            state["status"] = result.get("status", "FAIL")
            atomic_write_json(state_path, state)
            artifacts = write_review_artifacts(root, entry, run_id=run_id, run_status=state["status"], validation_status=state["status"], blockers=[str(result.get("error", "stage did not pass"))], warnings=[])
            state["review_artifacts"] = artifacts
            atomic_write_json(state_path, state)
            return state, 2 if state["status"] == "BLOCKED" else 4
    state["status"] = "PASS"
    atomic_write_json(state_path, state)
    artifacts = write_review_artifacts(root, entry, run_id=run_id, run_status="PASS", validation_status="PASS", blockers=[], warnings=[])
    state["review_artifacts"] = artifacts
    atomic_write_json(state_path, state)
    return state, 0
