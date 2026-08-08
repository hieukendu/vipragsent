from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from ..atomic import atomic_write_json, atomic_write_text
from ..protocol import validate_protocol_resolution
from .contracts import AZURE_STAGES, EXPERIMENT_STAGES, StageName
from .inventory import build_expected_runs

SEQUENTIAL_STAGES = EXPERIMENT_STAGES
AZURE_SEQUENTIAL_STAGES = AZURE_STAGES
AZURE_JOB_TYPES = (
    "rationale_generation",
    "pragmatic_zero_shot",
    "pragmatic_8_shot",
    "polarity_dedicated_prompt",
    "emotion_dedicated_prompt",
    "q3_budget_specific_pragmatic_8_shot",
)


REVIEW_FIELDS = (
    "run_id", "experiment_id", "azure_job_id", "research_question", "system_id", "display_name", "variant", "backbone",
    "seed", "budget", "execution_kind", "execution_mode", "run_status", "user_review_status", "next_run_allowed",
    "dataset_fingerprint", "split_hashes", "model_repository", "model_revision", "tokenizer_revision", "preprocessing_name",
    "preprocessing_version", "configuration_hash", "code_commit", "start_time", "end_time", "wall_clock_seconds", "warnings",
    "blockers", "validation_status", "artifact_paths", "artifact_sha256", "optimizer", "learning_rate", "weight_decay",
    "scheduler", "warmup_ratio", "precision", "physical_batch_size", "gradient_accumulation_steps", "effective_batch_size",
    "maximum_epochs", "actual_epochs", "best_epoch", "early_stopping_reason", "primary_dev_selection_metric", "best_dev_metric",
    "best_dev_loss", "checkpoint_path", "checkpoint_sha256", "frozen_thresholds", "per_label_dev_metrics", "per_label_test_metrics",
    "macro_pragmatic_f1", "per_label_pragmatic_ece", "macro_pragmatic_ece", "temperature_scaling", "bin_count",
    "probability_aggregation", "source_checkpoint_id", "source_prediction_hash", "azure_request_count", "azure_input_tokens",
    "azure_cached_input_tokens", "azure_non_cached_input_tokens", "azure_output_tokens", "azure_cost_usd",
    "azure_non_cached_input_cost_usd", "azure_cached_input_cost_usd", "azure_output_cost_usd", "azure_cost_accounting_method",
    "azure_cost_verification_status", "azure_usage_records_path", "azure_cost_ledger_path", "azure_invalid_output_rate",
    "azure_cache_hits", "azure_cache_misses", "azure_failed_requests", "azure_retried_requests", "peak_vram_gb",
    "successful_gpu_hours", "failed_or_retried_gpu_hours", "RUN_STATUS",
    "USER_REVIEW_STATUS", "NEXT_RUN_ALLOWED",
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
    errors.extend(f"master_run.yaml {key}={config.get(key)!r}, expected {value!r}" for key, value in expected.items() if config.get(key) != value)
    if errors:
        raise ValueError("Sequential execution policy is invalid: " + "; ".join(errors))
    return policy


def load_inventory(root: str | Path = ".") -> list[dict[str, Any]]:
    root = Path(root)
    path = root / "reports/expected_experiment_runs.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("rows", [])
            if isinstance(rows, list) and rows and all("execution_kind" in row for row in rows):
                return [dict(row) for row in rows]
        except (OSError, json.JSONDecodeError):
            pass
    return list(build_expected_runs(root)["rows"])


def find_experiment(root: str | Path, experiment_id: str) -> dict[str, Any]:
    matches = [row for row in load_inventory(root) if str(row.get("experiment_id") or row.get("run_id")) == experiment_id or str(row.get("run_id")) == experiment_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one inventory entry for {experiment_id!r}; found {len(matches)}")
    row = dict(matches[0])
    row.setdefault("experiment_id", row.get("run_id"))
    row.setdefault("system_id", row.get("system"))
    return row


def build_azure_job_inventory() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = [
        {"job_id": "azure_rationale_generation", "job_type": "rationale_generation", "research_question": "setup", "task": "rationale", "budget": ""},
        {"job_id": "azure_pragmatic_zero_shot", "job_type": "pragmatic_zero_shot", "research_question": "Q1a", "task": "pragmatic", "budget": ""},
        {"job_id": "azure_gpt41_mini_8shot", "job_type": "pragmatic_8_shot", "research_question": "Q1a", "task": "pragmatic", "budget": ""},
        {"job_id": "azure_polarity_dedicated", "job_type": "polarity_dedicated_prompt", "research_question": "Q1b", "task": "polarity", "budget": ""},
        {"job_id": "azure_emotion_dedicated", "job_type": "emotion_dedicated_prompt", "research_question": "Q1b", "task": "emotion", "budget": ""},
    ]
    for budget in ("32", "64", "128", "256", "512", "full"):
        jobs.append({"job_id": f"azure_q3_pragmatic_8_shot_{budget}", "job_type": "q3_budget_specific_pragmatic_8_shot", "research_question": "Q3", "task": "sarcasm", "budget": budget})
    for job in jobs:
        job.update({
            "display_name": job["job_id"].replace("_", " "),
            "execution_kind": "azure",
            "execution_policy": "sequential_review_gated",
            "required_phase15_assets": "azure_deployment;prompt_manifest",
            "dependencies": "preflight_validation",
            "split": "vipragsent_train" if job["job_type"] == "rationale_generation" else "vipragsent_test",
            "model": "gpt-4.1-mini",
            "backbone": "azure",
            "system_id": job["job_id"],
            "variant": job["job_type"],
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
    from .preflight_single import run_single_preflight

    report = run_single_preflight(root, entry, kind=kind, fixture=dry_run, dry_run=dry_run)
    return {
        "kind": kind,
        "entry_id": str(entry.get("experiment_id", entry.get("job_id"))),
        "policy": load_execution_policy(root),
        "protocol_resolution": validate_protocol_resolution(root)["resolution_status"],
        "blockers": report["blockers"],
        "warnings": report["warnings"],
        "checks": report["checks"],
        "passed": report["passed"],
        "code_commit": report["code_commit"],
        "preflight_hash": report["preflight_hash"],
    }


def render_review_summary(payload: Mapping[str, Any]) -> str:
    lines = ["# Sequential Run Review Summary", ""]
    for field in REVIEW_FIELDS:
        value = payload.get(field)
        if isinstance(value, (dict, list)):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = "NOT_APPLICABLE" if value is None else str(value)
        lines.extend([f"## {field.replace('_', ' ')}", rendered, ""])
    return "\n".join(lines)


def write_review_artifacts(root: str | Path, entry: Mapping[str, Any], *, run_id: str, run_status: str, validation_status: str, blockers: list[str] | tuple[str, ...] = (), warnings: list[str] | tuple[str, ...] = ()) -> dict[str, str]:
    """Compatibility writer for blocked/preflight reports; completed runs use the single-run summary builder."""
    root = Path(root)
    output = root / "results/runs" / run_id
    payload = {field: "NOT_APPLICABLE" for field in REVIEW_FIELDS}
    payload.update({
        "run_id": run_id,
        "experiment_id": entry.get("experiment_id"),
        "azure_job_id": entry.get("job_id"),
        "research_question": entry.get("research_question", "setup"),
        "system_id": entry.get("system_id", entry.get("system", run_id)),
        "display_name": entry.get("display_name", run_id),
        "variant": entry.get("variant", entry.get("job_type", "unknown")),
        "backbone": entry.get("backbone", "azure" if entry.get("job_id") else "unknown"),
        "seed": entry.get("seed"),
        "budget": entry.get("budget", ""),
        "execution_kind": entry.get("execution_kind", "azure" if entry.get("job_id") else "trainable"),
        "execution_mode": "single_run_sequential_review_gated",
        "run_status": run_status,
        "user_review_status": "PENDING",
        "next_run_allowed": "NO",
        "warnings": list(warnings),
        "blockers": list(blockers),
        "validation_status": validation_status,
        "RUN_STATUS": "BLOCKED" if run_status != "PASS" else "PASS",
        "USER_REVIEW_STATUS": "PENDING",
        "NEXT_RUN_ALLOWED": "NO",
        "artifact_paths": [],
        "artifact_sha256": {},
    })
    atomic_write_json(output / "review_summary.json", payload)
    atomic_write_text(output / "review_summary.md", render_review_summary(payload))
    atomic_write_json(output / "approval_status.json", {"run_id": run_id, "status": "PENDING_USER_APPROVAL", "approved_by": None, "approved_at": None})
    return {"review_summary_json": (output / "review_summary.json").relative_to(root).as_posix(), "review_summary_md": (output / "review_summary.md").relative_to(root).as_posix(), "approval_status": (output / "approval_status.json").relative_to(root).as_posix()}


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
    handlers: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Compatibility facade used by the two CLIs; the implementation lives in single_run."""
    if stage == "train_or_run":
        stage = StageName.TRAIN_OR_REUSE.value
    from .single_run import execute_single_run

    return execute_single_run(root, entry, kind=kind, stage=stage, run_id=run_id, resume=resume, dry_run=dry_run, fixture=fixture, injected_handlers=handlers)
