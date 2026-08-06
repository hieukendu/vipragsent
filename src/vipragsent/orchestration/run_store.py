from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, atomic_write_text, exclusive_lock
from ..hashing import sha256_file
from .contracts import (
    TERMINAL_STAGE_STATUSES,
    RunContext,
    RunStatus,
    StageStatus,
)
from .provenance import expected_inference_provenance


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit(root: Path) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def git_tree(root: Path) -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def git_worktree_clean(root: Path) -> bool:
    try:
        result = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return False
    return not bool(result.stdout.strip())


class RunStore:
    """Atomic state and artifact bookkeeping for exactly one sequential run."""

    def __init__(self, context: RunContext) -> None:
        self.context = context
        self.root = Path(context.run_root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.events_path = self.root / "stage_events.jsonl"
        self.checksums_path = self.root / "checksums.sha256"

    @property
    def required_stages(self) -> tuple[str, ...]:
        return self.context.entry.stages

    def initialize(self, *, resume: bool = False) -> dict[str, Any]:
        if self.state_path.exists() and resume:
            state = self.load()
            self.reconcile_resume_identity(state)
            self.recover_stale(state)
            self.invalidate_stale_preflight(state)
            state["code_commit"] = git_commit(self.context.root)
            state["code_tree"] = git_tree(self.context.root)
            self.save(state)
            return state
        if self.state_path.exists() and not resume:
            state = self.load()
            if state.get("run_id") != self.context.run_id:
                raise ValueError("Existing run directory belongs to a different run ID")
            if state.get("run_status") in {RunStatus.COMPLETED_PENDING_APPROVAL.value, RunStatus.APPROVED.value, RunStatus.REJECTED.value}:
                return state
            # A fresh command may continue an unfinished run only with --resume.
            raise RuntimeError("An unfinished run exists; resume only the same run")
        state = {
            "schema_version": 2,
            "run_id": self.context.run_id,
            "experiment_id": self.context.entry.run_id if not self.context.entry.is_azure else None,
            "azure_job_id": self.context.entry.run_id if self.context.entry.is_azure else None,
            "execution_kind": self.context.entry.execution_kind,
            "run_status": RunStatus.NOT_STARTED.value,
            "stages": {stage: {"status": StageStatus.NOT_STARTED.value} for stage in self.required_stages},
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "code_commit": git_commit(self.context.root),
            "code_tree": git_tree(self.context.root),
            "git_worktree_clean": git_worktree_clean(self.context.root),
            "fixture": self.context.fixture,
            "approval_status": "PENDING_USER_APPROVAL",
            "next_run_allowed": "NO",
        }
        self.save(state)
        self._write_baseline_files()
        self.append_event("run_initialized", {"run_status": state["run_status"], "fixture": self.context.fixture})
        return state

    def reconcile_resume_identity(self, state: dict[str, Any]) -> None:
        """Repair legacy pre-execution metadata without rewriting real run evidence."""
        expected = {
            "run_id": self.context.run_id,
            "experiment_id": self.context.entry.run_id if not self.context.entry.is_azure else None,
            "azure_job_id": self.context.entry.run_id if self.context.entry.is_azure else None,
            "execution_kind": self.context.entry.execution_kind,
        }
        identity_mismatches = {
            key: {"recorded": state.get(key), "expected": value}
            for key, value in expected.items()
            if state.get(key) != value
        }
        recorded_stages = state.get("stages", {})
        expected_stages = set(self.required_stages)
        stage_plan_mismatch = set(recorded_stages) != expected_stages
        if not identity_mismatches and not stage_plan_mismatch:
            return

        non_preflight_started = {
            name: details.get("status")
            for name, details in recorded_stages.items()
            if name != "preflight" and details.get("status") != StageStatus.NOT_STARTED.value
        }
        if non_preflight_started:
            raise RuntimeError(
                "resume identity or stage-plan conflict after execution began; "
                "preserving evidence and refusing silent metadata rewrite: "
                + json.dumps({"identity": identity_mismatches, "stages": non_preflight_started}, sort_keys=True)
            )

        previous_stages = dict(recorded_stages)
        state.update(expected)
        state["stages"] = {
            stage: dict(previous_stages.get(stage, {"status": StageStatus.NOT_STARTED.value}))
            for stage in self.required_stages
        }
        self._write_baseline_files()
        self.append_event(
            "run_identity_reconciled",
            {
                "identity_mismatches": identity_mismatches,
                "recorded_stage_names": sorted(previous_stages),
                "current_stage_names": sorted(expected_stages),
                "reason": "legacy pre-execution metadata reconciled from the current immutable inventory entry",
            },
        )

    def invalidate_stale_preflight(self, state: dict[str, Any]) -> None:
        """Force a fresh preflight when a resumable run crossed a code revision."""
        if state.get("run_status") in {
            RunStatus.COMPLETED_PENDING_APPROVAL.value,
            RunStatus.APPROVED.value,
            RunStatus.REJECTED.value,
        }:
            return
        preflight = state.get("stages", {}).get("preflight", {})
        if preflight.get("status") != StageStatus.PASS.value:
            return
        current_commit = git_commit(self.context.root)
        current_tree = git_tree(self.context.root)
        if state.get("code_commit") == current_commit and state.get("code_tree") == current_tree:
            return
        state["stages"]["preflight"] = {
            "status": StageStatus.NOT_STARTED.value,
            "invalidation_reason": "code commit or tree changed since the recorded preflight",
            "recorded_code_commit": state.get("code_commit"),
            "current_code_commit": current_commit,
            "recorded_code_tree": state.get("code_tree"),
            "current_code_tree": current_tree,
        }
        self.append_event(
            "preflight_invalidated",
            {
                "reason": "code commit or tree changed",
                "recorded_code_commit": state.get("code_commit"),
                "current_code_commit": current_commit,
                "recorded_code_tree": state.get("code_tree"),
                "current_code_tree": current_tree,
            },
        )

    def _write_baseline_files(self) -> None:
        entry = self.context.entry
        provenance = expected_inference_provenance(entry.system_id, execution_kind=entry.execution_kind)
        snapshot = {
            "run_id": entry.run_id,
            "execution_kind": entry.execution_kind,
            "research_question": entry.research_question,
            "system_id": entry.system_id,
            "configuration_source": "configs/master_run.yaml and inventory row",
            "entry": dict(entry.raw),
        }
        atomic_write_text(self.root / "config_snapshot.yaml", json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
        atomic_write_json(self.root / "environment.json", {
            "mode": "fixture" if self.context.fixture else "full",
            "python": os.sys.version,
            "code_commit": git_commit(self.context.root),
            "fixture": self.context.fixture,
        })
        atomic_write_json(self.root / "run_manifest.json", {
            "run_id": entry.run_id,
            "mode": "fixture" if self.context.fixture else "full",
            "synthetic_results": bool(self.context.fixture),
            "research_question": entry.research_question,
            "system": entry.system_id,
            "system_id": entry.system_id,
            "backbone": entry.backbone,
            "seed": entry.seed,
            "budget": entry.budget,
            "execution_kind": entry.execution_kind,
            "model_revision": entry.model_revision or ("fixture" if self.context.fixture else ""),
            "tokenizer_revision": entry.tokenizer_revision or ("fixture" if self.context.fixture else ""),
            "external_finetuning": False,
            **provenance,
            "code_commit": git_commit(self.context.root),
            "status": "NOT_STARTED",
        })
        atomic_write_json(self.root / "metrics.json", {"run_id": entry.run_id, "status": "NOT_STARTED", "mode": "fixture" if self.context.fixture else "full", "synthetic_results": bool(self.context.fixture)})
        atomic_write_json(self.root / "approval_status.json", {
            "run_id": entry.run_id,
            "status": "PENDING_USER_APPROVAL",
            "approved_by": None,
            "approved_at": None,
        })

    def load(self) -> dict[str, Any]:
        if not self.state_path.exists():
            raise FileNotFoundError(self.state_path)
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save(self, state: Mapping[str, Any]) -> None:
        payload = dict(state)
        payload["updated_at"] = utc_now()
        atomic_write_json(self.state_path, payload)

    def append_event(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        record = {"timestamp": utc_now(), "event": event, **dict(payload or {})}
        with exclusive_lock(self.root / ".stage_events.lock"):
            existing = self.events_path.read_text(encoding="utf-8") if self.events_path.exists() else ""
            atomic_write_text(self.events_path, existing + json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")

    def recover_stale(self, state: dict[str, Any]) -> None:
        changed = False
        for name, stage in state.get("stages", {}).items():
            if stage.get("status") == StageStatus.RUNNING.value:
                stage["status"] = StageStatus.RUNNING_STALE.value
                stage["recovered_at"] = utc_now()
                changed = True
        if changed and state.get("run_status") == RunStatus.RUNNING.value:
            state["run_status"] = RunStatus.RUNNING_STALE.value if hasattr(RunStatus, "RUNNING_STALE") else RunStatus.RUNNING.value
            self.append_event("stale_run_recovered", {"run_id": self.context.run_id})

    def start_stage(self, state: dict[str, Any], stage: str) -> None:
        if stage not in self.required_stages:
            raise ValueError(f"Stage {stage!r} is not applicable to {self.context.entry.run_id}")
        current = state["stages"].get(stage, {}).get("status", StageStatus.NOT_STARTED.value)
        if current in TERMINAL_STAGE_STATUSES:
            return
        state["run_status"] = RunStatus.RUNNING.value
        state["stages"][stage] = {"status": StageStatus.RUNNING.value, "started_at": utc_now()}
        self.save(state)
        self.append_event("stage_started", {"stage": stage})

    def complete_stage(self, state: dict[str, Any], stage: str, outcome: Mapping[str, Any]) -> None:
        status = str(outcome.get("status", StageStatus.FAIL.value))
        StageStatus(status)
        state["stages"][stage] = {**dict(outcome), "status": status, "ended_at": utc_now()}
        if status == StageStatus.BLOCKED.value:
            state["run_status"] = RunStatus.BLOCKED.value
        elif status == StageStatus.FAIL.value:
            state["run_status"] = RunStatus.FAIL.value
        self.save(state)
        self.append_event("stage_completed", {"stage": stage, "status": status})

    def mark_preflight_ready(self, state: dict[str, Any]) -> None:
        state["run_status"] = RunStatus.PREFLIGHT_READY.value
        self.save(state)
        self.append_event("preflight_ready", {})

    def mark_completed_pending_approval(self, state: dict[str, Any]) -> None:
        required = [state["stages"].get(stage, {}).get("status") for stage in self.required_stages]
        if any(status not in TERMINAL_STAGE_STATUSES for status in required):
            raise RuntimeError("Cannot complete a run before every required stage passes or is skipped by design")
        state["run_status"] = RunStatus.COMPLETED_PENDING_APPROVAL.value
        state["approval_status"] = "PENDING_USER_APPROVAL"
        state["next_run_allowed"] = "NO"
        self.save(state)
        self.append_event("run_completed_pending_approval", {})

    def artifact_paths(self) -> list[Path]:
        excluded = {self.state_path, self.events_path, self.checksums_path, self.root / "approval_status.json", self.root / "review_summary.json", self.root / "review_summary.md"}
        return sorted(path for path in self.root.rglob("*") if path.is_file() and path not in excluded and not path.name.endswith(".lock"))

    def write_checksums(self) -> dict[str, str]:
        records = {path.relative_to(self.root).as_posix(): sha256_file(path) for path in self.artifact_paths()}
        atomic_write_text(self.checksums_path, "".join(f"{digest}  {name}\n" for name, digest in sorted(records.items())))
        return records

    def validate_checksums(self) -> list[str]:
        if not self.checksums_path.exists():
            return ["checksums.sha256 is missing"]
        errors: list[str] = []
        expected: dict[str, str] = {}
        for line in self.checksums_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            expected[name] = digest
        actual = {path.relative_to(self.root).as_posix(): sha256_file(path) for path in self.artifact_paths()}
        for name, digest in expected.items():
            if actual.get(name) != digest:
                errors.append(f"checksum mismatch: {name}")
        errors.extend(f"checksum missing for artifact: {name}" for name in sorted(set(actual) - set(expected)))
        return errors

    def summary_sha256(self) -> str:
        return sha256_file(self.root / "review_summary.json")

    def checksum_file_sha256(self) -> str:
        return sha256_file(self.checksums_path)


def artifact_hashes(run_root: str | Path) -> dict[str, str]:
    root = Path(run_root)
    excluded = {root / "checksums.sha256", root / "state.json", root / "stage_events.jsonl", root / "approval_status.json", root / "review_summary.json", root / "review_summary.md"}
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(root.rglob("*")) if path.is_file() and path not in excluded and not path.name.endswith(".lock")}


def hash_file_list(paths: list[str | Path]) -> str:
    return hashlib.sha256("\n".join(sorted(str(Path(path)) for path in paths)).encode("utf-8")).hexdigest().upper()
