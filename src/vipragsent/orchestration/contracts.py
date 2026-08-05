from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class StageName(StrEnum):
    PREFLIGHT = "preflight"
    TRAIN = "train"
    TRAIN_OR_REUSE = "train_or_reuse"
    EXECUTE_COMPONENTS = "execute_components"
    COMBINE_COMPONENT_PREDICTIONS = "combine_component_predictions"
    EVALUATE_DEV = "evaluate_dev"
    FREEZE_SELECTION = "freeze_selection"
    FREEZE_COMPONENT_SELECTION = "freeze_component_selection"
    EVALUATE_TEST = "evaluate_test"
    TRAIN_GENERATION = "train_generation"
    GENERATE_DEV = "generate_dev"
    PARSE_DEV = "parse_dev"
    GENERATE_TEST = "generate_test"
    PARSE_TEST = "parse_test"
    RESOLVE_APPROVED_SOURCE = "resolve_approved_source"
    EVALUATE_EXTERNAL_TESTS = "evaluate_external_tests"
    VALIDATE_SOURCE_PREDICTIONS = "validate_source_predictions"
    EXTRACT_PRAGMATIC_CALIBRATION = "extract_pragmatic_calibration"
    EXTRACT_LEARNING_HISTORY = "extract_learning_history"
    EXECUTE_API_JOB = "execute_api_job"
    VALIDATE_RESPONSES = "validate_responses"
    EXPORT_ARTIFACTS = "export_artifacts"
    VALIDATE_ARTIFACTS = "validate_artifacts"
    GENERATE_REVIEW_SUMMARY = "generate_review_summary"


class StageStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    PASS = "PASS"
    BLOCKED = "BLOCKED"
    FAIL = "FAIL"
    SKIPPED_BY_DESIGN = "SKIPPED_BY_DESIGN"
    INTERRUPTED = "INTERRUPTED"
    RUNNING_STALE = "RUNNING_STALE"


class RunStatus(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    PREFLIGHT_READY = "PREFLIGHT_READY"
    RUNNING = "RUNNING"
    RUNNING_STALE = "RUNNING_STALE"
    BLOCKED = "BLOCKED"
    FAIL = "FAIL"
    COMPLETED_PENDING_APPROVAL = "COMPLETED_PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ExecutionKind(StrEnum):
    TRAINABLE = "trainable"
    COMPONENT_BUNDLE = "component_bundle"
    GENERATION = "generation"
    CHECKPOINT_REUSE = "checkpoint_reuse"
    EVALUATION_ONLY = "evaluation_only"
    AZURE = "azure"
    ARTIFACT_EXTRACTION = "artifact_extraction"


EXPERIMENT_STAGES: tuple[str, ...] = (
    StageName.PREFLIGHT.value,
    StageName.TRAIN_OR_REUSE.value,
    StageName.EVALUATE_DEV.value,
    StageName.FREEZE_SELECTION.value,
    StageName.EVALUATE_TEST.value,
    StageName.EXPORT_ARTIFACTS.value,
    StageName.VALIDATE_ARTIFACTS.value,
    StageName.GENERATE_REVIEW_SUMMARY.value,
)

AZURE_STAGES: tuple[str, ...] = (
    StageName.PREFLIGHT.value,
    StageName.EXECUTE_API_JOB.value,
    StageName.VALIDATE_RESPONSES.value,
    StageName.EXPORT_ARTIFACTS.value,
    StageName.VALIDATE_ARTIFACTS.value,
    StageName.GENERATE_REVIEW_SUMMARY.value,
)

VALID_EXECUTION_KINDS = {item.value for item in ExecutionKind}
TERMINAL_STAGE_STATUSES = {StageStatus.PASS.value, StageStatus.SKIPPED_BY_DESIGN.value}


@dataclass(frozen=True)
class RunEntry:
    """Typed view of one immutable inventory row."""

    run_id: str
    research_question: str
    system_id: str
    display_name: str
    variant: str
    backbone: str
    seed: int | str | None
    budget: str | int | None
    execution_kind: str
    task: str = ""
    split: str = ""
    dependencies: str = ""
    required_phase15_assets: str = ""
    model_repository: str | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    preprocessing_name: str | None = None
    preprocessing_version: str | None = None
    source_checkpoint_id: str | None = None
    q3_mask_path: str | None = None
    q3_mask_hash: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any], *, run_id: str | None = None) -> RunEntry:
        identifier = str(run_id or row.get("experiment_id") or row.get("run_id") or row.get("job_id") or "")
        if not identifier:
            raise ValueError("Inventory entry has no run identifier")
        return cls(
            run_id=identifier,
            research_question=str(row.get("research_question", "setup")),
            system_id=str(row.get("system_id") or row.get("system") or row.get("job_id") or identifier),
            display_name=str(row.get("display_name") or identifier),
            variant=str(row.get("variant") or row.get("job_type") or "unknown"),
            backbone=str(row.get("backbone") or ("azure" if row.get("job_id") else "")),
            seed=row.get("seed"),
            budget=row.get("budget"),
            execution_kind=str(row.get("execution_kind") or ("azure" if row.get("job_id") else "trainable")),
            task=str(row.get("task", "")),
            split=str(row.get("split", "")),
            dependencies=str(row.get("dependencies", "")),
            required_phase15_assets=str(row.get("required_phase15_assets", "")),
            model_repository=row.get("model_repository"),
            model_revision=row.get("model_revision"),
            tokenizer_revision=row.get("tokenizer_revision"),
            preprocessing_name=row.get("preprocessing_name"),
            preprocessing_version=row.get("preprocessing_version"),
            source_checkpoint_id=row.get("source_checkpoint_id") or row.get("reusable_checkpoint_key"),
            q3_mask_path=row.get("q3_mask_path"),
            q3_mask_hash=row.get("q3_mask_hash"),
            raw=dict(row),
        )

    @property
    def is_azure(self) -> bool:
        return self.execution_kind == ExecutionKind.AZURE.value or self.backbone == "azure"

    @property
    def stages(self) -> tuple[str, ...]:
        from .stage_plans import resolve_stage_plan

        return resolve_stage_plan(self.raw.get("_repository_root", "."), self.raw | {"execution_kind": self.execution_kind, "research_question": self.research_question, "system_id": self.system_id, "backbone": self.backbone}).stages


@dataclass
class RunContext:
    root: Path
    entry: RunEntry
    fixture: bool = False
    dry_run: bool = False
    run_root: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.run_root is None:
            base = self.root / "runs" / "fixture" / "results" / "runs" if self.fixture else self.root / "results" / "runs"
            self.run_root = base / self.entry.run_id
        self.run_root = Path(self.run_root)

    @property
    def run_id(self) -> str:
        return self.entry.run_id

    @property
    def execution_kind(self) -> str:
        return self.entry.execution_kind


@dataclass(frozen=True)
class StageOutcome:
    status: str
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    expected_files: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        StageStatus(self.status)
        if self.status == StageStatus.PASS.value and self.error:
            raise ValueError("PASS stage outcomes cannot carry an error")
        if self.status in {StageStatus.BLOCKED.value, StageStatus.FAIL.value} and not (self.error or self.blockers):
            raise ValueError("blocked/failed stage outcomes require an explanation")

    @classmethod
    def passed(cls, *, summary: Mapping[str, Any] | None = None, artifacts: list[str] | tuple[str, ...] = (), expected_files: list[str] | tuple[str, ...] = (), warnings: list[str] | tuple[str, ...] = ()) -> StageOutcome:
        return cls(StageStatus.PASS.value, dict(summary or {}), tuple(artifacts), tuple(expected_files), (), tuple(warnings), None)

    @classmethod
    def skipped(cls, reason: str) -> StageOutcome:
        return cls(StageStatus.SKIPPED_BY_DESIGN.value, {"reason": reason}, (), (), (), (), None)

    @classmethod
    def blocked(cls, *blockers: str) -> StageOutcome:
        return cls(StageStatus.BLOCKED.value, {}, (), (), tuple(blockers), (), blockers[0] if blockers else "blocked")

    @classmethod
    def failed(cls, *errors: str) -> StageOutcome:
        return cls(StageStatus.FAIL.value, {}, (), (), (), (), errors[0] if errors else "failed")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "artifacts": list(self.artifacts),
            "expected_files": list(self.expected_files),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "error": self.error,
        }


@dataclass(frozen=True)
class RunArtifactIndex:
    run_id: str
    paths: tuple[str, ...]
    sha256: Mapping[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "artifact_paths": list(self.paths), "artifact_sha256": dict(self.sha256)}


@dataclass(frozen=True)
class ApprovalRecord:
    run_id: str
    decision: str
    review_note: str
    approved_or_rejected_by: str
    timestamp: str
    review_summary_sha256: str
    artifact_checksum_file_sha256: str
    code_commit: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "decision": self.decision,
            "review_note": self.review_note,
            "approved_or_rejected_by": self.approved_or_rejected_by,
            "timestamp": self.timestamp,
            "review_summary_sha256": self.review_summary_sha256,
            "artifact_checksum_file_sha256": self.artifact_checksum_file_sha256,
            "code_commit": self.code_commit,
        }
