"""Pure runtime estimation that consumes the scheduler's typed workload state."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from .scheduler import (
    CampaignState,
    ResourcePolicy,
    StageRecord,
    StageSpec,
    StageStatus,
    build_dry_run_plan,
    validate_artifact,
)


GENERATION_FACTORS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


class EstimateStatus(str, Enum):
    REUSE = "REUSE"
    RESUME = "RESUME"
    TRAIN = "TRAIN"
    EVALUATE_ONLY = "EVALUATE_ONLY"
    ARTIFACT_ONLY = "ARTIFACT_ONLY"
    NOT_SCHEDULED_NAACL_BALANCED = "NOT_SCHEDULED_NAACL_BALANCED"
    BLOCKED = "BLOCKED"


ESTIMATE_STATUSES = tuple(item.value for item in EstimateStatus)


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    """An injected observation; no clock or artifact lookup occurs here."""

    stage_id: str
    wall_clock_minutes: float
    status: str = "SUCCEEDED"
    source_hash: str = ""

    def __post_init__(self) -> None:
        if self.wall_clock_minutes < 0:
            raise ValueError("observed wall-clock duration cannot be negative")


@dataclass(frozen=True, slots=True)
class EstimateRow:
    stage_id: str
    run_id: str
    status: EstimateStatus
    projected_minutes: float | None
    remaining_minutes: float | None
    reason: str
    gate_conditional: bool = False
    source_hashes: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        if self.source_hashes is None:
            object.__setattr__(self, "source_hashes", {})


@dataclass(frozen=True, slots=True)
class ReconciliationRow:
    stage_id: str
    before_status: str
    after_status: str
    projected_minutes: float | None
    observed_minutes: float | None
    delta_minutes: float | None


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    before_label: str
    after_label: str
    rows: tuple[ReconciliationRow, ...]
    changed_stage_count: int


@dataclass(frozen=True, slots=True)
class RuntimeEstimateReport:
    as_of: str
    source_hashes: Mapping[str, str]
    rows: tuple[EstimateRow, ...]
    lower_bound_minutes: float
    scheduler_makespan_minutes: float
    remaining_wall_clock_minutes: float
    generation_sensitivity: Mapping[float, float]
    phobert_concurrency_sensitivity: Mapping[int, float | None]
    reconciliation: ReconciliationReport
    assumptions: tuple[str, ...]
    projection_logic: str
    projection_status: str
    dry_run_plan: Any


def _observation_map(observations: Iterable[RuntimeObservation]) -> dict[str, RuntimeObservation]:
    result: dict[str, RuntimeObservation] = {}
    for observation in observations:
        if observation.stage_id in result:
            raise ValueError(f"duplicate runtime observation: {observation.stage_id}")
        result[observation.stage_id] = observation
    return result


def _completed(record_status: StageStatus) -> bool:
    return record_status in {StageStatus.SUCCEEDED, StageStatus.REUSED}


def _classify(spec: StageSpec, state: CampaignState, by_id: Mapping[str, StageSpec], authorization_ok: bool | None) -> tuple[EstimateStatus, str, bool]:
    if not spec.naacl_balanced:
        return EstimateStatus.NOT_SCHEDULED_NAACL_BALANCED, "excluded from balanced NAACL policy", False
    if authorization_ok is False:
        return EstimateStatus.BLOCKED, "campaign authorization is not bound", False
    record = state.record(spec.stage_id)
    if record.status == StageStatus.BLOCKED:
        return EstimateStatus.BLOCKED, "existing state is BLOCKED", False
    if record.status in {StageStatus.SUCCEEDED, StageStatus.REUSED} and spec.reusable_artifact:
        valid, _ = validate_artifact(record.artifact, expected_artifact_id=spec.artifact_id)
        if valid:
            return EstimateStatus.REUSE, "validated reusable artifact", False
        return EstimateStatus.BLOCKED, "completed artifact failed validation", False
    gate_ids = list(spec.dependencies)
    if spec.requires_dev_feedback and spec.dev_feedback_stage:
        gate_ids.append(spec.dev_feedback_stage)
    if spec.early_stopping_gate:
        gate_ids.append(spec.early_stopping_gate)
    unknown_gate = any(not _completed(state.record(gate_id).status) for gate_id in gate_ids)
    if any(gate_id not in by_id and not _completed(state.record(gate_id).status) for gate_id in gate_ids):
        return EstimateStatus.BLOCKED, "gate is not present in the supplied DAG/state", unknown_gate
    if spec.execution_mode == "reuse":
        valid, _ = validate_artifact(state.record(spec.stage_id).artifact, expected_artifact_id=spec.artifact_id)
        if not valid:
            return EstimateStatus.BLOCKED, "requested reuse artifact failed validation", unknown_gate
        return EstimateStatus.REUSE, "policy requests validated artifact reuse", unknown_gate
    if spec.execution_mode == "resume" or record.status in {StageStatus.RESUMABLE, StageStatus.STOPPED}:
        return EstimateStatus.RESUME, "checkpoint/state permits resume", unknown_gate
    if spec.execution_mode == "evaluate":
        return EstimateStatus.EVALUATE_ONLY, "evaluation-only stage", unknown_gate
    if spec.execution_mode == "artifact":
        return EstimateStatus.ARTIFACT_ONLY, "artifact extraction stage", unknown_gate
    return EstimateStatus.TRAIN, "trainable stage", unknown_gate


def _effective_duration(spec: StageSpec, observation: RuntimeObservation | None) -> float:
    return observation.wall_clock_minutes if observation is not None else spec.duration_minutes


def _lower_bound(specs: Sequence[StageSpec], durations: Mapping[str, float], policy: ResourcePolicy) -> float:
    if not specs:
        return 0.0
    capacities = policy.capacities()
    request_names = ("seven_b", "xlmr", "phobert", "cpu", "azure", "io")
    resource_work: dict[str, float] = {name: 0.0 for name in request_names}
    for spec in specs:
        request = spec.request
        for name in request_names:
            resource_work[name] += getattr(request, name) * durations[spec.stage_id]
    capacity_bound = max((resource_work[name] / capacities[name] for name in request_names if capacities[name]), default=0.0)
    longest: dict[str, float] = {}
    for spec in _topological(specs):
        deps = list(spec.dependencies)
        if spec.requires_dev_feedback and spec.dev_feedback_stage:
            deps.append(spec.dev_feedback_stage)
        if spec.early_stopping_gate:
            deps.append(spec.early_stopping_gate)
        longest[spec.stage_id] = durations[spec.stage_id] + max((longest.get(dep, 0.0) for dep in deps), default=0.0)
    return max(capacity_bound, max(longest.values(), default=0.0))


def _topological(specs: Sequence[StageSpec]) -> tuple[StageSpec, ...]:
    by_id = {spec.stage_id: spec for spec in specs}
    remaining = dict(by_id)
    ordered: list[StageSpec] = []
    while remaining:
        ready = [
            spec
            for spec in remaining.values()
            if (set(_stage_gates(spec)) & set(by_id)).issubset({item.stage_id for item in ordered})
        ]
        if not ready:
            # build_dry_run_plan gives the public error text for invalid graphs;
            # this fallback keeps the estimator's helper total for gate-only data.
            raise ValueError("stage graph contains a cycle or unknown dependency")
        for spec in sorted(ready, key=lambda item: item.stage_id):
            ordered.append(spec)
            del remaining[spec.stage_id]
    return tuple(ordered)


def _stage_gates(spec: StageSpec) -> tuple[str, ...]:
    return tuple(spec.dependencies) + tuple(
        gate for gate in (spec.dev_feedback_stage, spec.early_stopping_gate) if gate
    )


def _scaled(specs: Sequence[StageSpec], factor: float) -> tuple[StageSpec, ...]:
    return tuple(
        replace(spec, duration_minutes=spec.duration_minutes / factor)
        if spec.generation or "generation" in spec.kind.casefold()
        else spec
        for spec in specs
    )


def reconcile_before_after(*, before_label: str, after_label: str, specs: Sequence[StageSpec], before: Mapping[str, str], after: Mapping[str, str], observations: Iterable[RuntimeObservation] = ()) -> ReconciliationReport:
    observed = _observation_map(observations)
    rows: list[ReconciliationRow] = []
    for spec in specs:
        projected = spec.duration_minutes
        actual = observed.get(spec.stage_id)
        rows.append(ReconciliationRow(spec.stage_id, str(before.get(spec.stage_id, "PENDING")), str(after.get(spec.stage_id, before.get(spec.stage_id, "PENDING"))), projected, actual.wall_clock_minutes if actual else None, (actual.wall_clock_minutes - projected) if actual else None))
    return ReconciliationReport(before_label, after_label, tuple(rows), sum(row.before_status != row.after_status for row in rows))


def estimate_runtime(*, specs: Iterable[StageSpec], as_of: str, source_hashes: Mapping[str, str] | None = None, policy: ResourcePolicy | None = None, state: CampaignState | None = None, observations: Iterable[RuntimeObservation] = (), elapsed_minutes: float = 0.0, authorization_ok: bool | None = None, before_statuses: Mapping[str, str] | None = None, after_statuses: Mapping[str, str] | None = None) -> RuntimeEstimateReport:
    """Estimate policy-compatible work without observing or launching anything."""

    stages = tuple(specs)
    if not stages:
        chosen = policy or ResourcePolicy()
        empty = reconcile_before_after(before_label="before", after_label="after", specs=(), before={}, after={})
        return RuntimeEstimateReport(as_of, dict(source_hashes or {}), (), 0.0, 0.0, 0.0, {factor: 0.0 for factor in GENERATION_FACTORS}, {1: 0.0}, empty, ("No stages were supplied.",), "No gates supplied.", "PROJECTED", build_dry_run_plan((), policy=chosen))
    chosen = policy or ResourcePolicy()
    current_state = state or CampaignState(stages[0].campaign_id)
    by_id = {spec.stage_id: spec for spec in stages}
    observed = _observation_map(observations)
    rows: list[EstimateRow] = []
    durations: dict[str, float] = {}
    assumptions: list[str] = []
    for spec in stages:
        status, reason, conditional = _classify(spec, current_state, by_id, authorization_ok)
        observation = observed.get(spec.stage_id)
        if observation is None:
            assumptions.append(f"{spec.stage_id}: supplied duration is unmeasured")
        duration = _effective_duration(spec, observation)
        durations[spec.stage_id] = duration
        projected = None if status in {EstimateStatus.NOT_SCHEDULED_NAACL_BALANCED, EstimateStatus.BLOCKED, EstimateStatus.REUSE} else duration
        rows.append(EstimateRow(spec.stage_id, spec.run_id, status, projected, projected, reason, conditional, dict(source_hashes or {})))
    row_by_id = {row.stage_id: row for row in rows}
    # A downstream stage cannot be projected as runnable when a supplied
    # dependency is excluded or blocked.  Reused/completed dependencies are
    # retained as zero-duration planning anchors instead.
    for index, spec in enumerate(stages):
        row = row_by_id[spec.stage_id]
        if row.status in {EstimateStatus.BLOCKED, EstimateStatus.NOT_SCHEDULED_NAACL_BALANCED, EstimateStatus.REUSE}:
            continue
        bad_dependencies = [
            dependency for dependency in _stage_gates(spec)
            if dependency in row_by_id
            and row_by_id[dependency].status in {EstimateStatus.BLOCKED, EstimateStatus.NOT_SCHEDULED_NAACL_BALANCED}
            and not _completed(current_state.record(dependency).status)
        ]
        if bad_dependencies:
            row = replace(row, status=EstimateStatus.BLOCKED, projected_minutes=None, remaining_minutes=None, reason=f"dependency blocked: {sorted(bad_dependencies)}")
            rows[index] = row
            row_by_id[spec.stage_id] = row
    active = tuple(spec for spec, row in zip(stages, rows) if row.status not in {EstimateStatus.NOT_SCHEDULED_NAACL_BALANCED, EstimateStatus.BLOCKED, EstimateStatus.REUSE})
    active_durations = {spec.stage_id: durations[spec.stage_id] for spec in active}
    anchor_ids = {
        dependency
        for spec in active
        for dependency in _stage_gates(spec)
        if dependency in by_id and dependency not in {item.stage_id for item in active}
        and (row_by_id[dependency].status == EstimateStatus.REUSE or _completed(current_state.record(dependency).status))
    }
    anchors = tuple(replace(by_id[stage_id], duration_minutes=0.0) for stage_id in sorted(anchor_ids))
    measured_active = tuple(replace(spec, duration_minutes=active_durations[spec.stage_id]) for spec in active) + anchors
    planning_state = current_state
    for anchor_id in anchor_ids:
        if row_by_id[anchor_id].status == EstimateStatus.REUSE:
            planning_state = planning_state.with_record(StageRecord(anchor_id, StageStatus.REUSED))
    # A gate is retained in the plan so the makespan is conditional, while
    # excluded/reused work is removed from future workload.
    plan = build_dry_run_plan(measured_active, policy=chosen, state=planning_state)
    lower_bound = _lower_bound(active, active_durations, chosen)
    remaining = max(0.0, plan.makespan_minutes - max(0.0, elapsed_minutes))
    generation_sensitivity: dict[float, float] = {}
    for factor in GENERATION_FACTORS:
        scaled = _scaled(measured_active, factor)
        scaled_plan = build_dry_run_plan(scaled, policy=chosen, state=planning_state)
        generation_sensitivity[factor] = scaled_plan.makespan_minutes
    phobert_sensitivity: dict[int, float | None] = {1: plan.makespan_minutes}
    if chosen.phobert_profile is None:
        phobert_sensitivity[2] = None
        assumptions.append("PhoBERT concurrency=2 is unmeasured and blocked without an explicit validated >=25% profile.")
    else:
        concurrent_policy = replace(chosen, phobert_concurrency=2)
        phobert_sensitivity[2] = build_dry_run_plan(measured_active, policy=concurrent_policy, state=planning_state).makespan_minutes
    if chosen.resource_aware_enabled:
        assumptions.append("Resource-aware makespan assumes bounded lanes and deterministic dependency release.")
    else:
        assumptions.append("Legacy sequential_review_gated mode remains default-off for resource-aware overlap.")
    conditional = any(row.gate_conditional for row in rows)
    projection_logic = "PROJECTED_GATE_CONDITIONAL: DEV feedback and early-stopping gates are projected only when their injected state is incomplete." if conditional else "PROJECTED: all supplied gates are resolved or no gates are required."
    before = before_statuses or {spec.stage_id: current_state.record(spec.stage_id).status.value for spec in stages}
    after = after_statuses or {row.stage_id: row.status.value for row in rows}
    reconciliation = reconcile_before_after(before_label="before", after_label="after", specs=stages, before=before, after=after, observations=observations)
    return RuntimeEstimateReport(as_of, dict(source_hashes or {}), tuple(rows), lower_bound, plan.makespan_minutes, remaining, generation_sensitivity, phobert_sensitivity, reconciliation, tuple(dict.fromkeys(assumptions)), projection_logic, "PROJECTED_GATE_CONDITIONAL" if conditional else "PROJECTED", plan)


def estimate_campaign(**kwargs: Any) -> RuntimeEstimateReport:
    """Named façade for callers that keep scheduler and estimator separate."""

    return estimate_runtime(**kwargs)


__all__ = [
    "ESTIMATE_STATUSES", "GENERATION_FACTORS", "EstimateRow", "EstimateStatus", "ReconciliationReport", "ReconciliationRow",
    "RuntimeEstimateReport", "RuntimeObservation", "estimate_campaign", "estimate_runtime",
    "reconcile_before_after",
]
