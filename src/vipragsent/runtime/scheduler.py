"""Pure, opt-in resource scheduling policy for future LUNA campaigns.

This module deliberately stops at policy, state transitions, and dry-run
planning.  It does not launch processes, acquire cloud resources, inspect the
filesystem, or mutate a production run.  The small immutable records are
intended to be persisted by a caller that owns the durable storage boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
from typing import Any, Iterable, Mapping, Protocol, Sequence


DEFAULT_RESOURCE_AWARE_ENABLED = False
LEGACY_SCHEDULER_MODE = "sequential_review_gated"
MAX_CPU_LANES = 256
MAX_AZURE_LANES = 32
MAX_IO_LANES = 32
MIN_PHOBERT_PROFILE_CPU_FRACTION = 0.25


class SchedulerInvariantError(ValueError):
    """Raised when a policy or state transition would be unsafe."""


class StageStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    STOP_REQUESTED = "STOP_REQUESTED"
    STOPPED = "STOPPED"
    REUSED = "REUSED"
    RESUMABLE = "RESUMABLE"


class ResourceClass(str, Enum):
    SEVEN_B = "7b"
    XLM_R = "xlmr"
    PHOBERT = "phobert"
    CPU = "cpu"
    AZURE = "azure"
    IO = "io"


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """Validated host profile needed to run more than one PhoBERT job."""

    profile_id: str
    cpu_fraction: float
    memory_gb: float = 0.0
    validated: bool = True
    throughput_gain_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise SchedulerInvariantError("resource profile must have an id")
        if not 0.0 < self.cpu_fraction <= 1.0:
            raise SchedulerInvariantError("profile cpu_fraction must be in (0, 1]")
        if self.memory_gb < 0:
            raise SchedulerInvariantError("profile memory_gb cannot be negative")
        if not 0.0 <= self.throughput_gain_fraction <= 1.0:
            raise SchedulerInvariantError("profile throughput_gain_fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ResourcePolicy:
    """Bounded resource capacities; defaults preserve legacy sequencing."""

    resource_aware_enabled: bool = DEFAULT_RESOURCE_AWARE_ENABLED
    mode: str = LEGACY_SCHEDULER_MODE
    seven_b_exclusive: int = 1
    xlmr_exclusive: int = 1
    phobert_concurrency: int = 1
    cpu_lanes: int = 1
    azure_lanes: int = 1
    io_lanes: int = 1
    phobert_profile: ResourceProfile | None = None

    def __post_init__(self) -> None:
        if self.mode not in {LEGACY_SCHEDULER_MODE, "resource_aware"}:
            raise SchedulerInvariantError(f"unsupported scheduler mode: {self.mode}")
        if self.resource_aware_enabled and self.mode != "resource_aware":
            raise SchedulerInvariantError("resource-aware policy must use resource_aware mode")
        if not self.resource_aware_enabled and self.mode != LEGACY_SCHEDULER_MODE:
            raise SchedulerInvariantError("legacy policy must use sequential_review_gated mode")
        if self.seven_b_exclusive != 1 or self.xlmr_exclusive != 1:
            raise SchedulerInvariantError("7B and XLM-R are each exclusive single lanes")
        _bounded_int("phobert_concurrency", self.phobert_concurrency, 1, 2)
        _bounded_int("cpu_lanes", self.cpu_lanes, 1, MAX_CPU_LANES)
        _bounded_int("azure_lanes", self.azure_lanes, 1, MAX_AZURE_LANES)
        _bounded_int("io_lanes", self.io_lanes, 1, MAX_IO_LANES)
        if self.phobert_concurrency > 1:
            if self.phobert_profile is None or not self.phobert_profile.validated:
                raise SchedulerInvariantError(
                    "PhoBERT concurrency above one requires an explicit validated profile"
                )
            if self.phobert_profile.cpu_fraction < MIN_PHOBERT_PROFILE_CPU_FRACTION:
                raise SchedulerInvariantError(
                    "PhoBERT concurrency above one requires at least 25% CPU profile"
                )
            if self.phobert_profile.throughput_gain_fraction < MIN_PHOBERT_PROFILE_CPU_FRACTION:
                raise SchedulerInvariantError(
                    "PhoBERT concurrency above one requires at least 25% aggregate throughput gain"
                )

    @classmethod
    def legacy(cls) -> ResourcePolicy:
        return cls()

    @classmethod
    def resource_aware(cls, **kwargs: Any) -> ResourcePolicy:
        return cls(resource_aware_enabled=True, mode="resource_aware", **kwargs)

    def capacities(self) -> dict[str, int]:
        return {
            "seven_b": self.seven_b_exclusive,
            "xlmr": self.xlmr_exclusive,
            "phobert": self.phobert_concurrency,
            "cpu": self.cpu_lanes,
            "azure": self.azure_lanes,
            "io": self.io_lanes,
            ResourceClass.SEVEN_B.value: self.seven_b_exclusive,
            ResourceClass.XLM_R.value: self.xlmr_exclusive,
        }


def _bounded_int(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not minimum <= value <= maximum:
        raise SchedulerInvariantError(f"{name} must be an integer in [{minimum}, {maximum}]")


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """Integer lane demand for a stage."""

    seven_b: int = 0
    xlmr: int = 0
    phobert: int = 0
    cpu: int = 0
    azure: int = 0
    io: int = 0

    def __post_init__(self) -> None:
        for name in ("seven_b", "xlmr", "phobert", "cpu", "azure", "io"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0 or int(value) != value:
                raise SchedulerInvariantError(f"resource request {name} must be a non-negative integer")

    @classmethod
    def for_class(cls, resource_class: ResourceClass | str) -> ResourceRequest:
        name = ResourceClass(resource_class).value
        return cls(**{name.replace("7b", "seven_b").replace("xlmr", "xlmr"): 1})

    def fits(self, used: Mapping[str, int], capacity: Mapping[str, int]) -> bool:
        for name in ("seven_b", "xlmr", "phobert", "cpu", "azure", "io"):
            if used.get(name, 0) + getattr(self, name) > capacity.get(name, 0):
                return False
        return True

    def as_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name in ("seven_b", "xlmr", "phobert", "cpu", "azure", "io")}


@dataclass(frozen=True, slots=True)
class StageSpec:
    """A schedulable unit, carrying all policy-relevant dependencies."""

    stage_id: str
    campaign_id: str
    run_id: str
    kind: str
    duration_minutes: float
    resource_class: ResourceClass | str = ResourceClass.CPU
    dependencies: tuple[str, ...] = ()
    request: ResourceRequest | None = None
    dev_feedback_stage: str | None = None
    requires_dev_feedback: bool = False
    early_stopping_gate: str | None = None
    execution_mode: str = "train"
    reusable_artifact: bool = False
    resumable: bool = False
    artifact_id: str | None = None
    naacl_balanced: bool = True
    generation: bool = False

    def __post_init__(self) -> None:
        if not self.stage_id or not self.campaign_id or not self.run_id:
            raise SchedulerInvariantError("stage, campaign, and run identities are required")
        if self.duration_minutes < 0:
            raise SchedulerInvariantError("stage duration cannot be negative")
        if self.request is None:
            object.__setattr__(self, "request", ResourceRequest.for_class(self.resource_class))
        if len(set(self.dependencies)) != len(self.dependencies):
            raise SchedulerInvariantError(f"duplicate dependency for {self.stage_id}")
        if self.requires_dev_feedback and not self.dev_feedback_stage:
            raise SchedulerInvariantError("requires_dev_feedback needs dev_feedback_stage")
        if self.execution_mode not in {"train", "resume", "evaluate", "artifact", "reuse"}:
            raise SchedulerInvariantError(f"unsupported execution mode: {self.execution_mode}")


WorkloadSpec = StageSpec
JobSpec = StageSpec


@dataclass(frozen=True, slots=True)
class StageRecord:
    stage_id: str
    status: StageStatus = StageStatus.PENDING
    artifact: "ArtifactEvidence | None" = None
    safe_stop_requested: bool = False
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class JournalEntry:
    sequence: int
    campaign_id: str
    run_id: str
    stage_id: str
    from_status: str
    to_status: str
    event: str
    at: float
    actor: str


@dataclass(frozen=True, slots=True)
class DurableJournal:
    """Append-only, serializable journal; persistence is injected by callers."""

    entries: tuple[JournalEntry, ...] = ()

    def append(self, entry: JournalEntry) -> DurableJournal:
        expected = len(self.entries) + 1
        if entry.sequence != expected:
            raise SchedulerInvariantError("journal sequence is not contiguous")
        return replace(self, entries=self.entries + (entry,))

    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple({"sequence": entry.sequence, "campaign_id": entry.campaign_id, "run_id": entry.run_id, "stage_id": entry.stage_id, "from_status": entry.from_status, "to_status": entry.to_status, "event": entry.event, "at": entry.at, "actor": entry.actor} for entry in self.entries)


class JournalSink(Protocol):
    def append(self, entry: JournalEntry) -> None: ...


@dataclass(frozen=True, slots=True)
class CampaignState:
    campaign_id: str
    stages: tuple[StageRecord, ...] = ()
    journal: DurableJournal = field(default_factory=DurableJournal)
    authorized: bool = False
    safe_stop_requested: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.stages, Mapping):
            records = tuple(StageRecord(str(key), _status(value)) for key, value in self.stages.items())
            object.__setattr__(self, "stages", records)
        if len({record.stage_id for record in self.stages}) != len(self.stages):
            raise SchedulerInvariantError("campaign state contains duplicate stages")

    def record(self, stage_id: str) -> StageRecord:
        for record in self.stages:
            if record.stage_id == stage_id:
                return record
        return StageRecord(stage_id)

    def with_record(self, record: StageRecord) -> CampaignState:
        found = False
        records = []
        for current in self.stages:
            if current.stage_id == record.stage_id:
                records.append(record)
                found = True
            else:
                records.append(current)
        if not found:
            records.append(record)
        return replace(self, stages=tuple(records))


def _status(value: Any) -> StageStatus:
    return value if isinstance(value, StageStatus) else StageStatus(str(value))


_ALLOWED_TRANSITIONS: dict[StageStatus, frozenset[StageStatus]] = {
    StageStatus.PENDING: frozenset({StageStatus.READY, StageStatus.BLOCKED, StageStatus.REUSED, StageStatus.RESUMABLE}),
    StageStatus.READY: frozenset({StageStatus.RUNNING, StageStatus.BLOCKED}),
    StageStatus.RESUMABLE: frozenset({StageStatus.RUNNING, StageStatus.BLOCKED}),
    StageStatus.RUNNING: frozenset({StageStatus.SUCCEEDED, StageStatus.FAILED, StageStatus.STOP_REQUESTED}),
    StageStatus.STOP_REQUESTED: frozenset({StageStatus.STOPPED, StageStatus.SUCCEEDED}),
    StageStatus.FAILED: frozenset({StageStatus.RESUMABLE, StageStatus.BLOCKED, StageStatus.REUSED}),
    StageStatus.BLOCKED: frozenset({StageStatus.READY, StageStatus.REUSED}),
    StageStatus.SUCCEEDED: frozenset(),
    StageStatus.STOPPED: frozenset({StageStatus.RESUMABLE, StageStatus.BLOCKED}),
    StageStatus.REUSED: frozenset(),
}


def transition_stage(state: CampaignState, stage_id: str, to_status: StageStatus | str, *, event: str, at: float, actor: str, run_id: str | None = None, sink: JournalSink | None = None) -> CampaignState:
    """Apply one legal transition and return a new state plus journal record."""

    target = _status(to_status)
    current = state.record(stage_id)
    if target == current.status:
        raise SchedulerInvariantError("duplicate launch/state transition")
    if target not in _ALLOWED_TRANSITIONS[current.status]:
        raise SchedulerInvariantError(f"illegal transition {current.status.value} -> {target.value}")
    if not run_id:
        raise SchedulerInvariantError("journal transition requires run_id")
    entry = JournalEntry(len(state.journal.entries) + 1, state.campaign_id, run_id, stage_id, current.status.value, target.value, event, at, actor)
    updated_state = state.with_record(replace(current, status=target))
    updated = replace(updated_state, journal=state.journal.append(entry))
    if sink is not None:
        sink.append(entry)
    return updated


@dataclass(frozen=True, slots=True)
class LeaseIdentity:
    campaign_id: str
    run_id: str
    stage_id: str
    host: str
    pid: int
    instance_id: str
    heartbeat: float

    def __post_init__(self) -> None:
        if not all((self.campaign_id, self.run_id, self.stage_id, self.host, self.instance_id)):
            raise SchedulerInvariantError("complete lease identity is required")
        if self.pid <= 0:
            raise SchedulerInvariantError("lease pid must be positive")


@dataclass(frozen=True, slots=True)
class Lease:
    identity: LeaseIdentity
    acquired_at: float
    expires_at: float

    def stale(self, *, now: float, heartbeat_timeout: float) -> bool:
        return now - self.identity.heartbeat > heartbeat_timeout or now >= self.expires_at


@dataclass(frozen=True, slots=True)
class LeaseDecision:
    granted: bool
    lease: Lease | None
    reason: str
    recovered_stale: bool = False


def acquire_lease(existing: Lease | None, identity: LeaseIdentity, *, now: float, ttl: float = 900.0, heartbeat_timeout: float = 300.0) -> LeaseDecision:
    if ttl <= 0 or heartbeat_timeout <= 0:
        raise SchedulerInvariantError("lease TTL and heartbeat timeout must be positive")
    candidate = Lease(identity, now, now + ttl)
    if existing is None:
        return LeaseDecision(True, candidate, "ACQUIRED")
    if not existing.stale(now=now, heartbeat_timeout=heartbeat_timeout):
        return LeaseDecision(False, existing, "DUPLICATE_LAUNCH_PREVENTED")
    return LeaseDecision(True, candidate, "STALE_LEASE_RECOVERED", recovered_stale=True)


def renew_lease(lease: Lease, identity: LeaseIdentity, *, now: float, ttl: float = 900.0) -> Lease:
    if lease.identity != identity:
        raise SchedulerInvariantError("only the exact campaign/run/stage/host/PID/instance owner may renew")
    if now < lease.acquired_at:
        raise SchedulerInvariantError("lease renewal cannot move backward in time")
    return Lease(identity, lease.acquired_at, now + ttl)


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    artifact_id: str
    sha256: str
    size_bytes: int
    required_files: tuple[str, ...] = ()
    present_files: tuple[str, ...] = ()
    provenance_hash: str = ""


def validate_artifact(artifact: ArtifactEvidence | None, *, expected_artifact_id: str | None = None, expected_provenance_hash: str | None = None) -> tuple[bool, tuple[str, ...]]:
    errors: list[str] = []
    if artifact is None:
        errors.append("artifact evidence is missing")
    else:
        if expected_artifact_id is not None and artifact.artifact_id != expected_artifact_id:
            errors.append("artifact identity mismatch")
        if len(artifact.sha256) != 64 or any(char not in "0123456789abcdef" for char in artifact.sha256.lower()):
            errors.append("artifact sha256 is invalid")
        if artifact.size_bytes <= 0:
            errors.append("artifact is empty")
        missing = set(artifact.required_files) - set(artifact.present_files)
        if missing:
            errors.append(f"artifact files missing: {sorted(missing)}")
        if expected_provenance_hash is not None and artifact.provenance_hash != expected_provenance_hash:
            errors.append("artifact provenance mismatch")
    return not errors, tuple(errors)


@dataclass(frozen=True, slots=True)
class RetryDecision:
    allowed: bool
    reason: str
    artifact_valid: bool


def retry_gate(artifact: ArtifactEvidence | None, *, expected_artifact_id: str | None = None, expected_provenance_hash: str | None = None) -> RetryDecision:
    valid, errors = validate_artifact(artifact, expected_artifact_id=expected_artifact_id, expected_provenance_hash=expected_provenance_hash)
    if not valid:
        return RetryDecision(False, "ARTIFACT_VALIDATION_FAILED: " + "; ".join(errors), False)
    return RetryDecision(True, "ARTIFACT_VALIDATED_BEFORE_RETRY", True)


@dataclass(frozen=True, slots=True)
class LockOwner:
    campaign_id: str
    run_id: str
    stage_id: str
    host: str
    instance_id: str


@dataclass(frozen=True, slots=True)
class LockRecord:
    key: str
    owner: LockOwner


@dataclass(frozen=True, slots=True)
class LockDecision:
    acquired: bool
    lock: LockRecord | None
    reason: str


def acquire_lock(existing: LockRecord | None, key: str, owner: LockOwner) -> LockDecision:
    if not key:
        raise SchedulerInvariantError("lock key is required")
    if existing is not None:
        if existing.key == key and existing.owner == owner:
            return LockDecision(False, existing, "LOCK_ALREADY_OWNED")
        return LockDecision(False, existing, "LOCK_OWNERSHIP_CONFLICT")
    return LockDecision(True, LockRecord(key, owner), "LOCK_ACQUIRED")


def release_lock(existing: LockRecord | None, owner: LockOwner) -> LockDecision:
    if existing is None:
        return LockDecision(False, None, "LOCK_NOT_FOUND")
    if existing.owner != owner:
        return LockDecision(False, existing, "LOCK_OWNERSHIP_CONFLICT")
    return LockDecision(True, None, "LOCK_RELEASED")


@dataclass(frozen=True, slots=True)
class StorageRequirement:
    required_bytes: int
    reserve_bytes: int = 0

    def __post_init__(self) -> None:
        if self.required_bytes < 0 or self.reserve_bytes < 0:
            raise SchedulerInvariantError("storage requirements cannot be negative")


@dataclass(frozen=True, slots=True)
class StorageSnapshot:
    free_bytes: int
    existing_reserved_bytes: int = 0

    def __post_init__(self) -> None:
        if self.free_bytes < 0 or self.existing_reserved_bytes < 0:
            raise SchedulerInvariantError("storage snapshot values cannot be negative")


@dataclass(frozen=True, slots=True)
class StoragePreflight:
    passed: bool
    available_bytes: int
    required_bytes: int
    reason: str


def preflight_storage(snapshot: StorageSnapshot, requirement: StorageRequirement) -> StoragePreflight:
    available = snapshot.free_bytes - snapshot.existing_reserved_bytes
    required = requirement.required_bytes + requirement.reserve_bytes
    return StoragePreflight(available >= required, available, required, "PASS" if available >= required else "BLOCKED_STORAGE")


HASH_NAMES = ("inventory", "config", "source", "environment", "model", "data", "runtime_policy")


def _is_hash(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value.lower())


@dataclass(frozen=True, slots=True)
class AuthorizationBinding:
    campaign_id: str
    inventory_hash: str
    config_hash: str
    source_hash: str
    environment_hash: str
    model_hash: str
    data_hash: str
    runtime_policy_hash: str
    authorized: bool = False

    def __post_init__(self) -> None:
        if not self.campaign_id:
            raise SchedulerInvariantError("authorization campaign_id is required")
        if any(not _is_hash(getattr(self, f"{name}_hash")) for name in HASH_NAMES):
            raise SchedulerInvariantError("authorization requires all seven SHA-256 bindings")

    def digest(self) -> str:
        payload = {name: getattr(self, f"{name}_hash") for name in HASH_NAMES} | {"campaign_id": self.campaign_id}
        return _sha256(payload)


def build_authorization_binding(campaign_id: str, hashes: Mapping[str, str], *, authorized: bool = False) -> AuthorizationBinding:
    normalised = {name: hashes.get(name, hashes.get(f"{name}_hash", "")) for name in HASH_NAMES}
    missing = [name for name, value in normalised.items() if not value]
    if missing:
        raise SchedulerInvariantError(f"authorization hash fields missing: {missing}")
    return AuthorizationBinding(campaign_id, *(str(normalised[name]) for name in HASH_NAMES), authorized=authorized)


def authorize_campaign(binding: AuthorizationBinding, expected: AuthorizationBinding) -> AuthorizationBinding:
    if binding.campaign_id != expected.campaign_id or binding.digest() != expected.digest():
        raise SchedulerInvariantError("campaign authorization binding drifted")
    return replace(binding, authorized=True)


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SafeStopDecision:
    accepted: bool
    boundary: str
    reason: str


def safe_stop_request(*, stage_status: StageStatus | str, checkpoint_written: bool, artifact_validated: bool, at_boundary: bool) -> SafeStopDecision:
    status = _status(stage_status)
    if status not in {StageStatus.RUNNING, StageStatus.STOP_REQUESTED}:
        return SafeStopDecision(False, "not-running", "SAFE_STOP_NOT_APPLICABLE")
    if at_boundary and checkpoint_written and artifact_validated:
        return SafeStopDecision(True, "checkpoint-and-artifact-boundary", "SAFE_STOP_ACCEPTED")
    return SafeStopDecision(False, "next-safe-boundary", "SAFE_STOP_DEFERRED")


@dataclass(frozen=True, slots=True)
class Placement:
    stage_id: str
    status: str
    start_minute: float | None
    end_minute: float | None
    reason: str
    launch: bool = False


@dataclass(frozen=True, slots=True)
class DryRunPlan:
    campaign_id: str
    mode: str
    placements: tuple[Placement, ...]
    makespan_minutes: float
    launches_performed: int = 0

    @property
    def is_dry_run(self) -> bool:
        return self.launches_performed == 0


def _topological(specs: Sequence[StageSpec]) -> tuple[StageSpec, ...]:
    by_id = {spec.stage_id: spec for spec in specs}
    if len(by_id) != len(specs):
        raise SchedulerInvariantError("duplicate stage id")
    for spec in specs:
        unknown = set(_gate_dependencies(spec)) - set(by_id)
        if unknown:
            raise SchedulerInvariantError(f"unknown dependencies for {spec.stage_id}: {sorted(unknown)}")
    result: list[StageSpec] = []
    remaining = dict(by_id)
    while remaining:
        ready = [spec for spec in remaining.values() if set(_gate_dependencies(spec)).issubset({item.stage_id for item in result})]
        if not ready:
            raise SchedulerInvariantError("stage graph contains a cycle")
        for spec in sorted(ready, key=lambda item: item.stage_id):
            result.append(spec)
            del remaining[spec.stage_id]
    return tuple(result)


def _gate_dependencies(spec: StageSpec) -> tuple[str, ...]:
    gates = list(spec.dependencies)
    if spec.requires_dev_feedback and spec.dev_feedback_stage and spec.dev_feedback_stage not in gates:
        gates.append(spec.dev_feedback_stage)
    if spec.early_stopping_gate and spec.early_stopping_gate not in gates:
        gates.append(spec.early_stopping_gate)
    return tuple(gates)


def build_dry_run_plan(specs: Iterable[StageSpec], *, policy: ResourcePolicy | None = None, state: CampaignState | None = None, safe_stop: bool = False) -> DryRunPlan:
    """Create a deterministic plan.  This function can never launch a stage."""

    stages = _topological(tuple(specs))
    if not stages:
        return DryRunPlan(state.campaign_id if state else "", (policy or ResourcePolicy()).mode, (), 0.0)
    campaign_id = stages[0].campaign_id
    if any(spec.campaign_id != campaign_id for spec in stages):
        raise SchedulerInvariantError("all stages in a plan must belong to one campaign")
    chosen = policy or ResourcePolicy()
    records = state or CampaignState(campaign_id)
    placements: list[Placement] = []
    intervals: list[tuple[float, float, ResourceRequest]] = []
    ends: dict[str, float] = {}
    cursor = 0.0
    by_id = {stage.stage_id: stage for stage in stages}
    for stage in stages:
        record = records.record(stage.stage_id)
        if record.status in {StageStatus.SUCCEEDED, StageStatus.REUSED}:
            placements.append(Placement(stage.stage_id, record.status.value, None, None, "already-complete", False))
            ends[stage.stage_id] = 0.0
            continue
        if safe_stop or records.safe_stop_requested:
            placements.append(Placement(stage.stage_id, "NOT_SCHEDULED", None, None, "safe-stop-boundary", False))
            continue
        dependencies = _gate_dependencies(stage)
        if any(by_id[dependency].stage_id not in ends for dependency in dependencies):
            placements.append(Placement(stage.stage_id, "BLOCKED", None, None, "dependency-or-gate-not-complete", False))
            continue
        if not stage.request.fits({}, chosen.capacities()):
            placements.append(Placement(stage.stage_id, "BLOCKED", None, None, "resource-request-exceeds-policy-capacity", False))
            continue
        earliest = max((ends[dependency] for dependency in dependencies), default=0.0)
        if not chosen.resource_aware_enabled:
            start = max(cursor, earliest)
        else:
            start = earliest
            while True:
                overlapping = [(left, right, request) for left, right, request in intervals if left < start + stage.duration_minutes and start < right]
                used = {name: sum(getattr(request, name) for _, _, request in overlapping) for name in ("seven_b", "xlmr", "phobert", "cpu", "azure", "io")}
                if stage.request.fits(used, chosen.capacities()):
                    break
                start = min(right for left, right, _request in overlapping if right > start)
        end = start + stage.duration_minutes
        intervals.append((start, end, stage.request))
        ends[stage.stage_id] = end
        cursor = end
        placements.append(Placement(stage.stage_id, "SCHEDULED", start, end, "resource-aware" if chosen.resource_aware_enabled else "legacy-sequential", False))
    makespan = max((placement.end_minute or 0.0 for placement in placements), default=0.0)
    return DryRunPlan(campaign_id, chosen.mode, tuple(placements), makespan)


__all__ = [
    "AuthorizationBinding", "CampaignState", "DEFAULT_RESOURCE_AWARE_ENABLED", "DurableJournal", "DryRunPlan",
    "JobSpec", "JournalEntry", "LEGACY_SCHEDULER_MODE", "Lease", "LeaseDecision", "LeaseIdentity",
    "LockDecision", "LockOwner", "LockRecord", "Placement", "ResourceClass", "ResourcePolicy",
    "ResourceProfile", "ResourceRequest", "RetryDecision", "SafeStopDecision", "SchedulerInvariantError",
    "StageRecord", "StageSpec", "StageStatus", "StoragePreflight", "StorageRequirement", "StorageSnapshot",
    "WorkloadSpec", "acquire_lease", "acquire_lock", "authorize_campaign", "build_authorization_binding",
    "build_dry_run_plan", "preflight_storage", "release_lock", "renew_lease", "retry_gate", "safe_stop_request",
    "transition_stage", "validate_artifact",
]
