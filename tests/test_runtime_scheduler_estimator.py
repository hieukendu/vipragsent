from __future__ import annotations

import ast
from pathlib import Path

import pytest

from vipragsent.runtime.estimator import (
    ESTIMATE_STATUSES,
    GENERATION_FACTORS,
    EstimateStatus,
    RuntimeObservation,
    estimate_runtime,
)
from vipragsent.runtime.scheduler import (
    AuthorizationBinding,
    ArtifactEvidence,
    CampaignState,
    DurableJournal,
    LeaseIdentity,
    ResourcePolicy,
    ResourceProfile,
    SchedulerInvariantError,
    StageRecord,
    StageSpec,
    StageStatus,
    StorageRequirement,
    StorageSnapshot,
    acquire_lease,
    acquire_lock,
    build_authorization_binding,
    build_dry_run_plan,
    preflight_storage,
    release_lock,
    renew_lease,
    retry_gate,
    safe_stop_request,
    transition_stage,
    validate_artifact,
)


def stage(stage_id: str, minutes: float, resource: str, **kwargs: object) -> StageSpec:
    return StageSpec(
        stage_id=stage_id,
        campaign_id="campaign",
        run_id=stage_id,
        kind=kwargs.pop("kind", "train"),
        duration_minutes=minutes,
        resource_class=resource,
        **kwargs,
    )


def test_default_policy_is_legacy_and_resource_lanes_are_bounded() -> None:
    policy = ResourcePolicy()
    assert policy.resource_aware_enabled is False
    assert policy.mode == "sequential_review_gated"
    assert policy.capacities()["7b"] == 1
    assert policy.capacities()["xlmr"] == 1
    assert policy.capacities()["phobert"] == 1
    with pytest.raises(SchedulerInvariantError):
        ResourcePolicy(cpu_lanes=0)
    with pytest.raises(SchedulerInvariantError):
        ResourcePolicy(seven_b_exclusive=2)
    with pytest.raises(SchedulerInvariantError):
        ResourcePolicy(phobert_concurrency=2)
    with pytest.raises(SchedulerInvariantError):
        ResourcePolicy.resource_aware(phobert_concurrency=2, phobert_profile=ResourceProfile("weak", 0.2))
    with pytest.raises(SchedulerInvariantError):
        ResourcePolicy.resource_aware(
            phobert_concurrency=2,
            phobert_profile=ResourceProfile("validated-low-gain", 0.25, throughput_gain_fraction=0.24),
        )
    with pytest.raises(SchedulerInvariantError):
        ResourcePolicy.resource_aware(
            phobert_concurrency=2,
            phobert_profile=ResourceProfile("unvalidated", 0.25, validated=False, throughput_gain_fraction=0.25),
        )
    assert ResourcePolicy.resource_aware(
        phobert_concurrency=2,
        phobert_profile=ResourceProfile("validated", 0.25, throughput_gain_fraction=0.25),
    ).capacities()["phobert"] == 2


def test_legacy_mode_is_sequential_and_resource_aware_mode_overlaps_safe_lanes() -> None:
    specs = (stage("seven", 10, "7b"), stage("phobert", 4, "phobert"), stage("xlmr", 5, "xlmr"))
    legacy = build_dry_run_plan(specs)
    assert legacy.makespan_minutes == 19
    assert legacy.launches_performed == 0
    aware = build_dry_run_plan(specs, policy=ResourcePolicy.resource_aware())
    starts = {item.stage_id: item.start_minute for item in aware.placements}
    assert aware.makespan_minutes == 10
    assert starts == {"phobert": 0.0, "seven": 0.0, "xlmr": 0.0}


def test_exclusive_and_phobert_lanes_prevent_invalid_overlap() -> None:
    specs = (stage("p1", 10, "phobert"), stage("p2", 3, "phobert"))
    one = build_dry_run_plan(specs, policy=ResourcePolicy.resource_aware())
    assert one.makespan_minutes == 13
    two = build_dry_run_plan(
        specs,
        policy=ResourcePolicy.resource_aware(
            phobert_concurrency=2,
            phobert_profile=ResourceProfile("cpu-quarter", 0.25, throughput_gain_fraction=0.25),
        ),
    )
    assert two.makespan_minutes == 10


def test_dag_and_dev_feedback_gate_are_preserved_in_plan() -> None:
    dev = stage("dev", 5, "cpu")
    train = stage("train", 10, "7b", requires_dev_feedback=True, dev_feedback_stage="dev")
    plan = build_dry_run_plan((train, dev), policy=ResourcePolicy.resource_aware())
    placement = {item.stage_id: item for item in plan.placements}
    assert placement["train"].start_minute == 5
    with pytest.raises(SchedulerInvariantError):
        build_dry_run_plan((stage("bad", 1, "cpu", dependencies=("missing",)),))


def test_journal_transitions_are_append_only_and_duplicate_launch_is_rejected() -> None:
    state = CampaignState("campaign")
    state = transition_stage(state, "run", StageStatus.READY, event="ready", at=1, actor="planner", run_id="run")
    state = transition_stage(state, "run", StageStatus.RUNNING, event="launch", at=2, actor="planner", run_id="run")
    assert state.journal.entries[1].sequence == 2
    assert state.journal.records()[0]["to_status"] == "READY"
    with pytest.raises(SchedulerInvariantError):
        transition_stage(state, "run", StageStatus.RUNNING, event="duplicate", at=3, actor="planner", run_id="run")
    with pytest.raises(SchedulerInvariantError):
        DurableJournal().append(state.journal.entries[0].__class__(2, "c", "r", "s", "PENDING", "READY", "x", 0, "a"))


def test_lease_identity_lock_ownership_and_stale_recovery() -> None:
    identity = LeaseIdentity("c", "r", "s", "host-a", 101, "instance-a", 0)
    acquired = acquire_lease(None, identity, now=0, ttl=10, heartbeat_timeout=5)
    assert acquired.granted
    duplicate = acquire_lease(acquired.lease, identity, now=1)
    assert duplicate.reason == "DUPLICATE_LAUNCH_PREVENTED"
    other = LeaseIdentity("c", "r", "s", "host-b", 101, "instance-b", 20)
    recovered = acquire_lease(acquired.lease, other, now=20)
    assert recovered.recovered_stale
    with pytest.raises(SchedulerInvariantError):
        renew_lease(acquired.lease, other, now=2)
    owner = acquired.lease.identity
    from vipragsent.runtime.scheduler import LockOwner

    left = LockOwner(owner.campaign_id, owner.run_id, owner.stage_id, owner.host, owner.instance_id)
    right = LockOwner("c", "r", "s", "other", "other")
    lock = acquire_lock(None, "campaign-lock", left)
    assert lock.acquired
    assert acquire_lock(lock.lock, "campaign-lock", right).reason == "LOCK_OWNERSHIP_CONFLICT"
    assert release_lock(lock.lock, right).reason == "LOCK_OWNERSHIP_CONFLICT"
    assert release_lock(lock.lock, left).reason == "LOCK_RELEASED"


def test_artifact_validation_precedes_retry_and_storage_is_fail_closed() -> None:
    good = ArtifactEvidence("a", "a" * 64, 10, ("manifest",), ("manifest",), "p")
    assert validate_artifact(good, expected_artifact_id="a", expected_provenance_hash="p")[0]
    assert retry_gate(None).allowed is False
    assert retry_gate(good, expected_provenance_hash="wrong").artifact_valid is False
    assert preflight_storage(StorageSnapshot(100, 10), StorageRequirement(80, 10)).passed
    assert not preflight_storage(StorageSnapshot(100, 11), StorageRequirement(80, 10)).passed


def test_authorization_requires_all_campaign_bindings() -> None:
    hashes = {name: f"{index + 1:064x}" for index, name in enumerate(("inventory", "config", "source", "environment", "model", "data", "runtime_policy"))}
    binding = build_authorization_binding("campaign", hashes)
    assert isinstance(binding, AuthorizationBinding)
    assert binding.authorized is False
    assert build_authorization_binding("campaign", hashes, authorized=True).digest() == binding.digest()
    with pytest.raises(SchedulerInvariantError):
        build_authorization_binding("campaign", {"inventory": "a"})


def test_safe_stop_only_accepts_checkpoint_artifact_boundary() -> None:
    deferred = safe_stop_request(stage_status="RUNNING", checkpoint_written=False, artifact_validated=False, at_boundary=False)
    accepted = safe_stop_request(stage_status="RUNNING", checkpoint_written=True, artifact_validated=True, at_boundary=True)
    assert deferred.reason == "SAFE_STOP_DEFERRED"
    assert accepted.accepted


def test_estimator_has_exact_statuses_reconciliation_hashes_and_sensitivities() -> None:
    specs = (
        stage("generation", 10, "7b", kind="generation", generation=True),
        stage("eval", 3, "cpu", kind="evaluate", execution_mode="evaluate", dependencies=("generation",)),
        stage("excluded", 8, "phobert", naacl_balanced=False),
    )
    report = estimate_runtime(
        specs=specs,
        as_of="2026-08-16T00:00:00Z",
        source_hashes={"inventory": "i", "runtime_policy": "p"},
        policy=ResourcePolicy.resource_aware(),
        observations=(RuntimeObservation("generation", 12),),
        elapsed_minutes=2,
        before_statuses={"generation": "RUNNING"},
        after_statuses={"generation": "TRAIN"},
    )
    statuses = {row.stage_id: row.status for row in report.rows}
    assert statuses == {"generation": EstimateStatus.TRAIN, "eval": EstimateStatus.EVALUATE_ONLY, "excluded": EstimateStatus.NOT_SCHEDULED_NAACL_BALANCED}
    assert set(ESTIMATE_STATUSES) == {"REUSE", "RESUME", "TRAIN", "EVALUATE_ONLY", "ARTIFACT_ONLY", "NOT_SCHEDULED_NAACL_BALANCED", "BLOCKED"}
    assert report.as_of.endswith("Z")
    assert report.source_hashes["inventory"] == "i"
    assert tuple(report.generation_sensitivity) == GENERATION_FACTORS
    assert all(isinstance(factor, float) for factor in report.generation_sensitivity)
    assert report.generation_sensitivity[4.0] >= report.generation_sensitivity[1.0]
    assert report.phobert_concurrency_sensitivity[2] is None
    assert report.reconciliation.changed_stage_count == 1
    assert report.remaining_wall_clock_minutes >= 0
    assert report.projection_status == "PROJECTED_GATE_CONDITIONAL"
    assert "unmeasured" in " ".join(report.assumptions)


def test_estimator_blocks_unauthorized_work_and_supports_reuse_artifact() -> None:
    spec = stage("reuse", 5, "phobert", execution_mode="reuse", reusable_artifact=True, artifact_id="artifact")
    artifact = ArtifactEvidence("artifact", "b" * 64, 2)
    state = CampaignState("campaign").with_record(StageRecord("reuse", StageStatus.SUCCEEDED, artifact))
    reused = estimate_runtime(specs=(spec,), as_of="now", state=state)
    assert reused.rows[0].status == EstimateStatus.REUSE
    blocked = estimate_runtime(specs=(spec,), as_of="now", authorization_ok=False)
    assert blocked.rows[0].status == EstimateStatus.BLOCKED


def test_scheduler_and_estimator_are_static_side_effect_free_modules() -> None:
    root = Path(__file__).resolve().parents[1] / "src/vipragsent/runtime"
    forbidden_imports = {"subprocess", "requests", "torch", "transformers", "azure", "huggingface_hub"}
    forbidden_calls = {"open", "write_text", "write_bytes", "unlink", "mkdir", "rmtree", "run", "Popen"}
    for name in ("scheduler.py", "estimator.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] not in forbidden_imports for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or node.module.split(".")[0] not in forbidden_imports
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls
