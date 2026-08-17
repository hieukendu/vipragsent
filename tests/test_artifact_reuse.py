from __future__ import annotations

from dataclasses import dataclass

import pytest

from vipragsent.runtime.artifact_reuse import (
    EXACT_BINDING_FIELDS,
    LIVE_CODE_IDENTITY_UNCERTAIN,
    ReuseDecision,
    ReuseEvent,
    ReuseState,
    ReuseStatus,
    decide_reuse,
    plan_backup,
    plan_restore,
    transition,
)


def _metadata() -> dict[str, object]:
    return {
        "system": "system-a",
        "run": "run-001",
        "seed": 20260521,
        "checkpoint_sha": "checkpoint-sha-001",
        "config_sha": "config-sha-001",
        "dataset_sha": "dataset-sha-001",
        "model": "model-revision-001",
        "tokenizer": "tokenizer-revision-001",
        "code": "code-commit-001",
        "source": "source-run-001",
        "approval": "approval-sha-001",
        "checkpoint_epoch": 1,
        "checkpoint_kind": "best",
    }


def test_exact_match_is_reuse_and_hash_bound() -> None:
    metadata = _metadata()
    result = decide_reuse(metadata, dict(metadata), live_code_identity="code-commit-001")
    # VERIFIED is the compatibility spelling for normalized REUSE.
    assert result.status is ReuseStatus.REUSE
    assert result.status is ReuseStatus.VERIFIED
    assert result.reusable
    assert result.local_metadata_hash == result.remote_metadata_hash
    assert result.evidence_bearing
    assert set(result.matched_fields) == set(EXACT_BINDING_FIELDS)


def test_exact_paused_state_is_resume() -> None:
    result = decide_reuse(_metadata(), _metadata(), state={"status": "SAFELY_PAUSED"})
    assert result.status is ReuseStatus.RESUME
    assert result.resumable
    assert not result.reusable


def test_explicit_resume_does_not_use_checkpoint_filename() -> None:
    local = _metadata() | {"checkpoint_path": "checkpoints/epoch_1/model.pt"}
    remote = _metadata() | {"checkpoint_path": "checkpoints/latest/model.pt"}
    result = decide_reuse(local, remote, mode="resume")
    assert result.status is ReuseStatus.RESUME


def test_checkpoint_sha_alone_cannot_authorize_reuse() -> None:
    local = {"checkpoint_sha": "same-checkpoint"}
    remote = {"checkpoint_sha": "same-checkpoint"}
    result = decide_reuse(local, remote)
    assert result.status is ReuseStatus.BLOCKED
    assert any(reason.endswith(":system") for reason in result.reasons)


@pytest.mark.parametrize("field", EXACT_BINDING_FIELDS)
def test_each_missing_exact_binding_field_is_blocked(field: str) -> None:
    local = _metadata()
    remote = _metadata()
    del local[field]
    result = decide_reuse(local, remote)
    assert result.status is ReuseStatus.BLOCKED
    assert f"MISSING_BINDING_FIELD:local:{field}" in result.reasons


@pytest.mark.parametrize("field", EXACT_BINDING_FIELDS)
def test_each_conflicting_exact_binding_field_is_blocked(field: str) -> None:
    local = _metadata()
    remote = _metadata()
    value = remote[field]
    remote[field] = f"different-{field}" if isinstance(value, str) else value + 1  # type: ignore[operator]
    result = decide_reuse(local, remote)
    assert result.status is ReuseStatus.BLOCKED
    assert field in result.mismatched_fields
    assert f"BINDING_MISMATCH:{field}" in result.reasons


def test_conflicting_paused_and_in_progress_state_is_blocked() -> None:
    result = decide_reuse(
        _metadata(),
        _metadata(),
        local_state={"status": "SAFELY_PAUSED"},
        remote_state={"status": "IN_PROGRESS"},
    )
    assert result.status is ReuseStatus.BLOCKED
    assert "STATE_CONFLICT:PAUSED_AND_IN_PROGRESS" in result.reasons


def test_same_record_paused_and_in_progress_flags_are_blocked() -> None:
    result = decide_reuse(_metadata(), _metadata(), state={"paused": True, "in_progress": True})
    assert result.status is ReuseStatus.BLOCKED
    assert "STATE_CONFLICT:PAUSED_AND_IN_PROGRESS" in result.reasons


def test_in_progress_state_is_not_reusable() -> None:
    result = decide_reuse(_metadata(), _metadata(), state={"status": "RUNNING"})
    assert result.status is ReuseStatus.BLOCKED
    assert "IN_PROGRESS_STATE" in result.reasons


def test_changed_code_identity_is_blocked_even_with_caller_value() -> None:
    remote = _metadata() | {"code": "code-commit-other"}
    result = decide_reuse(_metadata(), remote, live_code_identity="code-commit-001")
    assert result.status is ReuseStatus.BLOCKED
    assert "code" in result.mismatched_fields
    assert LIVE_CODE_IDENTITY_UNCERTAIN not in result.reasons


def test_missing_code_identity_is_blocked_even_with_caller_value() -> None:
    local = _metadata()
    del local["code"]
    result = decide_reuse(local, _metadata(), live_code_identity="code-commit-001")
    assert result.status is ReuseStatus.BLOCKED
    assert LIVE_CODE_IDENTITY_UNCERTAIN in result.reasons
    assert "MISSING_BINDING_FIELD:local:code" in result.reasons


def test_aliases_are_explicit_and_normalize_to_reuse() -> None:
    metadata = {
        "system_id": "system-a",
        "run_id": "run-001",
        "seed": 20260521,
        "checkpoint_sha256": "checkpoint-sha-001",
        "config_sha256": "config-sha-001",
        "dataset_sha256": "dataset-sha-001",
        "model_revision": "model-revision-001",
        "tokenizer_revision": "tokenizer-revision-001",
        "code_commit": "code-commit-001",
        "source_run_id": "source-run-001",
        "approval_sha256": "approval-sha-001",
        "checkpoint_epoch_number": 1,
        "checkpoint_type": "best",
    }
    result = decide_reuse(metadata, dict(metadata))
    assert result.status is ReuseStatus.REUSE


@pytest.mark.parametrize(
    "malformed",
    ("digest:model", "a" * 63, "a" * 65, "g" * 64),
)
def test_malformed_equal_sha256_fields_are_blocked(malformed: str) -> None:
    metadata = _metadata() | {"model_hash": malformed}
    result = decide_reuse(metadata, dict(metadata))
    assert result.status is ReuseStatus.BLOCKED
    assert "INVALID_SHA256:local:model_hash" in result.reasons
    assert "INVALID_SHA256:remote:model_hash" in result.reasons


def test_exact_sha256_fields_are_verified_and_different_valid_fields_are_invalid() -> None:
    valid = decide_reuse(_metadata(), _metadata())
    assert valid.status is ReuseStatus.VERIFIED
    invalid = decide_reuse(_metadata(), _metadata() | {"model_hash": "7" * 64})
    assert invalid.status is ReuseStatus.INVALID
    assert "model_hash" in invalid.mismatched_fields


def test_extra_provenance_field_is_blocked() -> None:
    local = _metadata() | {"unapproved_provenance": "must-not-be-ignored"}
    result = decide_reuse(local, _metadata())
    assert result.status is ReuseStatus.BLOCKED
    assert "EXTRA_FIELD:local:unapproved_provenance" in result.reasons


def test_bare_verify_cannot_promote_blocked_or_invalid_state() -> None:
    blocked = ReuseState(status=ReuseStatus.BLOCKED)
    invalid = ReuseState(status=ReuseStatus.INVALID)
    assert transition(blocked, ReuseEvent.VERIFY).status is ReuseStatus.BLOCKED
    assert transition(invalid, ReuseEvent.VERIFY).status is ReuseStatus.BLOCKED


def test_verify_requires_reuse_evidence_bearing_decision() -> None:
    blocked = ReuseState(status=ReuseStatus.BLOCKED)
    bare_verified = transition(blocked, ReuseEvent.VERIFY, evidence=ReuseDecision(ReuseStatus.VERIFIED))
    assert bare_verified.status is ReuseStatus.BLOCKED

    evidence = decide_reuse(_metadata(), _metadata())
    promoted = transition(blocked, ReuseEvent.VERIFY, evidence=evidence)
    assert promoted.status is ReuseStatus.REUSE
    assert promoted.evidence == evidence


def test_resume_transition_requires_resume_evidence() -> None:
    state = ReuseState()
    evidence = decide_reuse(_metadata(), _metadata(), state={"status": "PAUSED"})
    resumed = transition(state, ReuseEvent.RESUME, evidence=evidence)
    assert resumed.status is ReuseStatus.RESUME
    assert resumed.evidence == evidence


def test_bare_resume_cannot_promote_without_exact_binding_evidence() -> None:
    state = ReuseState()
    bare_resume = transition(state, ReuseEvent.RESUME, evidence=ReuseDecision(ReuseStatus.RESUME))
    assert bare_resume.status is ReuseStatus.BLOCKED


def test_status_values_are_normalized_to_three_outcomes() -> None:
    assert {status.value for status in ReuseStatus} == {"REUSE", "RESUME", "BLOCKED"}
    assert ReuseStatus("verified") is ReuseStatus.REUSE
    assert ReuseStatus("invalid") is ReuseStatus.BLOCKED


@dataclass
class ForbiddenMutationAdapter:
    writes: int = 0

    def backup(self) -> None:
        self.writes += 1
        raise AssertionError("backup must not be called")

    def restore(self) -> None:
        self.writes += 1
        raise AssertionError("restore must not be called")


def test_backup_restore_are_forbidden_plans_and_adapter_is_untouched() -> None:
    adapter = ForbiddenMutationAdapter()
    assert plan_backup().allowed is False
    assert plan_restore().allowed is False
    assert adapter.writes == 0
