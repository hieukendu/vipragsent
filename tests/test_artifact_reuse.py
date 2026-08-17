from __future__ import annotations

from dataclasses import dataclass

import pytest

from vipragsent.runtime.artifact_reuse import (
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


def _metadata() -> dict[str, str]:
    return {
        "model": "model@sha256:model",
        "model_hash": "1" * 64,
        "tokenizer": "tokenizer@sha256:tokenizer",
        "tokenizer_hash": "2" * 64,
        "config": "config@sha256:config",
        "config_hash": "3" * 64,
        "data": "data@sha256:data",
        "data_hash": "4" * 64,
        "source": "source-run-001",
        "source_hash": "5" * 64,
        "live_code_identity": "git:abc123",
        "live_code_identity_hash": "6" * 64,
    }


def test_exact_match_is_verified_and_hash_bound() -> None:
    metadata = _metadata()
    result = decide_reuse(metadata, dict(metadata), live_code_identity="git:abc123")
    assert result.status is ReuseStatus.VERIFIED
    assert result.reusable
    assert result.local_metadata_hash == result.remote_metadata_hash
    assert result.evidence_bearing
    assert set(result.matched_fields) == set(metadata)


def test_mismatch_is_invalid() -> None:
    local = _metadata()
    remote = _metadata() | {"config": "config@sha256:other"}
    result = decide_reuse(local, remote, live_code_identity="git:abc123")
    assert result.status is ReuseStatus.INVALID
    assert "config" in result.mismatched_fields


def test_missing_live_code_field_is_blocked_even_with_caller_value() -> None:
    local = _metadata()
    remote = _metadata()
    del local["live_code_identity"]
    result = decide_reuse(local, remote, live_code_identity="git:abc123")
    assert result.status is ReuseStatus.BLOCKED
    assert LIVE_CODE_IDENTITY_UNCERTAIN in result.reasons
    assert "MISSING_FIELD:local:live_code_identity" in result.reasons


def test_missing_digest_is_blocked() -> None:
    remote = _metadata()
    del remote["config_hash"]
    result = decide_reuse(_metadata(), remote)
    assert result.status is ReuseStatus.BLOCKED
    assert "MISSING_FIELD:remote:config_hash" in result.reasons


def test_changed_live_code_identity_is_invalid() -> None:
    remote = _metadata() | {
        "live_code_identity": "git:def456",
        "live_code_identity_hash": "7" * 64,
    }
    result = decide_reuse(_metadata(), remote)
    assert result.status is ReuseStatus.INVALID
    assert "live_code_identity" in result.mismatched_fields
    assert "live_code_identity_hash" in result.mismatched_fields


def test_changed_digest_is_invalid() -> None:
    remote = _metadata() | {"data_hash": "7" * 64}
    result = decide_reuse(_metadata(), remote)
    assert result.status is ReuseStatus.INVALID
    assert "data_hash" in result.mismatched_fields
    assert "DIGEST_MISMATCH:data_hash" in result.reasons


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


def test_absent_source_identity_is_blocked() -> None:
    local = _metadata() | {"source": None}
    result = decide_reuse(local, _metadata(), live_code_identity="git:abc123")
    assert result.status is ReuseStatus.BLOCKED
    assert "SOURCE_IDENTITY_ABSENT" in result.reasons


def test_uncertain_live_code_identity_is_explicitly_blocked() -> None:
    result = decide_reuse(_metadata() | {"live_code_identity": None}, _metadata())
    assert result.status is ReuseStatus.BLOCKED
    assert LIVE_CODE_IDENTITY_UNCERTAIN in result.reasons


def test_state_transitions_are_idempotent() -> None:
    state = ReuseState()
    evidence = decide_reuse(_metadata(), _metadata())
    assert evidence.evidence_bearing
    verified = transition(transition(state, ReuseEvent.VERIFY, evidence=evidence), ReuseEvent.VERIFY, evidence=evidence)
    assert verified == transition(state, ReuseEvent.VERIFY, evidence=evidence)
    assert verified.history == ("verify",)


def test_bare_verify_cannot_promote_blocked_or_invalid_state() -> None:
    blocked = ReuseState(status=ReuseStatus.BLOCKED)
    invalid = ReuseState(status=ReuseStatus.INVALID)
    assert transition(blocked, ReuseEvent.VERIFY).status is ReuseStatus.BLOCKED
    assert transition(invalid, ReuseEvent.VERIFY).status is ReuseStatus.INVALID


def test_verify_requires_verified_evidence_bearing_decision() -> None:
    blocked = ReuseState(status=ReuseStatus.BLOCKED)
    bare_verified = transition(blocked, ReuseEvent.VERIFY, evidence=ReuseDecision(ReuseStatus.VERIFIED))
    assert bare_verified.status is ReuseStatus.BLOCKED

    evidence = decide_reuse(_metadata(), _metadata())
    promoted = transition(blocked, ReuseEvent.VERIFY, evidence=evidence)
    assert promoted.status is ReuseStatus.VERIFIED
    assert promoted.evidence == evidence


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
