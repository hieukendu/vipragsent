from __future__ import annotations

from dataclasses import dataclass

from vipragsent.runtime.artifact_reuse import (
    LIVE_CODE_IDENTITY_UNCERTAIN,
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
        "tokenizer": "tokenizer@sha256:tokenizer",
        "config": "config@sha256:config",
        "data": "data@sha256:data",
        "source": "source-run-001",
        "live_code_identity": "git:abc123",
    }


def test_exact_match_is_verified_and_hash_bound() -> None:
    metadata = _metadata()
    result = decide_reuse(metadata, dict(metadata), live_code_identity="git:abc123")
    assert result.status is ReuseStatus.VERIFIED
    assert result.reusable
    assert result.local_metadata_hash == result.remote_metadata_hash
    assert set(result.matched_fields) == {"model", "tokenizer", "config", "data", "source"}


def test_mismatch_is_invalid() -> None:
    local = _metadata()
    remote = _metadata() | {"config": "config@sha256:other"}
    result = decide_reuse(local, remote, live_code_identity="git:abc123")
    assert result.status is ReuseStatus.INVALID
    assert "config" in result.mismatched_fields


def test_absent_source_identity_is_blocked() -> None:
    local = _metadata() | {"source": None}
    result = decide_reuse(local, _metadata(), live_code_identity="git:abc123")
    assert result.status is ReuseStatus.BLOCKED
    assert "SOURCE_IDENTITY_ABSENT" in result.reasons


def test_uncertain_live_code_identity_is_explicitly_blocked() -> None:
    result = decide_reuse(_metadata(), _metadata())
    assert result.status is ReuseStatus.BLOCKED
    assert LIVE_CODE_IDENTITY_UNCERTAIN in result.reasons


def test_state_transitions_are_idempotent() -> None:
    state = ReuseState()
    verified = transition(transition(state, ReuseEvent.VERIFY), ReuseEvent.VERIFY)
    assert verified == transition(state, ReuseEvent.VERIFY)
    assert verified.history == ("verify",)


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
