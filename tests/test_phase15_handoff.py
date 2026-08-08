from __future__ import annotations

import json
from pathlib import Path

import pytest

from vipragsent.phase import PHASE15_SMOKE_TESTS, write_phase_handoff

FAMILY = "phobert_base"
REVISION = "locked-model-revision"
TOKENIZER_REVISION = "locked-tokenizer-revision"


def _write_status(root: Path, category: str, payload: dict[str, object], family: str = FAMILY) -> None:
    path = root / "data" / {"cache": "model_cache_status", "smoke": "model_smoke_status", "batch": "batch_probe_status"}[category] / f"{family}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_family": family, "category": category, **payload}, indent=2) + "\n", encoding="utf-8")


def _cache_payload() -> dict[str, object]:
    return {
        "status": "PASS",
        "repo_id": "fixture/phobert-base",
        "revision": REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "local_path": "data/model_cache/phobert_base",
        "manifest_hash": "cache-hash",
    }


def _smoke_payload(status: str = "PASS", blockers: list[str] | None = None) -> dict[str, object]:
    return {
        "status": status,
        "actual_local_loads": True,
        "revision": REVISION,
        "tokenizer_revision": TOKENIZER_REVISION,
        "checks": {name.replace(" ", "_"): True for name in PHASE15_SMOKE_TESTS},
        "blockers": blockers or [],
        "verification_hash": "smoke-hash" if status == "PASS" else None,
    }


def _batch_payload(status: str = "PASS", blockers: list[str] | None = None) -> dict[str, object]:
    return {
        "status": status,
        "frozen": status == "PASS",
        "fixture_probe": False,
        "successful_batch": 32 if status == "PASS" else None,
        "effective_batch_size": 32,
        "gradient_accumulation_steps": 1 if status == "PASS" else None,
        "hardware_identity": "test-gpu",
        "probe_hash": "batch-hash" if status == "PASS" else None,
        "blockers": blockers or [],
    }


def _payload(root: Path) -> dict[str, object]:
    return json.loads((root / "reports/phases/phase_15_handoff.json").read_text(encoding="utf-8"))


def _write_handoff(root: Path, *, status: str = "PASS", model_family: str | None = FAMILY) -> dict[str, object]:
    write_phase_handoff(
        "15",
        status,
        inputs_read=["configs/models/download_manifest.yaml"],
        files_created=["data/model_cache_manifest.json"],
        tests_run=["untrusted caller metadata"],
        tests_passed=True,
        blockers=[],
        next_phase_ready=True,
        report_root=root / "reports/phases",
        model_family=model_family,
    )
    return _payload(root)


def _write_all_pass(root: Path, family: str = FAMILY) -> None:
    _write_status(root, "cache", _cache_payload(), family)
    _write_status(root, "smoke", _smoke_payload(), family)
    _write_status(root, "batch", _batch_payload(), family)


def test_download_pass_alone_does_not_claim_all_phase15_tests_passed(tmp_path: Path) -> None:
    _write_status(tmp_path, "cache", _cache_payload())

    payload = _write_handoff(tmp_path)

    assert payload["status"] == "BLOCKED"
    assert payload["tests_passed"] is False
    assert payload["tests_run"] == ["locked cache/revision validation"]
    assert any("smoke status artifact is missing" in item for item in payload["blockers"])
    assert any("batch status artifact is missing" in item for item in payload["blockers"])


def test_download_smoke_batch_sequence_produces_audited_pass(tmp_path: Path) -> None:
    _write_status(tmp_path, "cache", _cache_payload())
    _write_handoff(tmp_path)
    _write_status(tmp_path, "smoke", _smoke_payload())
    _write_handoff(tmp_path)
    _write_status(tmp_path, "batch", _batch_payload())

    payload = _write_handoff(tmp_path)

    assert payload["status"] == "PASS"
    assert payload["tests_passed"] is True
    assert payload["tests_run"] == ["locked cache/revision validation", *PHASE15_SMOKE_TESTS, "physical batch probe"]
    assert payload["blockers"] == []
    evidence = payload["phase15_evidence"]
    assert evidence["model_family"] == FAMILY
    assert evidence["model_revision"] == REVISION
    assert evidence["tokenizer_revision"] == TOKENIZER_REVISION
    assert evidence["physical_batch"] == 32


def test_downloader_refresh_preserves_verified_smoke_and_batch_evidence(tmp_path: Path) -> None:
    _write_all_pass(tmp_path)
    first = _write_handoff(tmp_path)

    second = _write_handoff(tmp_path, status="PASS")
    first.pop("generated_at_utc", None)
    second.pop("generated_at_utc", None)

    assert second == first


@pytest.mark.parametrize("category", ["smoke", "batch"])
@pytest.mark.parametrize("failed_status", ["BLOCKED", "FAIL"])
def test_failed_smoke_or_batch_cannot_be_overwritten_by_download_pass(tmp_path: Path, category: str, failed_status: str) -> None:
    _write_all_pass(tmp_path)
    _write_status(
        tmp_path,
        category,
        (_smoke_payload if category == "smoke" else _batch_payload)(failed_status, [f"{category} evidence failed"]),
    )

    payload = _write_handoff(tmp_path, status="PASS")

    assert payload["status"] == failed_status
    assert payload["tests_passed"] is False
    assert f"{category} evidence failed" in payload["blockers"]


def test_repeated_phase15_persistence_is_idempotent(tmp_path: Path) -> None:
    _write_all_pass(tmp_path)
    first = _write_handoff(tmp_path)
    second = _write_handoff(tmp_path)
    third = _write_handoff(tmp_path, status="BLOCKED")
    for payload in (first, second, third):
        payload.pop("generated_at_utc", None)

    assert first == second == third


def test_unrelated_model_family_status_is_not_mixed_into_phobert_handoff(tmp_path: Path) -> None:
    _write_all_pass(tmp_path, FAMILY)
    _write_status(tmp_path, "cache", _cache_payload() | {"revision": "xlmr-revision"}, "xlmr_large")
    _write_status(tmp_path, "smoke", _smoke_payload("BLOCKED", ["xlmr smoke failed"]), "xlmr_large")
    _write_status(tmp_path, "batch", _batch_payload(), "xlmr_large")

    payload = _write_handoff(tmp_path, model_family=FAMILY)

    assert payload["status"] == "PASS"
    assert payload["tests_passed"] is True
    assert payload["phase15_evidence"]["model_family"] == FAMILY
    assert all("xlmr" not in item for item in payload["blockers"])
