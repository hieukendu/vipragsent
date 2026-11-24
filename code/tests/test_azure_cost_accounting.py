from __future__ import annotations

import json
from pathlib import Path

from vipragsent.orchestration.contracts import RunContext, RunEntry
from vipragsent.orchestration.run_store import RunStore
from vipragsent.orchestration.stage_registry import (
    _build_azure_cost_ledger,
    _load_azure_usage_records,
    _persist_azure_usage_record,
)
from vipragsent.profiling import azure_successful_usage_cost


def test_azure_cost_uses_non_cached_and_cached_input_separately() -> None:
    result = azure_successful_usage_cost({"input_tokens": 1_000_000, "input_tokens_details": {"cached_tokens": 250_000}, "output_tokens": 500_000})

    assert result["input_tokens"] == 1_000_000
    assert result["cached_input_tokens"] == 250_000
    assert result["non_cached_input_tokens"] == 750_000
    assert result["request_cost_usd"] == 1.125
    assert result["cost_status"] == "USAGE_AVAILABLE"


def test_azure_cost_does_not_fabricate_missing_usage() -> None:
    result = azure_successful_usage_cost({"input_tokens": 100})

    assert result["cost_status"] == "USAGE_UNAVAILABLE"
    assert result["request_cost_usd"] is None


def test_empty_azure_usage_ledger_is_not_a_pass() -> None:
    assert _build_azure_cost_ledger([], synthetic=False)["status"] == "NO_SUCCESSFUL_RESPONSES"


def _successful_result(*, cache_key: str, cache_hit: bool = False, retry_count: int = 0) -> dict[str, object]:
    return {
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "request_id": "request-1",
        "response_id": "response-1",
        "observed_model": "gpt-4.1-mini",
        "observed_model_version": "2025-04-14",
        "expected_model_family": "GPT-4.1-mini",
        "expected_model_version": "2025-04-14",
        "deployment": "gpt-4.1-mini",
        "retry_count": retry_count,
        "request_timestamp": "2026-08-07T00:00:00Z",
        "usage": {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 20}, "output_tokens": 10},
    }


def test_azure_cost_ledger_is_idempotent_and_does_not_charge_cache_reuse(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    result = _successful_result(cache_key="logical-1", retry_count=4)

    first = _persist_azure_usage_record(run_root, sample_id="sample-1", result=result)
    second = _persist_azure_usage_record(run_root, sample_id="sample-1", result=result | {"request_id": "request-2"})
    records = _load_azure_usage_records(run_root / "azure/usage_records.jsonl")
    ledger = json.loads((run_root / "azure/cost_ledger.json").read_text(encoding="utf-8"))

    assert first == second
    assert len(records) == 1
    assert records[0]["retry_count"] == 4
    assert ledger["successful_uncached_priced_requests"] == 1
    assert ledger["total_azure_cost_usd"] == first["request_cost_usd"]

    cached_root = tmp_path / "cached-run"
    _persist_azure_usage_record(cached_root, sample_id="sample-1", result=_successful_result(cache_key="logical-1", cache_hit=True))
    cached_ledger = json.loads((cached_root / "azure/cost_ledger.json").read_text(encoding="utf-8"))
    assert cached_ledger["cached_reuses"] == 1
    assert cached_ledger["successful_uncached_priced_requests"] == 0
    assert cached_ledger["total_azure_cost_usd"] == 0.0


def test_azure_resume_reopens_execute_after_response_validation_failure(tmp_path: Path) -> None:
    entry = RunEntry.from_mapping(
        {
            "job_id": "azure_retry",
            "research_question": "setup",
            "system_id": "azure_retry",
            "display_name": "azure retry",
            "variant": "rationale_generation",
            "backbone": "azure",
            "execution_kind": "azure",
            "stages": ["preflight", "execute_api_job", "validate_responses", "export_artifacts", "validate_artifacts", "generate_review_summary"],
        },
        run_id="azure_retry",
    )
    store = RunStore(RunContext(tmp_path, entry, fixture=True))
    state = store.initialize()
    state["stages"]["preflight"] = {"status": "PASS"}
    state["stages"]["execute_api_job"] = {"status": "PASS"}
    state["stages"]["validate_responses"] = {"status": "FAIL"}
    state["run_status"] = "FAIL"
    store.save(state)

    store.prepare_retry(state)
    resumed = store.load()

    assert resumed["stages"]["preflight"]["status"] == "PASS"
    assert resumed["stages"]["execute_api_job"]["status"] == "NOT_STARTED"
    assert resumed["stages"]["validate_responses"]["status"] == "NOT_STARTED"
    assert resumed["stages"]["export_artifacts"]["status"] == "NOT_STARTED"
