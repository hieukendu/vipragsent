from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vipragsent.azure.client import (
    AzureCache,
    AzureResponsesClient,
    AzureRetryableError,
    AzureSafetyBudgetError,
    AzureSafetyCeilings,
    AzureSafetyLedger,
    AzureSettings,
)
from vipragsent.azure.prompts import build_demo_manifest, validate_demo_manifest
from vipragsent.azure.schemas import (
    strict_label_schema,
    strict_rationale_schema,
    validate_structured_output,
)
from vipragsent.data.loaders import load_vipragsent
from vipragsent.orchestration.contracts import RunContext, RunEntry
from vipragsent.orchestration.stage_registry import _azure_execute


def test_azure_settings_reject_direct_openai_endpoint() -> None:
    with pytest.raises(ValueError):
        AzureSettings.from_env({"AZURE_OPENAI_ENDPOINT": "https://api.openai.com", "AZURE_OPENAI_BASE_URL": "https://api.openai.com/openai/v1/", "AZURE_OPENAI_DEPLOYMENT": "x", "AZURE_OPENAI_API_KEY": "secret"})


def test_structured_schema_has_exact_canonical_keys() -> None:
    schema = strict_label_schema()
    assert set(schema["required"]) == {"implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking", "polarity", "emotion"}
    value = {key: 0 for key in ("implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking")}
    value.update({"polarity": "neutral", "emotion": "other"})
    assert validate_structured_output(value) == value


def test_mocked_azure_retry_and_response_metadata() -> None:
    settings = AzureSettings("https://r.openai.azure.com", "https://r.openai.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    calls = {"n": 0}

    def transport(**_: object) -> dict[str, object]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise AzureRetryableError("429")
        return {"id": "resp_1", "model": "gpt-4.1-mini", "output": {"parsed": {"implicit_sentiment": 0, "sarcasm": 0, "irony": 0, "idiom_figurative": 0, "code_switching": 0, "mocking": 0, "polarity": "neutral", "emotion": "other"}}}

    client = AzureResponsesClient(settings, transport=transport, safety=AzureSafetyCeilings(allow_unknown_spend=True))
    result = client.create_structured(prompt="x", task="all", schema={"strict": True, "schema": strict_label_schema()}, max_output_tokens=32, sleep=lambda _: None)
    assert result["request_id"] == "resp_1"
    assert calls["n"] == 2


def test_azure_v1_transport_does_not_require_legacy_api_version(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    captured: dict[str, object] = {}

    class FakeResponses:
        def create(self, **kwargs: object) -> dict[str, object]:
            captured["request"] = kwargs
            return {"id": "resp_v1", "model": "gpt-4.1-mini", "output": {"parsed": {}}}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    settings = AzureSettings("https://r.openai.azure.com", "https://r.openai.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    result = AzureResponsesClient(settings)._default_transport(input="x")

    assert result["id"] == "resp_v1"
    assert captured["client"] == {"api_key": "secret", "base_url": settings.base_url, "max_retries": 0, "timeout": 300.0}
    assert captured["request"]["model"] == "dep"


def _structured_labels() -> dict[str, object]:
    return {
        "implicit_sentiment": 0,
        "sarcasm": 0,
        "irony": 0,
        "idiom_figurative": 0,
        "code_switching": 0,
        "mocking": 0,
        "polarity": "neutral",
        "emotion": "other",
    }


def _nested_response(labels: dict[str, object], *, response_id: str = "resp_nested") -> dict[str, object]:
    return {
        "id": response_id,
        "model": "gpt-4.1-mini-2025-04-14",
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(labels)}]}],
        "usage": {"input_tokens": 7, "output_tokens": 5},
    }


def test_public_client_parses_nested_responses_payload_and_caches(tmp_path: Path) -> None:
    calls = 0

    def transport(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _nested_response(_structured_labels())

    settings = AzureSettings("https://fixture.azure.com", "https://fixture.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    client = AzureResponsesClient(settings, transport=transport, cache=AzureCache(tmp_path / "cache"))
    kwargs = {"prompt": "nested", "task": "all", "schema": {"strict": True, "schema": strict_label_schema()}, "max_output_tokens": 32}
    first = client.create_structured(**kwargs)
    second = client.create_structured(**kwargs)

    assert first["valid"] is True
    assert first["labels"] == _structured_labels()
    assert first["observed_model_version"] == "2025-04-14"
    assert first["usage"]["input_tokens"] == 7
    assert second["cache_hit"] is True
    assert calls == 1


def test_public_client_ignores_top_level_request_text_when_parsing_rationale() -> None:
    payload = _nested_response({"rationale": "A valid rationale."})
    payload["text"] = {"format": {"type": "json_schema", "name": "vipragsent_rationale"}}
    settings = AzureSettings("https://r.openai.azure.com", "https://r.openai.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    client = AzureResponsesClient(settings, transport=lambda **_kwargs: payload)

    result = client.create_structured(prompt="reason", task="rationale", schema={"strict": True, "schema": strict_rationale_schema()}, max_output_tokens=256)

    assert result["labels"] == {"rationale": "A valid rationale."}


def test_public_client_retries_payload_status_and_uses_retry_after() -> None:
    calls = 0
    delays: list[float] = []

    def transport(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status_code": 429, "headers": {"retry-after": "0"}, "error": {"message": "busy"}}
        return _nested_response(_structured_labels(), response_id="resp_after_retry")

    settings = AzureSettings("https://fixture.azure.com", "https://fixture.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    client = AzureResponsesClient(settings, transport=transport)
    result = client.create_structured(prompt="retry", task="all", schema={"strict": True, "schema": strict_label_schema()}, max_output_tokens=32, sleep=delays.append)

    assert result["valid"] is True
    assert result["retry_count"] == 1
    assert calls == 2
    assert delays == [0.0]


def test_public_client_caches_terminal_invalid_response_without_retry(tmp_path: Path) -> None:
    calls = 0

    def transport(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return _nested_response({"not_a_label": 1}, response_id="resp_invalid")

    settings = AzureSettings("https://fixture.azure.com", "https://fixture.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    client = AzureResponsesClient(settings, transport=transport, cache=AzureCache(tmp_path / "cache"))
    kwargs = {"prompt": "invalid", "task": "all", "schema": {"strict": True, "schema": strict_label_schema()}, "max_output_tokens": 32, "return_invalid": True}
    first = client.create_structured(**kwargs)
    second = client.create_structured(**kwargs)

    assert first["valid"] is False
    assert first["invalid_stage"] == "structured_response"
    assert first["retry_count"] == 0
    assert second["valid"] is False
    assert second["cache_hit"] is True
    assert calls == 1


def test_retryable_outage_is_not_reused_across_client_runs(tmp_path: Path) -> None:
    settings = AzureSettings("https://fixture.azure.com", "https://fixture.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    cache = AzureCache(tmp_path / "cache")
    kwargs = {"prompt": "recover", "task": "all", "schema": {"strict": True, "schema": strict_label_schema()}, "max_output_tokens": 32, "return_invalid": True, "retries": 1, "sleep": lambda _: None}

    first_calls = 0

    def outage(**_: object) -> dict[str, object]:
        nonlocal first_calls
        first_calls += 1
        return {"status_code": 503, "error": "temporarily unavailable"}

    first = AzureResponsesClient(settings, transport=outage, cache=cache)
    failed = first.create_structured(**kwargs)
    assert failed["valid"] is False
    assert first_calls == 2

    recovery_calls = 0

    def recovery(**_: object) -> dict[str, object]:
        nonlocal recovery_calls
        recovery_calls += 1
        return _nested_response(_structured_labels(), response_id="recovered")

    second = AzureResponsesClient(settings, transport=recovery, cache=cache)
    recovered = second.create_structured(**kwargs)
    assert recovered["valid"] is True
    assert recovered["cache_hit"] is False
    assert recovery_calls == 1


def test_client_rejects_actual_usage_over_safety_ceiling() -> None:
    settings = AzureSettings("https://fixture.azure.com", "https://fixture.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    safety = AzureSafetyCeilings(max_output_tokens=1, max_total_tokens=2, max_verified_spend_usd=1.0)
    client = AzureResponsesClient(settings, transport=lambda **_: _nested_response(_structured_labels()), safety=safety, safety_ledger=AzureSafetyLedger(safety))
    with pytest.raises(AzureSafetyBudgetError, match="output-token ceiling"):
        client.create_structured(prompt="actual", task="all", schema={"strict": True, "schema": strict_label_schema()}, max_output_tokens=32)


def test_sync_actual_overrun_latches_shared_ledger_under_concurrency() -> None:
    settings = AzureSettings("https://fixture.azure.com", "https://fixture.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    safety = AzureSafetyCeilings(max_output_tokens=10, max_total_tokens=50, max_concurrency=2, max_retry_per_request=0)
    ledger = AzureSafetyLedger(safety)
    barrier = threading.Barrier(2)
    calls = 0
    calls_lock = threading.Lock()

    def transport(**_: object) -> dict[str, object]:
        nonlocal calls
        with calls_lock:
            calls += 1
        barrier.wait()
        payload = _nested_response(_structured_labels())
        payload["usage"] = {"input_tokens": 1, "output_tokens": 6}
        return payload

    client = AzureResponsesClient(settings, transport=transport, safety=safety, safety_ledger=ledger)

    def invoke() -> tuple[str, object]:
        try:
            return "ok", client.create_structured(prompt="concurrent", task="all", schema={"strict": True, "schema": strict_label_schema()}, max_output_tokens=4, retries=99)
        except AzureSafetyBudgetError as exc:
            return "error", exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: invoke(), range(2)))

    assert calls == 2
    assert sum(kind == "error" for kind, _ in outcomes) == 1
    assert ledger.tripped is True
    assert ledger.final_stop_reason == "actual_output_token_ceiling_exceeded"
    assert ledger.telemetry()["safety_overrun_count"] == 1
    with pytest.raises(AzureSafetyBudgetError, match="safety ledger stopped"):
        client.create_structured(prompt="after-stop", task="all", schema={"strict": True, "schema": strict_label_schema()}, max_output_tokens=4)
    assert calls == 2


def test_sync_missing_usage_fails_closed_and_blocks_later_admission() -> None:
    settings = AzureSettings("https://fixture.azure.com", "https://fixture.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    safety = AzureSafetyCeilings(max_output_tokens=32, max_total_tokens=64)
    ledger = AzureSafetyLedger(safety)
    calls = 0

    def transport(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        payload = _nested_response(_structured_labels())
        payload.pop("usage")
        return payload

    client = AzureResponsesClient(settings, transport=transport, safety=safety, safety_ledger=ledger)
    with pytest.raises(AzureSafetyBudgetError) as first:
        client.create_structured(prompt="missing-usage", task="all", schema={"strict": True, "schema": strict_label_schema()}, max_output_tokens=8)
    assert first.value.stop_reason == "unknown_usage"
    assert first.value.telemetry["unknown_spend_count"] == 1
    with pytest.raises(AzureSafetyBudgetError, match="safety ledger stopped"):
        client.create_structured(prompt="blocked", task="all", schema={"strict": True, "schema": strict_label_schema()}, max_output_tokens=8)
    assert calls == 1


def test_sync_retry_caller_is_clamped_by_safety_ceiling_alias() -> None:
    settings = AzureSettings("https://fixture.azure.com", "https://fixture.azure.com/openai/v1/", "dep", None, "api_key", "secret")
    safety = AzureSafetyCeilings.from_mapping({"max_retry_per_request": 1, "max_transport_attempts": 10})
    calls = 0

    def transport(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status_code": 503, "error": "temporary"}

    client = AzureResponsesClient(settings, transport=transport, safety=safety)
    result = client.create_structured(prompt="retry-cap", task="all", schema={"strict": True, "schema": strict_label_schema()}, max_output_tokens=8, retries=99, return_invalid=True, sleep=lambda _: None)
    assert calls == 2
    assert result["retry_count"] == 1
    assert AzureSafetyCeilings.from_mapping({"max_retries_per_request": 1}).retry_ceiling == 1


def test_stage_registry_preflights_logical_request_ceiling(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fixture.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://fixture.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    input_path = tmp_path / "data/processed/rationales/azure_rationale_input_train.jsonl"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("\n".join(json.dumps({"sample_id": f"sample-{i}", "comment": "bình luận"}) for i in range(2)) + "\n", encoding="utf-8")
    entry = RunEntry.from_mapping({"job_id": "azure_stage", "variant": "rationale_generation", "execution_kind": "azure", "backbone": "azure", "task": "rationale"}, run_id="azure_stage")
    calls = 0

    def transport(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {}

    context = RunContext(tmp_path, entry, run_root=tmp_path / "run", metadata={"azure_safety_ceilings": {"max_logical_requests": 1}, "azure_transport": transport})
    outcome = _azure_execute(context, entry)
    assert outcome.status == "BLOCKED"
    assert "logical-request ceiling" in str(outcome.error)
    assert calls == 0


def test_stage_registry_stops_after_actual_usage_overrun(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fixture.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", "https://fixture.azure.com/openai/v1/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "secret")
    calls = 0

    def transport(**_: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"model": "gpt-4.1-mini", "output": {"parsed": {"rationale": "valid rationale"}}, "usage": {"input_tokens": 1, "output_tokens": 257}}

    input_path = tmp_path / "data/processed/rationales/azure_rationale_input_train.jsonl"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps({"sample_id": "sample-0", "comment": "bình luận"}) + "\n", encoding="utf-8")
    entry = RunEntry.from_mapping({"job_id": "azure_stage", "variant": "rationale_generation", "execution_kind": "azure", "backbone": "azure", "task": "rationale"}, run_id="azure_stage")
    context = RunContext(tmp_path, entry, run_root=tmp_path / "run", metadata={"azure_safety_ceilings": {"max_output_tokens": 256, "max_total_tokens": 10_000}, "azure_transport": transport})
    outcome = _azure_execute(context, entry)
    assert outcome.status == "FAIL"
    assert "rejected" in str(outcome.error)
    response = json.loads((context.run_root / "azure/response_manifest.json").read_text(encoding="utf-8"))
    assert response["requested"] == response["invalid"] == 1


def test_cache_ignores_non_object_json_records(tmp_path: Path) -> None:
    cache = AzureCache(tmp_path / "cache")
    cache.path_for("corrupt").write_text("[]", encoding="utf-8")

    assert cache.get("corrupt", expected_model_family="GPT-4.1-mini", expected_model_version="2025-04-14") is None


def test_pragmatic_demo_manifest_has_eight_unique_train_examples() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = load_vipragsent(root / "data/processed/vipragsent")
    manifest = build_demo_manifest(bundle.train)
    validate_demo_manifest(manifest)
    assert len(manifest["sample_ids"]) == 8
