from __future__ import annotations

import json
from pathlib import Path

import pytest

from vipragsent.azure.client import (
    AzureCache,
    AzureResponsesClient,
    AzureRetryableError,
    AzureSettings,
)
from vipragsent.azure.prompts import build_demo_manifest, validate_demo_manifest
from vipragsent.azure.schemas import strict_label_schema, validate_structured_output
from vipragsent.data.loaders import load_vipragsent


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

    client = AzureResponsesClient(settings, transport=transport)
    result = client.create_structured(prompt="x", task="all", schema={"strict": True, "schema": strict_label_schema()}, max_output_tokens=32, sleep=lambda _: None)
    assert result["request_id"] == "resp_1"
    assert calls["n"] == 2


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
