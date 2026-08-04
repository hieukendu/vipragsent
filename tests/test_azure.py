from __future__ import annotations

from pathlib import Path

import pytest

from vipragsent.azure.client import AzureResponsesClient, AzureRetryableError, AzureSettings
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


def test_pragmatic_demo_manifest_has_eight_unique_train_examples() -> None:
    root = Path(__file__).resolve().parents[1]
    bundle = load_vipragsent(root / "data/processed/vipragsent")
    manifest = build_demo_manifest(bundle.train)
    validate_demo_manifest(manifest)
    assert len(manifest["sample_ids"]) == 8
