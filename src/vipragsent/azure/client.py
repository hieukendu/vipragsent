from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..hashing import sha256_json
from .schemas import validate_structured_output


@dataclass(frozen=True)
class AzureSettings:
    endpoint: str
    base_url: str
    deployment: str
    batch_deployment: str | None
    auth_mode: str
    api_key: str | None = None
    model_family: str = "GPT-4.1-mini"
    expected_model_version: str = "2025-04-14"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AzureSettings":
        env = env or os.environ
        endpoint = env.get("AZURE_OPENAI_ENDPOINT", "").strip()
        base_url = env.get("AZURE_OPENAI_BASE_URL", "").strip()
        deployment = env.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
        auth_mode = env.get("AZURE_OPENAI_AUTH_MODE", "api_key").strip().lower()
        if not endpoint or not base_url or not deployment:
            raise ValueError("AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_BASE_URL, and AZURE_OPENAI_DEPLOYMENT are required")
        if "api.openai.com" in endpoint or "api.openai.com" in base_url:
            raise ValueError("Direct OpenAI endpoints are prohibited")
        if not base_url.startswith("https://") or "/openai/v1/" not in base_url:
            raise ValueError("AZURE_OPENAI_BASE_URL must be an Azure OpenAI /openai/v1/ URL")
        if auth_mode not in {"api_key", "entra_id"}:
            raise ValueError("AZURE_OPENAI_AUTH_MODE must be api_key or entra_id")
        api_key = env.get("AZURE_OPENAI_API_KEY") if auth_mode == "api_key" else None
        if auth_mode == "api_key" and not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY is required in api_key mode")
        return cls(endpoint, base_url, deployment, env.get("AZURE_OPENAI_BATCH_DEPLOYMENT") or None, auth_mode, api_key)

    def redacted(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            "deployment": self.deployment,
            "batch_deployment": self.batch_deployment,
            "auth_mode": self.auth_mode,
            "model_family": self.model_family,
            "expected_model_version": self.expected_model_version,
        }


class AzureRetryableError(RuntimeError):
    pass


class AzureResponsesClient:
    """Responses API v1 client with injectable transport for tests and fixture runs."""

    def __init__(self, settings: AzureSettings, transport: Callable[..., Mapping[str, Any]] | None = None) -> None:
        self.settings = settings
        self.transport = transport

    def _default_transport(self, **kwargs: Any) -> Mapping[str, Any]:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the azure optional dependencies to call Azure") from exc
        if self.settings.auth_mode == "api_key":
            client = AzureOpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url, api_version="preview")
        else:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except ImportError as exc:
                raise RuntimeError("Install azure-identity for Entra ID authentication") from exc
            token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            client = AzureOpenAI(azure_ad_token_provider=token_provider, base_url=self.settings.base_url, api_version="preview")
        response = client.responses.create(model=self.settings.deployment, **kwargs)
        return response.model_dump() if hasattr(response, "model_dump") else dict(response)

    def create_structured(
        self,
        *,
        prompt: str,
        task: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        if not schema.get("strict", True):
            raise ValueError("Structured Outputs must be strict")
        transport = self.transport or self._default_transport
        request_id = str(uuid.uuid4())
        prompt_hash = sha256_json({"prompt": prompt, "schema": schema})
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                payload = transport(
                    input=prompt,
                    text={"format": {"type": "json_schema", "name": f"vipragsent_{task}", "strict": True, "schema": schema["schema"]}},
                    max_output_tokens=max_output_tokens,
                    temperature=0,
                    metadata={"request_id": request_id, "prompt_hash": prompt_hash},
                )
                output = payload.get("output", payload)
                if "parsed" in output:
                    parsed = output["parsed"]
                elif isinstance(output, Mapping) and "text" in output and isinstance(output["text"], Mapping):
                    parsed = output["text"]
                else:
                    parsed = output
                labels = validate_structured_output(dict(parsed), task)
                return {
                    "labels": labels,
                    "request_id": payload.get("id", request_id),
                    "prompt_hash": prompt_hash,
                    "attempt": attempt,
                    "model": payload.get("model", self.settings.model_family),
                    "deployment": self.settings.deployment,
                    "usage": payload.get("usage", {}),
                    "content_filter": payload.get("content_filter", None),
                }
            except (TimeoutError, ConnectionError, AzureRetryableError) as exc:
                last_error = exc
                if attempt >= retries:
                    break
                retry_after = getattr(exc, "retry_after", None)
                sleep(float(retry_after) if retry_after is not None else min(60.0, 2**attempt))
            except Exception:
                raise
        raise RuntimeError(f"Azure request failed after retries: {last_error}") from last_error

    def verify_deployment(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        model = str(metadata.get("model", ""))
        version = str(metadata.get("version", ""))
        if self.settings.model_family.lower() not in model.lower() and model:
            raise ValueError(f"Azure deployment model mismatch: {model}")
        if version and version != self.settings.expected_model_version:
            raise ValueError(f"Azure deployment version mismatch: {version}")
        return {"verified": True, **self.settings.redacted(), "metadata": dict(metadata)}
