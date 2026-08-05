from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, exclusive_lock
from ..hashing import sha256_json
from .schemas import validate_rationale_output, validate_structured_output


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
    def from_env(cls, env: Mapping[str, str] | None = None) -> AzureSettings:
        env = env or os.environ
        endpoint = env.get("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/") + "/"
        base_url = env.get("AZURE_OPENAI_BASE_URL", "").strip()
        deployment = env.get("AZURE_OPENAI_DEPLOYMENT", "").strip()
        auth_mode = env.get("AZURE_OPENAI_AUTH_MODE", "api_key").strip().lower()
        if not endpoint.strip("/") or not base_url or not deployment:
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
        return cls(endpoint, base_url.rstrip("/") + "/", deployment, env.get("AZURE_OPENAI_BATCH_DEPLOYMENT") or None, auth_mode, api_key)

    def redacted(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint, "base_url": self.base_url, "deployment": self.deployment, "batch_deployment": self.batch_deployment, "auth_mode": self.auth_mode, "model_family": self.model_family, "expected_model_version": self.expected_model_version}


class AzureRetryableError(RuntimeError):
    pass


def _status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return max(0.0, float(value)) if value is not None else None
        except (TypeError, ValueError):
            return None
    value = getattr(exc, "retry_after", None)
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, AzureRetryableError)):
        return True
    name = type(exc).__name__
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}:
        return True
    return name == "APIStatusError" and (_status_code(exc) == 429 or (_status_code(exc) or 0) >= 500)


class AzureCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, identity: Mapping[str, Any]) -> str:
        return sha256_json(identity)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str, *, expected_model_family: str, expected_model_version: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        observed_model = str(record.get("observed_model", ""))
        observed_version = str(record.get("observed_model_version", ""))
        if not observed_model or not observed_version:
            return None
        if expected_model_family.casefold() not in observed_model.casefold() or observed_version != expected_model_version:
            return None
        return record

    def put(self, key: str, record: Mapping[str, Any]) -> Path:
        path = self.path_for(key)
        with exclusive_lock(path.with_suffix(".lock")):
            atomic_write_json(path, dict(record))
        return path


class AzureResponsesClient:
    """Azure Responses API v1 client with injectable transport and idempotent cache."""

    def __init__(self, settings: AzureSettings, transport: Callable[..., Mapping[str, Any]] | None = None, *, cache: AzureCache | None = None) -> None:
        self.settings = settings
        self.transport = transport
        self.cache = cache

    def _default_transport(self, **kwargs: Any) -> Mapping[str, Any]:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:
            raise RuntimeError("Install the azure optional dependencies to call Azure") from exc
        if self.settings.auth_mode == "api_key":
            client = AzureOpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url)
        else:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except ImportError as exc:
                raise RuntimeError("Install azure-identity for Entra ID authentication") from exc
            token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            client = AzureOpenAI(azure_ad_token_provider=token_provider, base_url=self.settings.base_url)
        response = client.responses.create(model=self.settings.deployment, **kwargs)
        return response.model_dump() if hasattr(response, "model_dump") else dict(response)

    def create_structured(
        self,
        *,
        prompt: str,
        task: str,
        schema: dict[str, Any],
        max_output_tokens: int,
        sample_id: str | None = None,
        input_payload: Any | None = None,
        demonstration_manifest_hash: str | None = None,
        expected_model_version: str | None = None,
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        if not schema.get("strict", True):
            raise ValueError("Structured Outputs must be strict")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        expected_version = expected_model_version or self.settings.expected_model_version
        schema_hash = sha256_json(schema)
        prompt_hash = sha256_json({"prompt": prompt})
        identity = {
            "task": task,
            "sample_id": sample_id,
            "input_payload_hash": sha256_json(input_payload if input_payload is not None else prompt),
            "prompt_hash": prompt_hash,
            "demonstration_manifest_hash": demonstration_manifest_hash,
            "schema_hash": schema_hash,
            "deployment": self.settings.deployment,
            "model_family": self.settings.model_family,
            "expected_model_version": expected_version,
            "temperature": 0,
            "max_output_tokens": max_output_tokens,
        }
        cache_key = self.cache.key(identity) if self.cache else None
        if cache_key and self.cache:
            cached = self.cache.get(cache_key, expected_model_family=self.settings.model_family, expected_model_version=expected_version)
            if cached is not None:
                return dict(cached) | {"cache_hit": True, "retry_count": cached.get("retry_count", 0)}
        transport = self.transport or self._default_transport
        request_id = str(uuid.uuid4())
        last_error: Exception | None = None
        retry_count = 0
        for attempt in range(retries + 1):
            try:
                payload = transport(input=prompt, text={"format": {"type": "json_schema", "name": f"vipragsent_{task}", "strict": True, "schema": schema["schema"]}}, max_output_tokens=max_output_tokens, temperature=0, metadata={"request_id": request_id, "prompt_hash": prompt_hash, "schema_hash": schema_hash})
                output = payload.get("output", payload)
                if isinstance(output, Mapping) and "parsed" in output:
                    parsed = output["parsed"]
                elif isinstance(output, Mapping) and "text" in output and isinstance(output["text"], Mapping):
                    parsed = output["text"]
                else:
                    parsed = output
                if task == "rationale":
                    labels = {"rationale": validate_rationale_output(dict(parsed))}
                else:
                    labels = validate_structured_output(dict(parsed), task)
                observed_model = str(payload.get("model", ""))
                observed_version = str(payload.get("version", payload.get("metadata", {}).get("version", "")))
                if observed_model and self.settings.model_family.casefold() not in observed_model.casefold():
                    raise ValueError(f"Azure response model mismatch: {observed_model}")
                record = {
                    "labels": labels,
                    "request_id": payload.get("id", request_id),
                    "response_id": payload.get("id", request_id),
                    "deployment": self.settings.deployment,
                    "expected_model_family": self.settings.model_family,
                    "expected_model_version": expected_version,
                    "observed_model": observed_model or self.settings.model_family,
                    "observed_model_version": observed_version or expected_version,
                    "prompt_hash": prompt_hash,
                    "schema_hash": schema_hash,
                    "demonstration_manifest_hash": demonstration_manifest_hash,
                    "request_timestamp": datetime.now(UTC).isoformat(),
                    "retry_count": retry_count,
                    "usage": payload.get("usage", {}),
                    "content_filter": payload.get("content_filter"),
                    "cache_hit": False,
                }
                if cache_key and self.cache:
                    self.cache.put(cache_key, record)
                return record
            except Exception as exc:
                if not _is_retryable(exc):
                    raise
                last_error = exc
                if attempt >= retries:
                    break
                retry_count += 1
                delay = _retry_after(exc)
                sleep(min(60.0, delay if delay is not None else 2**attempt))
        message = str(last_error or "unknown Azure error")
        if self.settings.api_key:
            message = message.replace(self.settings.api_key, "<redacted>")
        raise RuntimeError(f"Azure request failed after retries: {message}") from last_error

    def verify_deployment(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        model = str(metadata.get("model", ""))
        version = str(metadata.get("version", ""))
        if self.settings.model_family.casefold() not in model.casefold() and model:
            raise ValueError(f"Azure deployment model mismatch: {model}")
        if version and version != self.settings.expected_model_version:
            raise ValueError(f"Azure deployment version mismatch: {version}")
        return {"verified": True, **self.settings.redacted(), "metadata": dict(metadata)}
