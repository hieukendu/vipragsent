from __future__ import annotations

import json
import os
import re
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

RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
AZURE_TRANSPORT_TIMEOUT_SECONDS = 300.0
_MISSING = object()


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
    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class AzureStructuredOutputError(ValueError):
    """Terminal structured-output failure with a machine-readable response record."""

    def __init__(self, message: str, *, payload: Mapping[str, Any] | None = None, content_filter: Any = None) -> None:
        super().__init__(message)
        self.payload = dict(payload or {})
        self.content_filter = content_filter


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


def _response_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method(mode="json") if method_name == "model_dump" else method()
            except TypeError:
                dumped = method()
            if isinstance(dumped, Mapping):
                return dict(dumped)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Responses API payload is not a mapping: {type(value).__name__}")


def _extract_structured_candidate(value: Any) -> Any:
    """Extract parsed or textual structured output from the Responses API shape."""
    if isinstance(value, Mapping):
        if "parsed" in value:
            return value["parsed"]
        if "labels" in value:
            return value["labels"]
        if "output_text" in value and isinstance(value["output_text"], (str, bytes, bytearray)) and value["output_text"].strip():
            return value["output_text"]
        if "text" in value and isinstance(value["text"], (str, bytes, bytearray)):
            return value["text"]
        for key in ("output", "response", "content"):
            if key in value:
                candidate = _extract_structured_candidate(value[key])
                if candidate is not _MISSING:
                    return candidate
        return _MISSING
    if isinstance(value, (list, tuple)):
        for item in value:
            candidate = _extract_structured_candidate(item)
            if candidate is not _MISSING:
                return candidate
        return _MISSING
    if isinstance(value, (str, bytes, bytearray)):
        return value
    return _MISSING


def extract_responses_structured_output(payload: Any) -> Any:
    """Return the strict structured object from a real or fake Responses payload."""
    candidate = _extract_structured_candidate(_response_mapping(payload))
    if candidate is _MISSING:
        raise AzureStructuredOutputError("Responses API payload has no structured output", payload=_response_mapping(payload))
    if isinstance(candidate, (bytes, bytearray)):
        candidate = bytes(candidate).decode("utf-8")
    for _ in range(3):
        if isinstance(candidate, Mapping):
            return dict(candidate)
        if not isinstance(candidate, str):
            raise AzureStructuredOutputError("structured output is not an object")
        text = candidate.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            if len(lines) < 3:
                raise AzureStructuredOutputError("structured output code fence is incomplete")
            text = "\n".join(lines[1:-1]).strip()
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AzureStructuredOutputError("structured output is not valid JSON") from exc
    if isinstance(candidate, Mapping):
        return dict(candidate)
    raise AzureStructuredOutputError("structured output did not resolve to an object")


def _response_model_version(payload: Mapping[str, Any], model: str) -> str:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("version"):
        return str(metadata["version"])
    if payload.get("version"):
        return str(payload["version"])
    match = re.search(r"(20\d{2}-\d{2}-\d{2})$", model)
    return match.group(1) if match else ""


def _content_filter_reason(payload: Mapping[str, Any]) -> Any:
    status = str(payload.get("status", "")).casefold()
    incomplete = payload.get("incomplete_details")
    reason = incomplete.get("reason") if isinstance(incomplete, Mapping) else None
    error = payload.get("error")
    error_text = json.dumps(error, ensure_ascii=False).casefold() if isinstance(error, Mapping) else str(error or "").casefold()
    if "content_filter" in status or "content_filter" in str(reason).casefold() or "content_filter" in error_text:
        return incomplete or error or payload.get("content_filter") or status
    if isinstance(payload.get("content_filter"), str) and payload["content_filter"]:
        return payload["content_filter"]
    return None


def _payload_transport_error(payload: Mapping[str, Any]) -> AzureRetryableError | None:
    status = payload.get("status_code", payload.get("status"))
    if status is None and isinstance(payload.get("error"), Mapping):
        status = payload["error"].get("status", payload["error"].get("status_code"))
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None
    if status_code is None or status_code < 400:
        return None
    headers = payload.get("headers")
    retry_after = payload.get("retry_after")
    if retry_after is None and isinstance(headers, Mapping):
        retry_after = headers.get("Retry-After", headers.get("retry-after"))
    try:
        retry_after_value = float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        retry_after_value = None
    message = payload.get("error", "Azure structured-output transport failure")
    return AzureRetryableError(str(message), status_code=status_code, retry_after=retry_after_value)


def _normalize_usage(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    usage = dict(value)
    if "input_tokens" not in usage and "prompt_tokens" in usage:
        usage["input_tokens"] = usage["prompt_tokens"]
    if "output_tokens" not in usage and "completion_tokens" in usage:
        usage["output_tokens"] = usage["completion_tokens"]
    return usage


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if isinstance(exc, AzureRetryableError):
        return exc.status_code is None or exc.status_code in RETRYABLE_STATUS_CODES
    name = type(exc).__name__
    if name in {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError"}:
        return True
    return name == "APIStatusError" and _status_code(exc) in RETRYABLE_STATUS_CODES


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
        if not isinstance(record, Mapping):
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

    def __init__(self, settings: AzureSettings, transport: Callable[..., Any] | None = None, *, cache: AzureCache | None = None) -> None:
        self.settings = settings
        self.transport = transport
        self.cache = cache

    def _default_transport(self, **kwargs: Any) -> Mapping[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the azure optional dependencies to call Azure") from exc
        if self.settings.auth_mode == "api_key":
            client = OpenAI(api_key=self.settings.api_key, base_url=self.settings.base_url, max_retries=0, timeout=AZURE_TRANSPORT_TIMEOUT_SECONDS)
        else:
            try:
                from azure.identity import DefaultAzureCredential, get_bearer_token_provider
            except ImportError as exc:
                raise RuntimeError("Install azure-identity for Entra ID authentication") from exc
            token_provider = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
            client = OpenAI(api_key=token_provider, base_url=self.settings.base_url, max_retries=0, timeout=AZURE_TRANSPORT_TIMEOUT_SECONDS)
        response = client.responses.create(model=self.settings.deployment, **kwargs)
        return _response_mapping(response)

    def _record(
        self,
        *,
        labels: Mapping[str, Any] | None,
        payload: Mapping[str, Any] | None,
        expected_version: str,
        prompt_hash: str,
        schema_hash: str,
        demonstration_manifest_hash: str | None,
        request_id: str,
        retry_count: int,
        cache_key: str | None,
        valid: bool,
        invalid_stage: str | None = None,
        invalid_reason: str | None = None,
        content_filter: Any = None,
    ) -> dict[str, Any]:
        payload = dict(payload or {})
        observed_model = str(payload.get("model") or self.settings.model_family)
        observed_version = _response_model_version(payload, observed_model) or expected_version
        return {
            "valid": valid,
            "labels": dict(labels) if labels is not None else None,
            "raw_response": payload or None,
            "invalid_stage": invalid_stage,
            "invalid_reason": invalid_reason,
            "request_id": payload.get("id", request_id),
            "response_id": payload.get("id", request_id),
            "deployment": self.settings.deployment,
            "expected_model_family": self.settings.model_family,
            "expected_model_version": expected_version,
            "observed_model": observed_model,
            "observed_model_version": observed_version,
            "prompt_hash": prompt_hash,
            "schema_hash": schema_hash,
            "demonstration_manifest_hash": demonstration_manifest_hash,
            "request_timestamp": datetime.now(UTC).isoformat(),
            "retry_count": retry_count,
            "usage": _normalize_usage(payload.get("usage", {})),
            "content_filter": content_filter,
            "cache_key": cache_key,
            "cache_hit": False,
        }

    def _cache_get(self, key: str | None, *, expected_version: str) -> dict[str, Any] | None:
        if key is None or self.cache is None:
            return None
        cached = self.cache.get(key, expected_model_family=self.settings.model_family, expected_model_version=expected_version)
        return dict(cached) | {"cache_hit": True, "retry_count": cached.get("retry_count", 0)} if cached is not None else None

    def _cache_put(self, key: str | None, record: Mapping[str, Any]) -> None:
        if key is not None and self.cache is not None:
            self.cache.put(key, record)

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
        cache_identity: Mapping[str, Any] | None = None,
        cache_key: str | None = None,
        output_validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        return_invalid: bool = False,
        terminal_invalid_stage: str = "structured_response",
        retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        if not schema.get("strict", True):
            raise ValueError("Structured Outputs must be strict")
        if not isinstance(schema.get("schema"), Mapping) or schema["schema"].get("additionalProperties") is not False:
            raise ValueError("Structured Outputs require an object schema with additionalProperties=false")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if retries < 0:
            raise ValueError("retries must be non-negative")
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
            "cache_identity": dict(cache_identity or {}),
        }
        resolved_cache_key = cache_key or (self.cache.key(identity) if self.cache else None)
        cached = self._cache_get(resolved_cache_key, expected_version=expected_version)
        if cached is not None:
            return cached
        transport = self.transport or self._default_transport
        request_id = str(uuid.uuid4())
        last_error: Exception | None = None
        retry_count = 0
        last_payload: dict[str, Any] = {}
        for attempt in range(retries + 1):
            try:
                payload = _response_mapping(transport(input=prompt, text={"format": {"type": "json_schema", "name": f"vipragsent_{task}", "strict": True, "schema": schema["schema"]}}, max_output_tokens=max_output_tokens, temperature=0, metadata={"request_id": request_id, "prompt_hash": prompt_hash, "schema_hash": schema_hash, "cache_key": resolved_cache_key, **dict(cache_identity or {})}))
                last_payload = payload
                transport_error = _payload_transport_error(payload)
                if transport_error is not None:
                    raise transport_error
                content_filter = _content_filter_reason(payload)
                if content_filter is not None:
                    raise AzureStructuredOutputError("Azure response was terminated by content filtering", payload=payload, content_filter=content_filter)
                parsed = extract_responses_structured_output(payload)
                try:
                    if task == "rationale":
                        labels = {"rationale": validate_rationale_output(dict(parsed))}
                    elif output_validator is not None:
                        labels = dict(output_validator(dict(parsed)))
                    else:
                        labels = validate_structured_output(dict(parsed), task)
                except (TypeError, ValueError) as exc:
                    raise AzureStructuredOutputError(str(exc), payload=payload) from exc
                observed_model = str(payload.get("model", ""))
                observed_version = _response_model_version(payload, observed_model)
                if observed_model and self.settings.model_family.casefold() not in observed_model.casefold():
                    raise AzureStructuredOutputError(f"Azure response model mismatch: {observed_model}", payload=payload)
                if observed_version and observed_version != expected_version:
                    raise AzureStructuredOutputError(f"Azure response version mismatch: {observed_version}", payload=payload)
                record = self._record(labels=labels, payload=payload, expected_version=expected_version, prompt_hash=prompt_hash, schema_hash=schema_hash, demonstration_manifest_hash=demonstration_manifest_hash, request_id=request_id, retry_count=retry_count, cache_key=resolved_cache_key, valid=True)
                self._cache_put(resolved_cache_key, record)
                return record
            except AzureStructuredOutputError as exc:
                if not return_invalid:
                    raise
                record = self._record(labels=None, payload=exc.payload or last_payload, expected_version=expected_version, prompt_hash=prompt_hash, schema_hash=schema_hash, demonstration_manifest_hash=demonstration_manifest_hash, request_id=request_id, retry_count=retry_count, cache_key=resolved_cache_key, valid=False, invalid_stage=terminal_invalid_stage, invalid_reason=str(exc), content_filter=exc.content_filter)
                self._cache_put(resolved_cache_key, record)
                return record
            except Exception as exc:
                if not _is_retryable(exc):
                    if not return_invalid:
                        raise
                    record = self._record(labels=None, payload=last_payload, expected_version=expected_version, prompt_hash=prompt_hash, schema_hash=schema_hash, demonstration_manifest_hash=demonstration_manifest_hash, request_id=request_id, retry_count=retry_count, cache_key=resolved_cache_key, valid=False, invalid_stage="judge_request", invalid_reason=str(exc))
                    self._cache_put(resolved_cache_key, record)
                    return record
                last_error = exc
                if attempt >= retries:
                    break
                retry_count += 1
                delay = _retry_after(exc)
                sleep(min(60.0, delay if delay is not None else (2, 4, 8, 16)[min(attempt, 3)]))
        if return_invalid:
            record = self._record(labels=None, payload=last_payload, expected_version=expected_version, prompt_hash=prompt_hash, schema_hash=schema_hash, demonstration_manifest_hash=demonstration_manifest_hash, request_id=request_id, retry_count=retry_count, cache_key=resolved_cache_key, valid=False, invalid_stage="judge_request", invalid_reason=str(last_error or "unknown Azure error"))
            self._cache_put(resolved_cache_key, record)
            return record
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
