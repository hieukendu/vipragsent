from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json, exclusive_lock
from ..hashing import sha256_json
from ..profiling import azure_successful_usage_cost
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


class AzureSafetyBudgetError(RuntimeError):
    """A finite Azure safety ceiling rejected a request or its actual usage."""


@dataclass(frozen=True)
class AzureSafetyCeilings:
    """Finite, process-wide ceilings for one production Azure execution."""

    max_logical_requests: int = 100_000
    max_transport_attempts: int = 500_000
    max_input_tokens: int = 50_000_000
    max_output_tokens: int = 50_000_000
    max_total_tokens: int = 100_000_000
    max_concurrency: int = 32
    max_verified_spend_usd: float = 100.0
    allow_unknown_spend: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            self.max_logical_requests,
            self.max_transport_attempts,
            self.max_input_tokens,
            self.max_output_tokens,
            self.max_total_tokens,
            self.max_concurrency,
        )
        if any(isinstance(value, bool) or int(value) <= 0 for value in integer_fields):
            raise ValueError("Azure safety ceilings must be positive")
        if float(self.max_verified_spend_usd) < 0:
            raise ValueError("max_verified_spend_usd must be non-negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> AzureSafetyCeilings:
        if not value:
            return cls()
        payload = dict(value)
        aliases = {"max_requests": "max_logical_requests", "max_attempts": "max_transport_attempts", "max_tokens": "max_total_tokens", "max_spend_usd": "max_verified_spend_usd"}
        for source, target in aliases.items():
            if source in payload and target not in payload:
                payload[target] = payload[source]
        return cls(**{field: payload[field] for field in cls.__dataclass_fields__ if field in payload})


@dataclass(frozen=True)
class _AzureAttemptReservation:
    input_estimate: int
    output_ceiling: int


class AzureSafetyLedger:
    """Thread-safe actual-usage ledger shared by all calls in one execution."""

    def __init__(self, ceilings: AzureSafetyCeilings | None = None) -> None:
        self.ceilings = ceilings or AzureSafetyCeilings()
        self.logical_requests = 0
        self.transport_attempts = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_tokens = 0
        self.verified_spend_usd = 0.0
        self.unknown_spend_count = 0
        self.active = 0
        self._reserved_input = 0
        self._reserved_output = 0
        self._lock = threading.RLock()

    def reserve_logical(self) -> None:
        with self._lock:
            if self.logical_requests >= self.ceilings.max_logical_requests:
                raise AzureSafetyBudgetError("Azure logical-request ceiling exhausted")
            self.logical_requests += 1

    def enter(self) -> None:
        with self._lock:
            if self.active >= self.ceilings.max_concurrency:
                raise AzureSafetyBudgetError("Azure concurrency ceiling exhausted")
            self.active += 1

    def leave(self) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)

    def reserve_attempt(self, input_estimate: int, output_ceiling: int) -> _AzureAttemptReservation:
        estimate = max(0, int(input_estimate))
        output = max(0, int(output_ceiling))
        with self._lock:
            if self.transport_attempts >= self.ceilings.max_transport_attempts:
                raise AzureSafetyBudgetError("Azure transport-attempt ceiling exhausted")
            if self.input_tokens + self._reserved_input + estimate > self.ceilings.max_input_tokens:
                raise AzureSafetyBudgetError("Azure input-token ceiling preflight failed")
            if self.output_tokens + self._reserved_output + output > self.ceilings.max_output_tokens:
                raise AzureSafetyBudgetError("Azure output-token ceiling preflight failed")
            if self.total_tokens + self._reserved_input + self._reserved_output + estimate + output > self.ceilings.max_total_tokens:
                raise AzureSafetyBudgetError("Azure total-token ceiling preflight failed")
            self.transport_attempts += 1
            self._reserved_input += estimate
            self._reserved_output += output
            return _AzureAttemptReservation(estimate, output)

    def cancel_attempt(self, reservation: _AzureAttemptReservation) -> None:
        with self._lock:
            self._reserved_input = max(0, self._reserved_input - reservation.input_estimate)
            self._reserved_output = max(0, self._reserved_output - reservation.output_ceiling)

    def commit_attempt(self, reservation: _AzureAttemptReservation, usage: Mapping[str, Any] | None) -> dict[str, Any]:
        cost = azure_successful_usage_cost(usage)
        input_tokens = cost.get("input_tokens")
        output_tokens = cost.get("output_tokens")
        with self._lock:
            self._reserved_input = max(0, self._reserved_input - reservation.input_estimate)
            self._reserved_output = max(0, self._reserved_output - reservation.output_ceiling)
            if input_tokens is None or output_tokens is None or cost.get("request_cost_usd") is None:
                self.unknown_spend_count += 1
                if not self.ceilings.allow_unknown_spend:
                    raise AzureSafetyBudgetError("Azure spend is unknown because response usage is incomplete")
                return {"spend_status": "UNKNOWN", **cost}
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
            total_tokens = input_tokens + output_tokens
            if self.input_tokens + input_tokens > self.ceilings.max_input_tokens:
                raise AzureSafetyBudgetError("Azure input-token ceiling exceeded by actual usage")
            if self.output_tokens + output_tokens > self.ceilings.max_output_tokens:
                raise AzureSafetyBudgetError("Azure output-token ceiling exceeded by actual usage")
            if self.total_tokens + total_tokens > self.ceilings.max_total_tokens:
                raise AzureSafetyBudgetError("Azure total-token ceiling exceeded by actual usage")
            spend = float(cost["request_cost_usd"])
            if self.verified_spend_usd + spend > self.ceilings.max_verified_spend_usd:
                raise AzureSafetyBudgetError("Azure verified-spend ceiling exceeded by actual usage")
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += total_tokens
            self.verified_spend_usd += spend
            return {"spend_status": "VERIFIED", **cost}


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
        if "output_text" in value and isinstance(value["output_text"], str | bytes | bytearray) and value["output_text"].strip():
            return value["output_text"]
        if "text" in value and isinstance(value["text"], str | bytes | bytearray):
            return value["text"]
        for key in ("output", "response", "content"):
            if key in value:
                candidate = _extract_structured_candidate(value[key])
                if candidate is not _MISSING:
                    return candidate
        return _MISSING
    if isinstance(value, list | tuple):
        for item in value:
            candidate = _extract_structured_candidate(item)
            if candidate is not _MISSING:
                return candidate
        return _MISSING
    if isinstance(value, str | bytes | bytearray):
        return value
    return _MISSING


def extract_responses_structured_output(payload: Any) -> Any:
    """Return the strict structured object from a real or fake Responses payload."""
    candidate = _extract_structured_candidate(_response_mapping(payload))
    if candidate is _MISSING:
        raise AzureStructuredOutputError("Responses API payload has no structured output", payload=_response_mapping(payload))
    if isinstance(candidate, bytes | bytearray):
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
    if isinstance(exc, TimeoutError | ConnectionError):
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
        if record.get("cacheable") is False or record.get("failure_kind") == "retryable_transport":
            return None
        # Records written by older clients did not distinguish terminal
        # validation failures from exhausted transport failures.  Never reuse
        # that ambiguous request-level failure as a production answer.
        if record.get("valid") is False and record.get("invalid_stage") == "judge_request":
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

    def __init__(
        self,
        settings: AzureSettings,
        transport: Callable[..., Any] | None = None,
        *,
        cache: AzureCache | None = None,
        safety: AzureSafetyCeilings | None = None,
        safety_ledger: AzureSafetyLedger | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.cache = cache
        self.safety = safety or AzureSafetyCeilings()
        self.safety_ledger = safety_ledger or AzureSafetyLedger(self.safety)

    def preflight_logical_requests(self, count: int) -> None:
        if count < 0:
            raise ValueError("logical request count must be non-negative")
        if self.safety_ledger.logical_requests + count > self.safety.max_logical_requests:
            raise AzureSafetyBudgetError("Azure logical-request ceiling preflight failed")

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
        """Execute one bounded logical request through the shared safety ledger."""
        self.safety_ledger.reserve_logical()
        self.safety_ledger.enter()
        try:
            return self._create_structured(
                prompt=prompt,
                task=task,
                schema=schema,
                max_output_tokens=max_output_tokens,
                sample_id=sample_id,
                input_payload=input_payload,
                demonstration_manifest_hash=demonstration_manifest_hash,
                expected_model_version=expected_model_version,
                cache_identity=cache_identity,
                cache_key=cache_key,
                output_validator=output_validator,
                return_invalid=return_invalid,
                terminal_invalid_stage=terminal_invalid_stage,
                retries=retries,
                sleep=sleep,
            )
        finally:
            self.safety_ledger.leave()

    def _create_structured(
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
            reservation = self.safety_ledger.reserve_attempt(max(1, len(prompt.split())), max_output_tokens)
            try:
                payload = _response_mapping(transport(input=prompt, text={"format": {"type": "json_schema", "name": f"vipragsent_{task}", "strict": True, "schema": schema["schema"]}}, max_output_tokens=max_output_tokens, temperature=0, metadata={"request_id": request_id, "prompt_hash": prompt_hash, "schema_hash": schema_hash, "cache_key": resolved_cache_key, **dict(cache_identity or {})}))
                last_payload = payload
                transport_error = _payload_transport_error(payload)
                if transport_error is not None:
                    raise transport_error
                # The estimate above is only admission control.  Actual
                # response usage is authoritative and is checked before any
                # result can be persisted or reused.
                safety_usage = self.safety_ledger.commit_attempt(reservation, _normalize_usage(payload.get("usage", {})))
                reservation = None
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
                record = self._record(labels=labels, payload=payload, expected_version=expected_version, prompt_hash=prompt_hash, schema_hash=schema_hash, demonstration_manifest_hash=demonstration_manifest_hash, request_id=request_id, retry_count=retry_count, cache_key=resolved_cache_key, valid=True) | safety_usage | {"cacheable": True, "failure_kind": None}
                self._cache_put(resolved_cache_key, record)
                return record
            except AzureSafetyBudgetError:
                raise
            except AzureStructuredOutputError as exc:
                if not return_invalid:
                    raise
                record = self._record(labels=None, payload=exc.payload or last_payload, expected_version=expected_version, prompt_hash=prompt_hash, schema_hash=schema_hash, demonstration_manifest_hash=demonstration_manifest_hash, request_id=request_id, retry_count=retry_count, cache_key=resolved_cache_key, valid=False, invalid_stage=terminal_invalid_stage, invalid_reason=str(exc), content_filter=exc.content_filter) | {"cacheable": True, "failure_kind": "terminal_semantic"}
                self._cache_put(resolved_cache_key, record)
                return record
            except Exception as exc:
                if reservation is not None:
                    self.safety_ledger.cancel_attempt(reservation)
                    reservation = None
                if not _is_retryable(exc):
                    if not return_invalid:
                        raise
                    record = self._record(labels=None, payload=last_payload, expected_version=expected_version, prompt_hash=prompt_hash, schema_hash=schema_hash, demonstration_manifest_hash=demonstration_manifest_hash, request_id=request_id, retry_count=retry_count, cache_key=resolved_cache_key, valid=False, invalid_stage="judge_request", invalid_reason=str(exc)) | {"cacheable": True, "failure_kind": "terminal_transport"}
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
            if last_error is not None and not _is_retryable(last_error):
                self._cache_put(resolved_cache_key, record | {"cacheable": True, "failure_kind": "terminal_transport"})
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
