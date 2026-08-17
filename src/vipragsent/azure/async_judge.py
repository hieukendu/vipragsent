"""Future-only, bounded reasoning-judge execution.

This module deliberately has no default Azure transport.  A caller must inject
the transport, which receives a reasoning-only request.  The module is useful
for exercising the eventual asynchronous runner with a mock transport while
keeping the locked judge/cache contract in one place.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import time
import unicodedata
from collections import deque
from collections.abc import AsyncIterable, Awaitable, Callable, Iterable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

from ..atomic import atomic_write_json, exclusive_lock
from ..constants import PRAGMATIC_LABELS
from ..evaluation.reasoning_judge import load_reasoning_protocol, validate_judge_labels
from ..hashing import sha256_json
from ..profiling import azure_successful_usage_cost

JUDGE_LABELS = tuple(PRAGMATIC_LABELS)
LOCKED_MODEL = "gpt-4.1-mini"
LOCKED_MODEL_VERSION = "2025-04-14"
LOCKED_TEMPERATURE = 0
LOCKED_MAX_ATTEMPTS = 5
RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


class JudgeTransport(Protocol):
    def __call__(self, payload: Mapping[str, Any]) -> Any: ...


AsyncOrSyncCallable: TypeAlias = Callable[..., Any]


class JudgeTransportError(RuntimeError):
    """A mock transport failure with optional HTTP-style retry evidence."""

    def __init__(self, message: str, *, status_code: int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class JudgeSemanticError(ValueError):
    """A response that violates the locked six-key binary JSON contract."""


class BudgetExhausted(RuntimeError):
    """Raised internally when a finite request/token budget is exhausted."""


class QuotaExceeded(RuntimeError):
    """A quota wait would exceed its configured bounded wait."""


def normalize_reasoning(value: str) -> str:
    """Use the existing cache normalization: NFC, LF endings, outer trim."""

    return unicodedata.normalize("NFC", str(value).replace("\r\n", "\n").replace("\r", "\n")).strip()


def reasoning_hash(value: str) -> str:
    return hashlib.sha256(normalize_reasoning(value).encode("utf-8")).hexdigest()


def _identity_dict(commit: JudgeCommit) -> dict[str, str]:
    return {
        "run_id": commit.run_id,
        "split": commit.split,
        "sample_id": commit.sample_id,
        "committed_reasoning_hash": commit.committed_reasoning_hash,
        "checkpoint_hash": commit.checkpoint_hash,
        "judge_identity": commit.judge_identity,
    }


@dataclass(frozen=True, slots=True)
class JudgeCommit:
    """A committed generation that is safe to enqueue for judging.

    The reasoning hash is verified at construction time.  Sentence and gold
    labels are intentionally not fields: neither can cross the client boundary.
    """

    run_id: str
    split: str
    sample_id: str
    reasoning: str
    committed_reasoning_hash: str
    checkpoint_hash: str
    judge_identity: str = "reasoning_judge_gpt41mini_zeroshot_v1"

    def __post_init__(self) -> None:
        for name in ("run_id", "split", "sample_id", "committed_reasoning_hash", "checkpoint_hash", "judge_identity"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        observed = reasoning_hash(self.reasoning)
        if observed != self.committed_reasoning_hash:
            raise ValueError("committed_reasoning_hash does not match normalized reasoning")

    @classmethod
    def from_reasoning(
        cls,
        run_id: str,
        split: str,
        sample_id: str,
        reasoning: str,
        checkpoint_hash: str,
        *,
        judge_identity: str = "reasoning_judge_gpt41mini_zeroshot_v1",
    ) -> JudgeCommit:
        return cls(run_id, split, sample_id, reasoning, reasoning_hash(reasoning), checkpoint_hash, judge_identity)

    @property
    def identity(self) -> dict[str, str]:
        return _identity_dict(self)

    @property
    def identity_hash(self) -> str:
        return sha256_json(self.identity)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    maximum_total_attempts: int = LOCKED_MAX_ATTEMPTS
    maximum_delay_seconds: float = 60.0
    fallback_delays_seconds: tuple[float, ...] = (2.0, 4.0, 8.0, 16.0)

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_total_attempts <= LOCKED_MAX_ATTEMPTS:
            raise ValueError("maximum_total_attempts must be between 1 and the locked five attempts")
        if self.maximum_delay_seconds < 0 or any(delay < 0 for delay in self.fallback_delays_seconds):
            raise ValueError("retry delays must be non-negative")


@dataclass(frozen=True, slots=True)
class QuotaConfig:
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    window_seconds: float = 60.0
    maximum_wait_seconds: float = 60.0

    def __post_init__(self) -> None:
        for value in (self.requests_per_minute, self.tokens_per_minute):
            if value is not None and value <= 0:
                raise ValueError("quota limits must be positive when enabled")
        if self.window_seconds <= 0 or self.maximum_wait_seconds < 0:
            raise ValueError("quota window must be positive and maximum wait non-negative")


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """Finite safety budgets; caller limits remain upper bounds, never overrides."""

    max_logical_items: int = 100_000
    max_requests: int = 500_000
    max_tokens: int = 50_000_000
    max_input_tokens: int = 50_000_000
    max_output_tokens: int = 50_000_000
    max_total_tokens: int = 50_000_000
    max_concurrency: int = 32
    max_verified_spend_usd: float = 100.0
    allow_unknown_spend: bool = True

    def __post_init__(self) -> None:
        if any(isinstance(value, bool) or int(value) <= 0 for value in (self.max_logical_items, self.max_requests, self.max_tokens, self.max_input_tokens, self.max_output_tokens, self.max_concurrency)):
            raise ValueError("all judge budgets must be positive")
        if isinstance(self.max_total_tokens, bool) or self.max_total_tokens <= 0:
            raise ValueError("max_total_tokens must be positive")
        if self.max_verified_spend_usd < 0:
            raise ValueError("max_verified_spend_usd must be non-negative")

    @property
    def total_token_limit(self) -> int:
        return min(self.max_tokens, self.max_total_tokens)


@dataclass(frozen=True, slots=True)
class JudgeResult:
    ordinal: int
    identity: dict[str, str]
    status: str
    valid: bool
    labels: dict[str, int] | None
    cache_hit: bool
    attempts: int
    retry_count: int
    failure_reason: str | None
    input_tokens: int
    output_tokens: int
    quota_wait_seconds: float = 0.0
    spend_status: str = "UNKNOWN"

    @property
    def accepted(self) -> bool:
        return self.status == "ok"

    def as_row(self) -> dict[str, Any]:
        return {
            **self.identity,
            "status": self.status,
            "valid": self.valid,
            "labels": dict(self.labels) if self.labels is not None else None,
            "cache_hit": self.cache_hit,
            "attempts": self.attempts,
            "retry_count": self.retry_count,
            "failure_reason": self.failure_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "quota_wait_seconds": self.quota_wait_seconds,
            "spend_status": self.spend_status,
        }


@dataclass(frozen=True, slots=True)
class JudgeRunReport:
    results: tuple[JudgeResult, ...]
    final_rows: tuple[dict[str, Any], ...]
    telemetry: dict[str, int | float]
    stopped_reason: str | None = None


class FileJudgeCache:
    """Sample-independent cache with atomic, idempotent persistence.

    A cache record contains contract identity and reasoning hash, but no run,
    split, sample, sentence, or gold labels.  ``put`` uses the repository's
    exclusive lock and atomic replace primitives.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.root / f"{key}.json"

    def get(self, key: str, *, contract: Mapping[str, str], normalized_hash: str) -> dict[str, Any] | None:
        path = self.path_for(key)
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(record, Mapping) or record.get("cache_key") != key:
            return None
        if record.get("cacheable") is False or record.get("failure_kind") == "retryable_transport":
            return None
        if any(record.get(name) != value for name, value in contract.items()):
            return None
        if record.get("normalized_reasoning_sha256") != normalized_hash or "valid" not in record:
            return None
        return dict(record)

    def put(self, key: str, record: Mapping[str, Any]) -> Path:
        path = self.path_for(key)
        payload = dict(record)
        with exclusive_lock(path.with_suffix(".lock")):
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                existing = None
            if existing != payload:
                atomic_write_json(path, payload)
        return path


class _QuotaLimiter:
    def __init__(self, config: QuotaConfig, *, clock: Callable[[], float], sleep: Callable[[float], Awaitable[None] | None]) -> None:
        self.config = config
        self.clock = clock
        self.sleep = sleep
        self.events: deque[tuple[float, int]] = deque()
        self.lock = asyncio.Lock()

    async def acquire(self, tokens: int) -> float:
        if self.config.requests_per_minute is None and self.config.tokens_per_minute is None:
            return 0.0
        started = self.clock()
        waited = 0.0
        async with self.lock:
            while True:
                now = self.clock()
                while self.events and now - self.events[0][0] >= self.config.window_seconds:
                    self.events.popleft()
                request_count = len(self.events)
                token_count = sum(item[1] for item in self.events)
                request_ok = self.config.requests_per_minute is None or request_count < self.config.requests_per_minute
                token_ok = self.config.tokens_per_minute is None or token_count + tokens <= self.config.tokens_per_minute
                if request_ok and token_ok:
                    self.events.append((now, max(0, tokens)))
                    return max(0.0, self.clock() - started)
                if not self.events:
                    raise QuotaExceeded("quota cannot admit the estimated request")
                delay = max(0.0, self.events[0][0] + self.config.window_seconds - now)
                if waited + delay > self.config.maximum_wait_seconds:
                    raise QuotaExceeded("quota wait exceeded maximum_wait_seconds")
                waited += delay
                await _maybe_await(self.sleep(delay))


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _status_from_exception(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", getattr(exc, "status", None))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _retry_after_from_exception(exc: Exception) -> float | None:
    value = getattr(exc, "retry_after", None)
    if value is None:
        headers = getattr(exc, "headers", None)
        if isinstance(headers, Mapping):
            value = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def _transport_error(response: Any) -> JudgeTransportError | None:
    if not isinstance(response, Mapping):
        return None
    value = response.get("status_code", response.get("status"))
    try:
        status = int(value) if value is not None else None
    except (TypeError, ValueError):
        status = None
    if status is not None and status >= 400:
        retry_after = response.get("retry_after")
        try:
            retry_after = float(retry_after) if retry_after is not None else None
        except (TypeError, ValueError):
            retry_after = None
        return JudgeTransportError(str(response.get("error") or "judge transport failure"), status_code=status, retry_after=retry_after)
    return None


def _retryable(exc: Exception) -> bool:
    status = _status_from_exception(exc)
    return isinstance(exc, TimeoutError | ConnectionError) or status in RETRYABLE_STATUS_CODES


def _parse_labels(response: Any) -> dict[str, int]:
    candidate: Any = response
    if isinstance(response, Mapping):
        for key in ("parsed", "output", "response", "labels"):
            if key in response:
                candidate = response[key]
                break
        else:
            candidate = response.get("output_text", response.get("text", response))
    if isinstance(candidate, bytes):
        candidate = candidate.decode("utf-8")
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise JudgeSemanticError("judge response is not valid JSON") from exc
    if not isinstance(candidate, Mapping):
        raise JudgeSemanticError("judge response does not contain the strict six-key object")
    try:
        return validate_judge_labels(candidate)
    except (TypeError, ValueError) as exc:
        raise JudgeSemanticError(str(exc)) from exc


def _usage(response: Any) -> tuple[int, int]:
    usage = response.get("usage", {}) if isinstance(response, Mapping) else {}
    if not isinstance(usage, Mapping):
        return 0, 0
    try:
        input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    except (TypeError, ValueError):
        return 0, 0
    return max(0, input_tokens), max(0, output_tokens)


@dataclass(frozen=True)
class _RequestReservation:
    input_estimate: int
    output_ceiling: int


@dataclass
class _BudgetLedger:
    config: BudgetConfig
    logical_items: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    verified_spend_usd: float = 0.0
    unknown_spend_count: int = 0
    reserved_input: int = 0
    reserved_output: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def reserve_logical(self) -> bool:
        async with self.lock:
            if self.logical_items >= self.config.max_logical_items:
                return False
            self.logical_items += 1
            return True

    async def reserve_request(self, estimated_tokens: int, output_ceiling: int) -> _RequestReservation | None:
        async with self.lock:
            estimate = max(0, int(estimated_tokens))
            output = max(0, int(output_ceiling))
            if self.requests >= self.config.max_requests:
                return None
            if self.input_tokens + self.reserved_input + estimate > self.config.max_input_tokens:
                return None
            if self.output_tokens + self.reserved_output + output > self.config.max_output_tokens:
                return None
            if self.total_tokens + self.reserved_input + self.reserved_output + estimate + output > self.config.total_token_limit:
                return None
            self.requests += 1
            self.reserved_input += estimate
            self.reserved_output += output
            return _RequestReservation(estimate, output)

    async def cancel_request(self, reservation: _RequestReservation) -> None:
        async with self.lock:
            self.reserved_input = max(0, self.reserved_input - reservation.input_estimate)
            self.reserved_output = max(0, self.reserved_output - reservation.output_ceiling)

    async def commit_request(self, reservation: _RequestReservation, usage: Mapping[str, Any]) -> str:
        cost = azure_successful_usage_cost(usage)
        input_tokens = cost.get("input_tokens")
        output_tokens = cost.get("output_tokens")
        async with self.lock:
            self.reserved_input = max(0, self.reserved_input - reservation.input_estimate)
            self.reserved_output = max(0, self.reserved_output - reservation.output_ceiling)
            if input_tokens is None or output_tokens is None or cost.get("request_cost_usd") is None:
                self.unknown_spend_count += 1
                if not self.config.allow_unknown_spend:
                    raise BudgetExhausted("actual usage has unknown spend")
                return "UNKNOWN"
            input_tokens = int(input_tokens)
            output_tokens = int(output_tokens)
            total_tokens = input_tokens + output_tokens
            if self.input_tokens + input_tokens > self.config.max_input_tokens:
                raise BudgetExhausted("actual input tokens exceed budget")
            if self.output_tokens + output_tokens > self.config.max_output_tokens:
                raise BudgetExhausted("actual output tokens exceed budget")
            if self.total_tokens + total_tokens > self.config.total_token_limit:
                raise BudgetExhausted("actual total tokens exceed budget")
            spend = float(cost["request_cost_usd"])
            if self.verified_spend_usd + spend > self.config.max_verified_spend_usd:
                raise BudgetExhausted("actual verified spend exceeds budget")
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.total_tokens += total_tokens
            self.verified_spend_usd += spend
            return "VERIFIED"


class AsyncJudgePipeline:
    """Bounded async judge runner for an explicitly injected mock transport."""

    def __init__(
        self,
        root: str | Path,
        *,
        transport: JudgeTransport,
        cache: FileJudgeCache | None = None,
        max_inflight: int = 4,
        max_committed_unjudged: int = 16,
        quota: QuotaConfig | None = None,
        retry: RetryPolicy | None = None,
        budget: BudgetConfig | None = None,
        sleep: Callable[[float], Awaitable[None] | None] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        token_estimator: Callable[[str], int] | None = None,
        aggregate_finalizer: Callable[[tuple[dict[str, Any], ...], Mapping[str, int | float]], Any] | None = None,
    ) -> None:
        if transport is None:
            raise ValueError("an explicit mock transport is required; live Azure is disabled")
        if max_inflight <= 0 or max_committed_unjudged <= 0:
            raise ValueError("max_inflight and max_committed_unjudged must be positive")
        self.root = Path(root)
        self.protocol = load_reasoning_protocol(self.root)
        if str(self.protocol.get("judge_model", "")).casefold() != "gpt-4.1-mini" or str(self.protocol.get("judge_model_version")) != LOCKED_MODEL_VERSION:
            raise ValueError("async judge requires the locked GPT-4.1-mini model/version")
        if int(self.protocol.get("judge_temperature", -1)) != LOCKED_TEMPERATURE:
            raise ValueError("async judge requires temperature 0")
        if self.protocol.get("judge_input") != "generated_reasoning_only" or self.protocol.get("original_sentence_visible") is not False:
            raise ValueError("async judge requires the reasoning-only input contract")
        if self.protocol.get("semantic_repair") is not False:
            raise ValueError("semantic repair is not allowed")
        self.prompt_template = (self.root / str(self.protocol["judge_prompt_path"])).read_text(encoding="utf-8")
        self.schema = json.loads((self.root / str(self.protocol["judge_schema_path"])).read_text(encoding="utf-8"))
        if self.schema.get("additionalProperties") is not False or set(self.schema.get("required", [])) != set(JUDGE_LABELS):
            raise ValueError("async judge requires the strict six-key schema")
        self.transport = transport
        self.cache = cache
        self.max_inflight = max_inflight
        self.max_committed_unjudged = max_committed_unjudged
        self.quota = _QuotaLimiter(quota or QuotaConfig(), clock=clock, sleep=sleep)
        self.retry = retry or RetryPolicy()
        self.budget = budget or BudgetConfig()
        self.sleep = sleep
        self.clock = clock
        self.token_estimator = token_estimator or (lambda text: max(1, len(text.split())))
        self.aggregate_finalizer = aggregate_finalizer
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._key_lock_users: dict[str, int] = {}
        self._key_locks_guard = asyncio.Lock()

    @property
    def judge_identity(self) -> str:
        return str(self.protocol["judge_protocol_id"])

    def contract_for(self, judge_identity: str | None = None) -> dict[str, str]:
        return {
            "judge_protocol_id": judge_identity or self.judge_identity,
            "model": LOCKED_MODEL,
            "model_version": LOCKED_MODEL_VERSION,
            "temperature": str(LOCKED_TEMPERATURE),
            "prompt_hash": str(self.protocol["judge_prompt_hash"]),
            "schema_hash": str(self.protocol["judge_schema_hash"]),
        }

    @property
    def contract(self) -> dict[str, str]:
        return self.contract_for()

    def cache_key(self, reasoning: str, judge_identity: str | None = None) -> str:
        normalized = normalize_reasoning(reasoning)
        identity = "".join(((judge_identity or self.judge_identity), LOCKED_MODEL_VERSION, str(LOCKED_TEMPERATURE), self.contract["prompt_hash"], self.contract["schema_hash"], normalized))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _render(self, reasoning: str) -> str:
        return self.prompt_template.replace("{GENERATED_REASONING}", normalize_reasoning(reasoning))

    @asynccontextmanager
    async def _cache_key_lock(self, key: str) -> Any:
        async with self._key_locks_guard:
            lock = self._key_locks.setdefault(key, asyncio.Lock())
            self._key_lock_users[key] = self._key_lock_users.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            async with self._key_locks_guard:
                users = self._key_lock_users.get(key, 1) - 1
                if users <= 0 and self._key_locks.get(key) is lock:
                    self._key_locks.pop(key, None)
                    self._key_lock_users.pop(key, None)
                else:
                    self._key_lock_users[key] = users

    def _cache_get(self, key: str, normalized: str, judge_identity: str) -> dict[str, Any] | None:
        return None if self.cache is None else self.cache.get(key, contract=self.contract_for(judge_identity), normalized_hash=reasoning_hash(normalized))

    def _cache_put(self, key: str, normalized: str, record: Mapping[str, Any], judge_identity: str) -> None:
        if self.cache is None:
            return
        self.cache.put(key, {"cache_key": key, **self.contract_for(judge_identity), "normalized_reasoning_sha256": reasoning_hash(normalized), **dict(record)})

    def _payload(self, prompt: str) -> dict[str, Any]:
        # This is the complete client boundary.  No local identity, sentence,
        # or gold labels are sent, making one stateless mock client shareable.
        return {
            "model": LOCKED_MODEL,
            "input": prompt,
            "temperature": LOCKED_TEMPERATURE,
            "max_output_tokens": int(self.protocol["judge_max_output_tokens"]),
            "text": {"format": {"type": "json_schema", "name": "vipragsent_reasoning_judge", "strict": True, "schema": self.schema}},
        }

    async def _invoke(self, payload: Mapping[str, Any]) -> Any:
        return await _maybe_await(self.transport(payload))

    def _result_from_cache(self, ordinal: int, commit: JudgeCommit, cached: Mapping[str, Any]) -> JudgeResult:
        labels = cached.get("labels")
        valid = cached.get("valid") is True
        if labels is not None:
            labels = validate_judge_labels(labels)
        # Historical attempts/usage belong to the cache producer; a cache hit
        # contributes no new request or token telemetry for this run.
        return JudgeResult(ordinal, commit.identity, "ok" if valid else "failed", valid, labels, True, 0, 0, cached.get("failure_reason"), 0, 0, 0.0, str(cached.get("spend_status", "UNKNOWN")))

    async def _judge_one(self, ordinal: int, commit: JudgeCommit, ledger: _BudgetLedger) -> JudgeResult:
        normalized = normalize_reasoning(commit.reasoning)
        key = self.cache_key(normalized, commit.judge_identity)
        async with self._cache_key_lock(key):
            cached = self._cache_get(key, normalized, commit.judge_identity)
            if cached is not None:
                return self._result_from_cache(ordinal, commit, cached)
            if not normalized:
                record = {"valid": False, "labels": None, "failure_reason": "empty_reasoning", "attempts": 0, "retry_count": 0, "input_tokens": 0, "output_tokens": 0, "cacheable": True, "failure_kind": "terminal_semantic", "spend_status": "VERIFIED"}
                self._cache_put(key, normalized, record, commit.judge_identity)
                return self._result_from_cache(ordinal, commit, record)
            prompt = self._render(normalized)
            estimated_tokens = max(1, int(self.token_estimator(prompt)))
            attempts = 0
            retries = 0
            quota_wait = 0.0
            input_tokens = 0
            output_tokens = 0
            spend_status = "UNKNOWN"
            last_error: Exception | None = None
            payload = self._payload(prompt)
            for attempt in range(self.retry.maximum_total_attempts):
                reservation = await ledger.reserve_request(estimated_tokens, int(payload["max_output_tokens"]))
                if reservation is None:
                    return JudgeResult(ordinal, commit.identity, "failed", False, None, False, attempts, retries, "request_or_token_budget_exhausted", input_tokens, output_tokens, quota_wait)
                try:
                    quota_wait += await self.quota.acquire(estimated_tokens)
                    response = await self._invoke(payload)
                    attempts += 1
                    transport_error = _transport_error(response)
                    if transport_error is not None:
                        raise transport_error
                    input_tokens, output_tokens = _usage(response)
                    spend_status = await ledger.commit_request(reservation, response.get("usage", {}) if isinstance(response, Mapping) else {})
                    reservation = None
                    labels = _parse_labels(response)
                    record = {"valid": True, "labels": labels, "failure_reason": None, "attempts": attempts, "retry_count": retries, "input_tokens": input_tokens, "output_tokens": output_tokens, "cacheable": True, "failure_kind": None, "spend_status": spend_status}
                    self._cache_put(key, normalized, record, commit.judge_identity)
                    return JudgeResult(ordinal, commit.identity, "ok", True, labels, False, attempts, retries, None, input_tokens, output_tokens, quota_wait, spend_status)
                except JudgeSemanticError as exc:
                    if reservation is not None:
                        await ledger.cancel_request(reservation)
                        reservation = None
                    attempts = max(attempts, 1)
                    last_error = exc
                    break
                except BudgetExhausted as exc:
                    # Actual usage is authoritative; never retry or cache an
                    # over-budget response, even when the estimate admitted it.
                    reservation = None
                    attempts = max(attempts, 1)
                    return JudgeResult(ordinal, commit.identity, "failed", False, None, False, attempts, retries, str(exc), input_tokens, output_tokens, quota_wait)
                except QuotaExceeded as exc:
                    await ledger.cancel_request(reservation)
                    reservation = None
                    return JudgeResult(ordinal, commit.identity, "failed", False, None, False, attempts, retries, str(exc), input_tokens, output_tokens, quota_wait)
                except Exception as exc:  # transport errors are terminal after bounded retries
                    await ledger.cancel_request(reservation)
                    reservation = None
                    attempts = max(attempts, 1)
                    last_error = exc
                    if not _retryable(exc) or attempt + 1 >= self.retry.maximum_total_attempts:
                        break
                    retries += 1
                    delay = _retry_after_from_exception(exc)
                    if delay is None:
                        delay = self.retry.fallback_delays_seconds[min(attempt, len(self.retry.fallback_delays_seconds) - 1)]
                    await _maybe_await(self.sleep(min(self.retry.maximum_delay_seconds, delay)))
            reason = str(last_error or "judge request failed")
            record = {"valid": False, "labels": None, "failure_reason": reason, "attempts": attempts, "retry_count": retries, "input_tokens": input_tokens, "output_tokens": output_tokens, "cacheable": True, "failure_kind": "terminal_semantic" if isinstance(last_error, JudgeSemanticError) else "terminal_transport", "spend_status": spend_status}
            if last_error is None or not _retryable(last_error):
                self._cache_put(key, normalized, record, commit.judge_identity)
            return JudgeResult(ordinal, commit.identity, "failed", False, None, False, attempts, retries, reason, input_tokens, output_tokens, quota_wait)

    @staticmethod
    def _budget_result(ordinal: int, commit: JudgeCommit) -> JudgeResult:
        return JudgeResult(ordinal, commit.identity, "failed", False, None, False, 0, 0, "logical_budget_exhausted", 0, 0)

    async def run(self, commits: Iterable[JudgeCommit] | AsyncIterable[JudgeCommit]) -> JudgeRunReport:
        """Consume commits with bounded queue/inflight state and deterministic merge."""

        queue: asyncio.Queue[tuple[int, JudgeCommit] | None] = asyncio.Queue(maxsize=self.max_committed_unjudged)
        result_queue: asyncio.Queue[JudgeResult | None] = asyncio.Queue(maxsize=self.max_committed_unjudged)
        slots = asyncio.Semaphore(self.max_committed_unjudged)
        ledger = _BudgetLedger(self.budget)
        submitted: list[JudgeCommit] = []
        results: dict[int, JudgeResult] = {}
        latest: dict[tuple[str, str, str], str] = {}
        stopped_reason: str | None = None
        worker_count = min(self.max_inflight, self.max_committed_unjudged, self.budget.max_concurrency)

        async def produce() -> None:
            nonlocal stopped_reason
            ordinal = 0
            is_async = isinstance(commits, AsyncIterable) or hasattr(commits, "__aiter__")

            async def enqueue(commit: JudgeCommit, *, slot_owned: bool) -> None:
                nonlocal ordinal, stopped_reason
                submitted.append(commit)
                key = (commit.run_id, commit.split, commit.sample_id)
                latest[key] = commit.identity_hash
                if not await ledger.reserve_logical():
                    stopped_reason = stopped_reason or "logical_budget_exhausted"
                    results[ordinal] = self._budget_result(ordinal, commit)
                    if slot_owned:
                        slots.release()
                    ordinal += 1
                    return
                if not slot_owned:
                    await slots.acquire()
                await queue.put((ordinal, commit))
                ordinal += 1

            if is_async:
                iterator = commits.__aiter__()  # type: ignore[union-attr]
                while True:
                    # Acquire before __anext__ so an async source cannot
                    # produce one hidden extra committed item above the HWM.
                    await slots.acquire()
                    try:
                        commit = await iterator.__anext__()
                    except StopAsyncIteration:
                        slots.release()
                        break
                    await enqueue(commit, slot_owned=True)
                    if stopped_reason:
                        break
            else:
                iterator = iter(commits)
                while True:
                    if stopped_reason:
                        # A synchronous fixture is finite; preserve one result
                        # per supplied item without making more requests.
                        try:
                            remainder = next(iterator)
                        except StopIteration:
                            break
                        submitted.append(remainder)
                        results[ordinal] = self._budget_result(ordinal, remainder)
                        ordinal += 1
                        continue
                    await slots.acquire()
                    try:
                        commit = next(iterator)
                    except StopIteration:
                        slots.release()
                        break
                    await enqueue(commit, slot_owned=True)
            for _ in range(worker_count):
                await queue.put(None)

        async def worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    await result_queue.put(None)
                    return
                ordinal, commit = item
                try:
                    result = await self._judge_one(ordinal, commit, ledger)
                except Exception as exc:
                    result = JudgeResult(ordinal, commit.identity, "failed", False, None, False, 0, 0, f"pipeline_error: {exc}", 0, 0)
                await result_queue.put(result)
                queue.task_done()

        async def finalize() -> None:
            finished_workers = 0
            while finished_workers < worker_count:
                item = await result_queue.get()
                if item is None:
                    finished_workers += 1
                else:
                    results[item.ordinal] = item
                    slots.release()
                result_queue.task_done()

        producer = asyncio.create_task(produce())
        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        finalizer = asyncio.create_task(finalize())
        await producer
        await queue.join()
        await asyncio.gather(*workers)
        await result_queue.join()
        await finalizer

        ordered = []
        for ordinal in sorted(results):
            result = results[ordinal]
            commit = submitted[ordinal]
            slot = (commit.run_id, commit.split, commit.sample_id)
            if slot in latest and latest[slot] != commit.identity_hash:
                result = JudgeResult(result.ordinal, result.identity, "stale", False, None, result.cache_hit, result.attempts, result.retry_count, "stale_result_identity", result.input_tokens, result.output_tokens, result.quota_wait_seconds)
            ordered.append(result)

        # Last current result wins for duplicate sample slots; order is still
        # stable by first occurrence, independent of completion timing.
        merged: dict[tuple[str, str, str], JudgeResult] = {}
        for result in ordered:
            if result.status == "stale":
                continue
            slot = (result.identity["run_id"], result.identity["split"], result.identity["sample_id"])
            merged[slot] = result
        final_results = tuple(sorted(merged.values(), key=lambda item: item.ordinal))
        telemetry: dict[str, int | float] = {
            "logical_items": len(submitted),
            "cache_hits": sum(int(item.cache_hit) for item in ordered),
            "cache_misses": sum(int(not item.cache_hit) for item in ordered if item.failure_reason != "logical_budget_exhausted"),
            "request_count": sum(item.attempts for item in ordered),
            "retry_count": sum(item.retry_count for item in ordered),
            "input_tokens": sum(item.input_tokens for item in ordered),
            "output_tokens": sum(item.output_tokens for item in ordered),
            "failure_count": sum(int(not item.valid) for item in final_results),
            "stale_result_count": sum(int(item.status == "stale") for item in ordered),
            "quota_wait_seconds": sum(item.quota_wait_seconds for item in ordered),
        }
        if stopped_reason is None:
            if any(item.failure_reason == "request_or_token_budget_exhausted" for item in ordered):
                stopped_reason = "request_or_token_budget_exhausted"
            elif any(item.failure_reason == "quota wait exceeded maximum_wait_seconds" for item in ordered):
                stopped_reason = "quota_exhausted"
        rows = tuple(item.as_row() for item in final_results)
        if self.aggregate_finalizer is not None:
            await _maybe_await(self.aggregate_finalizer(rows, telemetry))
        return JudgeRunReport(tuple(ordered), rows, telemetry, stopped_reason)


# Names that make the future-only boundary explicit to callers and tests.
AzureAsyncJudgePipeline = AsyncJudgePipeline
BoundedAsyncJudgePipeline = AsyncJudgePipeline
