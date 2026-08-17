from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

from vipragsent.azure.async_judge import (
    AsyncJudgePipeline,
    BudgetConfig,
    FileJudgeCache,
    JudgeCommit,
    QuotaConfig,
    RetryPolicy,
    reasoning_hash,
)

LABELS = ("implicit_sentiment", "sarcasm", "irony", "idiom_figurative", "code_switching", "mocking")
ROOT = Path(__file__).resolve().parents[1]


def _labels(value: int = 0) -> dict[str, int]:
    return {label: value for label in LABELS}


def _commit(sample_id: str, reasoning: str = "phân tích") -> JudgeCommit:
    return JudgeCommit.from_reasoning("run-a", "dev", sample_id, reasoning, "checkpoint-a")


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def _fixture_budget(**kwargs: object) -> BudgetConfig:
    """Explicitly permit legacy no-usage mock responses in fixture tests."""

    return BudgetConfig(allow_unknown_spend=True, **kwargs)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_async_budget_spend_ceiling_requires_finite_value(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        BudgetConfig(max_verified_spend_usd=value)
    with pytest.raises(ValueError, match="finite"):
        BudgetConfig.from_mapping({"max_verified_spend_usd": value})


@pytest.mark.parametrize(
    "field_name",
    [
        "max_logical_items",
        "max_requests",
        "max_tokens",
        "max_input_tokens",
        "max_output_tokens",
        "max_total_tokens",
        "max_concurrency",
    ],
)
def test_async_budget_positive_integer_ceilings_are_strict(field_name: str) -> None:
    for value in (True, 1.5, math.nan, math.inf, 0, -1):
        with pytest.raises(ValueError):
            BudgetConfig(**{field_name: value})
        with pytest.raises(ValueError):
            BudgetConfig.from_mapping({field_name: value})


@pytest.mark.parametrize("field_name", ["max_retry_per_request", "max_retries_per_request"])
def test_async_budget_retry_integer_ceilings_are_strict(field_name: str) -> None:
    for value in (True, 1.5, math.nan, math.inf, -1):
        with pytest.raises(ValueError):
            BudgetConfig(**{field_name: value})
        with pytest.raises(ValueError):
            BudgetConfig.from_mapping({field_name: value})


def test_async_budget_valid_mapping_keeps_integral_float_ceilings() -> None:
    budget = BudgetConfig.from_mapping({"max_total_tokens": 2048.0, "max_verified_spend_usd": 0.0, "max_retry_per_request": 0.0})
    assert budget.max_total_tokens == 2048
    assert budget.max_verified_spend_usd == 0.0
    assert budget.retry_ceiling == 0


def test_async_pipeline_rejects_nonfinite_budget_mapping() -> None:
    with pytest.raises(ValueError, match="finite"):
        AsyncJudgePipeline(ROOT, transport=lambda _payload: {"labels": _labels()}, budget={"max_verified_spend_usd": math.inf})


def test_cache_is_sample_independent_and_persisted(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        return {"labels": _labels(), "usage": {"input_tokens": 3, "output_tokens": 2}}

    pipeline = AsyncJudgePipeline(ROOT, transport=transport, cache=FileJudgeCache(tmp_path / "cache"), max_inflight=2)
    first = _run(pipeline.run([_commit("sample-a")]))
    second = _run(pipeline.run([_commit("sample-b")]))
    assert len(calls) == 1
    assert first.results[0].cache_hit is False
    assert second.results[0].cache_hit is True
    assert first.results[0].identity["sample_id"] == "sample-a"
    assert second.results[0].identity["sample_id"] == "sample-b"
    assert pipeline.cache_key("  phân tích\r\n") == pipeline.cache_key("phân tích\n")


def test_cache_key_lock_deduplicates_concurrent_requests(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []
    release = asyncio.Event()
    started = asyncio.Event()

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        calls.append(payload)
        started.set()
        await release.wait()
        return {"labels": _labels(1)}

    async def exercise() -> object:
        pipeline = AsyncJudgePipeline(ROOT, transport=transport, cache=FileJudgeCache(tmp_path / "cache"), max_inflight=2, budget=_fixture_budget())
        task = asyncio.create_task(pipeline.run([_commit("a"), _commit("b")]))
        await started.wait()
        assert len(calls) == 1
        release.set()
        return await task

    report = _run(exercise())
    assert len(calls) == 1
    assert report.telemetry["request_count"] == 1  # type: ignore[index]
    assert [item.cache_hit for item in report.results] == [False, True]  # type: ignore[union-attr]


def test_payload_is_reasoning_only_and_client_can_be_shared() -> None:
    payloads: list[dict[str, object]] = []

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        payloads.append(payload)
        return {"labels": _labels()}

    async def exercise() -> tuple[object, object]:
        left = AsyncJudgePipeline(ROOT, transport=transport, budget=_fixture_budget())
        right = AsyncJudgePipeline(ROOT, transport=transport, budget=_fixture_budget())
        return await asyncio.gather(left.run([_commit("left", "left reasoning")]), right.run([_commit("right", "right reasoning")]))  # type: ignore[return-value]

    _run(exercise())
    assert len(payloads) == 2
    for payload in payloads:
        assert set(payload) == {"model", "input", "temperature", "max_output_tokens", "text"}
        assert payload["model"] == "gpt-4.1-mini"
        assert payload["temperature"] == 0
        serialized = json.dumps(payload, ensure_ascii=False).casefold()
        assert "gold" not in serialized
        assert "sample_id" not in serialized
        assert "checkpoint" not in serialized


def test_backpressure_bounds_committed_unjudged_and_inflight() -> None:
    active = 0
    maximum_active = 0
    started = asyncio.Event()
    release = asyncio.Event()
    yielded = 0

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        started.set()
        await release.wait()
        active -= 1
        return {"labels": _labels()}

    async def commits():
        nonlocal yielded
        for index in range(5):
            yielded += 1
            yield _commit(str(index), f"reasoning-{index}")

    async def exercise() -> object:
        pipeline = AsyncJudgePipeline(ROOT, transport=transport, max_inflight=1, max_committed_unjudged=2, budget=_fixture_budget())
        task = asyncio.create_task(pipeline.run(commits()))
        await started.wait()
        await asyncio.sleep(0)
        assert yielded <= 2
        release.set()
        return await task

    report = _run(exercise())
    assert maximum_active == 1
    assert report.telemetry["request_count"] == 5  # type: ignore[index]


def test_retry_after_and_quota_are_bounded_and_telemetry_is_deterministic(tmp_path: Path) -> None:
    attempts: dict[str, int] = {}
    sleeps: list[float] = []
    now = 0.0

    def clock() -> float:
        return now

    async def sleep(seconds: float) -> None:
        nonlocal now
        sleeps.append(seconds)
        now += seconds

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        text = str(payload["input"])
        attempts[text] = attempts.get(text, 0) + 1
        if "one" in text and attempts[text] == 1:
            return {"status_code": 429, "retry_after": 0.25}
        return {"labels": _labels(), "usage": {"prompt_tokens": 2, "completion_tokens": 1}}

    pipeline = AsyncJudgePipeline(
        ROOT,
        transport=transport,
        cache=FileJudgeCache(tmp_path / "cache"),
        max_inflight=1,
        quota=QuotaConfig(requests_per_minute=2, maximum_wait_seconds=60),
        retry=RetryPolicy(maximum_total_attempts=2, maximum_delay_seconds=1),
        sleep=sleep,
        clock=clock,
    )
    report = _run(pipeline.run([_commit("a", "one"), _commit("b", "two")]))
    assert report.telemetry == {
        "logical_items": 2,
        "cache_hits": 0,
        "cache_misses": 2,
        "request_count": 3,
        "retry_count": 1,
        "input_tokens": 4,
        "output_tokens": 2,
        "failure_count": 0,
        "stale_result_count": 0,
        "quota_wait_seconds": 59.75,
    }
    assert sleeps == [0.25, 59.75]


def test_semantic_error_is_per_sample_and_not_retried() -> None:
    calls = 0

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"labels": {"sarcasm": 2}}

    pipeline = AsyncJudgePipeline(ROOT, transport=transport, retry=RetryPolicy(maximum_total_attempts=5), budget=_fixture_budget())
    report = _run(pipeline.run([_commit("bad")]))
    assert calls == 1
    assert report.results[0].status == "failed"  # type: ignore[union-attr]
    assert report.results[0].failure_reason  # type: ignore[union-attr]


def test_five_xx_is_retried_with_bounded_delay() -> None:
    calls = 0
    sleeps: list[float] = []

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"status_code": 503, "retry_after": 0.5}
        return {"labels": _labels()}

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    pipeline = AsyncJudgePipeline(ROOT, transport=transport, retry=RetryPolicy(maximum_total_attempts=2), budget=_fixture_budget(), sleep=sleep)
    report = _run(pipeline.run([_commit("server-error")]))
    assert calls == 2
    assert sleeps == [0.5]
    assert report.results[0].retry_count == 1  # type: ignore[union-attr]


def test_retryable_outage_is_not_reused_across_pipeline_runs(tmp_path: Path) -> None:
    calls = 0

    async def outage(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status_code": 503, "error": "temporarily unavailable"}

    first = AsyncJudgePipeline(ROOT, transport=outage, cache=FileJudgeCache(tmp_path / "cache"), retry=RetryPolicy(maximum_total_attempts=1))
    failed = _run(first.run([_commit("outage")]))
    assert failed.results[0].valid is False  # type: ignore[union-attr]

    async def recovery(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"labels": _labels(), "usage": {"input_tokens": 2, "output_tokens": 1}}

    second = AsyncJudgePipeline(ROOT, transport=recovery, cache=FileJudgeCache(tmp_path / "cache"), retry=RetryPolicy(maximum_total_attempts=1))
    recovered = _run(second.run([_commit("recovered")]))
    assert recovered.results[0].valid is True  # type: ignore[union-attr]
    assert recovered.results[0].cache_hit is False  # type: ignore[union-attr]
    assert calls == 2


def test_actual_usage_is_checked_after_estimate_admission() -> None:
    async def transport(payload: dict[str, object]) -> dict[str, object]:
        return {"labels": _labels(), "usage": {"input_tokens": 1, "output_tokens": 129}}

    pipeline = AsyncJudgePipeline(ROOT, transport=transport, budget=BudgetConfig(max_output_tokens=128, max_total_tokens=10_000))
    report = _run(pipeline.run([_commit("actual-budget")]))
    assert report.results[0].valid is False  # type: ignore[union-attr]
    assert "actual output tokens" in str(report.results[0].failure_reason)  # type: ignore[union-attr]


def test_concurrent_async_actual_overrun_latches_stop_and_telemetry() -> None:
    calls = 0
    release = asyncio.Event()

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            release.set()
        await release.wait()
        return {"labels": _labels(), "usage": {"input_tokens": 1, "output_tokens": 257}}

    pipeline = AsyncJudgePipeline(
        ROOT,
        transport=transport,
        max_inflight=2,
        max_committed_unjudged=2,
        retry=RetryPolicy(maximum_total_attempts=5),
        budget=BudgetConfig(max_output_tokens=256, max_total_tokens=10_000, max_requests=20, max_concurrency=2),
    )
    report = _run(pipeline.run([_commit("overrun-a", "reason-a"), _commit("overrun-b", "reason-b")]))

    assert calls == 2
    assert report.stopped_reason == "actual_output_token_ceiling_exceeded"
    assert report.telemetry["safety_tripped"] is True
    assert report.telemetry["safety_overrun_count"] == 1
    assert report.telemetry["safety_overrun_reason"] == "actual_output_token_ceiling_exceeded"
    assert report.telemetry["safety_stop_reason"] is not None


def test_async_missing_usage_fails_closed_by_default() -> None:
    calls = 0

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"labels": _labels()}

    pipeline = AsyncJudgePipeline(ROOT, transport=transport, max_inflight=1)
    report = _run(pipeline.run([_commit("missing-a"), _commit("missing-b")]))

    assert calls == 1
    assert report.stopped_reason == "unknown_usage"
    assert report.results[0].failure_reason == "actual usage has unknown spend"  # type: ignore[union-attr]
    assert report.telemetry["safety_tripped"] is True
    assert report.telemetry["unknown_spend_count"] == 1


def test_async_unknown_usage_opt_in_debits_the_reserved_ceiling() -> None:
    async def transport(payload: dict[str, object]) -> dict[str, object]:
        return {"labels": _labels()}

    pipeline = AsyncJudgePipeline(
        ROOT,
        transport=transport,
        budget=BudgetConfig(allow_unknown_spend=True, max_output_tokens=128, max_total_tokens=10_000),
    )
    report = _run(pipeline.run([_commit("unknown-opt-in")]))

    assert report.results[0].valid is True  # type: ignore[union-attr]
    assert report.results[0].spend_status == "UNKNOWN"  # type: ignore[union-attr]
    assert report.telemetry["unknown_spend_count"] == 1
    assert report.telemetry["accounted_output_tokens"] == 128
    assert report.telemetry["accounted_total_tokens"] >= 128


def test_stale_result_rejected_and_final_merge_is_ordered() -> None:
    release_first = asyncio.Event()
    calls: list[str] = []

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        text = str(payload["input"])
        calls.append(text)
        if "old reasoning" in text:
            await release_first.wait()
        return {"labels": _labels(1)}

    async def exercise() -> object:
        pipeline = AsyncJudgePipeline(ROOT, transport=transport, max_inflight=2, max_committed_unjudged=2, budget=_fixture_budget())
        old = _commit("same", "old reasoning")
        new = _commit("same", "new reasoning")
        task = asyncio.create_task(pipeline.run([old, new]))
        await asyncio.sleep(0)
        release_first.set()
        return await task

    report = _run(exercise())
    assert [item.status for item in report.results] == ["stale", "ok"]  # type: ignore[union-attr]
    assert [row["sample_id"] for row in report.final_rows] == ["same"]  # type: ignore[union-attr]
    assert report.telemetry["stale_result_count"] == 1  # type: ignore[index]
    assert report.results[0].identity["committed_reasoning_hash"] == reasoning_hash("old reasoning")  # type: ignore[union-attr]


def test_budget_stop_and_single_writer_finalizer() -> None:
    calls = 0
    finalizer_calls: list[tuple[tuple[dict[str, object], ...], dict[str, int | float]]] = []

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"labels": _labels()}

    def finalizer(rows: tuple[dict[str, object], ...], telemetry: dict[str, int | float]) -> None:
        finalizer_calls.append((rows, dict(telemetry)))

    pipeline = AsyncJudgePipeline(ROOT, transport=transport, budget=_fixture_budget(max_logical_items=2, max_requests=10, max_tokens=10_000), aggregate_finalizer=finalizer)
    report = _run(pipeline.run([_commit("0", "zero"), _commit("1", "one"), _commit("2", "two")]))
    assert calls == 2
    assert report.stopped_reason == "logical_budget_exhausted"
    assert len(report.results) == 3
    assert report.results[-1].failure_reason == "logical_budget_exhausted"  # type: ignore[union-attr]
    assert len(finalizer_calls) == 1
    assert [row["sample_id"] for row in finalizer_calls[0][0]] == ["0", "1", "2"]


def test_request_budget_stops_without_partial_extra_request() -> None:
    calls = 0

    async def transport(payload: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"labels": _labels()}

    pipeline = AsyncJudgePipeline(ROOT, transport=transport, max_inflight=1, budget=_fixture_budget(max_logical_items=3, max_requests=1, max_tokens=10_000))
    report = _run(pipeline.run([_commit("0", "zero"), _commit("1", "one")]))
    assert calls == 1
    assert report.results[1].failure_reason == "request_or_token_budget_exhausted"  # type: ignore[union-attr]
    assert report.stopped_reason == "request_or_token_budget_exhausted"


def test_commit_hash_and_checkpoint_identity_are_required() -> None:
    with pytest.raises(ValueError, match="committed_reasoning_hash"):
        JudgeCommit("r", "dev", "s", "reasoning", "wrong", "checkpoint")
    commit = _commit("s")
    assert set(commit.identity) == {"run_id", "split", "sample_id", "committed_reasoning_hash", "checkpoint_hash", "judge_identity"}
