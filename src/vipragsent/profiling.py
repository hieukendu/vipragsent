from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic import atomic_write_json

AZURE_COST_ACCOUNTING_METHOD = "USER_SUPPLIED_RATES_ACTUAL_SUCCESSFUL_USAGE"
AZURE_COST_VERIFICATION_STATUS = "LOCAL_USAGE_ACCOUNTING"
AZURE_USER_SUPPLIED_RATES_USD_PER_1M = {
    "input": 0.40,
    "cached_input": 0.10,
    "output": 1.60,
}


def _nonnegative_usage_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if numeric >= 0 else None


def _usage_value(usage: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _nonnegative_usage_int(usage.get(key))
        if value is not None:
            return value
    return None


def _nested_usage_value(usage: Mapping[str, Any], *keys: str) -> int | None:
    for parent in ("input_tokens_details", "prompt_tokens_details", "input_token_details", "prompt_token_details"):
        details = usage.get(parent)
        if isinstance(details, Mapping):
            value = _usage_value(details, *keys)
            if value is not None:
                return value
    return None


def azure_successful_usage_cost(usage: Mapping[str, Any] | None) -> dict[str, Any]:
    """Calculate one successful Azure response's local cost from reported usage.

    This intentionally does not estimate usage, multiply by retry attempts, or
    infer cost from request count.  Missing token usage remains unpriced.
    """

    payload = dict(usage) if isinstance(usage, Mapping) else {}
    input_tokens = _usage_value(payload, "input_tokens", "prompt_tokens")
    output_tokens = _usage_value(payload, "output_tokens", "completion_tokens")
    cached_input_tokens = _usage_value(payload, "cached_input_tokens", "cached_tokens")
    if cached_input_tokens is None:
        cached_input_tokens = _nested_usage_value(payload, "cached_input_tokens", "cached_tokens")
    cached_input_tokens = cached_input_tokens or 0
    result: dict[str, Any] = {
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "non_cached_input_tokens": max((input_tokens or 0) - cached_input_tokens, 0),
        "output_tokens": output_tokens,
        "cost_accounting_method": AZURE_COST_ACCOUNTING_METHOD,
        "cost_verification_status": AZURE_COST_VERIFICATION_STATUS,
    }
    if input_tokens is None or output_tokens is None:
        result.update({"cost_status": "USAGE_UNAVAILABLE", "request_cost_usd": None})
        return result
    input_cost = result["non_cached_input_tokens"] / 1_000_000 * AZURE_USER_SUPPLIED_RATES_USD_PER_1M["input"]
    cached_input_cost = cached_input_tokens / 1_000_000 * AZURE_USER_SUPPLIED_RATES_USD_PER_1M["cached_input"]
    output_cost = output_tokens / 1_000_000 * AZURE_USER_SUPPLIED_RATES_USD_PER_1M["output"]
    result.update({
        "cost_status": "USAGE_AVAILABLE",
        "non_cached_input_cost_usd": round(input_cost, 12),
        "cached_input_cost_usd": round(cached_input_cost, 12),
        "output_cost_usd": round(output_cost, 12),
        "request_cost_usd": round(input_cost + cached_input_cost + output_cost, 12),
    })
    return result


@dataclass(frozen=True)
class DeviceTelemetry:
    gpu_model: str | None
    mig_profile: str | None
    peak_vram_gb: float | None


@dataclass
class ProfileRecord:
    system: str
    successful_wall_seconds: float = 0.0
    successful_gpu_hours: float = 0.0
    failed_gpu_hours: float = 0.0
    retried_gpu_hours: float = 0.0
    peak_vram_gb: float | None = None
    gpu_model: str | None = None
    mig_profile: str | None = None
    trainable_parameters: int = 0
    batch1_latency_ms: float | None = None
    batch32_examples_per_second: float | None = None


class Profiler:
    def __init__(self, *, clock: Callable[[], float] = time.perf_counter, synchronize: Callable[[], None] | None = None) -> None:
        self.clock = clock
        self.synchronize = synchronize or (lambda: None)

    def measure_inference(self, fn: Callable[[int], None], *, examples: int, warmup_iterations: int = 50, repetitions: int = 3) -> dict[str, Any]:
        if examples < 500:
            raise ValueError("Production inference profiling requires at least 500 measured examples")
        for _ in range(warmup_iterations):
            fn(1)
        measurements: list[float] = []
        for _ in range(repetitions):
            self.synchronize()
            started = self.clock()
            fn(examples)
            self.synchronize()
            measurements.append((self.clock() - started) * 1000.0)
        ordered = sorted(measurements)
        return {"warmup_iterations": warmup_iterations, "measured_examples": examples, "repetitions": repetitions, "mean_ms": statistics.mean(measurements), "median_ms": statistics.median(measurements), "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))], "examples_per_second": examples / (statistics.mean(measurements) / 1000.0), "measurements_ms": measurements}

    @staticmethod
    def relative_cost(records: Iterable[ProfileRecord], denominator_system: str = "vipragsent_full_phobert") -> dict[str, float]:
        rows = list(records)
        denominator = next((record.successful_gpu_hours for record in rows if record.system == denominator_system), 0.0)
        if denominator <= 0:
            raise ValueError("Relative cost denominator must have measured successful GPU-hours")
        return {record.system: record.successful_gpu_hours / denominator for record in rows}


@dataclass
class AzureUsageLedger:
    rationale_requests: int = 0
    rationale_input_tokens: int = 0
    rationale_output_tokens: int = 0
    rationale_cost: float | None = None
    baseline_requests: int = 0
    baseline_input_tokens: int = 0
    baseline_output_tokens: int = 0
    baseline_cost: float | None = None

    def add(self, *, category: str, input_tokens: int, output_tokens: int, cost: float | None = None) -> None:
        if category == "rationale":
            self.rationale_requests += 1
            self.rationale_input_tokens += input_tokens
            self.rationale_output_tokens += output_tokens
            if cost is not None:
                self.rationale_cost = (self.rationale_cost or 0.0) + cost
        elif category == "baseline":
            self.baseline_requests += 1
            self.baseline_input_tokens += input_tokens
            self.baseline_output_tokens += output_tokens
            if cost is not None:
                self.baseline_cost = (self.baseline_cost or 0.0) + cost
        else:
            raise ValueError(f"Unknown Azure usage category: {category}")

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "rationale_cost_status": "available" if self.rationale_cost is not None else "monetary_cost_unavailable", "baseline_cost_status": "available" if self.baseline_cost is not None else "monetary_cost_unavailable"}


def validate_pricing_snapshot(snapshot: dict[str, Any]) -> None:
    required = {"currency", "effective_date", "pricing_source", "deployment_type", "rate_kind"}
    if not required.issubset(snapshot):
        raise ValueError(f"Pricing snapshot is missing {sorted(required - set(snapshot))}")
    if snapshot["rate_kind"] not in {"invoice", "estimate", "unavailable"}:
        raise ValueError("Pricing snapshot rate_kind must be invoice, estimate, or unavailable")


def write_usage_ledger(path: str | Path, ledger: AzureUsageLedger) -> None:
    atomic_write_json(path, ledger.as_dict())
