from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .atomic import atomic_write_json


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
