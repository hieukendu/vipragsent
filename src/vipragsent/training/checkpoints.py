"""Versioned checkpoint IO with explicit compatibility and load evidence."""

from __future__ import annotations

import gc
import os
import random
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..atomic import atomic_write_json
from ..orchestration.status import RuntimeBlocked

CHECKPOINT_SCHEMA_VERSION = 2
CANONICAL_CHECKPOINT_KEYS = (
    "schema_version",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "loss_aggregator_state_dict",
    "run_state",
    "rng_state",
    "metadata",
)


class CheckpointContractError(RuntimeBlocked):
    """A checkpoint cannot be proven to load the intended model."""


@dataclass(frozen=True)
class CheckpointLoadReport:
    path: str
    schema_version: int | None
    legacy_compatibility: bool
    status: str
    matched_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    allowed_missing_keys: tuple[str, ...]
    allowed_unexpected_keys: tuple[str, ...]
    matched_parameter_count: int
    expected_parameter_count: int
    matched_ratio: float
    required_head_prefixes: tuple[str, ...]
    missing_required_heads: tuple[str, ...]
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "schema_version": self.schema_version,
            "legacy_compatibility": self.legacy_compatibility,
            "status": self.status,
            "matched_keys": list(self.matched_keys),
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "allowed_missing_keys": list(self.allowed_missing_keys),
            "allowed_unexpected_keys": list(self.allowed_unexpected_keys),
            "matched_key_count": len(self.matched_keys),
            "missing_key_count": len(self.missing_keys),
            "unexpected_key_count": len(self.unexpected_keys),
            "matched_parameter_count": self.matched_parameter_count,
            "expected_parameter_count": self.expected_parameter_count,
            "matched_ratio": self.matched_ratio,
            "required_head_prefixes": list(self.required_head_prefixes),
            "missing_required_heads": list(self.missing_required_heads),
            "error": self.error,
        }


@dataclass(frozen=True)
class CheckpointLoadResult:
    """The validated payload and its persisted loading evidence."""

    payload: dict[str, Any]
    report: CheckpointLoadReport

    @property
    def run_state(self) -> Mapping[str, Any]:
        value = self.payload.get("run_state", {})
        return value if isinstance(value, Mapping) else {}


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().tolist(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [item.tolist() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if not state:
        return

    def as_tuple(value: Any) -> Any:
        return tuple(as_tuple(item) for item in value) if isinstance(value, list) else value

    if state.get("python") is not None:
        random.setstate(as_tuple(state["python"]))
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), numpy_state[2], numpy_state[3], numpy_state[4]))
    if state.get("torch") is not None:
        torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([torch.tensor(item, dtype=torch.uint8) for item in state["cuda"]])


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _value_count(value: Any) -> int:
    numel = getattr(value, "numel", None)
    return int(numel()) if callable(numel) else 1


def build_checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    loss_aggregator: nn.Module | None,
    run_state: Mapping[str, Any],
    *,
    metadata: Mapping[str, Any] | None = None,
    rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the only production checkpoint shape accepted by the loader."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "loss_aggregator_state_dict": loss_aggregator.state_dict() if loss_aggregator is not None else {},
        "run_state": dict(run_state),
        "rng_state": dict(rng_state or _rng_state()),
        "metadata": dict(metadata or {}),
    }


def save_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Persist a canonical v2 payload atomically."""
    path = Path(path)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointContractError("production checkpoints must use schema_version=2")
    missing_keys = [key for key in CANONICAL_CHECKPOINT_KEYS if key not in payload]
    if missing_keys:
        raise CheckpointContractError(f"checkpoint is missing canonical keys: {missing_keys}")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise CheckpointContractError("checkpoint model_state_dict is missing or empty")
    _atomic_torch_save(path, payload)
    return path


def _canonical_payload(raw: Any, *, allow_legacy_fixture: bool) -> tuple[dict[str, Any], bool]:
    if not isinstance(raw, Mapping):
        raise CheckpointContractError("checkpoint payload must be a mapping")
    payload = dict(raw)
    if payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION:
        missing_keys = [key for key in CANONICAL_CHECKPOINT_KEYS if key not in payload]
        if missing_keys:
            raise CheckpointContractError(f"checkpoint is missing canonical keys: {missing_keys}")
        state = payload.get("model_state_dict")
        if not isinstance(state, Mapping) or not state:
            raise CheckpointContractError("checkpoint model_state_dict is missing or empty")
        return payload, False
    if "model" in payload and allow_legacy_fixture:
        state = payload.get("model")
        if not isinstance(state, Mapping) or not state:
            raise CheckpointContractError("legacy checkpoint model state is missing or empty")
        return {
            "schema_version": 1,
            "model_state_dict": dict(state),
            "optimizer_state_dict": payload.get("optimizer"),
            "scheduler_state_dict": payload.get("scheduler"),
            "loss_aggregator_state_dict": payload.get("loss_aggregator", {}),
            "run_state": payload.get("state", {}),
            "rng_state": payload.get("rng_state", {}),
            "metadata": {"legacy_compatibility": True, "legacy_keys": sorted(payload)},
        }, True
    if "model" in payload:
        raise CheckpointContractError("legacy checkpoint requires explicit allow_legacy_fixture=True")
    if "model_state_dict" in payload:
        raise CheckpointContractError("unversioned model_state_dict checkpoint is not accepted")
    raise CheckpointContractError("checkpoint model state is missing")


def _matches(key: str, patterns: Sequence[str]) -> bool:
    return any(key == pattern or key.startswith(f"{pattern}.") for pattern in patterns)


def infer_required_head_prefixes(model: nn.Module) -> tuple[str, ...]:
    """Infer only task-head prefixes that are present in the target model."""
    keys = tuple(model.state_dict())
    config = getattr(model, "config", None)
    tasks = tuple(getattr(config, "active_tasks", ()) or ())
    prefixes: list[str] = []
    for task in tasks:
        candidates = (
            f"heads.{task}",
            f"{task}_head",
            f"classifier.{task}",
            f"classifier_{task}",
        )
        prefix = next((candidate for candidate in candidates if any(_matches(key, (candidate,)) for key in keys)), None)
        if prefix is not None:
            prefixes.append(prefix)
    return tuple(prefixes)


def _report_path(path: Path, report_path: str | Path | None) -> Path:
    return Path(report_path) if report_path is not None else path.with_suffix(".load_report.json")


def _write_report(path: Path, report: CheckpointLoadReport) -> None:
    atomic_write_json(path, report.as_dict())


def _failure_report(
    path: Path,
    *,
    schema_version: int | None,
    legacy: bool,
    expected_keys: Sequence[str] | Mapping[str, Any] = (),
    incoming_keys: Sequence[str] = (),
    required_head_prefixes: Sequence[str] = (),
    error: str,
) -> CheckpointLoadReport:
    expected_mapping = expected_keys if isinstance(expected_keys, Mapping) else {}
    expected = set(expected_mapping)
    incoming = set(incoming_keys)
    matched = sorted(expected & incoming)
    missing = sorted(expected - incoming)
    unexpected = sorted(incoming - expected)
    expected_count = sum(_value_count(value) for value in expected_mapping.values())
    matched_count = sum(_value_count(expected_mapping[key]) for key in matched)
    return CheckpointLoadReport(
        path=str(path),
        schema_version=schema_version,
        legacy_compatibility=legacy,
        status="FAIL",
        matched_keys=tuple(matched),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(unexpected),
        allowed_missing_keys=(),
        allowed_unexpected_keys=(),
        matched_parameter_count=matched_count,
        expected_parameter_count=expected_count,
        matched_ratio=(matched_count / expected_count if expected_count else 0.0),
        required_head_prefixes=tuple(required_head_prefixes),
        missing_required_heads=tuple(required_head_prefixes),
        error=error,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    loss_aggregator: nn.Module | None = None,
    restore_training_state: bool = False,
    allow_legacy_fixture: bool = False,
    allowed_missing_keys: Sequence[str] = (),
    allowed_unexpected_keys: Sequence[str] = (),
    required_head_prefixes: Sequence[str] = (),
    min_match_ratio: float = 0.5,
    report_path: str | Path | None = None,
    restore_rng: Callable[[Mapping[str, Any]], None] = _restore_rng_state,
) -> CheckpointLoadResult:
    """Load a checkpoint only after proving that it targets this model.

    ``allow_legacy_fixture`` is intentionally explicit. It is suitable for
    historical CPU fixtures and is never used by production writers.
    """
    path = Path(path)
    destination = _report_path(path, report_path)
    expected = model.state_dict()
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        payload, legacy = _canonical_payload(raw, allow_legacy_fixture=allow_legacy_fixture)
    except Exception as exc:
        report = _failure_report(path, schema_version=None, legacy=False, expected_keys=expected, error=str(exc))
        _write_report(destination, report)
        if isinstance(exc, CheckpointContractError):
            raise
        raise CheckpointContractError(f"unable to read checkpoint {path}: {exc}") from exc

    state = payload["model_state_dict"]
    incoming_keys = set(state)
    expected_keys = set(expected)
    matched = sorted(expected_keys & incoming_keys)
    raw_missing = sorted(expected_keys - incoming_keys)
    raw_unexpected = sorted(incoming_keys - expected_keys)
    allowed_missing = tuple(sorted(key for key in raw_missing if _matches(key, allowed_missing_keys)))
    allowed_unexpected = tuple(sorted(key for key in raw_unexpected if _matches(key, allowed_unexpected_keys)))
    missing = tuple(key for key in raw_missing if key not in allowed_missing)
    unexpected = tuple(key for key in raw_unexpected if key not in allowed_unexpected)
    missing_heads = tuple(
        prefix for prefix in required_head_prefixes if not any(_matches(key, (prefix,)) for key in incoming_keys)
    )
    expected_count = sum(_value_count(value) for value in expected.values())
    matched_count = sum(_value_count(expected[key]) for key in matched)
    matched_ratio = matched_count / expected_count if expected_count else 0.0
    report = CheckpointLoadReport(
        path=str(path),
        schema_version=int(payload.get("schema_version", 0)),
        legacy_compatibility=legacy,
        status="PASS",
        matched_keys=tuple(matched),
        missing_keys=missing,
        unexpected_keys=unexpected,
        allowed_missing_keys=allowed_missing,
        allowed_unexpected_keys=allowed_unexpected,
        matched_parameter_count=matched_count,
        expected_parameter_count=expected_count,
        matched_ratio=matched_ratio,
        required_head_prefixes=tuple(required_head_prefixes),
        missing_required_heads=missing_heads,
    )

    errors: list[str] = []
    if not expected_keys:
        errors.append("target model has no state keys")
    if not state:
        errors.append("checkpoint model state is empty")
    if not matched:
        errors.append("checkpoint has zero matching model keys")
    if missing:
        errors.append(f"missing model keys: {list(missing)}")
    if unexpected:
        errors.append(f"unexpected model keys: {list(unexpected)}")
    if missing_heads:
        errors.append(f"required task heads are absent: {list(missing_heads)}")
    if matched_ratio < min_match_ratio:
        errors.append(f"checkpoint match ratio {matched_ratio:.4f} is below {min_match_ratio:.4f}")
    if errors:
        report = replace(report, status="FAIL", error="; ".join(errors))
        _write_report(destination, report)
        raise CheckpointContractError("; ".join(errors))

    try:
        incompatible = model.load_state_dict(state, strict=False)
        loaded_missing = tuple(sorted(incompatible.missing_keys))
        loaded_unexpected = tuple(sorted(incompatible.unexpected_keys))
        if set(loaded_missing) != set(missing) or set(loaded_unexpected) != set(unexpected):
            raise CheckpointContractError("model loader reported keys different from preflight validation")
        if restore_training_state:
            if optimizer is not None and payload.get("optimizer_state_dict") is not None:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            if scheduler is not None and payload.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(payload["scheduler_state_dict"])
            if payload.get("rng_state"):
                restore_rng(payload["rng_state"])
        if loss_aggregator is not None:
            loss_aggregator.load_state_dict(payload.get("loss_aggregator_state_dict", {}))
    except Exception as exc:
        failed = replace(report, status="FAIL", error=str(exc))
        _write_report(destination, failed)
        if isinstance(exc, CheckpointContractError):
            raise
        raise CheckpointContractError(f"checkpoint state failed to load: {exc}") from exc

    _write_report(destination, report)
    return CheckpointLoadResult(payload=payload, report=report)


def release_model_resources(
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    loader: Any | None = None,
    *,
    clear_cuda_cache: bool = True,
) -> None:
    """Release references between one-at-a-time component executions."""
    if optimizer is not None:
        optimizer.state.clear()
    if loader is not None:
        close = getattr(loader, "close", None)
        if callable(close):
            close()
    if model is not None and not getattr(model, "_vipragsent_quantized", False):
        model.to("cpu")
    gc.collect()
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
